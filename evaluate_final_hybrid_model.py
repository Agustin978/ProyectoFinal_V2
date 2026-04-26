import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, roc_curve, auc
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from src.models.densenet import get_model
from src.utils.versioning import get_next_version

# =============================================================================
# CONFIGURACIÓN  (única sección a editar por el usuario)
# =============================================================================
DATA_DIR     = r"D:\Agustin\Facultad\ProyectoFinal\archive"
MODEL_DIR    = r"C:\Projects\ProyectoFinal_V2"   # directorio donde viven los .pth
TEST_SET_CSV = "holdout_test_set.csv"
#TEST_SET_CSV = "train_validation_set_40.csv"

CALIB_RATIO  = 0.50   # 50 % calibración → 50 % test final
RANDOM_SEED  = 42
NUM_CLASSES  = 14
IMAGE_SIZE   = 224

# Alias → nombre de archivo del modelo dentro de MODEL_DIR
MODELS = {
    'Base':     'best_model.pth',
    '20260402': 'best_model_20260402_161254.pth',
    '20260423': 'best_model_20260423_114736.pth',
    'T1':       'best_model_T1.pth',
    'T1_V2':    'best_model_T1_V2.pth',
    'T2':       'best_model_T2.pth',
    'T2_V2':    'best_model_T2_V2.pth',
    'T3':       'best_model_T3.pth',
    'T4':       'best_model_T4.pth',
    'V1':       'best_model_V1.pth',
    'V1_3':     'best_model_V1_3.pth',
}

# Override manual: fuerza el modelo indicado para esa patología, ignorando el
# torneo de Youden. El umbral de ese modelo sigue calibrándose en el calib set.
# Dejar vacío para que todas las patologías pasen por el Expert Router automático.
# Ejemplo: 'Pleural_Thickening': '20260423'
MANUAL_OVERRIDES = {
    'Atelectasis': 'T2_V2',
    'Cardiomegaly': 'T4',
    'Effusion': 'T1_V2',
    'Infiltration': 'T2',
    'Mass': 'T1',
    'Nodule': 'T3',
    'Pneumonia': 'T1',
    'Pneumothorax': '20260402',
    'Consolidation': 'T4',
    'Edema': 'Base',
    'Emphysema': 'T3',
    'Fibrosis': 'T3',
    'Pleural_Thickening': '20260423',
    'Hernia': 'V1_3',
}

ALL_LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
    'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema',
    'Fibrosis', 'Pleural_Thickening', 'Hernia',
]

OUTPUT_ROUTING_CSV     = get_next_version("final_hybrid_routing.csv")
OUTPUT_METRICS_CSV     = get_next_version("final_hybrid_metrics.csv")
OUTPUT_PREDICTIONS_CSV = get_next_version("final_hybrid_predictions.csv")
OUTPUT_ROC_PNG         = get_next_version("final_hybrid_roc_calib.png")

# =============================================================================
# UTILIDADES
# =============================================================================

def load_environment():
    try:
        import torch_directml
        if torch_directml.is_available():
            return torch_directml.device(), "dml (AMD GPU)"
    except (ImportError, Exception):
        pass
    if torch.cuda.is_available():
        return torch.device('cuda'), f"cuda ({torch.cuda.get_device_name(0)})"
    return torch.device('cpu'), "cpu"


def load_safe(filepath, model, device):
    """Carga pesos con parche automático Sequential → Linear."""
    data = torch.load(filepath, map_location='cpu', weights_only=False)
    state_dict = data['model_state_dict'] if (isinstance(data, dict) and 'model_state_dict' in data) else data

    if 'classifier.1.weight' in state_dict:
        state_dict['classifier.weight'] = state_dict.pop('classifier.1.weight')
        state_dict['classifier.bias']   = state_dict.pop('classifier.1.bias')
        extras = [k for k in list(state_dict.keys())
                  if k.startswith('classifier.') and k not in ('classifier.weight', 'classifier.bias')]
        for k in extras:
            del state_dict[k]

    model.load_state_dict(state_dict, strict=False)
    return model.to(device).eval()


def get_image_paths(data_dir):
    paths = {}
    search = os.path.join(data_dir, 'images')
    if not os.path.exists(search):
        search = data_dir
    for root, _, files in os.walk(search):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                paths[f] = os.path.join(root, f)
    return paths


def labels_to_vec(labels_str):
    vec = np.zeros(NUM_CLASSES, dtype=int)
    if pd.isna(labels_str) or labels_str == 'No Finding':
        return vec
    for i, label in enumerate(ALL_LABELS):
        if label in labels_str:
            vec[i] = 1
    return vec


def vec_to_labels(vec):
    found = [ALL_LABELS[i] for i in range(len(vec)) if vec[i] == 1]
    return " | ".join(found) if found else "No Finding"


def compute_youden_metrics(y_true, y_prob):
    """
    Retorna (fpr, tpr, best_threshold, best_fpr_pt, best_tpr_pt, roc_auc, youden_j).
    Devuelve (None, None, 0.5, 0, 0, 0.0, 0.0) si la clase no tiene positivos
    o todos son positivos (ROC indefinida).
    """
    if sum(y_true) == 0 or sum(y_true) == len(y_true):
        return None, None, 0.5, 0.0, 0.0, 0.0, 0.0
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    J = tpr + (1 - fpr) - 1
    idx = int(np.argmax(J))
    best_thresh = float(np.clip(thresholds[idx], 0.05, 0.95))
    roc_auc_val = float(auc(fpr, tpr))
    return fpr, tpr, best_thresh, float(fpr[idx]), float(tpr[idx]), roc_auc_val, float(J[idx])


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 65)
    print("  HÍBRIDO FINAL — 11 MODELOS CON SPLIT CALIBRACIÓN / TEST")
    print("=" * 65)
    print(f"  Split: {int(CALIB_RATIO*100)}% calibración  /  {int((1-CALIB_RATIO)*100)}% test final")
    print(f"  Overrides manuales: {list(MANUAL_OVERRIDES.keys()) or 'ninguno'}\n")

    device, device_name = load_environment()
    print(f"Dispositivo: {device_name}")

    # ------------------------------------------------------------------
    # 1. Cargar holdout, hacer shuffle determinista y dividir
    # ------------------------------------------------------------------
    if not os.path.exists(TEST_SET_CSV):
        print(f"Error: No se encontró {TEST_SET_CSV}")
        return

    df_full = pd.read_csv(TEST_SET_CSV).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    calib_size = int(len(df_full) * CALIB_RATIO)

    print(f"\nHoldout total : {len(df_full)} imágenes")
    print(f"Calibración   : {calib_size} imágenes  (~{calib_size // NUM_CLASSES} por patología)")
    print(f"Test final    : {len(df_full) - calib_size} imágenes  (~{(len(df_full) - calib_size) // NUM_CLASSES} por patología)\n")

    # ------------------------------------------------------------------
    # 2. Indexar imágenes en disco
    # ------------------------------------------------------------------
    print(f"Indexando imágenes en {DATA_DIR} ...")
    image_paths_dict = get_image_paths(DATA_DIR)
    if not image_paths_dict:
        print("Error: No se encontraron imágenes.")
        return
    print(f"  {len(image_paths_dict)} imágenes indexadas.\n")

    # ------------------------------------------------------------------
    # 3. Cargar modelos
    # ------------------------------------------------------------------
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    sigmoid = nn.Sigmoid()

    loaded_models = {}
    print("Cargando modelos...")
    for alias, filename in MODELS.items():
        filepath = os.path.join(MODEL_DIR, filename)
        if not os.path.exists(filepath):
            print(f"  [WARN] No encontrado: {filename}  →  se omite alias '{alias}'")
            continue
        m = get_model(num_classes=NUM_CLASSES, pretrained=False)
        loaded_models[alias] = load_safe(filepath, m, device)
        print(f"  [OK]   {alias:<12}  ←  {filename}")

    if not loaded_models:
        print("Error: No se cargó ningún modelo.")
        return

    model_names  = list(loaded_models.keys())
    model_colors = plt.cm.tab20(np.linspace(0, 1, len(model_names)))

    # ------------------------------------------------------------------
    # 4. Inferencia única sobre TODO el holdout (evita cargar imgs dos veces)
    # ------------------------------------------------------------------
    print("\nEjecutando inferencia sobre el holdout completo...")

    y_true_list       = []
    img_names_list    = []
    true_labels_list  = []
    probs_per_model   = {alias: [] for alias in model_names}

    with torch.no_grad():
        for _, row in df_full.iterrows():
            img_name  = row['Image Index']
            label_str = row.get('Original_Finding_Labels', row.get('Finding Labels', 'No Finding'))

            if img_name not in image_paths_dict:
                continue

            img    = Image.open(image_paths_dict[img_name]).convert('RGB')
            tensor = transform(img).unsqueeze(0).to(device)

            y_true_list.append(labels_to_vec(label_str))
            img_names_list.append(img_name)
            true_labels_list.append(label_str)

            for alias, model in loaded_models.items():
                probs = sigmoid(model(tensor)).cpu().squeeze().numpy()
                probs_per_model[alias].append(probs)

    y_all      = np.array(y_true_list)
    probs_all  = {alias: np.array(v) for alias, v in probs_per_model.items()}
    n_total    = len(y_all)
    calib_size = min(calib_size, n_total)

    # Dividir en calibración / test
    y_calib    = y_all[:calib_size]
    y_test     = y_all[calib_size:]
    probs_calib = {alias: probs_all[alias][:calib_size] for alias in model_names}
    probs_test  = {alias: probs_all[alias][calib_size:]  for alias in model_names}
    img_names_test   = img_names_list[calib_size:]
    true_labels_test = true_labels_list[calib_size:]

    print(f"  Inferencia completada sobre {n_total} imágenes válidas.")

    # ------------------------------------------------------------------
    # 5. Fase de calibración: selección de modelo + umbral por patología
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("  FASE DE CALIBRACIÓN")
    print("=" * 65)
    print(f"  {'Patología':<22} {'Estrategia':<30} {'Umbral':>7} {'J-calib':>8}")
    print("  " + "-" * 61)

    fig, axes = plt.subplots(4, 4, figsize=(26, 26))
    axes = axes.flatten()

    routing     = []
    final_preds = np.zeros_like(y_test)

    for p_idx, pathology in enumerate(ALL_LABELS):
        y_t = y_calib[:, p_idx]
        ax  = axes[p_idx]

        # Determinar modo de routing para esta patología
        use_override = (pathology in MANUAL_OVERRIDES) and (MANUAL_OVERRIDES[pathology] in loaded_models)

        if use_override:
            # --- OVERRIDE MANUAL -------------------------------------------
            forced_alias = MANUAL_OVERRIDES[pathology]
            _, _, best_thresh, best_fpr_pt, best_tpr_pt, roc_auc_val, j_score = \
                compute_youden_metrics(y_t, probs_calib[forced_alias][:, p_idx])
            winner_alias = forced_alias
            strategy     = f"Override ({forced_alias})"

            for i, alias in enumerate(model_names):
                fpr_m, tpr_m, *_ = compute_youden_metrics(y_t, probs_calib[alias][:, p_idx])
                if fpr_m is not None:
                    lw    = 2.5 if alias == winner_alias else 0.7
                    alpha = 1.0 if alias == winner_alias else 0.2
                    ax.plot(fpr_m, tpr_m, color=model_colors[i], lw=lw, alpha=alpha, label=alias)

            ax.set_title(f"{pathology}\nOverride ★ ({forced_alias})", fontweight='bold',
                         color='darkorange', fontsize=9)

        else:
            # --- EXPERT ROUTER: torneo Youden --------------------------------
            best_j       = -np.inf
            winner_alias = None
            winner_result = None
            all_results   = {}

            for alias in model_names:
                result = compute_youden_metrics(y_t, probs_calib[alias][:, p_idx])
                all_results[alias] = result
                if result[6] > best_j:
                    best_j        = result[6]
                    winner_alias  = alias
                    winner_result = result

            _, _, best_thresh, best_fpr_pt, best_tpr_pt, roc_auc_val, j_score = winner_result
            strategy = f"Expert Router ({winner_alias})"

            for i, alias in enumerate(model_names):
                fpr_m, tpr_m, *_ = all_results[alias]
                if fpr_m is not None:
                    is_win = (alias == winner_alias)
                    ax.plot(fpr_m, tpr_m,
                            color=model_colors[i],
                            lw=2.5 if is_win else 0.7,
                            alpha=1.0 if is_win else 0.25,
                            label=f"{alias} ★" if is_win else alias)

            ax.set_title(f"{pathology}\n{winner_alias} ★  (J={j_score:.3f})",
                         fontweight='bold', fontsize=9)

        print(f"  {pathology:<22} {strategy:<30} {best_thresh:>7.3f} {j_score:>8.3f}")

        # Guardar decisión de routing
        routing.append({
            'Pathology':        pathology,
            'Strategy':         strategy,
            'Winner_Model':     winner_alias,
            'Threshold_Calib':  round(best_thresh, 4),
            'Youden_J_Calib':   round(j_score, 4),
            'ROC_AUC_Calib':    round(roc_auc_val, 4),
        })

        # Aplicar al test set con las decisiones calibradas
        probs_winner_test = probs_test[winner_alias][:, p_idx]
        final_preds[:, p_idx] = (probs_winner_test >= best_thresh).astype(int)

        # Formato del subplot
        if best_fpr_pt is not None:
            ax.scatter(best_fpr_pt, best_tpr_pt, color='black', marker='*', s=220,
                       zorder=10, label='Óptimo')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, lw=0.8)
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
        ax.set_xlabel('FPR', fontsize=7)
        ax.set_ylabel('TPR', fontsize=7)
        ax.tick_params(labelsize=7)
        ax.legend(loc='lower right', fontsize=5.5, ncol=2)
        ax.grid(alpha=0.3)

    # Eliminar subplots vacíos (posición 14 y 15)
    for extra_idx in range(NUM_CLASSES, len(axes)):
        fig.delaxes(axes[extra_idx])

    plt.suptitle(
        f'Curvas ROC — Set de Calibración ({int(CALIB_RATIO*100)}% holdout, {calib_size} imgs)\n'
        f'Todos los modelos en gris · Ganador destacado en color · ★ = umbral óptimo',
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_ROC_PNG, dpi=150, bbox_inches='tight')
    print(f"\n  ROC de calibración guardado: {OUTPUT_ROC_PNG}")

    # ------------------------------------------------------------------
    # 6. Fase de evaluación: test ciego
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("  FASE DE EVALUACIÓN — TEST CIEGO")
    print("=" * 65)
    print(f"  {'Patología':<22} {'Modelo':>10} {'Sens%':>6} {'Spec%':>6} {'Youden':>7}  "
          f"{'TP':>4} {'FN':>4} {'TN':>4} {'FP':>4}")
    print("  " + "-" * 72)

    conf_matrices = multilabel_confusion_matrix(y_test, final_preds)
    metrics_list  = []

    for i, label in enumerate(ALL_LABELS):
        tn, fp, fn, tp = conf_matrices[i].ravel()
        sens = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0.0
        j    = (sens + spec) / 100 - 1

        r = next(r for r in routing if r['Pathology'] == label)
        print(f"  {label:<22} {r['Winner_Model']:>10} {sens:>6.1f} {spec:>6.1f} {j:>7.3f}  "
              f"{int(tp):>4} {int(fn):>4} {int(tn):>4} {int(fp):>4}")

        metrics_list.append({
            'Pathology':       label,
            'Winner_Model':    r['Winner_Model'],
            'Strategy':        r['Strategy'],
            'Threshold':       r['Threshold_Calib'],
            'TP': int(tp), 'FN': int(fn), 'TN': int(tn), 'FP': int(fp),
            'Sensitivity_%':   round(sens, 2),
            'Specificity_%':   round(spec, 2),
            'Youden_J':        round(j, 3),
        })

    # Promedios globales
    sens_vals = [m['Sensitivity_%'] for m in metrics_list]
    spec_vals = [m['Specificity_%'] for m in metrics_list]
    j_vals    = [m['Youden_J']      for m in metrics_list]
    print("  " + "-" * 72)
    print(f"  {'PROMEDIO':<22} {'':>10} {np.mean(sens_vals):>6.1f} {np.mean(spec_vals):>6.1f} "
          f"{np.mean(j_vals):>7.3f}")

    # ------------------------------------------------------------------
    # 7. Exportar archivos de resultados
    # ------------------------------------------------------------------
    pd.DataFrame(routing).to_csv(OUTPUT_ROUTING_CSV, index=False)
    pd.DataFrame(metrics_list).to_csv(OUTPUT_METRICS_CSV, index=False)

    pred_list = [
        {
            'Image_Name':        img_names_test[idx],
            'True_Labels':       true_labels_test[idx],
            'Predicted_Labels':  vec_to_labels(final_preds[idx]),
            'Exact_Match':       bool((y_test[idx] == final_preds[idx]).all()),
        }
        for idx in range(len(img_names_test))
    ]
    pd.DataFrame(pred_list).to_csv(OUTPUT_PREDICTIONS_CSV, index=False)

    print(f"\nArchivos exportados:")
    print(f"  {OUTPUT_ROUTING_CSV}")
    print(f"  {OUTPUT_METRICS_CSV}")
    print(f"  {OUTPUT_PREDICTIONS_CSV}")
    print(f"  {OUTPUT_ROC_PNG}")
    print("\n" + "=" * 65)
    print("  FIN")
    print("=" * 65)


if __name__ == '__main__':
    main()
