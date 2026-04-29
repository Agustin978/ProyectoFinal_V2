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

# Suprimir warnings de sklearn si hay clases con 0 muestras
warnings.filterwarnings("ignore", category=UserWarning)

# Imports locales
from src.models.densenet import get_model
from src.utils.versioning import get_next_version

# =============================================================================
# CONFIGURACIÓN  (única sección a editar)
# =============================================================================
DATA_DIR     = r"D:\Agustin\Facultad\ProyectoFinal\archive"
MODEL_PATH   = "best_model_T1_V2.pth"
TEST_SET_CSV = "holdout_test_set.csv"

# PREVENCIÓN DE LEAKAGE (STRICT MODE ON)
# True  → 50% calibración de umbral / 50% test ciego  [recomendado]
# False → 100% del holdout para calibrar y evaluar    [solo para debugging]
STRICT_EVALUATION_MODE = True

OUTPUT_RESULTS_CSV = get_next_version("demo_predictions.csv")
OUTPUT_METRICS_CSV = get_next_version("demo_metrics.csv")
OUTPUT_ROC_PLOT    = get_next_version("demo_roc_calib.png")

IMAGE_SIZE  = 224
NUM_CLASSES = 14

ALL_LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
    'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema',
    'Fibrosis', 'Pleural_Thickening', 'Hernia',
]

# =============================================================================
# UTILIDADES
# =============================================================================

def load_environment():
    """Configura el dispositivo (AMD DirectML / CUDA / CPU) de manera segura."""
    try:
        import torch_directml
        if torch_directml.is_available():
            print("Usando dispositivo: dml (AMD GPU)")
            return torch_directml.device()
    except ImportError:
        pass
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")
    return device


def load_safe(filepath, model, device):
    """Parche Cirujano Universal: acopla cualquier arquitectura guardada
    (Sequential ↔ Linear en el clasificador final)."""
    data = torch.load(filepath, map_location=torch.device('cpu'), weights_only=False)
    state_dict = data['model_state_dict'] if (isinstance(data, dict) and 'model_state_dict' in data) else data

    has_dropout_local = any('classifier.1.weight' in k for k in model.state_dict().keys())

    if not has_dropout_local and 'classifier.1.weight' in state_dict:
        state_dict['classifier.weight'] = state_dict.pop('classifier.1.weight')
        state_dict['classifier.bias']   = state_dict.pop('classifier.1.bias')
    elif has_dropout_local and 'classifier.weight' in state_dict:
        state_dict['classifier.1.weight'] = state_dict.pop('classifier.weight')
        state_dict['classifier.1.bias']   = state_dict.pop('classifier.bias')

    keys_to_delete = [
        k for k in list(state_dict.keys())
        if k.startswith('classifier.')
        and k not in ('classifier.1.weight', 'classifier.1.bias',
                      'classifier.weight', 'classifier.bias')
    ]
    for k in keys_to_delete:
        del state_dict[k]

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model


def get_image_paths(data_dir):
    """Mapea recursivamente nombres de archivo a rutas absolutas."""
    image_paths = {}
    search_dir = os.path.join(data_dir, 'images')
    if not os.path.exists(search_dir):
        search_dir = data_dir
    for root, _, files in os.walk(search_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths[file] = os.path.join(root, file)
    return image_paths


def labels_to_binary_vector(labels_str):
    """String de etiquetas separadas por '|' → vector binario de 14 posiciones."""
    vec = np.zeros(NUM_CLASSES, dtype=int)
    if pd.isna(labels_str) or labels_str.strip() == 'No Finding':
        return vec
    for i, label in enumerate(ALL_LABELS):
        if label in labels_str:
            vec[i] = 1
    return vec


def binary_vector_to_labels(vec):
    """Vector binario → string descriptivo."""
    found = [ALL_LABELS[i] for i in range(len(vec)) if vec[i] == 1]
    return " | ".join(found) if found else "No Finding"


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=== INICIANDO DEMO MODEL CON CONTROL DE LEAKAGE ===")

    # ------------------------------------------------------------------
    # 1. Entorno y rutas
    # ------------------------------------------------------------------
    device = load_environment()

    if not os.path.exists(TEST_SET_CSV):
        print(f"Error Crítico: No existe {TEST_SET_CSV}.")
        return

    df_test = pd.read_csv(TEST_SET_CSV)

    # Desordenar rigurosamente para evitar bloques de enfermedades seguidas
    df_test = df_test.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Indexando imágenes ({DATA_DIR})...")
    image_paths_dict = get_image_paths(DATA_DIR)

    if len(image_paths_dict) == 0:
        print("ERROR: No se encontraron imágenes en el directorio.")
        return

    print(f"Cargadas {len(df_test)} imágenes a evaluar.\n")

    # ------------------------------------------------------------------
    # 2. Cargar modelo
    # ------------------------------------------------------------------
    if not os.path.exists(MODEL_PATH):
        print(f"Error Crítico: No existe el modelo en {MODEL_PATH}.")
        return

    print(f"Cargando modelo → {MODEL_PATH}...")
    model = get_model(num_classes=NUM_CLASSES, pretrained=False)
    model = load_safe(MODEL_PATH, model, device)
    print("Modelo cargado exitosamente.\n")

    # ------------------------------------------------------------------
    # 3. Inferencia sobre TODO el holdout (una sola pasada)
    # ------------------------------------------------------------------
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    all_img_names        = []
    all_true_labels_strs = []
    y_true_all           = []
    y_pred_probs_all     = []

    print("Comenzando inferencia...")
    with torch.no_grad():
        for idx, row in df_test.iterrows():
            img_name        = row['Image Index']
            true_labels_str = (row['Original_Finding_Labels']
                               if 'Original_Finding_Labels' in row
                               else row['Finding Labels'])

            if img_name not in image_paths_dict:
                continue

            image  = Image.open(image_paths_dict[img_name]).convert('RGB')
            tensor = transform(image).unsqueeze(0).to(device)
            probs  = torch.sigmoid(model(tensor).cpu()).numpy()[0]

            all_img_names.append(img_name)
            all_true_labels_strs.append(true_labels_str)
            y_true_all.append(labels_to_binary_vector(true_labels_str))
            y_pred_probs_all.append(probs)

            if (idx + 1) % 50 == 0:
                print(f"  Procesadas {idx + 1}/{len(df_test)} imágenes...")

    y_true_all       = np.array(y_true_all)
    y_pred_probs_all = np.array(y_pred_probs_all)

    # ------------------------------------------------------------------
    # 4. Split calib / test  ← FIX: evita data leakage
    # ------------------------------------------------------------------
    if STRICT_EVALUATION_MODE:
        calib_size = len(y_true_all) // 2
        print(f"\n[STRICT MODE] Calibrando umbrales en {calib_size} imágenes"
              f" | Test ciego en {len(y_true_all) - calib_size} imágenes.")
    else:
        calib_size = len(y_true_all)
        print("\n[LAXO] Umbral calibrado con el 100% del holdout (solo para debugging).")

    y_calib    = y_true_all[:calib_size, :]
    probs_calib = y_pred_probs_all[:calib_size, :]

    y_test      = y_true_all[calib_size:, :] if STRICT_EVALUATION_MODE else y_true_all
    probs_test  = y_pred_probs_all[calib_size:, :] if STRICT_EVALUATION_MODE else y_pred_probs_all

    # ------------------------------------------------------------------
    # 5. Calibración de umbrales (Youden) — SOLO sobre calib set
    # ------------------------------------------------------------------
    print("\nOptimizando umbrales (Índice de Youden) — excluyendo set de test...")
    optimal_thresholds = np.zeros(NUM_CLASSES)
    plt.figure(figsize=(15, 12))

    for i in range(NUM_CLASSES):
        # Sin positivos en calib: umbral por defecto
        if sum(y_calib[:, i]) == 0:
            optimal_thresholds[i] = 0.5
            continue

        fpr, tpr, thresholds = roc_curve(y_calib[:, i], probs_calib[:, i])
        youden_index          = tpr + (1 - fpr) - 1
        best_threshold_idx    = np.argmax(youden_index)
        best_threshold        = thresholds[best_threshold_idx]

        # Clip [0.05, 0.95]: evita extremos inestables sin restringir el rango útil
        optimal_thresholds[i] = np.clip(best_threshold, 0.05, 0.95)

        roc_auc_cal = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2,
                 label=f'{ALL_LABELS[i]} (AUC_Cal={roc_auc_cal:.2f}, T={optimal_thresholds[i]:.2f})')
        plt.scatter(fpr[best_threshold_idx], tpr[best_threshold_idx],
                    marker='o', color='black', s=50, zorder=5)

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos (1 − Especificidad)')
    plt.ylabel('Tasa de Verdaderos Positivos (Sensibilidad)')
    plt.title(f'Curvas ROC — Set de Calibración ({calib_size} imágenes)\n'
              f'Modelo: {MODEL_PATH}  ·  ● = umbral óptimo por Youden J')
    plt.legend(loc='lower right', fontsize='small', ncol=2)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_ROC_PLOT, dpi=150, bbox_inches='tight')
    print(f"-> Gráfico ROC de calibración guardado: {OUTPUT_ROC_PLOT}")

    # ------------------------------------------------------------------
    # 6. Evaluación sobre test ciego (umbrales ya fijos)
    # ------------------------------------------------------------------
    y_pred_binary_test = np.zeros_like(probs_test, dtype=int)
    for i in range(NUM_CLASSES):
        y_pred_binary_test[:, i] = (probs_test[:, i] >= optimal_thresholds[i]).astype(int)

    img_names_test        = all_img_names[calib_size:]        if STRICT_EVALUATION_MODE else all_img_names
    true_labels_strs_test = all_true_labels_strs[calib_size:] if STRICT_EVALUATION_MODE else all_true_labels_strs

    # Predicciones imagen por imagen
    results_list = [
        {
            'Image_Name':       img_names_test[k],
            'True_Labels':      true_labels_strs_test[k],
            'Predicted_Labels': binary_vector_to_labels(y_pred_binary_test[k]),
            'Exact_Match':      bool((y_test[k] == y_pred_binary_test[k]).all()),
        }
        for k in range(len(img_names_test))
    ]
    pd.DataFrame(results_list).to_csv(OUTPUT_RESULTS_CSV, index=False)
    print(f"-> Predicciones por imagen guardadas: {OUTPUT_RESULTS_CSV}")

    # Matrices de confusión y métricas por patología
    print("\n" + "=" * 55)
    print("  MATRICES DE CONFUSIÓN — TEST CIEGO")
    print("=" * 55)

    mcm          = multilabel_confusion_matrix(y_test, y_pred_binary_test)
    metrics_list = []

    for i, label in enumerate(ALL_LABELS):
        tn, fp, fn, tp = mcm[i].ravel()
        sensitivity = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0.0
        specificity = (tn / (tn + fp) * 100) if (tn + fp) > 0 else 0.0
        youden_j    = (sensitivity / 100) + (specificity / 100) - 1

        print(f"\n--- {label:<20} (Umbral: {optimal_thresholds[i]:.3f})")
        print(f"  Real: NO → TN: {tn:<5} FP: {fp:<5} | Especificidad: {specificity:.1f}%")
        print(f"  Real: SÍ → FN: {fn:<5} TP: {tp:<5} | Sensibilidad:  {sensitivity:.1f}%")
        print(f"  Youden J: {youden_j:.3f}")

        metrics_list.append({
            'Pathology':       label,
            'Threshold':       round(float(optimal_thresholds[i]), 4),
            'TP': int(tp), 'FN': int(fn), 'TN': int(tn), 'FP': int(fp),
            'Sensitivity_%':   round(sensitivity, 2),
            'Specificity_%':   round(specificity, 2),
            'Youden_J':        round(youden_j, 3),
        })

    pd.DataFrame(metrics_list).to_csv(OUTPUT_METRICS_CSV, index=False)
    print(f"\n-> Métricas por patología guardadas: {OUTPUT_METRICS_CSV}")

    # Promedio global
    sens_mean = np.mean([m['Sensitivity_%'] for m in metrics_list])
    spec_mean = np.mean([m['Specificity_%'] for m in metrics_list])
    j_mean    = np.mean([m['Youden_J']      for m in metrics_list])
    print(f"\n  PROMEDIO GLOBAL → Sens: {sens_mean:.1f}%  Spec: {spec_mean:.1f}%  Youden J: {j_mean:.3f}")

    print("\n  Leyenda:")
    print("  TP = Verdadero Positivo  → patología detectada correctamente.")
    print("  TN = Verdadero Negativo  → ausencia descartada correctamente.")
    print("  FP = Falso Positivo      → falsa alarma.")
    print("  FN = Falso Negativo      → error clínico grave (enfermedad omitida).")
    print("\n[OK] Evaluación completada sin data leakage.")

if __name__ == "__main__":
    main()
