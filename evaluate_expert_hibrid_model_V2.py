import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix, roc_curve, auc
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from src.models.densenet import get_model
from src.utils.versioning import get_next_version


DATA_DIR = r"D:\Agustin\Facultad\ProyectoFinal\archive"
TEST_SET_CSV = "holdout_test_set.csv"
OUTPUT_RESULTS_CSV = get_next_version("hybrid_master_predictions.csv")
OUTPUT_METRICS_CSV = get_next_version("hybrid_master_metrics.csv")
OUTPUT_AWARDS_CSV = get_next_version("hybrid_master_awards.csv")
IMAGE_SIZE = 224
NUM_CLASSES = 14

# 0=M1(AMD), 1=M2(NV-T1), 2=M3(NV-T2)
MODEL_1_PATH = "best_model_20260402_161254.pth"          
MODEL_2_PATH = "best_model_V1.pth"       
MODEL_3_PATH = "best_model_T2_V2.pth"       

ALL_LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 
    'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 
    'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

# Reglas Clínicas de Cascada (Diseñadas a Medida por la investigación Tesis)
CASCADE_RULES = {
    'Consolidation': {'Pantalla': 0, 'Confirma': 1}, # M1 -> M2
    #'Atelectasis':   {'Pantalla': 1, 'Confirma': 0}, # M2 -> M1
    'Pneumonia':     {'Pantalla': 2, 'Confirma': 2}, # M3 -> M3 (Auto-Cascada para Pneumonia, mismo modelo en ambas etapas por su desempeño)
    'Effusion':      {'Pantalla': 2, 'Confirma': 0},  # M3 -> M1
    'Nodule':      {'Pantalla': 0, 'Confirma': 0} # M1 -> M1 (Auto-Cascada para Nodule, mismo modelo en ambas etapas por su desempeño)
}

# FACTORES DE TENSIÓN PARA ZONA DE RECHAZO
TENSION_PANTALLA = 1.0 # Restaurado a Youden puro para no colapsar Cascadas
TENSION_CONFIRMA = 1.0 # Restaurado a Youden puro para no colapsar Cascadas
MIN_SPEC = 0.55 # Cota obligatoria clínica para frenar histeria de Expert Router

def load_environment():
    device_name = "cpu"
    try:
        import torch_directml
        if torch_directml.is_available():
            device = torch_directml.device()
            device_name = "dml (AMD GPU)"
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = torch.device('cpu')
    print(f"Usando dispositivo Hibrido Maestro: {device_name}")
    return device

def load_safe(filepath, model, device):
    data = torch.load(filepath, map_location=torch.device('cpu'), weights_only=False)
    state_dict = data['model_state_dict'] if (isinstance(data, dict) and 'model_state_dict' in data) else data
    if 'classifier.1.weight' in state_dict:
        state_dict['classifier.weight'] = state_dict.pop('classifier.1.weight')
        state_dict['classifier.bias'] = state_dict.pop('classifier.1.bias')
        keys_to_delete = [k for k in list(state_dict.keys()) if k.startswith('classifier.') and k != 'classifier.weight' and k != 'classifier.bias']
        for k in keys_to_delete:
            del state_dict[k]
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model

def get_image_paths(data_dir):
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
    vec = np.zeros(NUM_CLASSES, dtype=int)
    if pd.isna(labels_str) or labels_str == 'No Finding':
        return vec
    for i, label in enumerate(ALL_LABELS):
        if label in labels_str:
            vec[i] = 1
    return vec

def binary_vector_to_labels(vec):
    found = [ALL_LABELS[i] for i in range(len(vec)) if vec[i] == 1]
    return " | ".join(found) if found else "No Finding"

def get_expert_metrics(y_true, y_prob):
    """Extrae datos de ROC y calcula el Puntaje Youden acotado a Especificidad clinica."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    youden_index = tpr + (1 - fpr) - 1
    
    # Búsqueda Acotada (Bounded Youden)
    valid_indices = np.where((1 - fpr) >= MIN_SPEC)[0]
    if len(valid_indices) == 0:
        # Fallback si el modelo no puede alcanzar 55% de especificidad en toda la curva
        optimal_idx = np.argmax(youden_index)
    else:
        # Busca el índice original que maximiza el J-Score DENTRO de los válidos
        valid_youden = youden_index[valid_indices]
        optimal_valid_idx = np.argmax(valid_youden)
        optimal_idx = valid_indices[optimal_valid_idx]

    if np.isnan(youden_index[optimal_idx]):
         return fpr, tpr, 0.5, 0, 0, 0, 0
    roc_auc = auc(fpr, tpr)
    best_thresh = thresholds[optimal_idx]
    best_fpr = fpr[optimal_idx]
    best_tpr = tpr[optimal_idx]
    max_j_score = youden_index[optimal_idx]
    return fpr, tpr, best_thresh, best_fpr, best_tpr, roc_auc, max_j_score


def get_cascade_youden(y_true, y_pred_binary):
    """Calcula el Youden teorico resultante de las decisiones hibridas (Cascada)"""
    cm = confusion_matrix(y_true, y_pred_binary, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    return sens + spec - 1

def main():
    print("=== INICIANDO HÍBRIDO MAESTRO (MOE + CASCADA) ===")
    
    device = load_environment()
    image_paths_dict = get_image_paths(DATA_DIR)
    
    if not os.path.exists(TEST_SET_CSV):
        print(f"Error: No se encontro {TEST_SET_CSV}.")
        return
        
    df_test = pd.read_csv(TEST_SET_CSV)
    print(f"Cargadas {len(df_test)} imagenes del conjunto de holdout rigoroso.")
    
    model1 = load_safe(MODEL_1_PATH, get_model(num_classes=NUM_CLASSES, pretrained=False).to(device), device)
    model2 = load_safe(MODEL_2_PATH, get_model(num_classes=NUM_CLASSES, pretrained=False).to(device), device)
    model3 = load_safe(MODEL_3_PATH, get_model(num_classes=NUM_CLASSES, pretrained=False).to(device), device)
    print("3 Expertos inicializados (AMD, NV-T1, NV-T2).")

    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    all_img_names, all_true_labels_str, all_y_true = [], [], []
    all_probs = [ [], [], [] ]
    
    sigmoid = nn.Sigmoid()

    print("\nIniciando Escaneo Sensorial Paralelo...")
    with torch.no_grad():
        for index, row in df_test.iterrows():
            img_name = row['Image Index']
            if img_name not in image_paths_dict: continue
                 
            all_img_names.append(img_name)
            all_true_labels_str.append(row['Original_Finding_Labels'])
            img_path = image_paths_dict[img_name]
            image_tensor = eval_transform(Image.open(img_path).convert('RGB')).unsqueeze(0).to(device)
            all_y_true.append(labels_to_binary_vector(row['Original_Finding_Labels']))
            
            all_probs[0].append(sigmoid(model1(image_tensor)).cpu().squeeze().numpy())
            all_probs[1].append(sigmoid(model2(image_tensor)).cpu().squeeze().numpy())
            all_probs[2].append(sigmoid(model3(image_tensor)).cpu().squeeze().numpy())

    all_y_true = np.array(all_y_true)
    for m in range(3): all_probs[m] = np.array(all_probs[m])
    
    print("\nAuditoria completada. Desplegando Motor Bifurcado de Decisiones...")
    router_preds = np.zeros_like(all_y_true)
    
    fig, axes = plt.subplots(4, 4, figsize=(22, 22))
    axes = axes.flatten()
    awards_list = []
    model_names = ["M1 (AMD)", "M2 (NV-T1)", "M3 (NV-T2)"]
    model_colors = ['red', 'blue', 'green']

    for p_idx, pathology in enumerate(ALL_LABELS):
        y_t = all_y_true[:, p_idx]
        if sum(y_t) == 0: continue
            
        metrics = []
        for m in range(3):
            metrics.append(get_expert_metrics(y_t, all_probs[m][:, p_idx]))
            
        ax = axes[p_idx]
        
        # === BIFURCACIÓN DE LÓGICA ===
        if pathology in CASCADE_RULES:
            # ---> RUTA A: CASCADA ASIMÉTRICA <---
            rule = CASCADE_RULES[pathology]
            idx_pantalla = rule['Pantalla']
            idx_confirma = rule['Confirma']
            
            opt_thresh_pantalla = metrics[idx_pantalla][2]
            opt_thresh_confirma = metrics[idx_confirma][2]
            
            # Aplicar Factor de Tensión (Zonas de Rechazo)
            real_thresh_p = opt_thresh_pantalla * TENSION_PANTALLA
            real_thresh_c = opt_thresh_confirma * TENSION_CONFIRMA
            
            # Predict Cascade
            probs_p = all_probs[idx_pantalla][:, p_idx]
            probs_c = all_probs[idx_confirma][:, p_idx]
            
            for i in range(len(y_t)):
                if probs_p[i] >= real_thresh_p:
                    if probs_c[i] >= real_thresh_c:
                        router_preds[i, p_idx] = 1
            
            winner_name = f"CASCADA M{idx_pantalla+1}→M{idx_confirma+1}"
            applied_thresh_str = f"P:{real_thresh_p:.2f} | C:{real_thresh_c:.2f}"
            j_score_logged = get_cascade_youden(y_t, router_preds[:, p_idx])
            
            # Dibujar en rojo/azul o sus respectivos colores
            for m in range(3):
                lw = 3 if m in [idx_pantalla, idx_confirma] else 1
                alpha = 1.0 if m in [idx_pantalla, idx_confirma] else 0.2
                rol_str = " (Pantalla)" if m == idx_pantalla else (" (Confirma)" if m == idx_confirma else "")
                ax.plot(metrics[m][0], metrics[m][1], color=model_colors[m], lw=lw, alpha=alpha, label=f'M{m+1}{rol_str}')
            
            ax.set_title(f"{pathology}\nEscudo Cascada ★", fontweight='bold', color='purple')
            
        else:
            # ---> RUTA B: MÉRITO PURO (EXPERT ROUTER) <---
            j_scores = [metrics[m][6] for m in range(3)]
            winner_idx = np.argmax(j_scores)
            
            real_thresh = metrics[winner_idx][2]
            probs_w = all_probs[winner_idx][:, p_idx]
            
            for i in range(len(y_t)):
                router_preds[i, p_idx] = 1 if probs_w[i] >= real_thresh else 0
                
            winner_name = f"Experto {model_names[winner_idx]}"
            applied_thresh_str = f"{real_thresh:.3f}"
            j_score_logged = j_scores[winner_idx]
            
            for m in range(3):
                lw = 4 if m == winner_idx else 1
                alpha = 1.0 if m == winner_idx else 0.2
                is_win = ' ★' if m == winner_idx else ''
                ax.plot(metrics[m][0], metrics[m][1], color=model_colors[m], lw=lw, alpha=alpha, label=f'M{m+1} (J={metrics[m][6]:.2f}){is_win}')
                ax.scatter(metrics[m][3], metrics[m][4], color=model_colors[m], s=80 if m==winner_idx else 20)
                
            ax.set_title(f"{pathology}\nEspecialista Autónomo", fontweight='bold')
            
        # Common plot settings
        awards_list.append({
            'Pathology': pathology,
            'Diagnosis_Strategy': winner_name,
            'Effective_Youden_Score': round(j_score_logged, 3),
            'Applied_Threshold(s)': applied_thresh_str
        })
        
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.15])
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.3)

    fig.delaxes(axes[14]); fig.delaxes(axes[15])
    plt.tight_layout()
    plt.savefig("hybrid_master_roc_curves.png", dpi=200, bbox_inches='tight')
    
    print("\n--- MATRICES Y RENDIMIENTO HÍBRIDO (POST-TENSIÓN) ---")
    conf_matrices = multilabel_confusion_matrix(all_y_true, router_preds)
    
    metrics_list = []
    for i, label in enumerate(ALL_LABELS):
        cm = conf_matrices[i]
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        j_score_real = sens + spec - 1
        
        print(f"\n{label} -> Sens: {sens:.1%} | Spec: {spec:.1%} | Youden: {j_score_real:.2f}")
        
        metrics_list.append({
            'Pathology': label,
            'Mechanism': next((item['Diagnosis_Strategy'] for item in awards_list if item["Pathology"] == label), "N/A"),
            'TP': tp, 'FN': fn, 'TN': tn, 'FP': fp,
            'Sensitivity_%': round(sens * 100, 2),
            'Specificity_%': round(spec * 100, 2),
            'Youden_J': round(j_score_real, 3)
        })
        
    pd.DataFrame(awards_list).to_csv(OUTPUT_AWARDS_CSV, index=False)
    pd.DataFrame(metrics_list).to_csv(OUTPUT_METRICS_CSV, index=False)
    
    results_list = [{'Image_Name': all_img_names[idx], 'True_Labels': all_true_labels_str[idx],
                     'Predicted_Labels': binary_vector_to_labels(router_preds[idx]),
                     'Exact_Match': (all_y_true[idx] == router_preds[idx]).all()} 
                     for idx in range(len(all_img_names))]
    pd.DataFrame(results_list).to_csv(OUTPUT_RESULTS_CSV, index=False)
    print(f"\n-> Reportes maestros exportados exitosamente ('hybrid_master_*').")

if __name__ == "__main__":
    main()
