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

# Imports locales
from src.models.densenet import get_model
from src.utils.versioning import get_next_version

DATA_DIR = r"D:\Agustin\Facultad\ProyectoFinal\archive"
TEST_SET_CSV = "holdout_test_set.csv"
OUTPUT_RESULTS_CSV = get_next_version("expert_router_predictions.csv")
OUTPUT_METRICS_CSV = get_next_version("expert_router_metrics.csv")
OUTPUT_AWARDS_CSV = get_next_version("expert_router_awards.csv")
OUTPUT_ROC_PLOT = get_next_version("expert_roc_curves.png")
IMAGE_SIZE = 224
NUM_CLASSES = 14

MODEL_1_PATH = "best_model.pth"          
MODEL_2_PATH = "best_model_T1.pth"       
MODEL_3_PATH = "best_model_T2.pth"       

ALL_LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 
    'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 
    'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

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
    print(f"Usando dispositivo para evaluacion Experta: {device_name}")
    return device

def load_safe(filepath, model, device):
    data = torch.load(filepath, map_location=torch.device('cpu'), weights_only=False)
    state_dict = data['model_state_dict'] if (isinstance(data, dict) and 'model_state_dict' in data) else data
    if 'classifier.1.weight' in state_dict:
        print(f"  [Parche Activado] Adaptando estructura neuronal del compañero en {os.path.basename(filepath)}...")
        state_dict['classifier.weight'] = state_dict.pop('classifier.1.weight')
        state_dict['classifier.bias'] = state_dict.pop('classifier.1.bias')
        keys_to_delete = [k for k in list(state_dict.keys()) if k.startswith('classifier.') and k != 'classifier.weight' and k != 'classifier.bias']
        for k in keys_to_delete:
            del state_dict[k]
    # PROTEGER CONTRA EL "SILENT KILLER" (Pesos ignorados)
    load_status = model.load_state_dict(state_dict, strict=False)
    
    if load_status.missing_keys or load_status.unexpected_keys:
        print(f"\n[⚠️ ADVERTENCIA CRITICA] Problema de Mapeo en {os.path.basename(filepath)}:")
        if load_status.missing_keys:
            print(f"  -> {len(load_status.missing_keys)} LLAVES FALTANTES (¡Se usan pesos aleatorios iniciales!):")
            print(f"     Ej: {load_status.missing_keys[:5]}")
        if load_status.unexpected_keys:
            print(f"  -> {len(load_status.unexpected_keys)} LLAVES SOBRANTES (Se ignoraron):")
            print(f"     Ej: {load_status.unexpected_keys[:5]}")
            
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
    """Extrae datos de ROC y calcula el Puntaje Youden máximo."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    youden_index = tpr + (1 - fpr) - 1
    optimal_idx = np.argmax(youden_index)
    
    if np.isnan(youden_index[optimal_idx]):
         return fpr, tpr, 0.5, 0, 0, 0, 0
        
    roc_auc = auc(fpr, tpr)
    best_thresh = thresholds[optimal_idx]
    best_fpr = fpr[optimal_idx]
    best_tpr = tpr[optimal_idx]
    max_j_score = youden_index[optimal_idx]
    
    return fpr, tpr, best_thresh, best_fpr, best_tpr, roc_auc, max_j_score

def main():
    print("=== INICIANDO ENRUTADOR DE EXPERTOS (EXPERT ROUTER) ===")
    
    device = load_environment()
    image_paths_dict = get_image_paths(DATA_DIR)
    
    if not os.path.exists(TEST_SET_CSV):
        print(f"Error: No se encontro {TEST_SET_CSV}. Ejecute create_test_set.py primero.")
        return
        
    df_test = pd.read_csv(TEST_SET_CSV)
    print(f"Cargadas {len(df_test)} imagenes del conjunto de holdout rigoroso.")
    
    print("\nVerificando y cargando los 3 modelos...")
    if not (os.path.exists(MODEL_1_PATH) and os.path.exists(MODEL_2_PATH) and os.path.exists(MODEL_3_PATH)):
        print("ERROR FATAL: No se encontraron los 3 archivos de modelos (.pth).")
        return

    model1 = get_model(num_classes=NUM_CLASSES, pretrained=False).to(device)
    model2 = get_model(num_classes=NUM_CLASSES, pretrained=False).to(device)
    model3 = get_model(num_classes=NUM_CLASSES, pretrained=False).to(device)

    model1 = load_safe(MODEL_1_PATH, model1, device)
    model2 = load_safe(MODEL_2_PATH, model2, device)
    model3 = load_safe(MODEL_3_PATH, model3, device)
    print("Modelos inicializados y cargados en memoria exitosamente.")

    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    all_img_names = []
    all_true_labels_str = []
    all_y_true = []
    all_probs_m1 = []
    all_probs_m2 = []
    all_probs_m3 = []
    
    sigmoid = nn.Sigmoid()

    print("\nIniciando escaneos masivos paralelos...")
    with torch.no_grad():
        for index, row in df_test.iterrows():
            if index % 50 == 0 and index > 0:
                print(f"  Procesados {index}/{len(df_test)} pacientes...")
                
            img_name = row['Image Index']
            true_labels_str = row['Original_Finding_Labels']
            
            if img_name not in image_paths_dict:
                 continue
                 
            all_img_names.append(img_name)
            all_true_labels_str.append(true_labels_str)
                 
            img_path = image_paths_dict[img_name]
            image = Image.open(img_path).convert('RGB')
            image_tensor = eval_transform(image).unsqueeze(0).to(device)
            
            true_vec = labels_to_binary_vector(true_labels_str)
            all_y_true.append(true_vec)
            
            prob1 = sigmoid(model1(image_tensor)).cpu().squeeze().numpy()
            prob2 = sigmoid(model2(image_tensor)).cpu().squeeze().numpy()
            prob3 = sigmoid(model3(image_tensor)).cpu().squeeze().numpy()
            
            all_probs_m1.append(prob1)
            all_probs_m2.append(prob2)
            all_probs_m3.append(prob3)

    all_y_true = np.array(all_y_true)
    all_probs_m1 = np.array(all_probs_m1)
    all_probs_m2 = np.array(all_probs_m2)
    all_probs_m3 = np.array(all_probs_m3)
    
    print("\nEscaneos completos. Iniciando Torneo de Modelos (Torneo de Youden)...")
    router_preds = np.zeros_like(all_y_true)
    
    fig, axes = plt.subplots(4, 4, figsize=(22, 22))
    axes = axes.flatten()
    
    awards_list = []

    for p_idx, pathology in enumerate(ALL_LABELS):
        y_t = all_y_true[:, p_idx]
        if sum(y_t) == 0: continue
            
        # 1. Torneo de Expertos
        fpr1, tpr1, th1, opx1, opy1, auc1, j1 = get_expert_metrics(y_t, all_probs_m1[:, p_idx])
        fpr2, tpr2, th2, opx2, opy2, auc2, j2 = get_expert_metrics(y_t, all_probs_m2[:, p_idx])
        fpr3, tpr3, th3, opx3, opy3, auc3, j3 = get_expert_metrics(y_t, all_probs_m3[:, p_idx])
        
        # Seleccionar el ganador (Mejor Score Youden)
        scores = [j1, j2, j3]
        winner_idx = np.argmax(scores) # 0, 1 o 2 (M1, M2 o M3)
        
        # Enrutamiento de las predicciones
        if winner_idx == 0:
            winner_name = "Modelo 1 (AMD)"
            winner_thresh = th1
            # Imponer la voluntad del Modelo 1 sobre todos los pacientes para ESTA patologia
            for i in range(len(y_t)): router_preds[i, p_idx] = 1 if all_probs_m1[i, p_idx] >= th1 else 0
        elif winner_idx == 1:
            winner_name = "Modelo 2 (NV-T1)"
            winner_thresh = th2
            for i in range(len(y_t)): router_preds[i, p_idx] = 1 if all_probs_m2[i, p_idx] >= th2 else 0
        else:
            winner_name = "Modelo 3 (NV-T2)"
            winner_thresh = th3
            for i in range(len(y_t)): router_preds[i, p_idx] = 1 if all_probs_m3[i, p_idx] >= th3 else 0
            
        awards_list.append({
            'Pathology': pathology,
            'Expert_Winner': winner_name,
            'Winning_Youden_Score': round(scores[winner_idx], 3),
            'Applied_Threshold': round(winner_thresh, 3)
        })
        
        # DIBUJAR EN EL PANEL ROC (Resaltando al ganador)
        ax = axes[p_idx]
        lw1 = 4 if winner_idx == 0 else 1
        ax.plot(fpr1, tpr1, color='red', lw=lw1, alpha=1.0 if winner_idx==0 else 0.3, label=f'M1 (J={j1:.2f})' + ('★' if winner_idx==0 else ''))
        ax.scatter(opx1, opy1, color='red', marker='o', s=80 if winner_idx==0 else 30)
        
        lw2 = 4 if winner_idx == 1 else 1
        ax.plot(fpr2, tpr2, color='blue', lw=lw2, alpha=1.0 if winner_idx==1 else 0.3, label=f'M2 (J={j2:.2f})' + ('★' if winner_idx==1 else ''))
        ax.scatter(opx2, opy2, color='blue', marker='s', s=80 if winner_idx==1 else 30)
        
        lw3 = 4 if winner_idx == 2 else 1
        ax.plot(fpr3, tpr3, color='green', lw=lw3, alpha=1.0 if winner_idx==2 else 0.3, label=f'M3 (J={j3:.2f})' + ('★' if winner_idx==2 else ''))
        ax.scatter(opx3, opy3, color='green', marker='^', s=80 if winner_idx==2 else 30)
        
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax.set_title(f"{pathology}\nExperto: {winner_name} ★", fontweight='bold')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.15])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.3)

    fig.delaxes(axes[14])
    fig.delaxes(axes[15])
    plt.tight_layout()
    plt.savefig(OUTPUT_ROC_PLOT, dpi=200, bbox_inches='tight')
    print(f"-> Gráfico Espectacular de 14 Paneles (Grid ROC) guardado como: {OUTPUT_ROC_PLOT}")
    
    print("\n" + "="*50)
    print("GUARDANDO REPORTES DE PERSISTENCIA (CSV)")
    print("="*50)
    
    # Exportar Medallas
    pd.DataFrame(awards_list).to_csv(OUTPUT_AWARDS_CSV, index=False)
    print(f"-> Generada Tabla de Premios por Patologia: {OUTPUT_AWARDS_CSV}")
    
    # Guardar Predicciones Individuales
    results_list = []
    for idx in range(len(all_img_names)):
        y_pred_vec = router_preds[idx]
        pred_labels_str = binary_vector_to_labels(y_pred_vec)
        
        results_list.append({
            'Image_Name': all_img_names[idx],
            'True_Labels': all_true_labels_str[idx],
            'Predicted_Labels': pred_labels_str,
            'Exact_Match': (all_y_true[idx] == y_pred_vec).all()
        })
        
    pd.DataFrame(results_list).to_csv(OUTPUT_RESULTS_CSV, index=False)
    print(f"-> Generada Bitacora de Pacientes Ruteados: {OUTPUT_RESULTS_CSV}")

    # Generar Reporte de Matriz
    print("\n--- RESULTADOS DEL ENRUTADOR DE EXPERTOS ---")
    conf_matrices = multilabel_confusion_matrix(all_y_true, router_preds)
    
    metrics_list = []
    for i, label in enumerate(ALL_LABELS):
        cm = conf_matrices[i]
        tn, fp, fn, tp = cm.ravel()
        
        sensibilidad = tp / (tp + fn) if (tp + fn) > 0 else 0
        especificidad = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        print(f"\nPatologia: {label}")
        print(f"  True Positives (TP): {tp} | False Negatives (FN): {fn}")
        print(f"  True Negatives (TN): {tn} | False Positives (FP): {fp}")
        print(f"  -> Sensibilidad Definitiva: {sensibilidad:.1%}")
        print(f"  -> Especificidad Definitiva: {especificidad:.1%}")
        
        metrics_list.append({
            'Pathology': label,
            'Winner_Model': next((item['Expert_Winner'] for item in awards_list if item["Pathology"] == label), "N/A"),
            'TP': tp, 'FN': fn, 'TN': tn, 'FP': fp,
            'Sensitivity_%': round(sensibilidad * 100, 2),
            'Specificity_%': round(especificidad * 100, 2)
        })
        
    pd.DataFrame(metrics_list).to_csv(OUTPUT_METRICS_CSV, index=False)
    print(f"\n-> Generado Reporte de Metricas Finales en: {OUTPUT_METRICS_CSV}")

if __name__ == "__main__":
    main()
