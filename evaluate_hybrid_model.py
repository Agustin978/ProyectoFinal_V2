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

# Suprimir warnings de sklearn si hay clases con 0 muestras en batches chicos
warnings.filterwarnings("ignore", category=UserWarning)

# Imports locales
from src.models.densenet import get_model

# Configuración
DATA_DIR = r"D:\Agustin\Facultad\ProyectoFinal\archive"
TEST_SET_CSV = "holdout_test_set.csv"
OUTPUT_RESULTS_CSV = "hybrid_evaluation_results.csv"
OUTPUT_METRICS_CSV = "hybrid_evaluation_metrics.csv"
IMAGE_SIZE = 224
NUM_CLASSES = 14

# Aquí definimos las rutas de los 3 modelos (Ajustar según donde los guarden)
# En caso de no existir, el script avisará.
MODEL_1_PATH = "best_model.pth"          # Alto Recall (Tu modelo con Oversampling)
MODEL_2_PATH = "best_model_T1.pth" # Alta Especif. (Modelo sin tanto oversampling)
MODEL_3_PATH = "best_model_T2.pth"     # Modelo Balanceado General

ALL_LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 
    'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 
    'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

def load_environment():
    """Configura el dispositivo (GPU/CPU) de manera segura."""
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
    
    print(f"Usando dispositivo para evaluacion híbrida: {device_name}")
    return device

def load_safe(filepath, model, device):
    """Carga pesos cross-hardware e inyecta parches de arquitectura."""
    # MAP TO CPU FIRST to strip hardware-specific internal tensors
    data = torch.load(filepath, map_location=torch.device('cpu'), weights_only=False)
    
    # Extraer el diccionario de estado real
    state_dict = data['model_state_dict'] if (isinstance(data, dict) and 'model_state_dict' in data) else data
    
    # --- PARCHE CIRUJANO: COMPATIBILIDAD DE ARQUITECTURAS ---
    # Si el modelo tiene armada su cabeza de red usando un nn.Sequential, renombramos sus tensores 
    # finales para que calcen exactamente en nuestra arquitectura nn.Linear simple.
    if 'classifier.1.weight' in state_dict:
        print(f"  [Parche Activado] Adaptando estructura neuronal del modelo en {os.path.basename(filepath)}...")
        state_dict['classifier.weight'] = state_dict.pop('classifier.1.weight')
        state_dict['classifier.bias'] = state_dict.pop('classifier.1.bias')
        
        # Eliminar cualquier otro rastro de su Sequential (como classifier.0) para evitar que estorben
        keys_to_delete = [k for k in list(state_dict.keys()) if k.startswith('classifier.') and k != 'classifier.weight' and k != 'classifier.bias']
        for k in keys_to_delete:
            del state_dict[k]
    # --------------------------------------------------------
    
    # Cargar con strict=False permite incrustar la matrix aunque haya ligeras variables huérfanas
    model.load_state_dict(state_dict, strict=False)
        
    model = model.to(device)
    model.eval()
    return model

def get_image_paths(data_dir):
    """Mapea recursivamente los nombres de archivo a sus rutas absolutas."""
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
    """Convierte un vector binario a un string descriptivo."""
    found = [ALL_LABELS[i] for i in range(len(vec)) if vec[i] == 1]
    return " | ".join(found) if found else "No Finding"

def get_roc_data_and_threshold(y_true, y_prob):
    """Calcula el Índice de Youden exacto para la curva ROC y extrae todos los puntos de ploteo."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    youden_index = tpr + (1 - fpr) - 1
    optimal_idx = np.argmax(youden_index)
    
    # Prevenir que busque el umbral con 0 samples de algo
    if np.isnan(youden_index[optimal_idx]):
         return fpr, tpr, 0.5, 0, 0, 0
        
    roc_auc = auc(fpr, tpr)
    best_thresh = thresholds[optimal_idx]
    best_fpr = fpr[optimal_idx]
    best_tpr = tpr[optimal_idx]
    
    return fpr, tpr, best_thresh, best_fpr, best_tpr, roc_auc

def main():
    print("=== INICIANDO COMITÉ MÉDICO ARTIFICIAL (HÍBRIDO CASCADA) ===")
    
    device = load_environment()
    image_paths_dict = get_image_paths(DATA_DIR)
    
    if not os.path.exists(TEST_SET_CSV):
        print(f"Error: No se encontro {TEST_SET_CSV}. Ejecute create_test_set.py primero.")
        return
        
    df_test = pd.read_csv(TEST_SET_CSV)
    print(f"Cargadas {len(df_test)} imagenes del conjunto de holdout rigoroso.")
    
    print("\nVerificando y cargando los 3 modelos (Traducción Cross-Hardware activada)...")
    if not (os.path.exists(MODEL_1_PATH) and os.path.exists(MODEL_2_PATH) and os.path.exists(MODEL_3_PATH)):
        print("ERROR FATAL: No se encontraron los 3 archivos de modelos (.pth).")
        print("Renombrar el nombre de los modelos a:")
        print(f"1: {MODEL_1_PATH}\n2: {MODEL_2_PATH}\n3: {MODEL_3_PATH}")
        return

    # Inicialización de bases
    model1 = get_model(num_classes=NUM_CLASSES, pretrained=False).to(device)
    model2 = get_model(num_classes=NUM_CLASSES, pretrained=False).to(device)
    model3 = get_model(num_classes=NUM_CLASSES, pretrained=False).to(device)

    # Carga Segura Cross-Hardware
    model1 = load_safe(MODEL_1_PATH, model1, device)
    model2 = load_safe(MODEL_2_PATH, model2, device)
    model3 = load_safe(MODEL_3_PATH, model3, device)
    print("Modelos inicializados y cargados en memoria exitosamente.")

    # Transformaciones de evaluacion estandar
    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Arrays de almacenamiento
    all_img_names = []
    all_true_labels_str = []
    all_y_true = []
    all_probs_m1 = []
    all_probs_m2 = []
    all_probs_m3 = []
    
    sigmoid = nn.Sigmoid()

    # Evaluación y recopilación de probabilidades de los 3 doctores
    print("\nIniciando escaneos paralelos en las imagenes...")
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
            
            # Ground truth
            true_vec = labels_to_binary_vector(true_labels_str)
            all_y_true.append(true_vec)
            
            # Forward passes de los 3 modelos simultáneos
            output1 = model1(image_tensor)
            output2 = model2(image_tensor)
            output3 = model3(image_tensor)
            
            # Pasar a probabilidades y llevar a CPU RAM
            prob1 = sigmoid(output1).cpu().squeeze().numpy()
            prob2 = sigmoid(output2).cpu().squeeze().numpy()
            prob3 = sigmoid(output3).cpu().squeeze().numpy()
            
            all_probs_m1.append(prob1)
            all_probs_m2.append(prob2)
            all_probs_m3.append(prob3)

    all_y_true = np.array(all_y_true)
    all_probs_m1 = np.array(all_probs_m1)
    all_probs_m2 = np.array(all_probs_m2)
    all_probs_m3 = np.array(all_probs_m3)
    
    print("\nEscaneos completos. Dibujando Panel ROC Multi-Modelo y procesando Cascada...")
    hybrid_preds = np.zeros_like(all_y_true)
    
    # Crear Panel Gráfico 4x4 (Para acomodar las 14 patologías)
    fig, axes = plt.subplots(4, 4, figsize=(22, 22))
    axes = axes.flatten()
    
    # Evaluamos patología por patología matemáticamente
    for p_idx, pathology in enumerate(ALL_LABELS):
        y_t = all_y_true[:, p_idx]
        
        # Evitar errores si no hay ejemplos activos de esta patologia
        if sum(y_t) == 0:
            continue
            
        # 1. Obtener Umbrales Múltiples y datos de ROC
        fpr1, tpr1, thresh1, op_x1, op_y1, auc1 = get_roc_data_and_threshold(y_t, all_probs_m1[:, p_idx])
        fpr2, tpr2, thresh2, op_x2, op_y2, auc2 = get_roc_data_and_threshold(y_t, all_probs_m2[:, p_idx])
        fpr3, tpr3, thresh3, op_x3, op_y3, auc3 = get_roc_data_and_threshold(y_t, all_probs_m3[:, p_idx])
        
        # DIBUJAR EN EL PANEL ROC
        ax = axes[p_idx]
        # Tu Modelo (M1) - Alarmista
        ax.plot(fpr1, tpr1, color='red', lw=2, label=f'Triaje Local (AUC={auc1:.2f}, TH={thresh1:.2f})')
        ax.scatter(op_x1, op_y1, color='red', marker='o', s=80, zorder=5) # Youden M1
        
        # Modelo Conservador (M2) 
        ax.plot(fpr2, tpr2, color='blue', lw=2, label=f'Confirm. M2 (AUC={auc2:.2f}, TH={thresh2:.2f})')
        ax.scatter(op_x2, op_y2, color='blue', marker='s', s=80, zorder=5) # Youden M2
        
        # Modelo Balanceado (M3)
        ax.plot(fpr3, tpr3, color='green', lw=2, label=f'Balanc. M3 (AUC={auc3:.2f}, TH={thresh3:.2f})')
        ax.scatter(op_x3, op_y3, color='green', marker='^', s=80, zorder=5) # Youden M3
        
        # Grid decorativa
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax.set_title(pathology, fontweight='bold')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate (FPR)')
        ax.set_ylabel('True Positive Rate (TPR)')
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.3)
        
        # 2. Aplicar la Lógica Clínica en Cascada sobre todas las imágenes
        for i in range(len(y_t)):
            p1 = all_probs_m1[i, p_idx]
            p2 = all_probs_m2[i, p_idx]
            p3 = all_probs_m3[i, p_idx]
            
            # Paso A: Triage (M1 - Alto Recall)
            if p1 < thresh1:
                hybrid_preds[i, p_idx] = 0
            else:
                # Paso B: Confirmación (M2 - Alta Especificidad)
                if p2 >= thresh2:
                    hybrid_preds[i, p_idx] = 1
                else:
                    # Paso C: Desempate (M3)
                    hybrid_preds[i, p_idx] = 1 if p3 >= thresh3 else 0

    # Apagar y borrar las gráficas 15 y 16 vacías de la cuadrícula
    fig.delaxes(axes[14])
    fig.delaxes(axes[15])
    
    # Guardar Foto del Panel
    plt.tight_layout()
    plt.savefig("hybrid_roc_curves.png", dpi=200, bbox_inches='tight')
    print("-> Gráfico Espectacular de 14 Paneles (Grid ROC) guardado como: hybrid_roc_curves.png")

    print("\n" + "="*50)
    print("GUARDANDO REPORTES DE PERSISTENCIA (CSV)")
    print("="*50)
    
    # 3. Guardar Tabla de Predicciones Individuales
    results_list = []
    for idx in range(len(all_img_names)):
        y_pred_vec = hybrid_preds[idx]
        pred_labels_str = binary_vector_to_labels(y_pred_vec)
        
        results_list.append({
            'Image_Name': all_img_names[idx],
            'True_Labels': all_true_labels_str[idx],
            'Predicted_Labels': pred_labels_str,
            'Exact_Match': (all_y_true[idx] == y_pred_vec).all()
        })
        
    pd.DataFrame(results_list).to_csv(OUTPUT_RESULTS_CSV, index=False)
    print(f"-> Generada bitacora de pacientes: {OUTPUT_RESULTS_CSV}")

    # Generar Reporte de Matriz de Confusión del Ensamblado
    print("\n--- RESULTADOS DEL COMITÉ MÉDICO (CASCADA) ---")
    conf_matrices = multilabel_confusion_matrix(all_y_true, hybrid_preds)
    
    metrics_list = []
    for i, label in enumerate(ALL_LABELS):
        cm = conf_matrices[i]
        tn, fp, fn, tp = cm.ravel()
        
        sensibilidad = tp / (tp + fn) if (tp + fn) > 0 else 0
        especificidad = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        print(f"\nPatologia: {label}")
        print(f"  True Positives (TP): {tp} | False Negatives (FN): {fn}")
        print(f"  True Negatives (TN): {tn} | False Positives (FP): {fp}")
        print(f"  -> Sensibilidad Final (Triage Superado): {sensibilidad:.1%}")
        print(f"  -> Especificidad Final (Falsas Alarmas Evitadas): {especificidad:.1%}")
        
        metrics_list.append({
            'Pathology': label,
            'TP': tp, 'FN': fn, 'TN': tn, 'FP': fp,
            'Sensitivity_%': round(sensibilidad * 100, 2),
            'Specificity_%': round(especificidad * 100, 2)
        })
        
    pd.DataFrame(metrics_list).to_csv(OUTPUT_METRICS_CSV, index=False)
    print(f"\n-> Generado reporte de metricas finales en: {OUTPUT_METRICS_CSV}")

if __name__ == "__main__":
    main()
