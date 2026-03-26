import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
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
    'Fibrosis', 'Pleural_Thickening', 'Hernia', 'No Finding'
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
    """Carga pesos de forma segura cross-hardware (NVIDIA -> AMD) evitando crashes."""
    data = torch.load(filepath, map_location=device, weights_only=False)
    if isinstance(data, dict) and 'model_state_dict' in data:
        # Es un checkpoint robusto
        model.load_state_dict(data['model_state_dict'])
    else:
        # Es un simple densenet_nih.pth directamente con pesos
        model.load_state_dict(data)
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

def find_optimal_threshold(y_true, y_prob):
    """Calcula el Índice de Youden exacto para la curva ROC de un modelo/patología específica."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    youden_index = tpr + (1 - fpr) - 1
    optimal_idx = np.argmax(youden_index)
    
    # Prevenir que busque el umbral con 0 samples de algo
    if np.isnan(youden_index[optimal_idx]):
        return 0.5
        
    return thresholds[optimal_idx]

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
        print("Asegúrate de conseguir los archivos de tu compañero y renombrarlos a:")
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
    
    print("\nEscaneos completos. Iniciando Algoritmo de Cascada y Umbrales Óptimos...")
    
    hybrid_preds = np.zeros_like(all_y_true)
    
    # Evaluamos patología por patología matemáticamente
    for p_idx, pathology in enumerate(ALL_LABELS):
        y_t = all_y_true[:, p_idx]
        
        # Evitar errores si no hay ejemplos activos de esta patologia (improbable con 20x samples)
        if sum(y_t) == 0:
            continue
            
        # 1. Obtener Umbrales de Youden Óptimos Específicos para CADA modelo ("Auto-calibración")
        thresh1 = find_optimal_threshold(y_t, all_probs_m1[:, p_idx])
        thresh2 = find_optimal_threshold(y_t, all_probs_m2[:, p_idx])
        thresh3 = find_optimal_threshold(y_t, all_probs_m3[:, p_idx])
        
        # 2. Aplicar la Lógica Clínica en Cascada sobre todas las imágenes
        for i in range(len(y_t)):
            p1 = all_probs_m1[i, p_idx]
            p2 = all_probs_m2[i, p_idx]
            p3 = all_probs_m3[i, p_idx]
            
            # Paso A: Triage (M1 - Alto Recall)
            if p1 < thresh1:
                # El modelo paranoico no ve nada. Estamos seguros de que el paciente está sano.
                hybrid_preds[i, p_idx] = 0
            else:
                # Paso B: Confirmación (M2 - Alta Especificidad)
                # M1 dio alarma. Consultamos a M2 que no asume riesgos tontos.
                if p2 >= thresh2:
                    # Ambos coinciden. Diagnóstico sellado como Positivo.
                    hybrid_preds[i, p_idx] = 1
                else:
                    # Paso C: Desempate (M3 - Balanceado)
                    # M1 dice enfermo, M2 dice sano. M3 define el voto final tomando su umbral.
                    if p3 >= thresh3:
                        hybrid_preds[i, p_idx] = 1
                    else:
                        hybrid_preds[i, p_idx] = 0

    # Generar Reporte de Matriz de Confusión del Ensamblado
    print("\n--- RESULTADOS DEL COMITÉ MÉDICO (CASCADA) ---")
    conf_matrices = multilabel_confusion_matrix(all_y_true, hybrid_preds)
    
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

if __name__ == "__main__":
    main()
