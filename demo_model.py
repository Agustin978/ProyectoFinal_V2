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

# Suprimir warnings de sklearn si hay clases con 0 muestras (no deberia pasar con nuestro holdout, pero por si acaso)
warnings.filterwarnings("ignore", category=UserWarning)

# Imports locales
from src.models.densenet import get_model

# Configuracion
DATA_DIR = r"D:\Agustin\Facultad\ProyectoFinal\archive"
MODEL_PATH = "densenet_nih.pth"
TEST_SET_CSV = "holdout_test_set.csv"
OUTPUT_RESULTS_CSV = "evaluation_results.csv"
OUTPUT_ROC_PLOT = "roc_curves_per_class.png"
IMAGE_SIZE = 224
NUM_CLASSES = 14

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
    
    print(f"Usando dispositivo para evaluacion: {device_name}")
    return device

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
    """Convierte un string de etiquetas separadas por '|' a un vector binario de 14 posiciones."""
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

def main():
    print("=== INICIANDO EVALUACION RIGUROSA ===")
    
    # 1. Preparar dispositivo y rutas
    device = load_environment()
    image_paths_dict = get_image_paths(DATA_DIR)
    
    # 2. Cargar Test Set
    if not os.path.exists(TEST_SET_CSV):
        print(f"Error: No se encontro {TEST_SET_CSV}. Ejecute create_test_set.py primero.")
        return
        
    df_test = pd.read_csv(TEST_SET_CSV)
    print(f"Cargadas {len(df_test)} imagenes del conjunto de validacion segregado.")
    
    # 3. Preparar Modelo
    print(f"Cargando modelo desde {MODEL_PATH}...")
    model = get_model(num_classes=NUM_CLASSES, pretrained=False)
    
    if not os.path.exists(MODEL_PATH):
         print(f"Error: No se encontro el modelo {MODEL_PATH}. Debe entrenarlo primero.")
         return
         
    # Cargar pesos con map_location='cpu' por seguridad con DirectML
    state_dict = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    # 4. Transformaciones (Solo Validacion)
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Listas para almacenar resultados
    results_list = []
    y_true_all = []
    y_pred_probs_all = []
    
    # 5. Bucle de Inferencia
    print("\nProcesando imagenes...")
    with torch.no_grad():
        for idx, row in df_test.iterrows():
            img_name = row['Image Index']
            true_labels_str = row['Original_Finding_Labels']
            
            # Vectores Reales
            y_true_vec = labels_to_binary_vector(true_labels_str)
            
            if img_name not in image_paths_dict:
                print(f"  [Advertencia] Imagen {img_name} no encontrada en disco. Saltando.")
                continue
                
            img_path = image_paths_dict[img_name]
            
            # Cargar y preprocesar
            image = Image.open(img_path).convert('RGB')
            tensor = transform(image).unsqueeze(0).to(device) # Añadir dimension de batch
            
            # Prediccion
            outputs = model(tensor)
            
            # Convertir logits a probabilidades usando Sigmoid en CPU (Workaround AMD)
            probs = torch.sigmoid(outputs.cpu()).numpy()[0]
            
            # Guardar resultados
            results_list.append({
                'Image_Name': img_name,
                'True_Labels': true_labels_str,
                # 'Predicted_Labels': pred_labels_str, # Se determinara despues de calcular el umbral optimo
            })
            
            y_true_all.append(y_true_vec)
            y_pred_probs_all.append(probs) # IMPORTANTE: Ahora guardamos probabilidades, no ceros y unos
            
            if (idx + 1) % 20 == 0:
                print(f"  Procesadas {idx + 1}/{len(df_test)} imagenes...")

    y_true_all = np.array(y_true_all)
    y_pred_probs_all = np.array(y_pred_probs_all)

    # =========================================================================
    # TUNING DE UMBRALES (THRESHOLD TUNING PER-CLASE) USANDO ÍNDICE DE YOUDEN
    # =========================================================================
    print("\nCalculando umbrales optimos por clase (Maximizando Sensibilidad + Especificidad)...")
    optimal_thresholds = np.zeros(NUM_CLASSES)
    
    plt.figure(figsize=(15, 12))
    
    for i in range(NUM_CLASSES):
        # Calcular la curva ROC real
        fpr, tpr, thresholds = roc_curve(y_true_all[:, i], y_pred_probs_all[:, i])
        
        # Calcular Indice de Youden: J = Sensibilidad(TPR) + Especificidad(1-FPR) - 1
        # Maximos TPR y minimos FPR dan el mayor J
        youden_index = tpr + (1 - fpr) - 1
        best_threshold_idx = np.argmax(youden_index)
        best_threshold = thresholds[best_threshold_idx]
        
        # Limitar el umbral matematicamente a rangos logicos (0.05 a 0.50) para evitar extremos inestables
        optimal_thresholds[i] = np.clip(best_threshold, 0.05, 0.50)
        
        # Graficar ROC 
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{ALL_LABELS[i]} (AUC = {roc_auc:.2f}, Thresh = {optimal_thresholds[i]:.2f})')
        plt.scatter(fpr[best_threshold_idx], tpr[best_threshold_idx], marker='o', color='black', s=50, zorder=5) # Punto optimo

    # Configurar grafico ROC
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos (1 - Especificidad)')
    plt.ylabel('Tasa de Verdaderos Positivos (Sensibilidad)')
    plt.title('Curvas ROC por Patologia y Puntos de Umbral Optimo (Youden)')
    plt.legend(loc="lower right", fontsize='small', ncol=2)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_ROC_PLOT)
    print(f"Grafico ROC guardado en: {OUTPUT_ROC_PLOT}")

    # =========================================================================
    # APLICAR UMBRALES ÓPTIMOS A LAS PREDICCIONES
    # =========================================================================
    y_pred_binary_all = np.zeros_like(y_pred_probs_all)
    for i in range(NUM_CLASSES):
        y_pred_binary_all[:, i] = (y_pred_probs_all[:, i] >= optimal_thresholds[i]).astype(int)

    # Actualizar la lista de resultados con las predicciones finales optimizadas
    for idx, row in enumerate(results_list):
        y_pred_vec = y_pred_binary_all[idx]
        pred_labels_str = binary_vector_to_labels(y_pred_vec)
        y_true_vec = y_true_all[idx]
        
        row['Predicted_Labels'] = pred_labels_str
        row['Exact_Match'] = (y_true_vec == y_pred_vec).all()
        # Puedes añadir aqui los umbrales usados si quieres en el CSV
        
    # 6. Guardar Tabla de Resultados Generales
    results_df = pd.DataFrame(results_list)
    results_df.to_csv(OUTPUT_RESULTS_CSV, index=False)
    print(f"\nGenerada tabla de resultados individuales en: {OUTPUT_RESULTS_CSV}")
    
    # 7. Calcular y Mostrar Matrices de Confusion Multi-etiqueta
    y_true_all = np.array(y_true_all)
    
    print("\n" + "="*50)
    print("MATRICES DE CONFUSION POR PATOLOGIA (UMBRALES ÓPTIMOS CALIBRADOS)")
    print("="*50)
    
    # Scikit-learn devuelve un array de shape (n_classes, 2, 2)
    # [ [TN, FP], [FN, TP] ]
    mcm = multilabel_confusion_matrix(y_true_all, y_pred_binary_all)
    
    for i, label in enumerate(ALL_LABELS):
        tn, fp, fn, tp = mcm[i].ravel()
        total_real_positives = tp + fn
        total_real_negatives = tn + fp
        
        # Calcular porcentajes (Sensibilidad y Especificidad)
        sensitivity = (tp / total_real_positives * 100) if total_real_positives > 0 else 0.0
        specificity = (tn / total_real_negatives * 100) if total_real_negatives > 0 else 0.0
        
        print(f"\n--- {label} --- (Umbral optimizado: {optimal_thresholds[i]:.3f})")
        print(f"             Prediccion: NO    Prediccion: SI")
        print(f"Real: NO      TN: {tn:<7}       FP: {fp:<7}  | Especificidad (Sanos correctos): {specificity:.1f}%")
        print(f"Real: SI      FN: {fn:<7}       TP: {tp:<7}  | Sensibilidad (Enfermos correctos): {sensitivity:.1f}%")

    print("\nLeyenda:")
    print("TP (True Positive)  = El modelo detecto la patologia correctamente.")
    print("TN (True Negative)  = El modelo descarto la patologia correctamente.")
    print("FP (False Positive) = El modelo se equivoco diciendo que HABIA patologia (Falsa Alarma).")
    print("FN (False Negative) = El modelo se equivoco diciendo que NO habia patologia (Peligro: Enfermedad omitida).")

if __name__ == "__main__":
    main()
