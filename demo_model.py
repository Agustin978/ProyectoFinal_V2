import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import random
import numpy as np

# Imports locales
from src.models.densenet import get_model
from src.data.dataset import NIHChestXRayDataset

def predict_random_images(model_path, data_dir, num_images=10):
    """
    Carga el modelo y realiza predicciones sobre 10 imagenes aleatorias del dataset.
    """
    print(f"--- Iniciando Demostracion de Inferencia ---")
    
    # 1. Configuracion
    try:
        import torch_directml
        if torch_directml.is_available():
            device = torch_directml.device()
            device_name = "dml (AMD GPU)"
        else:
            print("  [INFO] torch_directml instalado pero .is_available() retorno False.")
            if torch.cuda.is_available():
                 device = torch.device('cuda')
                 device_name = f"cuda ({torch.cuda.get_device_name(0)})"
            else:
                 print("  [INFO] CUDA no disponible.")
                 device = torch.device('cpu')
                 device_name = "cpu"
    except ImportError:
        print("  [INFO] Modulo 'torch_directml' no encontrado.")
        if torch.cuda.is_available():
             device = torch.device('cuda')
             device_name = f"cuda ({torch.cuda.get_device_name(0)})"
        else:
             print("  [INFO] CUDA no disponible.")
             device = torch.device('cpu')
             device_name = "cpu"
    except Exception as e:
        print(f"  [WARN] Error inesperado detectando DirectML: {e}")
        device = torch.device('cpu')
        device_name = "cpu"

    print(f"Usando dispositivo: {device_name}")
    
    # 2. Definir transformaciones (Mismas que Validacion)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    # 3. Cargar Dataset (solo para obtener rutas de imagenes reales y etiquetas reales para comparar)
    # No necesitamos cargar todo el CSV si solo queremos 10 rutas, pero usar la clase Dataset facilita todo.
    print("Cargando indice del dataset...")
    try:
        # Usamos no_finding_keep_frac=1.0 para tener el pool completo para elegir
        dataset = NIHChestXRayDataset(data_dir=data_dir, transform=transform, no_finding_keep_frac=1.0)
    except Exception as e:
        print(f"Error cargando dataset: {e}")
        return

    # 4. Cargar Modelo
    print(f"Cargando modelo desde {model_path}...")
    model = get_model(num_classes=14, pretrained=False) # Pretrained=False porque cargaremos nuestros propios pesos
    
    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
    except FileNotFoundError:
        print(f"Error: No se encontro el archivo de pesos {model_path}. Asegurate de haber entrenado el modelo primero.")
        return
        
    model = model.to(device)
    model.eval()
    
    # 5. Seleccionar imagenes aleatorias
    indices = random.sample(range(len(dataset)), num_images)
    
    print(f"\nRealizando prediccion sobre {num_images} imagenes aleatorias:\n")
    print(f"{'Imagen':<20} | {'Etiquetas Reales':<40} | {'Predicciones (Prob > 0.5)':<40}")
    print("-" * 110)
    
    classes = dataset.all_labels
    
    with torch.no_grad():
        for idx in indices:
            image_tensor, label_tensor = dataset[idx]
            img_name = dataset.df.iloc[idx]['Image Index']
            
            # Preparar input
            image_tensor = image_tensor.unsqueeze(0).to(device) # Batch dimension
            
            # Inferencia
            output = model(image_tensor)
            probs = torch.sigmoid(output).squeeze(0).cpu().numpy()
            
            # Decodificar etiquetas reales
            real_labels = []
            for i, val in enumerate(label_tensor.numpy()):
                if val == 1:
                    real_labels.append(classes[i])
            if not real_labels:
                real_labels = ["No Finding"]
                
            # Decodificar predicciones (Umbral 0.5)
            pred_labels = []
            for i, prob in enumerate(probs):
                if prob > 0.5:
                    pred_labels.append(f"{classes[i]} ({prob:.2f})")
            
            if not pred_labels:
                pred_labels = ["No Finding"]
                
            # Formatear salida
            real_str = ", ".join(real_labels)
            pred_str = ", ".join(pred_labels)
            
            print(f"{img_name:<20} | {real_str[:40]:<40} | {pred_str:<40}")

if __name__ == "__main__":
    # Ajusta estas rutas segun tu entorno
    MODEL_PATH = "densenet_nih.pth"
    DATA_DIR = r"D:\Agustin\Facultad\ProyectoFinal\archive" # Misma ruta que en main.py
    
    if not os.path.exists(DATA_DIR):
         # Fallback para intentar encontrarlo si la ruta hardcoded falla en entorno de prueba
         # Pero asumo que el usuario tiene la ruta correcta o la editara.
         pass

    predict_random_images(MODEL_PATH, DATA_DIR)
