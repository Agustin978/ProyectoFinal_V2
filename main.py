import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset, WeightedRandomSampler
from torchvision import transforms
import pandas as pd
import numpy as np
import time

from src.data.dataset import NIHChestXRayDataset
from src.data.transforms import RandomGaussianBlur, RandomUnsharpMask

from src.data.dataset import NIHChestXRayDataset
from src.models.densenet import get_model
from src.training.trainer import Trainer

# CONFIGURACION
DATA_DIR = r"D:\Agustin\Facultad\ProyectoFinal\archive"
BATCH_SIZE = 8 # Ajustado segun VRAM (1024x1024 input original -> Resized to 224)()
LEARNING_RATE = 1e-4
EPOCHS = 15
NUM_CLASSES = 14
IMAGE_SIZE = 224
UNDERSAMPLE_RATE = 0.30 # Mantener 30% de 'No Finding'
CSV_FILE = "results.csv"
CHECKPOINT_FILE = "checkpoint.pth"
EXCLUDE_LIST_FILE = "holdout_test_set.csv" # Imagenes reservadas estrictamente para test final

class AugmentedDataset(Dataset):
    """Envolvedor para aplicar transformaciones a un subconjunto."""
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, idx):
        x, y = self.subset[idx]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)

def calculate_sampler_weights(subset, dataset):
    """
    Calcula los pesos para el WeightedRandomSampler.
    Asigna un peso basado en la clase más rara presente en la imagen.
    Se enfoca en clases minoritarias (aprox. < 2000 muestras).
    """
    # 1. Contar frecuencias globales en el subconjunto de entrenamiento (para ser precisos)
    #    o simplemente usar las frecuencias del dataframe completo para simplificar trabajando con índices.
    #    Usemos los índices del subconjunto para buscar en el dataframe original.
    
    df = dataset.df.iloc[subset.indices]
    all_labels = dataset.all_labels
    
    # Calcular conteos por clase en este subconjunto
    label_counts = {}
    for label in all_labels:
        count = df['Finding Labels'].str.contains(label, regex=False).sum()
        label_counts[label] = count
        
    print("Conteo de clases en Training Subset:", label_counts)
    
    # Calcular peso por clase (Frecuencia Inversa)
    # N = len(df)
    # class_weight = N / count
    class_weights = {}
    for label, count in label_counts.items():
        if count > 0:
            class_weights[label] = 1.0 / count
        else:
            class_weights[label] = 0.0
            
    # Asignar peso específico a cada muestra
    sample_weights = []
    
    # Pre-calcular mapa para mayor velocidad
    def get_max_weight(labels_str):
        if labels_str == 'No Finding':
            return 0.05 / len(df) # Peso bajo para No Finding relativo a patologias
            # O simplemente 1/count_no_finding, pero queremos boostear las raras.
        
        w = 0.0
        for label in all_labels:
            if label in labels_str:
                w = max(w, class_weights[label])
        return w

    print("Calculando pesos de muestreo...")
    # Usando list comprehension simple para mayor velocidad en series de pandas
    labels_series = df['Finding Labels']
    sample_weights_list = labels_series.apply(get_max_weight).tolist()
    
    return sample_weights_list

def main():
    # Deteccion de dispositivo con soporte para DirectML (AMD en Windows)
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

    # 1. Definir Transformaciones
    # Transformaciones Base (Validacion)
    val_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

    # Transformaciones Aumentadas (Entrenamiento)
    # Incluye geometricas y de frecuencia
    train_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        # transforms.RandomCrop -> Cuidado con perder info, mejor rotacion leve
        # RandomGaussianBlur(p=0.3), # Transformación de Frecuencia Personalizada
        RandomUnsharpMask(p=0.3),  # Transformación de Frecuencia Personalizada
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

    # 2. Cargar Dataset Base (Sin transformaciones aun)
    try:
        print(f"Cargando datos desde {DATA_DIR}...")
        # Pasamos transform=None para obtener PIL Images crudas
        # Aplicamos la exclusión estricta de las imágenes separadas en holdout_test_set.csv
        full_dataset_raw = NIHChestXRayDataset(
            data_dir=DATA_DIR, 
            transform=None, 
            no_finding_keep_frac=UNDERSAMPLE_RATE,
            exclude_list_path=EXCLUDE_LIST_FILE
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # 3. Dividir en Train/Validation (80/20)
    train_size = int(0.8 * len(full_dataset_raw))
    val_size = len(full_dataset_raw) - train_size
    train_subset, val_subset = random_split(full_dataset_raw, [train_size, val_size])
    
    # 4. Envolver subsets con sus respectivas transformaciones
    train_dataset = AugmentedDataset(train_subset, transform=train_transforms)
    val_dataset = AugmentedDataset(val_subset, transform=val_transforms)
    
    print(f"Datos de entrenamiento: {len(train_dataset)}")
    print(f"Datos de validacion: {len(val_dataset)}")

    # 5. Configurar Sampler para Entrenamiento
    sample_weights = calculate_sampler_weights(train_subset, full_dataset_raw)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset), # Mismo tamaño, pero con reemplazo (oversampling)
        replacement=True
    )

    # 6. DataLoaders
    # Shuffle debe ser False cuando usamos sampler
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, sampler=sampler, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # 5. Modelo
    model = get_model(num_classes=NUM_CLASSES, pretrained=True)
    model = model.to(device)

    # 6. Loss y Optimizador
    # Calcular pesos de clase (Weighted Loss)
    # Nota: Idealmente deberiamos calcular esto solo sobre train_dataset para evitar leakage tambien en la Loss,
    # pero usar full_dataset es una aproximacion estandar aceptable.
    raw_pos_weights = full_dataset_raw.get_pos_weight()
    pos_weights = torch.sqrt(raw_pos_weights) # Relajar pesos para evitar over-correction junto con Sampler
    pos_weights = pos_weights.to(device)
    
    # Para multi-label classification usamos BCEWithLogitsLoss con pos_weight
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights) 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 6.5 Cargar Checkpoint si existe
    start_epoch = 1
    if os.path.exists(CHECKPOINT_FILE):
        print(f"Cargando checkpoint desde {CHECKPOINT_FILE}...")
        try:
            checkpoint = torch.load(CHECKPOINT_FILE, map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resumiendo entrenamiento desde la epoca {start_epoch}")
        except Exception as e:
            print(f"Error cargando checkpoint: {e}. Se iniciara desde cero.")
    else:
        print("No se encontro checkpoint. Iniciando entrenamiento desde cero.")

    # 7. Entrenador
    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, device)

    # 8. Bucle principal
    results = []
    
    for epoch in range(start_epoch, EPOCHS + 1):
        start_time = time.time()
        train_loss = trainer.train_one_epoch(epoch)
        val_loss, val_auc = trainer.validate(epoch)
        end_time = time.time()
        
        epoch_duration = end_time - start_time
        
        print(f"Epoch {epoch}/{EPOCHS} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val AUC: {val_auc:.4f} - Time: {epoch_duration:.2f}s")
        
        results.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_auc': val_auc,
            'duration_sec': epoch_duration
        })
        
        # Guardar CSV cada epoca
        # Si estamos resumiendo, deberiamos idealmente hacer append, pero por simplicidad reescribimos con lo que tenemos en memoria
        # (Nota: 'results' empieza vacio aqu. Si quisieramos historial completo en CSV, deberiamos leer el CSV existente antes)
        pd.DataFrame(results).to_csv(CSV_FILE, index=False)
        
        # Guardar Checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss
        }
        torch.save(checkpoint, CHECKPOINT_FILE)
        
        # Guardar modelo final (solo pesos, para compatibilidad con inferencia)
        torch.save(model.state_dict(), "densenet_nih.pth")
        
    print("Entrenamiento finalizado y modelo guardado.")

if __name__ == '__main__':
    # Fix para multiprocessing en Windows
    import multiprocessing
    multiprocessing.freeze_support()
    main()
