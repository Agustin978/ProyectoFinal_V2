import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torchvision import transforms
import pandas as pd
import numpy as np
import time

from src.data.dataset import NIHChestXRayDataset
from src.data.transforms import RandomGaussianBlur, RandomUnsharpMask

from src.models.densenet import get_model
from src.training.trainer import Trainer

# CONFIGURACION
DATA_DIR = r"D:\Agustin\Facultad\ProyectoFinal\archive"
BATCH_SIZE = 8 # Ajustado segun VRAM (1024x1024 input original -> Resized to 224)()
LEARNING_RATE = 1e-4
EPOCHS = 10
NUM_CLASSES = 14
IMAGE_SIZE = 224
UNDERSAMPLE_RATE = 0.25 # Mantener 25% de 'No Finding'
RESUME_FROM = "best_model_T2.pth" # Ej: "best_model.pth". Aplica Transfer Learning inyectando pesos previos.
CSV_FILE = "results.csv"
CHECKPOINT_FILE = "checkpoint.pth"
EXCLUDE_LIST_FILE = "holdout_test_set.csv" # Imagenes reservadas estrictamente para test final
RANDOM_SEED = 42 # Semilla estandar izada para el split

def get_next_version(base_name):
    """Genera un sufijo _Vn para evitar sobreescribir archivos existentes."""
    if not os.path.exists(base_name):
        return base_name
    name, ext = os.path.splitext(base_name)
    version = 1
    new_name = f"{name}_V{version}{ext}"
    while os.path.exists(new_name):
        version += 1
        new_name = f"{name}_V{version}{ext}"
    return new_name

def load_checkpoint_safe(filepath, model):
    """Carga pesos via Transfer Learning inyectando parches de arquitectura y aborto duro."""
    print(f"\n[TRANSFER LEARNING] Intentando cargar pesos base desde: {filepath}")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f" No se encontró el modelo base: {filepath}")
        
    data = torch.load(filepath, map_location=torch.device('cpu'), weights_only=False)
    state_dict = data['model_state_dict'] if (isinstance(data, dict) and 'model_state_dict' in data) else data
    
    if 'classifier.1.weight' in state_dict:
        print(f"  [Parche Activado] Adaptando estructura neuronal Sequential a Linear...")
        state_dict['classifier.weight'] = state_dict.pop('classifier.1.weight')
        state_dict['classifier.bias'] = state_dict.pop('classifier.1.bias')
        keys_to_delete = [k for k in list(state_dict.keys()) if k.startswith('classifier.') and k not in ['classifier.weight', 'classifier.bias']]
        for k in keys_to_delete:
            del state_dict[k]
            
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    # Validación Dura (Hard-Abort)
    bad_keys = [k for k in missing_keys if 'classifier.weight' in k or 'classifier.bias' in k]
    if bad_keys:
        raise RuntimeError(f"\n[!] ABORTO DURO: Incompatibilidad de Arquitectura. \n"
                           f"Se perdieron las capas vitales de decisión al cargar '{filepath}'. \n"
                           f"Llaves perdidas: {bad_keys}\n"
                           f"El entrenamiento se aboró para evitar corromper los pesos sanos.")
                           
    print("  -> Transfer Learning exitoso. El modelo arrancará con este conocimiento previo.\n")
    return model

class EarlyStopping:
    """Detiene el entrenamiento si la métrica de validación no mejora luego de una paciencia dada."""
    def __init__(self, patience=5, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_auc):
        if self.best_score is None:
            self.best_score = val_auc
        elif val_auc < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping: paciencia en {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_auc
            self.counter = 0

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
    
    # Conteo correcto para la clase sana
    count_no_finding = (df['Finding Labels'] == 'No Finding').sum()
    no_finding_weight = 1.0 / count_no_finding if count_no_finding > 0 else 0.0
    
    # Pre-calcular mapa para mayor velocidad
    def get_mean_weight(labels_str):
        if labels_str == 'No Finding':
            return no_finding_weight # Corregido: Ya no se anula al grupo Sano
        
        total_weight = 0.0
        active_labels = 0
        for label in all_labels:
            if label in labels_str:
                total_weight += class_weights[label]
                active_labels += 1
                
        if active_labels > 0:
             return total_weight / active_labels
        return 0.0

    print("Calculando pesos de muestreo (Media)...")
    # Usando list comprehension simple para mayor velocidad en series de pandas
    labels_series = df['Finding Labels']
    sample_weights_list = labels_series.apply(get_mean_weight).tolist()
    
    return sample_weights_list

def main():
    global CSV_FILE, CHECKPOINT_FILE
    
    # Aplicar Auto-Versionado para no pisar ejecuciones pasadas
    CSV_FILE = get_next_version(CSV_FILE)
    CHECKPOINT_FILE = get_next_version(CHECKPOINT_FILE)
    BEST_MODEL_OUT = get_next_version("best_model.pth")
    FINAL_MODEL_OUT = get_next_version("densenet_nih.pth")
    
    print(f"Archivos de salida asignados:\n - {BEST_MODEL_OUT}\n - {CSV_FILE}\n")

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

    # 3. Dividir en Train/Validation (80/20) de forma determinista
    train_size = int(0.8 * len(full_dataset_raw))
    val_size = len(full_dataset_raw) - train_size
    
    # IMPORTANTE: Forzar una semilla estricta para garantizar que el set de validación
    # y entrenamiento sean siempres LOS MISMOS en cada reinicio.
    seed_generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_subset, val_subset = random_split(full_dataset_raw, [train_size, val_size], generator=seed_generator)
    
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
    
    # Fase 6: Inyectar conocimiento ajeno (Transfer Learning de companeros/Kaggle)
    if RESUME_FROM is not None:
        model = load_checkpoint_safe(RESUME_FROM, model)
        
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
    
    # Schedulers y Controladores de Entrenamiento
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    early_stopping = EarlyStopping(patience=5, delta=0.001)
    best_val_auc = 0.0
    results = []

    # 6.5 Cargar Checkpoint para reanudar mid-run (Ignorado al usar _Vn)
    start_epoch = 1
    # Nota: Como ahora auto-versionamos CHECKPOINT_FILE, os.path.exists
    # siempre será Falso a menos que se fuerce el nombre manualmente.
    # El usuario pidió empezar limpio siempre vía RESUME_FROM.
    print(f"El entrenamiento volcará progresos en {CHECKPOINT_FILE} desde cero.")

    # 7. Entrenador
    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, device)

    # 8. Bucle principal
    
    for epoch in range(start_epoch, EPOCHS + 1):
        start_time = time.time()
        train_loss = trainer.train_one_epoch(epoch)
        val_loss, val_auc = trainer.validate(epoch)
        end_time = time.time()
        
        epoch_duration = end_time - start_time
        
        print(f"Epoch {epoch}/{EPOCHS} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val AUC: {val_auc:.4f} - LR: {scheduler.get_last_lr()[0]:.6f} - Time: {epoch_duration:.2f}s")
        
        # Actualizar Schedulers y Early Stopping
        scheduler.step()
        early_stopping(val_auc)
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), BEST_MODEL_OUT)
            print(f"  -> Nuevo mejor modelo guardado ({BEST_MODEL_OUT}) | AUC: {best_val_auc:.4f}")
        
        results.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_auc': val_auc,
            'lr': scheduler.get_last_lr()[0],
            'duration_sec': epoch_duration
        })
        
        # Guardar CSV cada epoca
        pd.DataFrame(results).to_csv(CSV_FILE, index=False)
        
        # Guardar Checkpoint Robusto
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': train_loss,
            'best_val_auc': best_val_auc,
            'config': {
                 'random_seed': RANDOM_SEED,
                 'batch_size': BATCH_SIZE,
                 'learning_rate': LEARNING_RATE,
                 'undersample_rate': UNDERSAMPLE_RATE
            }
        }
        torch.save(checkpoint, CHECKPOINT_FILE)
        
        # Guardar modelo final en cada iteracion por prevencion
        torch.save(model.state_dict(), FINAL_MODEL_OUT)
        
        if early_stopping.early_stop:
            print(f"\n[!] Entrenenamiento detenido tempranamente por Early Stopping (Paciencia {early_stopping.patience} alcanzada).")
            break
        
    print(f"Entrenamiento finalizado. El mejor modelo esta guardado como '{BEST_MODEL_OUT}'.")

if __name__ == '__main__':
    # Fix para multiprocessing en Windows
    import multiprocessing
    multiprocessing.freeze_support()
    main()
