import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from src.data.dataset import NIHChestXRayDataset
from src.models.densenet import get_model
from src.training.trainer import Trainer

# CONFIGURACION
DATA_DIR = r"D:\Agustin\Facultad\ProyectoFinal\archive"
BATCH_SIZE = 8 # Ajustar segun VRAM (1024x1024 input original -> Resized to 224)
LEARNING_RATE = 1e-4
EPOCHS = 10
NUM_CLASSES = 14
IMAGE_SIZE = 224

def main():
    # Deteccion de dispositivo con soporte para DirectML (AMD en Windows)
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

    print(f"Usando dispositivo: {device_name}")
    # 1. Definir Transformaciones
    # Es crucial redimensionar a 224x224 para DenseNet preentrenada
    data_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

    # 2. Cargar Dataset
    try:
        print(f"Cargando datos desde {DATA_DIR}...")
        full_dataset = NIHChestXRayDataset(data_dir=DATA_DIR, transform=data_transforms)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # 3. Dividir en Train/Validation (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Datos de entrenamiento: {len(train_dataset)}")
    print(f"Datos de validacion: {len(val_dataset)}")

    # 4. DataLoaders
    # num_workers=0 para evitar problemas en Windows con spawn/fork en scripts simples, ajustar segun necesidad
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # 5. Modelo
    model = get_model(num_classes=NUM_CLASSES, pretrained=True)
    model = model.to(device)

    # 6. Loss y Optimizador
    # Para multi-label classification usamos BCEWithLogitsLoss
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 7. Entrenador
    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, device)

    # 8. Bucle principal
    for epoch in range(1, EPOCHS + 1):
        train_loss = trainer.train_one_epoch(epoch)
        val_loss = trainer.validate(epoch)
        
        print(f"Epoch {epoch}/{EPOCHS} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
        
    # Guardar modelo final
    torch.save(model.state_dict(), "densenet_nih.pth")
    print("Entrenamiento finalizado y modelo guardado.")

if __name__ == '__main__':
    # Fix para multiprocessing en Windows
    import multiprocessing
    multiprocessing.freeze_support()
    main()
