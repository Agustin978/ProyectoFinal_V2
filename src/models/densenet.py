import torch
import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights

def get_model(num_classes=14, pretrained=True):
    """
    Retorna el modelo DenseNet-121 modificado para clasificacion multi-etiqueta.
    
    Args:
        num_classes (int): Numero de clases de salida (defecto 14 para NIH).
        pretrained (bool): Si usar pesos pre-entrenados en ImageNet.
    """
    weights = DenseNet121_Weights.DEFAULT if pretrained else None
    model = densenet121(weights=weights)
    
    # DenseNet-121 tiene una capa 'classifier' al final:
    # (classifier): Linear(in_features=1024, out_features=1000, bias=True)
    
    num_ftrs = model.classifier.in_features
    
    # Reemplazamos la capa lineal
    # No usamos Sigmoid aqui si usamos BCEWithLogitsLoss en el entrenamiento (recomendado para estabilidad numerica)
    model.classifier = nn.Linear(num_ftrs, num_classes)
    
    return model
