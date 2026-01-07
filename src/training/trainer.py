import torch
import torch.nn as nn
from tqdm import tqdm
import time

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device, pos_weight=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.pos_weight = pos_weight # Para futuro balanceo de clases

    def train_one_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            pbar.set_postfix({'loss': loss.item()})
            
        epoch_loss = running_loss / len(self.train_loader.dataset)
        return epoch_loss

    def validate(self, epoch):
        self.model.eval()
        running_loss = 0.0
        
        # Aqui podriamos acumular predicciones para calcular ROC-AUC despues
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Valid]")
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * images.size(0)
                pbar.set_postfix({'loss': loss.item()})
                
        epoch_loss = running_loss / len(self.val_loader.dataset)
        return epoch_loss
