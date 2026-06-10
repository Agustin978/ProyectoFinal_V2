import torch
import torch.nn as nn
from tqdm import tqdm
import time
import numpy as np
from sklearn.metrics import roc_auc_score

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device, pos_weight=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.pos_weight = pos_weight # Tensor de pesos positivos por clase

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
        
        all_labels = []
        all_preds = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Valid]")
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Acumular para AUC
                preds = torch.sigmoid(outputs)
                all_labels.append(labels.cpu().numpy())
                all_preds.append(preds.detach().cpu().numpy())
                
                running_loss += loss.item() * images.size(0)
                pbar.set_postfix({'loss': loss.item()})
                
        epoch_loss = running_loss / len(self.val_loader.dataset)
        
        # Calcular AUC
        all_labels = np.concatenate(all_labels)
        all_preds = np.concatenate(all_preds)
        
        try:
            epoch_auc = roc_auc_score(all_labels, all_preds, average='macro')
            if np.isnan(epoch_auc):
                 print("Warning: AUC is NaN (possibly due to missing classes in validation set).")
                 epoch_auc = 0.0
        except ValueError as e:
            print(f"Warning: Error calculating AUC: {e}")
            epoch_auc = 0.0
            
        return epoch_loss, epoch_auc
