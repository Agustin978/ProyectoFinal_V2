import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np

class NIHChestXRayDataset(Dataset):
    def __init__(self, data_dir, csv_file='Data_Entry_2017.csv', transform=None, images_dir='images'):
        """
        Args:
            data_dir (str): Directorio raíz donde se encuentran los datos.
            csv_file (str): Nombre del archivo CSV con las etiquetas.
            transform (callable, optional): Transformaciones opcionales a aplicar en la imagen.
            images_dir (str): Nombre de la carpeta que contiene las imágenes (o carpetas de imágenes).
        """
        self.data_dir = data_dir
        self.transform = transform
        
        # Ruta completa al CSV
        self.csv_path = os.path.join(data_dir, csv_file)
        
        # Cargar el dataframe
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"No se encontró el archivo CSV en: {self.csv_path}")
            
        self.df = pd.read_csv(self.csv_path)
        
        # Lista de las 14 patologías oficiales + 'Hernia' (15 en total si incluimos Hernia explícitamente, pero el estándar es 14)
        # El usuario mencionó explícitamente 'Hernia' como desbalanceada.
        # Las 14 clases estándar del dataset NIH son:
        self.all_labels = [
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
            'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
            'Pleural_Thickening', 'Hernia'
        ]
        
        # Mapeo de imagen a ruta completa
        # El dataset original a veces viene con imágenes en múltiples subcarpetas (images_001, images_002, etc.)
        # O todas en una carpeta 'images'.
        # Vamos a intentar indexar todas las imágenes en el directorio de imágenes.
        self.image_paths = {}
        full_images_dir = os.path.join(data_dir, images_dir)
        
        # Si la carpeta 'images' existe, buscamos ahí. Si no, asumimos que images_dir es vacío y buscamos en data_dir
        search_dir = full_images_dir if os.path.exists(full_images_dir) else data_dir
        
        print(f"Indexando imágenes en {search_dir}...")
        for root, dirs, files in os.walk(search_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths[file] = os.path.join(root, file)
        
        print(f"Encontradas {len(self.image_paths)} imágenes.")
        
        # Filtramos el dataframe para mantener solo las imágenes que encontramos
        # (Esto es útil si no se descargó el dataset completo o si hay discrepancias)
        initial_len = len(self.df)
        self.df = self.df[self.df['Image Index'].isin(self.image_paths.keys())].reset_index(drop=True)
        print(f"DataFrame filtrado de {initial_len} a {len(self.df)} entradas basado en imágenes disponibles.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['Image Index']
        img_path = self.image_paths[img_name]
        
        # Cargar imagen y convertir a RGB (DenseNet espera 3 canales)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # Procesar etiquetas
        labels = row['Finding Labels']
        label_vec = np.zeros(len(self.all_labels), dtype=np.float32)
        
        if labels != 'No Finding':
            for i, disease in enumerate(self.all_labels):
                if disease in labels:
                    label_vec[i] = 1.0
                    
        return image, torch.tensor(label_vec)
