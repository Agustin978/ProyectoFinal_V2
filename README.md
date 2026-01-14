# Documentación del Proyecto: Clasificación de Rayos X de Tórax NIH con DenseNet-121

## 1. Introducción
Este proyecto implementa un sistema de aprendizaje profundo (Deep Learning) para la clasificación multi-etiqueta de 14 patologías torácicas comunes utilizando imágenes de Rayos X. El modelo base es **DenseNet-121**, pre-entrenado en ImageNet, adaptado mediante Transfer Learning.

**Nuevas Características:**
*   **Manejo de Desbalance de Clases:** Implementación de **Weighted Loss** y **Undersampling** de la clase mayoritaria ("No Finding").
*   **Métricas Avanzadas:** Cálculo de **AUC-ROC** (Area Under the Receiver Operating Characteristic Curve) para evaluar mejor el rendimiento en datos desbalanceados.
*   **Registro de Resultados:** Exportación automática de métricas por época a un archivo CSV.

## 2. Estructura del Proyecto

```
nih_chest_xray_classification/
│
├── main.py                  # Punto de entrada principal para el entrenamiento
├── verify_setup.py          # Script de verificación de entorno
├── requirements.txt         # Dependencias del proyecto
│
└── src/
    ├── data/
    │   └── dataset.py       # Definición de la clase Dataset (Pytorch)
    ├── models/
    │   └── densenet.py      # Definición de la arquitectura del modelo
    └── training/
        └── trainer.py       # Lógica del bucle de entrenamiento y validación
```

## 3. Detalle del Código y Componentes

A continuación se detalla el propósito de cada módulo, función y variable importante.

### 3.1 `main.py`
Es el script orquestador del entrenamiento.

**Constantes de Configuración:**
*   `DATA_DIR`: Ruta al directorio donde se encuentra el dataset (imágenes y CSV).
*   `BATCH_SIZE`: Tamaño del lote de imágenes (ej. 16). Ajustar según la memoria VRAM disponible.
*   `LEARNING_RATE`: Tasa de aprendizaje para el optimizador (ej. 1e-4).
*   `EPOCHS`: Número total de épocas de entrenamiento.
*   `NUM_CLASSES`: 14, correspondiente a las patologías del dataset NIH.
*   `IMAGE_SIZE`: 224, resolución de entrada requerida por DenseNet.
*   `UNDERSAMPLE_RATE`: Fracción (0.0 - 1.0) de la clase "No Finding" a mantener para reducir el desbalance (ej. 0.25).
*   `CSV_FILE`: Nombre del archivo donde se guardarán los resultados detallados del entrenamiento.

**Flujo Principal (`main()`):**
1.  **Configuración de Dispositivo**:
    *   Intenta usar **DirectML** (para AMD en Windows) si está disponible (`dml`).
    *   Si no, busca **CUDA** (NVIDIA).
    *   Por defecto cae en **CPU**.
2.  **Transformaciones**: Define `transforms.Compose` incluyendo redimensionamiento a 224x224, conversión a Tensor y normalización con medias/desviaciones estándar de ImageNet.
3.  **Carga de Datos**: Instancia `NIHChestXRayDataset` aplicando **undersampling** a la clase "No Finding" según `UNDERSAMPLE_RATE`.
4.  **División**: Separa el dataset en 80% entrenamiento y 20% validación usando `random_split`.
5.  **DataLoaders**: Crea iteradores para entrenamiento y validación.
6.  **Modelo**: Instancia el modelo usando `get_model`.
7.  **Loss y Optimizador**: 
    *   Calcula los pesos positivos (`pos_weights`) del dataset para darle más importancia a las clases menos frecuentes.
    *   Usa `BCEWithLogitsLoss` con estos pesos (`pos_weight`) para mitigar el desbalance.
    *   Usa el optimizador `Adam`.
8.  **Bucle de Entrenamiento**: 
    *   Itera por las épocas llamando a `trainer.train_one_epoch` y `trainer.validate`.
    *   Calcula métricas como **AUC-ROC** promedio.
    *   Guarda los resultados (Loss Train, Loss Val, AUC Val, Tiempo) en un archivo **CSV**.

### 3.2 `src/data/dataset.py`
Contiene la clase `NIHChestXRayDataset`, encargada de leer las imágenes y procesar las etiquetas.

**Clase `NIHChestXRayDataset`:**
*   `__init__(data_dir, csv_file, transform, images_dir, no_finding_keep_frac)`: Constructor actualizado.
    *   `no_finding_keep_frac`: Controla qué porcentaje de muestras con "No Finding" se conservan. Ayuda a reducir el desbalance extremo de esta clase.
    *   `self.all_labels`: Lista con los nombres de las 14 patologías (Atelectasis, Cardiomegaly, etc.).
    *   `self.image_paths`: Diccionario que mapea el nombre del archivo de imagen a su ruta absoluta en disco.
    *   `self.df`: DataFrame de Pandas filtrado que contiene solo las imágenes encontradas.
*   `__len__()`: Retorna la cantidad total de imágenes disponibles.
*   `__getitem__(idx)`: Método para obtener una muestra.
    *   Carga la imagen usando PIL y la convierte a RGB.
    *   Aplica transformaciones (resize, normalize).
    *   Procesa las etiquetas de texto (ej. "Infiltration|Mass") a un vector One-Hot (ej. `[0, 0, 1, 0, 1, ...]`).
*   `get_pos_weight()`: Nuevo método que calcula los pesos para la función de pérdida.
    *   Devuelve un tensor con el peso para cada clase, calculado como `(N_negativos) / (N_positivos)`. Esto fuerza al modelo a prestar más atención a las patologías reales.

### 3.3 `src/models/densenet.py`
Define la arquitectura del modelo.

**Función `get_model(num_classes, pretrained)`:**
*   Carga `densenet121` de `torchvision.models`.
*   Si `pretrained=True`, carga los pesos aprendidos en ImageNet, lo cual acelera la convergencia y mejora la precisión (Transfer Learning).
*   **Modificación Clave**: Reemplaza la última capa lineal (`classifier`) que originalmente tiene 1000 salidas, por una nueva capa `nn.Linear` con `num_classes` (14) salidas.

### 3.4 `src/training/trainer.py`
Encapsula la lógica de entrenamiento para mantener `main.py` limpio.

**Clase `Trainer`:**
*   `__init(...)`: Guarda referencias al modelo, loaders, criterio y optimizador.
*   `train_one_epoch(epoch)`:
    *   Pone el modelo en modo entrenamiento (`model.train()`).
    *   Itera sobre el `train_loader`.
    *   Realiza el paso forward (predicción), cálculo de loss, backward (gradientes) y optimización (`optimizer.step()`).
    *   Retorna el loss promedio de la época.
*   `validate(epoch)`:
    *   Pone el modelo en modo evaluación (`model.eval()`).
    *   Desactiva el cálculo de gradientes (`torch.no_grad()`) para ahorrar memoria.
    *   Calcula el loss sobre el conjunto de validación.
    *   **Nuevo**: Calcula el AUC-ROC (Area Under the Curve) promedio usando `sklearn.metrics.roc_auc_score` para evaluar la calidad de las predicciones independientemente del umbral de decisión.
    *   Retorna `val_loss` y `val_auc`.

## 4. Diagrama de Funcionamiento

El siguiente diagrama muestra cómo fluyen los datos desde el disco hasta el entrenamiento del modelo.

```mermaid
graph TD
    subgraph "Datos"
        A[Imágenes Rayos X] --> B[Dataset (dataset.py)]
        C[CSV Etiquetas] --> B
    end

    subgraph "Procesamiento"
        B --> D[Transformaciones]
        D -->|Resize 224x224 & Norm| E[Tensores]
        E --> F[DataLoader (batching)]
    end

    subgraph "Modelo (DenseNet-121)"
        F --> G[Capas Convolucionales]
        G --> H[Capa Clasificadora Personalizada]
        H --> I[Logits (14 Clases)]
    end

    subgraph "Entrenamiento (Trainer)"
        I --> J[Cálculo de Loss (BCEWithLogits)]
        J -->|Con Pesos de Clase| K[Backpropagation]
        K --> L[Optimizador (Adam)]
        L -->|Actualizar Pesos| G
        J -.-> M[Cálculo de AUC-ROC]
        M -.-> N[CSV Log]
    end
```

## 5. Cómo Ejecutarlo

### Requisitos Previos
Asegúrate de tener instalado Python y las dependencias (se recomienda usar un entorno virtual):

```bash
pip install -r requirements.txt
```

*(El archivo `requirements.txt` contiene: torch, torchvision, pandas, pillow, tqdm y torch-directml para AMD)*

### Configuración
Abre `main.py` y verifica las variables de configuración al inicio:

```python
DATA_DIR = r"Ruta\A\Tu\Dataset"  # Ajusta esto a donde descomprimiste el dataset
BATCH_SIZE = 16                  # Bajar si tienes poca memoria de video (VRAM)
```

### Ejecutar Entrenamiento
Desde la terminal, en la carpeta raíz del proyecto:

```bash
python main.py
```

El script mostrará el progreso época por época, imprimiendo el Loss de entrenamiento y validación. Al finalizar, guardará el modelo entrenado como `densenet_nih.pth`.
