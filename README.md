# Documentación del Proyecto: Clasificación de Rayos X de Tórax NIH con DenseNet-121

## 1. Introducción
Este proyecto implementa un sistema de aprendizaje profundo (Deep Learning) para la clasificación multi-etiqueta de 14 patologías torácicas comunes utilizando imágenes de Rayos X. El modelo base es **DenseNet-121**, pre-entrenado en ImageNet, adaptado mediante Transfer Learning.

**Nuevas Características Implementadas:**
*   **Manejo de Desbalance Avanzado:** Uso combinado de **Undersampling** (clase mayoritaria "No Finding") y **WeightedRandomSampler** para forzar lotes equilibrados.
*   **Aumentación de Datos (Data Augmentation):** Aplicación de transformaciones geométricas (Flips, Rotaciones) y filtros de frecuencia (UnsharpMask) para generalización.
*   **Entrenamiento Reanudable (Checkpoints Robustos):** Sistema de guardado y carga automática (`checkpoint.pth`) de modelo, optimizador, scheduler y configuración de semillas (`random_split`) garantizando 100% de reproducibilidad ante interrupciones.
*   **Optimizadores de Entrenamiento Avanzado:** Implementación de `CosineAnnealingWarmRestarts` para la tasa de aprendizaje, y `EarlyStopping` preventivo (paciencia de 5 épocas) para evitar *overfitting* al monitorear el Val AUC.
*   **Sampling sin Sesgo Multietiqueta:** Uso de peso promedio (`get_mean_weight`) en el `WeightedRandomSampler` para no sobredimensionar y sesgar las co-ocurrencias de enfermedades raras.
*   **Soporte Multi-Dispositivo Robusto:** Detección de gráficas AMD en Windows vía `torch-directml` con workaround específico para cálculos de pérdida matemáticamente incompatibles.
*   **Métricas Dinámicas y Calibración:** Cálculo de **AUC-ROC** y Sensibilidad Per-Clase en vivo durante el entrenamiento, con búsqueda final del umbral óptimo (Índice de Youden) pos-entrenamiento.

## 2. Estructura del Proyecto

```
nih_chest_xray_classification/
│
├── main.py                  # Punto de entrada principal para el entrenamiento
├── evaluate_model.py        # Evaluación rigurosa sobre el Test Set de reserva
├── verify_setup.py          # Script de verificación de entorno
├── requirements.txt         # Dependencias del proyecto
│
└── src/
    ├── data/
    │   ├── create_test_set.py # Script para extraer el Test Set puro (Holdout)
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
*   `CHECKPOINT_FILE`: Archivo (`checkpoint.pth`) para guardar el estado completo y permitir retomar el entrenamiento si se interrumpe.
*   `EXCLUDE_LIST_FILE`: Archivo CSV (`holdout_test_set.csv`) que contiene las imágenes extraídas previamente que **bajo ninguna circunstancia** deben incluirse en el entrenamiento.

**Flujo Principal (`main()`):**
1.  **Configuración de Dispositivo**:
    *   Intenta usar **DirectML** (para AMD en Windows) si está disponible (`dml`). Maneja fallos informativos.
    *   Si no, busca **CUDA** (NVIDIA), y por último cae en **CPU**.
2.  **Transformaciones**: Define separadamente las `val_transforms` (Solo redimensionar y normalizar) y las `train_transforms` (Agregando **Data Augmentation** activo como RandomHorizontalFlip, RandomRotation y filtros).
3.  **Carga de Datos**: 
    *   Excluye estrictamente las imágenes listadas en `EXCLUDE_LIST_FILE` garantizando 0% de fuga de datos (Data Leakage).
    *   Instancia `NIHChestXRayDataset` aplicando **undersampling** a la clase "No Finding" según `UNDERSAMPLE_RATE`.
4.  **División**: Separa el dataset restante en 80% entrenamiento y 20% validación (Epoch-level) usando `random_split`.
5.  **DataLoaders y Samplers**: 
    *   Crea un `WeightedRandomSampler` que usa `get_mean_weight()`. Esto calcula el promedio de rareza anatómica real de la imagen, combatiendo el desbalance sin perjudicar imágenes con patologías múltiples.
    *   Crea los iteradores de entrenamiento (usando el Sampler) y validación.
6.  **Modelo**: Instancia el modelo (`get_model`).
7.  **Loss y Optimizador**: 
    *   Calcula los pesos del dataset pero los *relaja* con una raíz cuadrada (`torch.sqrt`) para evitar que el Sampler provoque alucinaciones (Falsos Positivos).
    *   Usa el optimizador `Adam`.
8.  **Reanudación Automática**: Revisa si existe `checkpoint.pth` y restablece todo a como estaba (época, pesos, estado adaptativo de Adam).
9.  **Bucle de Entrenamiento**: 
    *   Entrena y valida por épocas.
    *   Mide Loss y **AUC-ROC**.
    *   Guarda métricas (CSV) y sobreescribe el `checkpoint.pth` para proteger el progreso.

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
    *   Se usa `roc_auc_score` para ver qué tan bueno es el ranking del modelo (independientemente de un punto de corte fijo).
    *   Retorna `val_loss` y `val_auc`.

### 3.5 Segregación y Evaluación Científica
*   **`src/data/create_test_set.py`**: Este script debe ejecutarse antes de cualquier entrenamiento. Recorre el CSV original, selecciona matemáticamente **20 imágenes aleatorias por cada una de las 14 patologías** y genera un listado "sagrado" (`holdout_test_set.csv`) de ~280 imágenes que jamás serán vistas por la red neuronal durante su estudio. Con 20 muestras, el intervalo de confianza estadístico mejora drásticamente (±22%).
*   **`evaluate_model.py`**: Es el auditor final. Carga el `densenet_nih.pth` ya entrenado y lo evalúa **únicamente** sobre las ~280 imágenes de la lista segregada. Por cada patología, busca el Umbral Óptimo matemáticamente (Índice de Youden), traza las Curvas ROC, y emite una **Sensibilidad** y **Especificidad** precisas, además de guardar una tabla auditable (`evaluation_results.csv`). Activa `model.eval()` y desactiva gradientes descartando fuga de datos.

## 4. Diagrama de Funcionamiento

El siguiente diagrama muestra cómo fluyen los datos desde el disco hasta el entrenamiento del modelo.

```mermaid
flowchart TD
    subgraph Datos["Datos"]
        A["Imágenes Rayos X"] --> B["Dataset (dataset.py)"]
        C["CSV Etiquetas"] --> B
        P["create_test_set.py"] -->|"Genera 140 Test"| Q["holdout_test_set.csv"]
        Q -->|"Lista de Exclusión"| B
    end

    subgraph Procesamiento["Procesamiento"]
        B --> D["Transformaciones + Augmentation"]
        D -->|"Resize, Flip, Blur, Norm"| E["Tensores"]
        E -->|"Weighted Sampler"| F["DataLoader Batching"]
    end

    subgraph Modelo["Modelo (DenseNet-121)"]
        F --> G["Capas Convolucionales"]
        G --> H["Capa Clasificadora Personalizada"]
        H --> I["Logits (14 Clases)"]
    end

    subgraph Entrenamiento["Entrenamiento y Evaluación"]
        I --> J["Cálculo de Loss BCE"]
        J -.->|"Workaround AMD"| O["Cálculo en CPU"] 
        O -.-> J
        J -->|"Weighted BCE"| K["Backpropagation"]
        K --> L["Optimizador Adam"]
        L -->|"Actualizar Pesos"| G
        J -.-> M["AUC-ROC"]
        M -.-> N["CSV + Checkpoint.pth"]
        N -.->|"Modelo Final.pth"| R["evaluate_model.py"]
        Q -.->|"140 Imágenes Vírgenes"| R
        R -.-> S["Matrices de Confusión + Tabla"]
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

### Ejecución Cronológica

**Paso 1: Generar el Test Set (Vital)**
Antes de empezar a entrenar para fines científicos, aísla un set de pruebas puro:
```bash
python src/data/create_test_set.py
```

**Paso 2: Ejecutar Entrenamiento**
Desde la terminal, en la carpeta raíz del proyecto, inicia o reanuda el entrenamiento. El script ignorará las imágenes aisladas en el Paso 1:
```bash
python main.py
```
El script mostrará el progreso época por época, imprimiendo el Loss de entrenamiento y validación. Al finalizar las épocas estipuladas, guardará el modelo entrenado terminado como `densenet_nih.pth`. Si interrumpes el proceso, guardará un `checkpoint.pth`.

**Paso 3: Evaluación Rigurosa (Auditoría Final)**
Una vez finalizado tu entrenamiento y teniendo tu `densenet_nih.pth` listo, somételo a la validación rigurosa sobre las imágenes jamás vistas:
```bash
python evaluate_model.py
```
Se imprimirán por consola 14 Matrices de Confusión (Sensibilidad y Especificidad por enfermedad) y se exportará el archivo `evaluation_results.csv`.
