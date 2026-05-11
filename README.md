# Documentación del Proyecto: Clasificación de Rayos X de Tórax NIH con DenseNet-121

## 1. Introducción

Este proyecto implementa un sistema de aprendizaje profundo para la clasificación multi-etiqueta de 14 patologías torácicas utilizando imágenes de Rayos X del dataset NIH ChestX-ray14. El modelo base es **DenseNet-121**, pre-entrenado en ImageNet y adaptado mediante Transfer Learning. El sistema evoluciona desde un evaluador de modelo individual hasta un **sistema híbrido multi-modelo** que selecciona el mejor experto por patología.

**Características principales:**

- **Manejo de desbalance avanzado:** Undersampling de la clase mayoritaria ("No Finding") combinado con `WeightedRandomSampler` para forzar lotes equilibrados durante el entrenamiento.
- **Aumentación de datos:** Transformaciones geométricas (flips, rotaciones) y de frecuencia (UnsharpMask) para mejorar la generalización.
- **Entrenamiento reanudable:** Sistema de checkpoints robustos (`checkpoint.pth`) que guarda modelo, optimizador, scheduler y semilla aleatoria, garantizando reproducibilidad ante interrupciones.
- **Optimización del aprendizaje:** `CosineAnnealingWarmRestarts` para la tasa de aprendizaje y `EarlyStopping` con paciencia de 5 épocas monitoreando el AUC-ROC de validación (delta mínimo de mejora: 0.001).
- **Soporte multi-dispositivo:** Detección automática de GPU AMD vía `torch-directml` con workaround para operaciones incompatibles con DirectML, y fallback a CUDA/CPU.
- **Control estricto de Data Leakage:** Segregación del holdout antes de cualquier entrenamiento, y split calibración/test dentro del evaluador para calibrar umbrales sin contaminar la evaluación final.
- **Sistema híbrido multi-modelo:** Expert Router por competencia de Youden + sobreescrituras manuales para asignar el modelo más robusto a cada patología.

---

## 2. Estructura del Proyecto

```
ProyectoFinal_V2/
│
├── main.py                        # Entrenamiento principal (DenseNet-121)
├── demo_model.py                  # Evaluador individual de un modelo sobre el holdout
├── evaluate_final_hybrid_model.py # Evaluador híbrido: 11 modelos compiten por patología
├── requirements.txt               # Dependencias del proyecto
│
└── src/
    ├── data/
    │   ├── create_test_set.py     # Genera el holdout (ejecutar UNA SOLA VEZ antes de entrenar)
    │   ├── dataset.py             # Clase Dataset de PyTorch con exclusión del holdout
    │   └── transforms.py          # Transformaciones personalizadas (UnsharpMask, GaussianBlur)
    ├── models/
    │   └── densenet.py            # Arquitectura DenseNet-121 adaptada a 14 clases
    ├── training/
    │   └── trainer.py             # Lógica del bucle de entrenamiento y validación
    └── utils/
        └── versioning.py          # Auto-versionado de archivos de salida (evita sobrescrituras)
```

**Archivos generados en runtime:**

Los archivos de la tabla siguiente son creados automáticamente por los scripts al ejecutarse; no forman parte del código fuente. No deben añadirse al repositorio de Git (con `git add` / `git commit`) porque los modelos `.pth` pueden superar 1 GB y los CSV/PNG son resultados derivados de una ejecución concreta que se pueden regenerar. Lo habitual es listarlos en el `.gitignore` del proyecto para que Git los ignore automáticamente.

| Archivo | Generado por | Descripción |
|---|---|---|
| `holdout_test_set.csv` | `create_test_set.py` | Lista sagrada de imágenes reservadas para test |
| `best_model_Vn.pth` | `main.py` | Mejor checkpoint por AUC-ROC durante entrenamiento |
| `results_Vn.csv` | `main.py` | Historial de métricas por época |
| `demo_predictions_Vn.csv` | `demo_model.py` | Predicciones imagen por imagen (evaluador individual) |
| `demo_metrics_Vn.csv` | `demo_model.py` | Métricas por patología (evaluador individual) |
| `demo_roc_calib_Vn.png` | `demo_model.py` | Curvas ROC del set de calibración |
| `final_hybrid_routing_Vn.csv` | `evaluate_final_hybrid_model.py` | Decisión de routing y umbral por patología |
| `final_hybrid_metrics_Vn.csv` | `evaluate_final_hybrid_model.py` | Métricas en test ciego del sistema híbrido |
| `final_hybrid_predictions_Vn.csv` | `evaluate_final_hybrid_model.py` | Predicciones del sistema híbrido imagen por imagen |
| `final_hybrid_roc_calib_Vn.png` | `evaluate_final_hybrid_model.py` | Curvas ROC de todos los modelos por patología |

---

## 3. Detalle del Código y Componentes

### 3.1 `src/data/create_test_set.py` — Generación del Holdout

**Propósito:** Extraer, antes de cualquier entrenamiento, un conjunto de imágenes que jamás serán vistas por ningún modelo durante el entrenamiento ni la validación interna. Este paso debe ejecutarse una única vez.

**Constantes de configuración:**

| Variable | Valor | Descripción |
|---|---|---|
| `SAMPLES_PER_PATHOLOGY` | 20 | Imágenes a extraer por cada una de las 14 patologías (+1 clase "No Finding") |
| `RANDOM_SEED` | 42 | Semilla fijada para `random.seed()` garantizando reproducibilidad exacta del muestreo |
| `OUTPUT_FILE` | `holdout_test_set.csv` | Archivo de salida con la lista de exclusión |

**Algoritmo:**

1. Carga el CSV original (`Data_Entry_2017.csv`, ~112.000 registros).
2. Fija la semilla aleatoria global con `random.seed(RANDOM_SEED)`.
3. Construye un índice `{patología → [índices de filas]}` recorriendo el DataFrame una sola vez.
4. Para cada patología (en orden definido por `ALL_LABELS`), extrae `SAMPLES_PER_PATHOLOGY` imágenes con `random.sample()`, excluyendo índices ya seleccionados por otra patología para evitar duplicados.
5. Guarda el resultado en `holdout_test_set.csv` con tres columnas:
   - `Target_Pathology_For_Extraction`: patología objetivo de esa extracción.
   - `Image Index`: nombre del archivo de imagen.
   - `Original_Finding_Labels`: etiquetas reales de la imagen (puede incluir co-morbilidades).

**Reproducibilidad:** La semilla `RANDOM_SEED = 42` en `random.seed()` garantiza que, partiendo del mismo CSV, `random.sample()` producirá exactamente el mismo subconjunto en cualquier ejecución futura.

> ⚠️ **Ejecutar este script una sola vez**, antes del primer entrenamiento. El archivo `holdout_test_set.csv` generado debe mantenerse intacto durante toda la vida del proyecto.

---

### 3.2 `main.py` — Entrenamiento

**Propósito:** Script orquestador de todo el ciclo de entrenamiento de un modelo DenseNet-121.

**Constantes de configuración:**

| Variable | Descripción |
|---|---|
| `DATA_DIR` | Ruta al directorio raíz del dataset (imágenes y CSV). |
| `BATCH_SIZE` | Imágenes por lote (ajustar según VRAM disponible). |
| `LEARNING_RATE` | Tasa de aprendizaje inicial del optimizador Adam (ej. `1e-4`). |
| `EPOCHS` | Número máximo de épocas de entrenamiento. |
| `IMAGE_SIZE` | Resolución de entrada fija requerida por DenseNet-121 (224 × 224 px). |
| `UNDERSAMPLE_RATE` | Fracción de muestras "No Finding" a conservar (ej. `0.35` = 35%). |
| `RANDOM_SEED` | Semilla (`42`) usada en `torch.Generator().manual_seed()` para el split 80/20. |
| `RESUME_FROM` | Ruta a un `.pth` previo para Transfer Learning inter-modelos. `None` si se entrena desde ImageNet. |
| `EXCLUDE_LIST_FILE` | Nombre del CSV del holdout (`holdout_test_set.csv`). Se pasa al Dataset para exclusión estricta. |
| `CHECKPOINT_FILE` | Archivo versionado donde se guarda el estado completo por época. |
| `BEST_MODEL_FILE` | Archivo versionado donde se guarda el mejor modelo según AUC-ROC de validación. |

**Reproducibilidad del split 80/20:**

```python
seed_generator = torch.Generator().manual_seed(RANDOM_SEED)
train_subset, val_subset = random_split(full_dataset_raw,
                                        [train_size, val_size],
                                        generator=seed_generator)
```

Al fijar el generador de PyTorch con `manual_seed(42)`, el split es idéntico en toda ejecución que parta del mismo dataset (con el holdout ya excluido). Este mismo `RANDOM_SEED` queda serializado dentro del `checkpoint.pth` bajo la clave `config.random_seed`, permitiendo auditarlo a posteriori.

**Flujo principal (`main()`):**

1. **Auto-versionado:** `get_next_version()` asigna sufijos `_V2`, `_V3`, etc. a todos los archivos de salida para evitar sobrescribir ejecuciones anteriores.
2. **Selección de dispositivo:** Intenta `torch_directml` (AMD), luego `cuda` (NVIDIA), y finalmente `cpu`.
3. **Carga del dataset:** Instancia `NIHChestXRayDataset` pasando `exclude_list_path=EXCLUDE_LIST_FILE`, lo que excluye el holdout antes de cualquier procesamiento.
4. **Split 80/20 determinista:** con `torch.Generator().manual_seed(RANDOM_SEED)`.
5. **Transformaciones diferenciadas:** `val_transforms` (solo Resize + Normalize) para validación; `train_transforms` (+ RandomHorizontalFlip, RandomRotation, RandomUnsharpMask) para entrenamiento.
6. **WeightedRandomSampler:** La función `calculate_sampler_weights()` asigna a cada imagen un peso igual al promedio de la rareza inversa de sus patologías presentes (`get_mean_weight`). Imágenes con patologías raras son muestreadas más frecuentemente.
7. **Loss relajada:** Los `pos_weights` calculados por el Dataset se suavizan con `torch.sqrt()` para evitar sobrecompensación doble (ya existe el Sampler).
8. **Scheduler:** `CosineAnnealingWarmRestarts(T_0=10, T_mult=2)` — ciclos coseno con reinicios progresivos.
9. **EarlyStopping:** Detiene el entrenamiento si el AUC-ROC de validación no mejora en al menos `delta=0.001` durante `patience=5` épocas consecutivas.
10. **Checkpoint robusto:** Cada época sobrescribe `checkpoint.pth` con `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `best_val_auc` y el bloque `config` (semilla, batch size, lr, undersample rate).
11. **Mejor modelo:** Se guarda en `best_model_Vn.pth` cada vez que el AUC de validación supera el máximo histórico.

**Clases auxiliares:**

- **`EarlyStopping`:** Acumula un contador cuando `val_auc < best_score + delta`; lo resetea cuando hay mejora suficiente. Activa `early_stop = True` al alcanzar `patience`.
- **`AugmentedDataset`:** Wrapper que aplica las transformaciones diferenciales a los subsets retornados por `random_split`, ya que estos no admiten transformaciones directamente.
- **`load_checkpoint_safe()`:** Parche de arquitectura para Transfer Learning inter-modelos: detecta y renombra las llaves `classifier.1.weight` → `classifier.weight` cuando el checkpoint proviene de una arquitectura con capa Dropout en el clasificador. Incluye abort duro si las capas vitales de decisión no se cargan correctamente.

---

### 3.3 `src/data/dataset.py` — Dataset

**Clase `NIHChestXRayDataset`:**

- `__init__(data_dir, transform, no_finding_keep_frac, exclude_list_path)`:
  - Escanea el disco construyendo un diccionario `{nombre_archivo → ruta_absoluta}`.
  - Lee `Data_Entry_2017.csv` y filtra a imágenes existentes en disco.
  - **Exclusión estricta:** Lee `exclude_list_path` y elimina esas imágenes del DataFrame antes de cualquier otra operación, garantizando 0% de fuga del holdout.
  - **Undersampling:** Separa filas "No Finding" y conserva solo `no_finding_keep_frac` de ellas, reduciendo el desbalance sin afectar las imágenes con patologías.
- `__getitem__(idx)`: Carga la imagen con PIL, aplica transformaciones, y convierte la cadena de etiquetas (ej. `"Infiltration|Mass"`) a un vector binario de 14 posiciones.
- `get_pos_weight()`: Calcula `N_negativos / N_positivos` por clase como tensor de pesos para `BCEWithLogitsLoss`.

---

### 3.4 `src/models/densenet.py` — Modelo

**Función `get_model(num_classes, pretrained)`:**

- Carga `densenet121` de `torchvision.models`.
- Si `pretrained=True`: inicializa con pesos ImageNet (Transfer Learning de dominio general).
- Reemplaza el clasificador final (`classifier`, 1000 salidas) por `nn.Linear(1024, num_classes)` con 14 salidas, habilitando la clasificación multi-etiqueta.

---

### 3.5 `src/training/trainer.py` — Entrenador

**Clase `Trainer`:**

- `train_one_epoch(epoch)`: Modo entrenamiento (`model.train()`). Itera lotes, ejecuta forward pass, calcula `BCEWithLogitsLoss`, retropropaga gradientes y actualiza pesos con Adam.
- `validate(epoch)`: Modo evaluación (`model.eval()`, `torch.no_grad()`). Calcula loss y **AUC-ROC macro** sobre el set de validación usando `roc_auc_score` de scikit-learn. Retorna `(val_loss, val_auc)`.

---

### 3.6 `demo_model.py` — Evaluador Individual de Modelo

**Propósito:** Evaluar un único modelo entrenado sobre el holdout con control estricto de Data Leakage. Implementa un split interno calibración/test dentro del propio holdout para que los umbrales de decisión no sean evaluados sobre los mismos datos con los que se calibran.

**Constantes de configuración (única sección a editar):**

| Variable | Descripción |
|---|---|
| `MODEL_PATH` | Ruta al archivo `.pth` del modelo a evaluar. |
| `TEST_SET_CSV` | CSV del holdout (`holdout_test_set.csv`). |
| `STRICT_EVALUATION_MODE` | `True` (recomendado): split 50% calibración / 50% test ciego. `False`: solo para debugging. |
| `IMAGE_SIZE` | 224 (debe coincidir con el entrenamiento). |
| `NUM_CLASSES` | 14. |

**Mecanismo anti-leakage:**

```python
# Shuffle determinista del holdout
df_test = df_test.sample(frac=1, random_state=42).reset_index(drop=True)

# Split calibración / test ciego
calib_size = len(y_true_all) // 2
y_calib    = y_true_all[:calib_size, :]
y_test     = y_true_all[calib_size:, :]   # nunca tocado durante calibración
```

El shuffle con `random_state=42` garantiza que, dado el mismo holdout, el split sea siempre idéntico.

**Flujo de ejecución:**

1. **Inferencia única:** El modelo procesa todas las imágenes del holdout en una sola pasada, acumulando probabilidades `[0, 1]` para las 14 clases.
2. **Calibración de umbrales (solo sobre calib set):** Para cada patología, si hay al menos un positivo en el calib set, se ejecuta `roc_curve()` y se busca el threshold que maximiza el **Índice de Youden** (`J = Sensibilidad + Especificidad − 1`). El umbral se recorta al rango `[0.05, 0.95]` para evitar valores extremos inestables.
3. **Evaluación ciega (sobre test set):** Los umbrales calibrados se aplican al 50% restante del holdout, que el modelo nunca "vio" durante la calibración. Se calculan matrices de confusión, Sensibilidad, Especificidad y Youden J por patología.
4. **Exportación de resultados (versionados automáticamente):**
   - `demo_predictions_Vn.csv`: predicción imagen por imagen con etiquetas reales y predichas.
   - `demo_metrics_Vn.csv`: tabla de métricas por patología (TP, FN, TN, FP, Sens%, Spec%, Youden J, Umbral).
   - `demo_roc_calib_Vn.png`: curvas ROC del set de calibración con el punto óptimo de Youden marcado.

**Función `load_safe()`:** Parche universal de arquitectura que permite cargar cualquier checkpoint independientemente de si el clasificador fue guardado como `Sequential` (con capa Dropout, llaves `classifier.1.weight`) o como `Linear` directo (llaves `classifier.weight`). Realiza la conversión automática en ambas direcciones.

---

### 3.7 `evaluate_final_hybrid_model.py` — Evaluador Híbrido Multi-Modelo

**Propósito:** Evaluar un sistema híbrido donde múltiples modelos compiten para ser el "experto" asignado a cada patología. Para cada patología, el script selecciona el modelo más robusto usando el 50% de calibración del holdout y lo evalúa en el 50% de test ciego.

**Constantes de configuración:**

| Variable | Descripción |
|---|---|
| `DATA_DIR` | Ruta al directorio del dataset. |
| `MODEL_DIR` | Directorio donde residen los archivos `.pth` de todos los modelos. |
| `TEST_SET_CSV` | CSV del holdout a evaluar. |
| `CALIB_RATIO` | Fracción destinada a calibración (default: `0.50`). |
| `RANDOM_SEED` | Semilla para el shuffle determinista del holdout (`42`). |
| `MODELS` | Diccionario `{alias → nombre_archivo.pth}` con los 11 modelos candidatos. |
| `MANUAL_OVERRIDES` | Diccionario `{patología → alias}` para forzar un modelo específico sin torneo. Vacío = modo Expert Router puro. |

**Registro de los 11 modelos candidatos:**

| Alias | Hardware de entrenamiento |
|---|---|
| `Base` | AMD (torch-directml) |
| `20260402` | AMD (torch-directml) |
| `20260423` | NVIDIA (CUDA / Kaggle) |
| `T1`, `T1_V2`, `T2`, `T2_V2`, `T3`, `T4` | NVIDIA (CUDA / Kaggle) |
| `V1`, `V1_3` | AMD (torch-directml) |

**Reproducibilidad del shuffle y split:**

```python
df_full = pd.read_csv(TEST_SET_CSV).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
calib_size = int(len(df_full) * CALIB_RATIO)
```

Con `random_state=42` y el mismo CSV de entrada, el orden de imágenes es siempre idéntico, garantizando que el mismo subconjunto de imágenes quede en calibración y en test en toda ejecución.

**Flujo de ejecución:**

#### Paso 1 — Inferencia única sobre todo el holdout

Todos los modelos procesan todas las imágenes en una única pasada conjunta, evitando cargar las imágenes múltiples veces. Por cada imagen se ejecuta:

```python
for alias, model in loaded_models.items():
    probs = sigmoid(model(tensor)).cpu().squeeze().numpy()
    probs_per_model[alias].append(probs)
```

Resultado: matriz `probs_all[alias]` de forma `(N_imágenes, 14)` por cada modelo. Luego se divide en calibración y test:

```python
y_calib     = y_all[:calib_size]         # 50% para calibrar umbrales
y_test      = y_all[calib_size:]          # 50% para test ciego (nunca consultado durante calibración)
probs_calib = {alias: probs_all[alias][:calib_size] for alias in model_names}
probs_test  = {alias: probs_all[alias][calib_size:]  for alias in model_names}
```

#### Paso 2 — Fase de calibración: selección de modelo y umbral por patología

Para cada una de las 14 patologías se ejecuta la función `compute_youden_metrics(y_true, y_prob)`:

```python
def compute_youden_metrics(y_true, y_prob):
    if sum(y_true) == 0 or sum(y_true) == len(y_true):
        return None, None, 0.5, 0.0, 0.0, 0.0, 0.0
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    J = tpr + (1 - fpr) - 1
    idx = int(np.argmax(J))
    best_thresh = float(np.clip(thresholds[idx], 0.05, 0.95))
    roc_auc_val = float(auc(fpr, tpr))
    return fpr, tpr, best_thresh, float(fpr[idx]), float(tpr[idx]), roc_auc_val, float(J[idx])
```

`roc_curve()` evalúa cada valor de probabilidad del calib set como candidato a umbral, generando pares `(FPR, TPR)`. El umbral óptimo maximiza `J = TPR + (1 − FPR) − 1`. El recorte `[0.05, 0.95]` previene umbrales degenerados que clasificarían casi toda imagen como positiva o negativa.

**Modo Expert Router (por defecto):** Los 11 modelos compiten sobre el calib set. El ganador es el que produce el mayor Youden J:

```python
for alias in model_names:
    result = compute_youden_metrics(y_t, probs_calib[alias][:, p_idx])
    if result[6] > best_j:        # result[6] = J_calib
        best_j       = result[6]
        winner_alias = alias
```

**Modo Manual Override:** El modelo asignado en `MANUAL_OVERRIDES` es forzado sin torneo. Sin embargo, su umbral de decisión **sí se calibra** sobre el calib set mediante `compute_youden_metrics()`. La diferencia respecto al Expert Router es únicamente la selección del modelo, no el proceso de calibración del umbral.

En ambos modos, la decisión de routing se registra en el CSV de salida:

| Campo | Significado |
|---|---|
| `Pathology` | Nombre de la patología |
| `Strategy` | `"Expert Router (alias)"` o `"Override (alias)"` |
| `Winner_Model` | Alias del modelo seleccionado |
| `Threshold_Calib` | Umbral óptimo calibrado sobre el calib set |
| `Youden_J_Calib` | J del modelo ganador **en el calib set** (optimista, in-sample) |
| `ROC_AUC_Calib` | AUC-ROC del ganador en el calib set |

> **Nota sobre la diferencia entre `Youden_J_Calib` y `Youden_J` en métricas:** El primero se obtiene evaluando en el mismo conjunto donde se buscó el umbral óptimo (in-sample, sesgado hacia arriba). El segundo se calcula aritméticamente a partir de la matriz de confusión en el test ciego (`J = Sens/100 + Spec/100 − 1`), siendo la medida honesta del rendimiento real. Con ~10 positivos por patología en el calib set, las curvas ROC son inestables y el J de calibración puede ser significativamente más alto que el J de test.

#### Paso 3 — Fase de evaluación: test ciego

Los umbrales ya fijos se aplican a las probabilidades del test set, que no fue consultado en ningún momento anterior:

```python
final_preds[:, p_idx] = (probs_test[winner_alias][:, p_idx] >= best_thresh).astype(int)
```

Se calculan matrices de confusión con `multilabel_confusion_matrix()` y se reportan Sensibilidad, Especificidad y Youden J por patología.

#### Paso 4 — Exportación de resultados (todos versionados automáticamente)

| Archivo | Contenido |
|---|---|
| `final_hybrid_routing_Vn.csv` | Tabla de routing: modelo ganador, umbral y J-calib por patología |
| `final_hybrid_metrics_Vn.csv` | Métricas en test ciego: TP, FN, TN, FP, Sens%, Spec%, Youden J |
| `final_hybrid_predictions_Vn.csv` | Predicción imagen por imagen del test ciego |
| `final_hybrid_roc_calib_Vn.png` | Subplots 4×4 con curvas ROC de todos los modelos por patología; ganador resaltado |

---

## 4. Diagrama de Funcionamiento

```
FASE 0 — PREPARACIÓN (ejecutar una sola vez)
─────────────────────────────────────────────
create_test_set.py
  └─ random.seed(42) + random.sample(20 por patología)
  └─ holdout_test_set.csv  ←─ lista sagrada (~300 imágenes)

FASE 1 — ENTRENAMIENTO (repetido por cada modelo)
─────────────────────────────────────────────────
Data_Entry_2017.csv + Imágenes
  └─ NIHChestXRayDataset
       ├─ Excluye holdout_test_set.csv
       ├─ Undersampling "No Finding" (35%)
       └─ random_split 80/20 (Generator seed=42)
            ├─ train_subset → AugmentedDataset → WeightedRandomSampler → DataLoader
            └─ val_subset   → AugmentedDataset → DataLoader
  └─ DenseNet-121 (ImageNet → 14 clases)
  └─ BCEWithLogitsLoss (pos_weight relajado con sqrt)
  └─ Adam + CosineAnnealingWarmRestarts + EarlyStopping(patience=5, delta=0.001)
  └─ best_model_Vn.pth  (guardado cuando val_AUC mejora ≥ 0.001)

FASE 2A — EVALUACIÓN INDIVIDUAL (demo_model.py)
────────────────────────────────────────────────
holdout_test_set.csv
  └─ shuffle(random_state=42)
  └─ 50% calib | 50% test ciego
       ├─ Calib → roc_curve() → argmax(Youden J) → threshold ∈ [0.05, 0.95]
       └─ Test  → aplicar thresholds → TP/FN/TN/FP → Sens/Spec/Youden J
  └─ demo_predictions_Vn.csv | demo_metrics_Vn.csv | demo_roc_calib_Vn.png

FASE 2B — EVALUACIÓN HÍBRIDA (evaluate_final_hybrid_model.py)
───────────────────────────────────────────────────────────────
holdout_test_set.csv
  └─ shuffle(random_state=42)
  └─ 50% calib | 50% test ciego
  └─ Inferencia única: 11 modelos × N imágenes → probs_all[alias](N, 14)
       CALIB SET por patología:
         ├─ Expert Router: argmax_alias(Youden_J_calib) → winner + threshold
         └─ Manual Override: alias forzado → threshold calibrado igual
       TEST SET:
         └─ Aplicar [winner, threshold] → multilabel_confusion_matrix
  └─ final_hybrid_routing_Vn.csv | final_hybrid_metrics_Vn.csv
     final_hybrid_predictions_Vn.csv | final_hybrid_roc_calib_Vn.png
```

---

## 5. Cómo Ejecutarlo

### Requisitos previos

```bash
pip install -r requirements.txt
```

Dependencias principales: `torch`, `torchvision`, `torch-directml` (solo AMD), `pandas`, `pillow`, `scikit-learn`, `matplotlib`, `tqdm`, `openpyxl`.

### Configuración

Editar únicamente la sección de configuración al inicio de cada script:

```python
DATA_DIR = r"Ruta\Al\Dataset"   # directorio con las imágenes y Data_Entry_2017.csv
```

### Ejecución cronológica

**Paso 0 — Generar el holdout (una sola vez, antes del primer entrenamiento)**

```bash
python src/data/create_test_set.py
```

Genera `holdout_test_set.csv`. Este archivo debe conservarse sin modificaciones durante todo el proyecto.

**Paso 1 — Entrenar modelos**

```bash
python main.py
```

Configura `RESUME_FROM` para Transfer Learning desde un modelo previo. El script asigna automáticamente nombres versionados a todos los archivos de salida.

**Paso 2A — Evaluar un modelo individual**

```bash
python demo_model.py
```

Editar `MODEL_PATH` para apuntar al `.pth` deseado. Con `STRICT_EVALUATION_MODE = True` se usa el split 50/50 anti-leakage.

**Paso 2B — Evaluar el sistema híbrido multi-modelo**

```bash
python evaluate_final_hybrid_model.py
```

Todos los modelos registrados en `MODELS` deben existir en `MODEL_DIR`. Para activar sobreescrituras manuales, descomentar las líneas correspondientes en `MANUAL_OVERRIDES`.

---

## 6. Garantías de Reproducibilidad

| Mecanismo | Script | Efecto |
|---|---|---|
| `random.seed(42)` | `create_test_set.py` | Mismo holdout en cualquier ejecución sobre el mismo CSV |
| `torch.Generator().manual_seed(42)` | `main.py` | Mismo split 80/20 en cada entrenamiento |
| `RANDOM_SEED=42` serializado en checkpoint | `main.py` | Auditable post-facto en `checkpoint.pth['config']['random_seed']` |
| `sample(frac=1, random_state=42)` | `demo_model.py` | Mismo orden de imágenes → mismo split calib/test |
| `sample(frac=1, random_state=42)` | `evaluate_final_hybrid_model.py` | Mismo split calib/test para todos los modelos |
| Auto-versionado (`get_next_version`) | Todos | Nunca sobrescribe resultados de ejecuciones anteriores |
