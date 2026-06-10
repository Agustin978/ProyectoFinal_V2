# Flujo de Ejecución del Código: Paso a Paso

Este documento detalla qué sucede exactamente en cada etapa del pipeline, desde la generación del holdout hasta la evaluación del sistema híbrido multi-modelo. Se explican los archivos involucrados, las funciones clave, las variables de control y los mecanismos de reproducibilidad en cada fase.

---

## Fase 0 — Aislamiento del Holdout: `src/data/create_test_set.py`

Este script debe ejecutarse **una única vez**, antes de cualquier entrenamiento. Su misión es separar un subconjunto de imágenes que ningún modelo verá jamás durante el entrenamiento ni la validación interna.

### ¿Por qué antes del entrenamiento?

Si el holdout se separa después de entrenar, existe el riesgo de que el modelo haya ajustado sus parámetros sobre imágenes que luego se pretendan usar como "nunca vistas". La separación previa garantiza que el holdout sea verdaderamente virgen.

### Qué hace el script paso a paso

1. **Carga el CSV original** (`Data_Entry_2017.csv`, ~112.000 registros) con todas las etiquetas del dataset NIH.

2. **Fija la semilla aleatoria global:**
   ```python
   RANDOM_SEED = 42
   random.seed(RANDOM_SEED)
   ```
   Esto garantiza que `random.sample()` produzca exactamente el mismo subconjunto en cualquier ejecución futura sobre el mismo CSV.

3. **Construye un índice por patología:** Recorre el DataFrame una sola vez, mapeando cada etiqueta a la lista de índices de fila que la contienen.

4. **Extrae 20 imágenes por patología (`SAMPLES_PER_PATHOLOGY = 20`):**
   - Para cada patología (en el orden fijo de `ALL_LABELS`), se llama a `random.sample(available_indices, 20)`.
   - Se excluyen índices ya seleccionados por patologías previas, evitando duplicados en el holdout aunque una imagen tenga co-morbilidades.
   - Si no hay 20 imágenes disponibles no solapadas, se toman todas las disponibles con aviso.

5. **Guarda `holdout_test_set.csv`** con tres columnas:
   - `Target_Pathology_For_Extraction`: patología que motivó la extracción de esa fila.
   - `Image Index`: nombre del archivo de imagen.
   - `Original_Finding_Labels`: etiquetas reales completas (puede incluir múltiples patologías).

### Resultado

Un archivo CSV con aproximadamente 300 imágenes únicas (20 × 15 clases, con solapamientos entre patologías reduciendo el total exacto). Este archivo actúa como **lista negra** permanente durante todo el proyecto.

---

## Fase 1 — Entrenamiento: `main.py`

El entrenamiento puede ejecutarse múltiples veces con distintas configuraciones. Cada ejecución genera archivos de salida con sufijo de versión automática (`_V2`, `_V3`, etc.) para no pisar resultados previos.

### 1.1 Configuración inicial y auto-versionado

```python
CSV_FILE       = get_next_version("results.csv")      # → results_V2.csv, etc.
CHECKPOINT_FILE = get_next_version("checkpoint.pth")
BEST_MODEL_FILE = get_next_version("best_model.pth")
RANDOM_SEED     = 42
```

`get_next_version()` busca el primer sufijo disponible en disco, garantizando que cada entrenamiento tenga su propio historial de métricas.

### 1.2 Selección de dispositivo

El código intenta en orden de prioridad:
1. `torch_directml.device()` — GPU AMD en Windows vía DirectML.
2. `torch.device('cuda')` — GPU NVIDIA.
3. `torch.device('cpu')` — CPU como último recurso.

La variable `device` resultante se usa en todas las operaciones tensoriales.

### 1.3 Carga del dataset con exclusión estricta del holdout

```python
full_dataset_raw = NIHChestXRayDataset(
    data_dir=DATA_DIR,
    transform=None,
    no_finding_keep_frac=UNDERSAMPLE_RATE,   # ej. 0.35 → conserva el 35% de "No Finding"
    exclude_list_path=EXCLUDE_LIST_FILE       # holdout_test_set.csv
)
```

Dentro de `NIHChestXRayDataset.__init__()`, antes de cualquier otra operación:
1. Se lee `holdout_test_set.csv`.
2. Se eliminan esas imágenes del DataFrame interno (`self.df`).
3. El modelo nunca podrá acceder a ellas durante el entrenamiento.

### 1.4 Split 80/20 determinista

```python
seed_generator = torch.Generator().manual_seed(RANDOM_SEED)
train_subset, val_subset = random_split(
    full_dataset_raw,
    [train_size, val_size],
    generator=seed_generator
)
```

Al fijar el generador de PyTorch con `manual_seed(42)`, el split es idéntico en toda ejecución que parta del mismo dataset (con el holdout ya excluido). El mismo valor de `RANDOM_SEED` queda serializado en el `checkpoint.pth` bajo `config['random_seed']`, permitiendo auditarlo en el futuro.

### 1.5 Transformaciones diferenciadas

- **Validación** (`val_transforms`): solo `Resize(224)` + `ToTensor()` + `Normalize(ImageNet)`.
- **Entrenamiento** (`train_transforms`): añade `RandomHorizontalFlip(p=0.5)`, `RandomRotation(10°)` y `RandomUnsharpMask(p=0.3)`.

Ambas se aplican a través de `AugmentedDataset`, un wrapper necesario porque `random_split` retorna subsets que no admiten transformaciones directamente.

### 1.6 WeightedRandomSampler

La función `calculate_sampler_weights()` calcula el peso de cada imagen de entrenamiento:
- Por cada imagen, se promedian los pesos de rareza inversa (`1 / count_clase`) de sus patologías presentes.
- Imágenes con patologías raras aparecen más frecuentemente en los lotes.
- Las imágenes "No Finding" reciben su propio peso basado en su conteo real en el subconjunto de entrenamiento.

El `WeightedRandomSampler` usa estos pesos con `replacement=True`, generando lotes de tamaño `BATCH_SIZE` con distribución balanceada.

### 1.7 Loss y optimizador

```python
raw_pos_weights = full_dataset_raw.get_pos_weight()   # N_neg / N_pos por clase
pos_weights     = torch.sqrt(raw_pos_weights)          # relajación para evitar doble corrección
criterion       = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
optimizer       = optim.Adam(model.parameters(), lr=LEARNING_RATE)
```

La raíz cuadrada sobre los `pos_weights` evita sobrecompensación: el Sampler ya favorece las minorías; sin relajar la Loss, el modelo tendería a generar excesivos falsos positivos.

### 1.8 Scheduler y EarlyStopping

```python
scheduler     = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
early_stopping = EarlyStopping(patience=5, delta=0.001)
```

- **CosineAnnealingWarmRestarts:** Ciclo coseno de 10 épocas, que se duplica en cada reinicio (`T_mult=2`). Permite escapar de mínimos locales periódicamente.
- **EarlyStopping:** Si el AUC-ROC de validación no mejora en al menos `0.001` durante 5 épocas consecutivas, el entrenamiento se detiene automáticamente para prevenir overfitting.

### 1.9 Bucle de entrenamiento

Cada época:
1. `trainer.train_one_epoch(epoch)` → loss de entrenamiento.
2. `trainer.validate(epoch)` → loss de validación + AUC-ROC macro.
3. `scheduler.step()` → actualización coseno de la tasa de aprendizaje.
4. Si `val_auc > best_val_auc` → se guarda `best_model_Vn.pth`.
5. Se agrega la fila de métricas al `results_Vn.csv`.
6. Se sobrescribe el `checkpoint.pth` con el estado completo.
7. `early_stopping(val_auc)` → si `early_stop == True`, se interrumpe el bucle.

### 1.10 Archivos de salida

| Archivo | Contenido |
|---|---|
| `best_model_Vn.pth` | Mejores pesos (guardados solo cuando AUC mejora) |
| `results_Vn.csv` | Historial épocas: train_loss, val_loss, val_auc, lr, duración |
| `checkpoint_Vn.pth` | Estado completo para reanudar: model + optimizer + scheduler + config |

---

## Fase 2A — Evaluador Individual: `demo_model.py`

Evalúa un único modelo `.pth` sobre el holdout completo, implementando un split interno calibración/test para evitar que los umbrales se calibren sobre los mismos datos donde se miden.

### 2A.1 Configuración

```python
MODEL_PATH           = "best_model_T1.pth"      # modelo a evaluar
TEST_SET_CSV         = "holdout_test_set.csv"
STRICT_EVALUATION_MODE = True                    # True: 50/50  |  False: solo para debugging
```

### 2A.2 Shuffle determinista del holdout

```python
df_test = pd.read_csv(TEST_SET_CSV)
df_test = df_test.sample(frac=1, random_state=42).reset_index(drop=True)
```

El shuffle con semilla fija garantiza que el mismo CSV produzca siempre el mismo orden de imágenes, y por ende el mismo split calib/test.

### 2A.3 Inferencia única sobre todo el holdout

El modelo procesa todas las imágenes en una sola pasada con `torch.no_grad()`:
- Se aplica `transforms.Compose([Resize(224), ToTensor(), Normalize(ImageNet)])`.
- `torch.sigmoid(model(tensor))` convierte los logits crudos a probabilidades `[0, 1]` por clase.
- Se acumulan los vectores de probabilidades en `y_pred_probs_all` (forma `N × 14`).

### 2A.4 Split calibración/test

```python
calib_size   = len(y_true_all) // 2
y_calib      = y_true_all[:calib_size, :]
probs_calib  = y_pred_probs_all[:calib_size, :]
y_test       = y_true_all[calib_size:, :]       # nunca consultado durante calibración
probs_test   = y_pred_probs_all[calib_size:, :]
```

### 2A.5 Calibración de umbrales (solo sobre calib set)

Para cada patología `i`:
1. Si no hay positivos en el calib set, se asigna umbral por defecto `0.5`.
2. `roc_curve(y_calib[:, i], probs_calib[:, i])` evalúa cada valor de probabilidad como candidato a umbral.
3. Se calcula `Youden J = TPR + (1 − FPR) − 1` para cada punto de la curva.
4. El umbral óptimo es el que maximiza J, recortado al rango `[0.05, 0.95]`.

```python
optimal_thresholds[i] = np.clip(best_threshold, 0.05, 0.95)
```

El recorte previene umbrales degenerados: un umbral de 0.01 clasificaría casi toda imagen como positiva (alta sensibilidad espuria); uno de 0.99 la clasificaría como negativa.

### 2A.6 Evaluación sobre test ciego

Los umbrales fijos se aplican a `probs_test`:
```python
y_pred_binary_test[:, i] = (probs_test[:, i] >= optimal_thresholds[i]).astype(int)
```

`multilabel_confusion_matrix()` produce TN, FP, FN, TP por patología. Se calculan Sensibilidad, Especificidad y Youden J como medidas finales.

### 2A.7 Carga de modelos con parche de arquitectura (`load_safe`)

```python
def load_safe(filepath, model, device):
    data = torch.load(filepath, map_location='cpu', weights_only=False)
    state_dict = (data['model_state_dict'] if isinstance(data, dict)
                  and 'model_state_dict' in data else data)

    # Parche Sequential → Linear
    if 'classifier.1.weight' in state_dict:
        state_dict['classifier.weight'] = state_dict.pop('classifier.1.weight')
        state_dict['classifier.bias']   = state_dict.pop('classifier.1.bias')
        # eliminar llaves extra del Sequential (dropout, etc.)

    # Parche Linear → Sequential
    elif 'classifier.weight' in state_dict and has_dropout_local:
        state_dict['classifier.1.weight'] = state_dict.pop('classifier.weight')
        state_dict['classifier.1.bias']   = state_dict.pop('classifier.bias')

    model.load_state_dict(state_dict, strict=False)
    return model.to(device).eval()
```

Este parche permite cargar cualquier checkpoint independientemente de si fue guardado con una capa Dropout en el clasificador (llaves `classifier.1.*`) o sin ella (llaves `classifier.*`), sin necesidad de conocer a priori la arquitectura exacta del checkpoint.

---

## Fase 2B — Evaluador Híbrido Multi-Modelo: `evaluate_final_hybrid_model.py`

Evalúa un sistema donde 11 modelos compiten por ser el "experto" asignado a cada patología. El proceso garantiza control de leakage: la selección del modelo y la calibración del umbral usan el calib set; la evaluación final usa únicamente el test set.

### 2B.1 Configuración y registro de modelos

```python
CALIB_RATIO  = 0.50
RANDOM_SEED  = 42

MODELS = {
    'Base':     'best_model.pth',
    '20260402': 'best_model_20260402_161254.pth',
    '20260423': 'best_model_20260423_114736.pth',
    'T1':       'best_model_T1.pth',
    'T1_V2':    'best_model_T1_V2.pth',
    'T2':       'best_model_T2.pth',
    'T2_V2':    'best_model_T2_V2.pth',
    'T3':       'best_model_T3.pth',
    'T4':       'best_model_T4.pth',
    'V1':       'best_model_V1.pth',
    'V1_3':     'best_model_V1_3.pth',
}

MANUAL_OVERRIDES = {
    # Vacío → Expert Router puro para todas las patologías.
    # Ejemplo con overrides:
    # 'Hernia': 'V1_3',
    # 'Atelectasis': 'T2_V2',
}
```

### 2B.2 Shuffle determinista y división del holdout

```python
df_full    = pd.read_csv(TEST_SET_CSV).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
calib_size = int(len(df_full) * CALIB_RATIO)
```

Idéntico mecanismo que en `demo_model.py`: semilla fija garantiza el mismo split en toda ejecución.

### 2B.3 Carga de los 11 modelos

Cada modelo se carga usando `load_safe()` (mismo parche de arquitectura Sequential↔Linear que en el evaluador individual). Los modelos se mantienen en memoria simultáneamente como diccionario `loaded_models[alias]`.

### 2B.4 Inferencia única: todos los modelos sobre todo el holdout

```python
with torch.no_grad():
    for _, row in df_full.iterrows():
        img    = Image.open(image_paths_dict[img_name]).convert('RGB')
        tensor = transform(img).unsqueeze(0).to(device)

        for alias, model in loaded_models.items():
            probs = sigmoid(model(tensor)).cpu().squeeze().numpy()
            probs_per_model[alias].append(probs)
```

Este enfoque carga cada imagen **una sola vez** y pasa el tensor por los 11 modelos simultáneamente, minimizando el I/O de disco. El resultado es `probs_all[alias]` de forma `(N_imágenes, 14)` por cada modelo.

La división en calib/test se realiza sobre los arrays ya acumulados:

```python
probs_calib = {alias: probs_all[alias][:calib_size] for alias in model_names}
probs_test  = {alias: probs_all[alias][calib_size:]  for alias in model_names}
```

### 2B.5 Fase de calibración por patología

Para cada una de las 14 patologías se determina el modelo ganador y su umbral óptimo. Hay dos rutas:

#### Ruta A — Expert Router (torneo automático)

```python
for alias in model_names:
    result = compute_youden_metrics(y_calib[:, p_idx], probs_calib[alias][:, p_idx])
    if result[6] > best_j:   # result[6] = Youden J en calib
        winner_alias = alias
        winner_result = result
```

Se evalúan los 11 modelos sobre el calib set. El modelo que produce el mayor Youden J es coronado como ganador para esa patología.

#### Ruta B — Override Manual

```python
if pathology in MANUAL_OVERRIDES and MANUAL_OVERRIDES[pathology] in loaded_models:
    forced_alias = MANUAL_OVERRIDES[pathology]
    result = compute_youden_metrics(y_calib[:, p_idx], probs_calib[forced_alias][:, p_idx])
    winner_alias = forced_alias
```

El modelo es pre-asignado sin torneo, pero su umbral de decisión **sí se calibra** sobre el calib set mediante `compute_youden_metrics()`. La diferencia respecto al Expert Router es únicamente la selección del modelo, no el proceso de calibración.

#### Función `compute_youden_metrics`

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

`roc_curve()` de scikit-learn prueba cada valor único de probabilidad presente en el calib set como candidato a umbral, generando la curva completa. El Índice de Youden `J = TPR + (1 − FPR) − 1` se maximiza encontrando el umbral de mejor equilibrio entre sensibilidad y especificidad. El clip `[0.05, 0.95]` previene umbrales degenerados.

La función retorna `None` (umbral = 0.5 por defecto) si la clase no tiene ningún positivo o todos son positivos en el calib set, ya que en ese caso la curva ROC no está definida.

#### Registro de la decisión de routing

Para cada patología se almacena:

| Campo | Descripción |
|---|---|
| `Strategy` | `"Expert Router (alias)"` o `"Override (alias)"` |
| `Winner_Model` | Alias del modelo seleccionado |
| `Threshold_Calib` | Umbral óptimo calibrado (redondeado a 4 decimales) |
| `Youden_J_Calib` | J del ganador **en el calib set** — valor in-sample, optimista |
| `ROC_AUC_Calib` | AUC-ROC del ganador en el calib set |

> **Interpretación del Youden_J_Calib vs Youden_J en métricas:** El `Youden_J_Calib` se obtiene evaluando en el mismo conjunto donde se buscó el umbral óptimo (in-sample, sesgado hacia arriba). El `Youden_J` de las métricas finales se calcula aritméticamente a partir de la matriz de confusión en el test ciego (`J = Sens/100 + Spec/100 − 1`), siendo la medida honesta del rendimiento real. Con ~10 positivos por patología en el calib set, las curvas ROC son inestables y la diferencia entre ambos valores puede ser significativa.

### 2B.6 Fase de evaluación sobre test ciego

```python
final_preds[:, p_idx] = (probs_test[winner_alias][:, p_idx] >= best_thresh).astype(int)
```

Los umbrales ya fijos se aplican al 50% del holdout que no participó en ninguna decisión anterior. `multilabel_confusion_matrix(y_test, final_preds)` produce TN, FP, FN, TP por patología. Las métricas finales son:

```
Sensibilidad (%) = TP / (TP + FN) × 100
Especificidad (%) = TN / (TN + FP) × 100
Youden J         = Sensibilidad/100 + Especificidad/100 − 1
```

### 2B.7 Visualización: curvas ROC de calibración

El script genera un panel 4×4 de subplots (uno por patología). En cada subplot:
- Las curvas de todos los modelos se muestran en gris tenue.
- La curva del modelo ganador se resalta en color.
- El punto óptimo de Youden del ganador se marca con una estrella (★).
- El título del subplot indica el alias ganador y su J de calibración.

Los overrides manuales se distinguen con el encabezado `"Override ★"` en color naranja.

### 2B.8 Archivos de salida (todos versionados automáticamente)

| Archivo | Contenido |
|---|---|
| `final_hybrid_routing_Vn.csv` | Modelo ganador, umbral calibrado y J-calib por patología |
| `final_hybrid_metrics_Vn.csv` | Métricas en test ciego: TP, FN, TN, FP, Sens%, Spec%, Youden J |
| `final_hybrid_predictions_Vn.csv` | Predicción imagen por imagen del test ciego con match exacto |
| `final_hybrid_roc_calib_Vn.png` | Panel 4×4 de curvas ROC de calibración |

---

## Resumen Cronológico del Pipeline Completo

```
Paso 0 — create_test_set.py
   └─ random.seed(42) → mismo holdout siempre
   └─ Genera holdout_test_set.csv (~300 imágenes)
   └─ [Ejecutar UNA sola vez]

Paso 1 — main.py  (repetir por cada modelo)
   └─ NIHChestXRayDataset excluye holdout_test_set.csv estrictamente
   └─ random_split 80/20 con Generator(seed=42) → mismo split siempre
   └─ WeightedRandomSampler → lotes balanceados por rareza
   └─ BCEWithLogitsLoss(pos_weight=sqrt(N_neg/N_pos))
   └─ Adam + CosineAnnealingWarmRestarts + EarlyStopping(p=5, δ=0.001)
   └─ Guarda best_model_Vn.pth cuando val_AUC mejora ≥ 0.001

Paso 2A — demo_model.py  (evaluador individual)
   └─ sample(random_state=42) → mismo orden de holdout siempre
   └─ 50% calib → roc_curve() → argmax(Youden J) → clip[0.05, 0.95]
   └─ 50% test ciego → aplicar umbrales → métricas finales
   └─ Exporta: demo_predictions_Vn.csv, demo_metrics_Vn.csv, demo_roc_calib_Vn.png

Paso 2B — evaluate_final_hybrid_model.py  (sistema híbrido)
   └─ sample(random_state=42) → mismo orden de holdout siempre
   └─ Carga 11 modelos simultáneamente
   └─ Inferencia única: 11 modelos × N imágenes → probs_all[alias](N, 14)
   └─ 50% calib:
       ├─ Expert Router: argmax_alias(J_calib) por patología
       └─ Manual Override: alias forzado, umbral calibrado igual
   └─ 50% test ciego:
       └─ Aplicar [winner_alias, threshold] → multilabel_confusion_matrix
   └─ Exporta: routing_Vn.csv, metrics_Vn.csv, predictions_Vn.csv, roc_calib_Vn.png
```

---

## Garantías de Reproducibilidad

| Semilla / Mecanismo | Ubicación | Efecto garantizado |
|---|---|---|
| `random.seed(42)` | `create_test_set.py` | Mismo holdout exacto en cualquier máquina con el mismo CSV |
| `torch.Generator().manual_seed(42)` | `main.py` | Mismo split 80/20 en cada entrenamiento del mismo dataset |
| `RANDOM_SEED` en `checkpoint.pth['config']` | `main.py` | Auditable a posteriori: qué semilla usó cada modelo |
| `df.sample(frac=1, random_state=42)` | `demo_model.py` | Mismo split calib/test del holdout siempre |
| `df.sample(frac=1, random_state=42)` | `evaluate_final_hybrid_model.py` | Mismo split calib/test para todos los modelos del torneo |
| `get_next_version()` en todos los scripts | Todos | Nunca sobrescribe resultados de ejecuciones anteriores |
