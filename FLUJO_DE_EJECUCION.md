# Flujo de Ejecución del Código: Paso a Paso

Este documento detalla qué sucede exactamente cuando ejecutas el comando `python main.py`, explicando cada archivo, función y variable importante en el proceso.

## 1. Fase Previa: Aislamiento Clínico (`src/data/create_test_set.py`)

Antes de que la red neuronal vea cualquier imagen, debemos separar rigurosamente un grupo de pacientes para el examen final. Este paso evita el Data Leakage (Fuga de datos).
1.  **Lectura Inicial**: Abre el CSV original con los 112,000 registros.
2.  **Muestreo Equitativo**: Selecciona 10 imágenes al azar por cada una de las 14 patologías usando una semilla matemática fija.
3.  **La Lista Negra**: Guarda los nombres de esas 140 imágenes extraídas en el archivo `holdout_test_set.csv`.

---

## 2. El Director de Orquesta: `main.py`

El entrenamiento real comienza aquí. Este script coordina todos los componentes de la red neuronal.

### Fase de Configuración (Líneas 14-22)
Antes de nada, el código define las "reglas del juego" mediante constantes:
*   `DATA_DIR`: Dónde buscar las imágenes.
*   `BATCH_SIZE = 8`: Cuántas imágenes procesar a la vez. Si tu computadora se queda sin memoria, este número baja.
*   `IMAGE_SIZE = 224`: Las redes neuronales necesitan entradas de tamaño fijo. Las imágenes originales (1024x1024) se encogerán a 224x224.
*   `UNDERSAMPLE_RATE = 0.25`: Se descartará el 75% de las imágenes "sanas" (No Finding) para que el modelo no aprenda solo a decir "sano".
*   `CHECKPOINT_FILE`: El archivo donde se guardará el estado exacto del entrenamiento (pesos, optimizador, época) para poder reanudarlo si se corta.

### Inicio de `main()`
1.  **Selección de Dispositivo (Hardware)**:
    *   El código comprueba si tienes una GPU AMD (`dml`), NVIDIA (`cuda`) o si debe usar el procesador (`cpu`). Esto es vital para la velocidad.
    *   *Variable*: `device` guarda esta elección.

1.  **Transformaciones (Validación vs Entrenamiento) (Líneas 114-135)**:
    *   **Validación (`val_transforms`)**: Solo prepara la imagen para que el modelo la entienda:
        1.  `Resize(224)`: Achicar la imagen.
        2.  `ToTensor()`: Convertir la imagen de píxeles (0-255) a números matemáticos (0.0-1.0).
        3.  `Normalize(...)`: Estandarizar colores.
    *   **Entrenamiento (`train_transforms`)**: Aplica la función anterior PLUS **Data Augmentation** (Aumentación de Datos):
        *   `RandomHorizontalFlip`: Voltea la imagen como un espejo aleatoriamente.
        *   `RandomRotation`: Gira la imagen ligeramente.
        *   `RandomUnsharpMask`: Aplica un filtro de enfoque.
        *   *Objetivo*: Hacer que el modelo vea variaciones de la misma imagen para que no memorice el dataset (evitar Sobreajuste/Overfitting).

2.  **Preparación de los Lotes (DataLoaders y Sampler)**:
    *   Primero, se separa en Train (80%) y Validation (20%).
    *   **`WeightedRandomSampler`**: Es un selector inteligente para el set de Entrenamiento. Calcula qué patologías son las más raras y las elige a propósito con mucha más frecuencia. Esto asegura que cada paquete (batch) tenga ejemplos de casi todas las enfermedades, combatiendo el desbalance.
    *   **`DataLoader`**: Empaqueta todo en grupos de tamaño `BATCH_SIZE` (ej. 8 imágenes a la vez).

3.  **Inicialización del Modelo y Pérdida (Líneas 171-185)**:
    *   Llama a `get_model(...)` para cargar **DenseNet-121**.
    *   Calcula `pos_weights`: Determina qué pesos aplicar a la función de error.
    *   *Nota técnica:* Para evitar una doble sobre-corrección (ya que usamos el `Sampler` inteligente), se relajan los pesos aplicando una raíz cuadrada (`torch.sqrt`).
    *   Define `criterion = BCEWithLogitsLoss` pasando esos pesos relajados.
    *   Crea el `optimizer = Adam` para ajustar los tornillos (pesos) del modelo.

4.  **Recuperación de Respaldo - Resumable Training (Líneas 188-201)**:
    *   Busca si existe el archivo `checkpoint.pth`.
    *   **Si existe**: Carga la memoria del modelo y el estado del optimizador, y reanuda desde la época donde se interrumpió.
    *   **Si no existe**: Empieza el entrenamiento desde la época 1.

5.  **El Bucle Principal (Líneas 207-244)**:
    *   Un bucle `for` que se repite hasta la época final marcada por `EPOCHS`.
    *   En cada vuelta (época):
        1.  `trainer.train_one_epoch(...)`: El modelo estudia (Train).
        2.  `trainer.validate(...)`: El modelo realiza un examen de práctica (Valid).
        3.  Se guarda el progreso de la época (Loss, AUC, Tiempo) en `results.csv`.
        4.  **Guardado Seguro (Checkpoint)**: Salva `checkpoint.pth` para poder reanudar si se apaga la PC.
    *   Al final de todo el proceso de las 10 épocas, guarda una versión "limpia" de los pesos `densenet_nih.pth` lista para inferencia en producción.

---

## 3. El Gestor de Datos: `src/data/dataset.py`

Esta clase (`NIHChestXRayDataset`) actúa como un bibliotecario. No lee todos los libros a la vez, sino que sabe dónde están y entrega uno cuando se le pide.

### `__init__` (Al arrancar)
1.  **Lectura del CSV**: Carga `Data_Entry_2017.csv` en memoria (`self.df`).
2.  **Búsqueda de Imágenes**: Escanea el disco duro para ver qué imágenes existen realmente (`self.image_paths`).
3.  **Filtrado Base**: Elimina del CSV las filas de imágenes que no se encontraron en el disco.
4.  **Exclusión Estricta (Holdout Set)**: *¡Paso Crítico!* Lee la lista negra (`holdout_test_set.csv`) y **borra** esas 140 imágenes de la memoria antes de hacer nada más. La red neuronal nunca interactuará con ellas.
5.  **Undersampling (Líneas 82-89)**:
    *   Separa las filas con 'No Finding' (sanos) y las enfermedades.
    *   Toma solo una fracción de los sanos (ej. el 25% o lo que dicte `UNDERSAMPLE_RATE`).
    *   Vuelve a juntar todo. Esto equilibra el juego a favor de las enfermedades.

### `__getitem__` (Pedido bajo demanda)
Esta función se llama miles de veces, una por cada imagen.
1.  **Entrada**: Un número `idx` (ej. "dame la imagen número 105").
2.  **Carga**: Abre el archivo de imagen correspondiente con `PIL.Image`.
3.  **Transformación**: Aplica el `data_transforms` definido en `main.py` (Resize -> Tensor -> Normalize).
4.  **Etiquetado**:
    *   Lee la columna 'Finding Labels' (ej. "Infiltration|Mass").
    *   Crea un vector de ceros y unos.
    *   Ejemplo para 3 enfermedades: Si tiene la 1 y la 3, devuelve `[1, 0, 1]`.
5.  **Salida**: Entrega el par `(imagen_procesada, etiquetas)`.

---

## 4. El Cerebro: `src/models/densenet.py`

### `get_model`
1.  **Base**: Descarga `densenet121` de internet. Esta red ya sabe reconocer gatos, perros, coches, etc. (ImageNet). Esto significa que ya sabe identificar bordes, texturas y formas complejas ("Know-how").
2.  **Cirugía (Línea 29)**:
    *   La red original termina en una capa que clasifica 1000 cosas.
    *   Le cortamos esa cabeza y le ponemos una nueva capa lineal (`nn.Linear`) que tiene `num_classes` (14) salidas.
    *   Ahora la red usará su conocimiento previo para aprender específicamente sobre rayos X.

---

## 5. El Entrenador: `src/training/trainer.py`

Esta clase `Trainer` hace el trabajo sucio del bucle de entrenamiento.

### `train_one_epoch` (Estudiar)
1.  **`model.train()`**: Le dice al modelo "ponte en modo aprendizaje". Algunas capas (como Dropout) se comportan diferente.
2.  **Iteración**: Va pidiendo lotes al `train_loader`.
3.  **Pasos Clave**:
    *   `optimizer.zero_grad()`: Borra los cálculos de corrección de la vuelta anterior.
    *   `outputs = model(images)`: El modelo mira el paquete de 8 imágenes e intenta diagnosticar.
    *   `loss = criterion(...)`: Se compara el diagnóstico adivinado contra la respuesta real.
        *   *(Workaround DirectML)*: Como las tarjetas AMD pueden rebotar en esta operación matemática interna (`aten::log_sigmoid_forward`), parte de este cálculo se desplaza temporalmente al Procesador (`CPU`) y vuelve a la GPU limpiamente.
    *   `loss.backward()`: **Backpropagation**. El algoritmo averigua quién fue el "culpable" de la equivocación a lo largo de toda la red.
    *   `optimizer.step()`: Se ajustan los "tornillos" (pesos) del modelo en la dirección correcta para reducir el error la próxima vez.

### `validate` (Examen)
1.  **`model.eval()`**: Modo examen. "No cambies nada, solo responde".
2.  **`with torch.no_grad()`**: Apaga la calculadora de gradientes. Esto ahorra mucha memoria y hace que vaya más rápido, porque no vamos a aprender, solo a medir.
3.  **Cálculo de AUC**:
    *   Se guardan todas las predicciones y etiquetas reales.
    *   Se usa `roc_auc_score` para ver qué tan bueno es el ranking del modelo (independientemente de un punto de corte fijo).

---

## 6. La Auditoría Final: `evaluate_model.py`

Una vez terminado el entrenamiento (cuando `main.py` finaliza todas sus épocas y escupe el archivo `densenet_nih.pth`), entra en juego este script para evaluar científicamente al modelo.

1.  **Carga del Test Set**: Lee la lista negra (`holdout_test_set.csv`) aislando únicamente las 140 imágenes.
2.  **Modo Examen Estricto**: Carga el modelo guardado y ejecuta `model.eval()` y `torch.no_grad()`. Esto apaga por completo los motores de aprendizaje de la red neuronal, garantizando que el modelo **no pueda alterar sus pesos** mientras las examina.
3.  **Predicción Pura**: El modelo diagnostica las 140 imágenes basándose en su `densenet_nih.pth`, sin trucos, ni Samplers, ni Data Augmentation.
4.  **Veredicto**: Genera 14 Matrices de Confusión por consola midiendo *Sensibilidad* y *Especificidad* de cada patología, además de exportar una tabla auditable (`evaluation_results.csv`) registro por registro.

---

## Resumen Final del Flujo de Datos Cronológico

1.  `create_test_set.py` -> Aísla 140 archivos críticos al `holdout_test_set.csv`
2.  **Inicio de `main.py`:**
3.  **Disco Duro** -> `dataset.py` (Lee archivo general)
4.  `dataset.py` -> Chequea `holdout_test_set.csv` y purga esas 140 imágenes.
5.  `dataset.py` -> Reduce la clase mayoritaria (Undersampling).
6.  `dataset.py` -> `transforms` (Voltea, Rota y Afila bordes al vuelo).
7.  `transforms` -> `WeightedRandomSampler` (Fuerza aparición de Minorías Clínicas).
8.  `Sampler` -> `DataLoader` (Agrupa en paquetes de 8).
9.  `DataLoader` -> `trainer.py` (Envía a la Tarjeta Gráfica/DirectML).
10. `trainer.py` -> `model` (Predice usando Pesos Relajados).
11. `model` -> `loss` (Calcula error temporalmente en CPU y corrige en GPU).
12. `loss` -> `optimizer` (Ajusta modelo y aprende).
13. `metrics` -> `csv` (Guarda historia) + `checkpoint.pth` (Sobrescribe progreso).
14. **Fin de `main.py`:** Genera modelo final `densenet_nih.pth`.
15. **Auditoría Final:** `evaluate_model.py` usa `densenet_nih.pth` contra `holdout_test_set.csv` generando la tabla auditada.
