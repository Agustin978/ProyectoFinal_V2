# Flujo de Ejecución del Código: Paso a Paso

Este documento detalla qué sucede exactamente cuando ejecutas el comando `python main.py`, explicando cada archivo, función y variable importante en el proceso.

## 1. El Director de Orquesta: `main.py`

Todo comienza aquí. Este script coordina todos los componentes.

### Fase de Configuración (Líneas 14-22)
Antes de nada, el código define las "reglas del juego" mediante constantes:
*   `DATA_DIR`: Dónde buscar las imágenes.
*   `BATCH_SIZE = 8`: Cuántas imágenes procesar a la vez. Si tu computadora se queda sin memoria, este número baja.
*   `IMAGE_SIZE = 224`: Las redes neuronales necesitan entradas de tamaño fijo. Las imágenes originales (1024x1024) se encogerán a 224x224.
*   `UNDERSAMPLE_RATE = 0.30`: Se descartará el 70% de las imágenes "sanas" (No Finding) para que el modelo no aprenda solo a decir "sano".
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

## 2. El Gestor de Datos: `src/data/dataset.py`

Esta clase (`NIHChestXRayDataset`) actúa como un bibliotecario. No lee todos los libros a la vez, sino que sabe dónde están y entrega uno cuando se le pide.

### `__init__` (Al arrancar)
1.  **Lectura del CSV**: Carga `Data_Entry_2017.csv` en memoria (`self.df`).
2.  **Búsqueda de Imágenes**: Escanea el disco duro para ver qué imágenes existen realmente (`self.image_paths`).
3.  **Filtrado**: Elimina del CSV las filas de imágenes que no se encontraron en el disco.
4.  **Undersampling (Líneas 63-73)**:
    *   Separa las filas con 'No Finding' (sanos) y las enfermedades.
    *   Toma solo una fracción de los sanos.
    *   Vuelve a juntar todo. Esto equilibra el juego.

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

## 3. El Cerebro: `src/models/densenet.py`

### `get_model`
1.  **Base**: Descarga `densenet121` de internet. Esta red ya sabe reconocer gatos, perros, coches, etc. (ImageNet). Esto significa que ya sabe identificar bordes, texturas y formas complejas ("Know-how").
2.  **Cirugía (Línea 29)**:
    *   La red original termina en una capa que clasifica 1000 cosas.
    *   Le cortamos esa cabeza y le ponemos una nueva capa lineal (`nn.Linear`) que tiene `num_classes` (14) salidas.
    *   Ahora la red usará su conocimiento previo para aprender específicamente sobre rayos X.

---

## 4. El Entrenador: `src/training/trainer.py`

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

## Resumen de Flujo de Datos

1.  **Disco Duro** -> `dataset.py` (Lee archivo)
2.  `dataset.py` -> `transforms` (Redimensiona/Normaliza)
3.  `transforms` -> `DataLoader` (Agrupa en paquetes de 8)
4.  `DataLoader` -> `trainer.py` (Envía a GPU)
5.  `trainer.py` -> `model` (Predice)
6.  `model` -> `loss` (Calcula error)
7.  `loss` -> `optimizer` (Mejora modelo)
8.  `metrics` -> `csv` (Guarda historia)
