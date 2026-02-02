# Flujo de Ejecución del Código: Paso a Paso

Este documento detalla qué sucede exactamente cuando ejecutas el comando `python main.py`, explicando cada archivo, función y variable importante en el proceso.

## 1. El Director de Orquesta: `main.py`

Todo comienza aquí. Este script coordina todos los componentes.

### Fase de Configuración (Líneas 14-22)
Antes de nada, el código define las "reglas del juego" mediante constantes:
*   `DATA_DIR`: Dónde buscar las imágenes.
*   `BATCH_SIZE = 8`: Cuántas imágenes procesar a la vez. Si tu computadora se queda sin memoria, este número baja.
*   `IMAGE_SIZE = 224`: Las redes neuronales necesitan entradas de tamaño fijo. Las imágenes originales (1024x1024) se encogerán a 224x224.
*   `UNDERSAMPLE_RATE = 0.25`: Se decidió descartar el 75% de las imágenes "sanas" (No Finding) para que el modelo no aprenda solo a decir "sano".

### Inicio de `main()`
1.  **Selección de Dispositivo (Hardware)**:
    *   El código comprueba si tienes una GPU AMD (`dml`), NVIDIA (`cuda`) o si debe usar el procesador (`cpu`). Esto es vital para la velocidad.
    *   *Variable*: `device` guarda esta elección.

2.  **Preparación de Transformaciones (Líneas 40-47)**:
    *   Se crea `data_transforms`. Es una "receta" que se aplicará a cada imagen *en el momento de cargarla*:
        1.  `Resize(224)`: Achicar la imagen.
        2.  `ToTensor()`: Convertir la imagen de píxeles (0-255) a números matemáticos (0.0-1.0) entendibles por PyTorch.
        3.  `Normalize(...)`: Restar la media y dividir por la desviación estándar de ImageNet. Esto ayuda a que el modelo aprenda más rápido.

    *   `DataLoader`: Es un cargador inteligente. En lugar de darte todo de golpe, te da paquetes (`batches`) de 8 imágenes.
    *   `shuffle=True` (solo en train): Baraja las cartas en cada época para que el modelo no memorice el orden.

6.  **Inicialización del Modelo (Línea 71)**:
    *   Llama a `get_model(...)` en `src/models/densenet.py`.

7.  **Definición de Loss y Optimizador**:
    *   `pos_weights`: Se calculan pesos para dar más importancia a las enfermedades raras.
    *   `criterion = BCEWithLogitsLoss`: La fórmula matemática que mide el error. "BCE" (Binary Cross Entropy) es ideal para preguntas de Sí/No (¿Tiene Neumonía? Sí/No. ¿Tiene Hernia? Sí/No), aplicado a las 14 enfermedades a la vez.
    *   `optimizer = Adam`: El algoritmo que ajusta los "tornillos" (pesos) del modelo basándose en el error reportado por `criterion`.

8.  **El Bucle Principal (Líneas 89-112)**:
    *   Un bucle `for` que se repite `EPOCHS` veces (10).
    *   En cada vuelta (época):
        1.  `trainer.train_one_epoch(...)`: El modelo estudia.
        2.  `trainer.validate(...)`: El modelo es evaluado.
        3.  Se guarda el progreso en `results.csv`.
    *   Al final, `torch.save` guarda el cerebro entrenado en un archivo `.pth`.

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
    *   `optimizer.zero_grad()`: Borra los cálculos de la vuelta anterior.
    *   `outputs = model(images)`: El modelo mira las imágenes y adivina.
    *   `loss = criterion(outputs, labels)`: Se compara la adivinanza con la realidad.
    *   `loss.backward()`: **Backpropagation**. Calcula de quién fue la culpa del error (gradientes).
    *   `optimizer.step()`: Ajusta los pesos para reducir el error la próxima vez.

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
