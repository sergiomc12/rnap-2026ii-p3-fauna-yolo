# Paso 1 — Datos: anotaciones y descarga

Obtiene Caltech Camera Traps desde LILA BC y deja las anotaciones legibles desde
código.

## Clases y módulos de este paso

| Archivo | Clase / función | Qué hace |
|---|---|---|
| [`anotaciones.py`](anotaciones.py) | `Caja`<br>`Imagen`<br>`Particion`<br>`cargar_particion()`<br>`cargar_todas()`<br>`archivos_necesarios()` | Lee el formato COCO Camera Traps. `Caja.a_yolo()` convierte de esquina+tamaño a centro+tamaño normalizado |
| [`descargar.py`](descargar.py) | `Progreso`<br>`bajar_una()`<br>`descargar()` | Descarga paralela y reanudable desde LILA BC |

## Comandos

Desde la raíz del repositorio.

### Descargar las imágenes

```bash
.venv/Scripts/python.exe pregunta3/paso1_datos/descargar.py
.venv/Scripts/python.exe pregunta3/paso1_datos/descargar.py --particiones train cis_val
.venv/Scripts/python.exe pregunta3/paso1_datos/descargar.py --limite 100
```

**Reanudable**: una imagen ya en disco se salta. La descarga completa supera las
tres horas, más de lo que dura una sesión, así que la reanudación no es una
comodidad sino un requisito — basta volver a correr el mismo comando.

### Usar las clases desde Python

```python
from pregunta3.paso1_datos.anotaciones import cargar_particion, cargar_todas

train = cargar_particion("train")
print(len(train.imagenes), len(train.con_cajas), train.n_cajas)
print(train.ubicaciones)          # ubicaciones de camara presentes
print(train.prevalencia())        # cajas por especie, de mas a menos

imagen = train.con_cajas[0]
caja = imagen.cajas[0]
print(caja.a_yolo(imagen.ancho, imagen.alto))   # (cx, cy, w, h) en [0,1]
```

---

## Lo que se hizo

### 1. Encontrar las URLs reales

La página de LILA no expone los enlaces de forma obvia y las rutas que se
adivinan devuelven 403 o 404. Los metadatos salen de:

```
https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/
  labels/caltech_bboxes_20200316.json      37 MB
  CaltechCameraTrapsSplits_v0.json          8 KB
  eccv_18_annotations.tar.gz                3 MB
```

Y las imágenes, de un contenedor de Azure cuyo prefijo se obtuvo **listando el
contenedor**, no leyendo la documentación:

```bash
curl "https://lilawildlife.blob.core.windows.net/lila-wildlife?restype=container&comp=list&prefix=caltech&maxresults=25"
```

Prefijo correcto:
`https://lilawildlife.blob.core.windows.net/lila-wildlife/caltech-unzipped/cct_images/`

### 2. Decidir cuántas imágenes bajar

| Escenario | Imágenes | GB | Tiempo |
|---|---|---|---|
| Submuestreo a 4,000 | 4,000 | 1.7 | 16 min |
| train + ambos val | 16,053 | 6.8 | 64 min |
| **todas las que tienen caja** | **46,026** | **20.5** | **192 min** |

Se bajan **todas**, y la razón decisiva son las clases raras. `fox` tiene
**cinco** cajas de entrenamiento en total y `badger` nueve. Submuestrear a un
tercio dejaría a `fox` con uno o dos ejemplos: no es que el modelo aprenda peor,
es que el entregable #2 —analizar el efecto del desbalance— se queda sin caso
que analizar.

Es distinto de la Pregunta 1, donde había 4,407 estudios y quedarse con 576
dejaba 73 positivos en la etiqueta más rara. Acá el piso ya está en cinco.

El costo de no submuestrear son tres horas desatendidas, contra las 6h25m que
costó bajar 1.39 GB de Kaggle en la Pregunta 1.

### 3. Medir la velocidad antes de comprometerse

| Modo | Velocidad |
|---|---|
| Secuencial | 0.47 MB/s |
| **Paralelo, 12 hilos** | **1.78 MB/s** |

La descarga está limitada por **latencia**, no por ancho de banda: doce hilos la
cuadruplican. Tamaño medio por imagen: **0.43 MB**.

### 4. Escritura atómica, por la lección de la Pregunta 1

Cada imagen se baja a `.parcial` y se renombra al completarse. En la Pregunta 1,
una corrida interrumpida dejó un CSV al 95% que el pipeline dio por completo y
habría entrenado con datos faltantes, en silencio. Acá un corte no deja JPEGs
truncados que la siguiente corrida acepte como buenos.

---

## Hallazgos

### La partición oficial resuelve el entregable #3

`cis` son las mismas ubicaciones de cámara que el entrenamiento; `trans`, otras.
La brecha entre ambas es exactamente el experimento de generalización geográfica
que pide el enunciado, sobre la partición que los autores diseñaron para eso.

### `trans_val` tiene una sola ubicación

1,533 imágenes de **una única cámara**. Cualquier métrica sobre ese conjunto mide
el desempeño en esa cámara concreta, no "en ubicaciones nuevas" en general. Para
el experimento del entregable #3 hay que usar `trans_test`, que tiene nueve.

Es el mismo problema que los 56 estudios dorados de la Pregunta 1: un conjunto de
evaluación chico y poco diverso que infla la incertidumbre.

### Las imágenes son grandes: 2048×1494

Entrenar a 640 px exige reducirlas más de tres veces. Un animal lejano, que ya
ocupaba pocos píxeles, queda diminuto. Es justamente lo que el entregable #4 pide
analizar.

### `deer` desaparece en `trans_test`

44 cajas en train, 140 en `cis_test`, **cero** en `trans_test`. Su mAP ahí no
existe, igual que un AUC sin positivos. Hay que reportarlo como indefinido y no
como cero, o el promedio por especie queda sesgado hacia abajo sin avisar.

---

## Lo que se cerró

### La descarga terminó

**46,026 imágenes, 18 GB** en disco. La descarga sobrevivió cuatro
interrupciones: `bajar_una()` salta cualquier archivo ya presente con tamaño
mayor a cero, y escribe a `<nombre>.jpg.parcial` antes de renombrar, así que un
corte a mitad de una transferencia no deja un JPEG truncado que la corrida
siguiente daría por bueno.

### Ninguna imagen quedó corrupta

Verificadas las **46,026** con PIL: `0 corruptas`. La verificación hace dos
pasadas por archivo y las dos hacen falta:

```python
with Image.open(ruta) as img:
    img.verify()      # valida la estructura del JPEG
with Image.open(ruta) as img:
    img.load()        # fuerza la decodificacion: detecta el truncamiento final
```

`verify()` sola **no** detecta un archivo cortado al final, que es exactamente la
forma en que falla una descarga interrumpida. Con solo la primera pasada, el
resultado habría sido el mismo "0 corruptas" sin haber comprobado nada.

Es la confirmación de que la escritura atómica funcionó: sin ella, cuatro cortes
sobre 46 mil archivos habrían dejado alguno a medias con casi total seguridad.

### El solapamiento de ráfagas está medido

`cis_val` comparte **224 ráfagas** con `train`, lo que contamina 75 de sus 1,665
imágenes con caja (4.5%). Las particiones `trans_*` y `cis_test` no comparten
ninguna. El filtro vive en el [paso 2](../paso2_preprocesamiento/), que es donde
se escriben las listas de cada partición.

---

## Pendiente

| | Qué falta | Por qué importa |
|---|---|---|
| 1 | Bajar también las imágenes **vacías** con anotación de clase (las 243,100 del conjunto completo) | Hoy solo están las 46,026 con caja; las vacías harían falta para la ablación de negativos explícitos del [paso 3](../paso3_modelo/) |
| 2 | Estadísticas de tamaño de caja por partición | El [paso 5](../paso5_generalizacion/) las calcula al evaluar, pero tenerlas acá describiría el conjunto antes de modelar |
