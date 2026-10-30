# Paso 2 — Preprocesamiento: COCO a formato YOLO

Convierte las anotaciones y arma la estructura que Ultralytics espera.

## Clases y módulos de este paso

| Archivo | Clase / función | Qué hace |
|---|---|---|
| [`conversion.py`](conversion.py) | `MapaClases` | Traduce los IDs de COCO a los índices contiguos de YOLO |
| | `convertir_imagen()`<br>`LineasEtiqueta` | Produce las líneas `clase cx cy w h` de una imagen |
| | `escribir_etiquetas()`<br>`ResumenConversion` | Escribe un `.txt` por imagen y reporta descartes |
| | `escribir_data_yaml()` | Genera el `data.yaml` de Ultralytics |
| [`preparar.py`](preparar.py) | `main()`<br>`preparar_carpeta_imagenes()`<br>`escribir_lista()` | Script: arma el conjunto completo |

## Comandos

```bash
# convertir todo lo que haya anotado
.venv/Scripts/python.exe pregunta3/paso2_preprocesamiento/preparar.py

# convertir solo lo que ya esta descargado (util mientras baja)
.venv/Scripts/python.exe pregunta3/paso2_preprocesamiento/preparar.py --solo-presentes
```

Produce en `dataset/`:

```
dataset/
├── train.txt  cis_val.txt  ...             listas de rutas absolutas
├── train.txt  cis_val.txt  cis_test.txt  trans_val.txt  trans_test.txt
├── data_base.yaml  data_cis.yaml  data_trans.yaml  data_trans_val.yaml
└── resumen.json
```

### Usar las clases desde Python

```python
from pregunta3.paso1_datos.anotaciones import cargar_todas
from pregunta3.paso2_preprocesamiento.conversion import MapaClases, convertir_imagen

particiones = cargar_todas()
mapa = MapaClases.desde_particiones(list(particiones.values()))
print(mapa.n, mapa.nombres)        # 15 clases, ordenadas por frecuencia

imagen = particiones["train"].con_cajas[0]
etiqueta = convertir_imagen(imagen, mapa)
print(etiqueta.lineas)             # ('6 0.918125 0.586881 0.146250 0.186774',)
```

---

## Lo que se hizo

### 1. La conversión de coordenadas

COCO guarda **esquina superior izquierda más tamaño, en píxeles**. YOLO guarda
**centro más tamaño, normalizado a [0, 1]**.

Confundirlos desplaza cada caja media caja hacia arriba y a la izquierda, y el
entrenamiento sigue adelante sin ningún error visible: el modelo aprende a
predecir cajas corridas. Verificado sobre una caja real:

```
COCO:  x=1730.56  y=737.28  w=299.52  h=279.04   imagen 2048×1494
YOLO:  clase 6    0.918125  0.586881  0.146250   0.186774

  cx = (1730.56 + 299.52/2) / 2048 = 0.918125
  cy = (737.28  + 279.04/2) / 1494 = 0.586881
  w  =  299.52 / 2048             = 0.146250
  h  =  279.04 / 1494             = 0.186774
```

Los valores de COCO van **sin redondear**: con `x=1730.6` y `w=299.5` la cuenta
da 0.918140, no 0.918125, y la verificación dejaría de reproducir. Es el tipo de
detalle que hace que un número "casi igual" pase por correcto.

### 2. El remapeo de clases

Los identificadores de COCO Camera Traps no son contiguos ni empiezan en cero.
YOLO exige 0..N−1 en el orden exacto de la lista `names` del `data.yaml`, porque
**el índice de clase es la posición en esa lista**.

Se ordenan por frecuencia descendente, de modo que las clases con datos quedan
juntas al principio y una matriz de confusión se lee mejor:

```
0:opossum  1:raccoon  2:rabbit  3:coyote  4:bobcat  5:cat  6:dog  7:car
8:squirrel  9:bird  10:skunk  11:rodent  12:deer  13:badger  14:fox
```

**`empty` se excluye.** En detección, la ausencia de cajas ya expresa "acá no hay
nada"; crear una clase de fondo le enseñaría al modelo a dibujar una caja
alrededor de la nada. Quedan **15 clases**.

### 3. Las imágenes no se copian por partición

Ultralytics deduce la ruta de las etiquetas reemplazando `/images/` por
`/labels/`, así que esos nombres de carpeta son obligatorios y deben estar al
mismo nivel. Pero **no exige copiar las imágenes**: cada partición puede ser un
archivo de texto con una ruta por línea.

Las 46,026 imágenes viven una sola vez. Copiarlas por partición duplicaría 20 GB
cinco veces, y además una misma imagen aparece en más de una partición.

Las listas apuntan directamente a `pregunta3/paso1_datos/images/`, que convive
con `labels/` como carpeta hermana — el nombre exacto que Ultralytics necesita.
Un primer intento con un enlace de directorio falló en este sistema de archivos,
y copiar 20 GB solo para renombrar una carpeta no tenía sentido.

### 4. Cuatro `data.yaml`, uno por pregunta del enunciado

| Archivo | train | val | Para qué |
|---|---|---|---|
| `data_base.yaml` | train | cis_val | Entrenamiento y validación normales |
| `data_cis.yaml` | train | cis_test | Prueba en ubicaciones **vistas** |
| `data_trans.yaml` | train | trans_test | Prueba en ubicaciones **nunca vistas** ← entregable #3 |
| `data_trans_val.yaml` | train | trans_val | Validación en ubicaciones nuevas (una cámara) |

El mismo modelo entrenado se evalúa contra los cuatro con `model.val(data=...)`.
La brecha entre `data_cis` y `data_trans` **es** el experimento de generalización
geográfica.

### 5. Validación de geometría

`_geometria_valida()` descarta cajas con tamaño cero o centro fuera de la imagen.
Ultralytics las acepta sin quejarse y después entrena con ellas, de modo que
filtrarlas acá es la única oportunidad.

Resultado sobre las primeras 2,104 imágenes descargadas: **cero descartes** por
clase y **cero** por geometría. Las anotaciones de CCT están limpias.

---

## Hallazgos

### Las anotaciones no necesitan limpieza

Ninguna caja degenerada ni fuera de rango en la muestra verificada. Es inusual en
conjuntos de este tamaño y conviene decirlo: no se invirtió esfuerzo en limpieza
porque, medido, no hacía falta.

### El filtro `--solo-presentes` es necesario mientras baja

Etiquetar una imagen que todavía no está en disco dejaría a Ultralytics buscando
un archivo inexistente y abortando la época. La bandera limita la conversión a lo
descargado, de modo que el conjunto es utilizable en cualquier momento de la
descarga.

---

## Lo que se cerró después

### La descarga terminó: el conjunto ya no depende de `--solo-presentes`

`resumen.json` reporta `sin_imagen_en_disco: 0` en las cinco particiones. Las
46,026 imágenes están, y las listas cubren el conjunto completo:

| Partición | Imágenes con caja | Cajas | Descartadas por fuga |
|---|---:|---:|---:|
| `train` | 12,099 | 12,617 | — |
| `cis_val` | 1,590 | 1,686 | **75** |
| `cis_test` | 12,696 | 13,264 | 0 |
| `trans_val` | 1,533 | 1,673 | 0 |
| `trans_test` | 18,033 | 18,786 | 0 |

Cero descartes por clase y cero por geometría en todo el conjunto, no solo en la
muestra verificada al principio.

### La fuga de ráfagas estaba y se filtró

`cis_val` comparte **224 ráfagas** con `train`, lo que contamina 75 de sus 1,665
imágenes con caja (4.5%). Esas 75 se excluyen por defecto. Los dos conjuntos de
prueba y `trans_val` **no comparten ninguna** ráfaga con el entrenamiento, lo
cual era esperable: están en otras ubicaciones de cámara, y una ráfaga no puede
cruzar cámaras.

El efecto se puede medir con `--permitir-fuga`, que desactiva el filtro. Sin él,
el mAP de `cis_val` incluiría 75 imágenes casi memorizadas.

### Las imágenes vacías: la decisión quedó tomada

`empty` se excluye y no se agregan negativos explícitos en esta corrida. La razón
completa está en el [README del paso 3](../paso3_modelo/README.md), que es donde
el entregable #1 pide documentarla.

---

## Pendiente

| | Qué falta | Por qué importa |
|---|---|---|
| 1 | Ablación con negativos explícitos (las imágenes vacías como fondo) | La decisión del entregable #1 quedó argumentada pero no medida; corresponde al [paso 3](../paso3_modelo/) |
| 2 | Medir cuánto infla la fuga de ráfagas, con `--permitir-fuga` | Se filtró sin cuantificar el efecto; el número sirve de advertencia metodológica en el informe |
