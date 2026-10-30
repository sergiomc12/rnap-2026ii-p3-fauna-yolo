# Paso 4 — Evaluación con desglose por especie · entregable #2

> Entregable #2: *"mAP@0.5 y mAP@0.5:0.95 con desglose por especie, señalando el
> efecto del desbalance"*.

Las métricas se **reimplementan** en vez de copiar las de Ultralytics. No es
desconfianza: es que el enunciado pide tres cosas que la salida estándar no da.

## Clases y módulos de este paso

| Archivo | Clase / función | Qué hace |
|---|---|---|
| [`metricas.py`](metricas.py) | `Prediccion`, `Verdad`, `Muestra` | Una imagen con lo que el modelo predijo y lo que había |
| | `iou()` | Intersección sobre unión de dos cajas en píxeles |
| | `ap_todos_los_puntos()` | Área bajo la curva P-R con envolvente monótona (VOC2010+/COCO) |
| | `average_precision()` | AP de una clase, una por umbral de IoU; `None` si no hay positivos |
| | `evaluar()` → `ResultadoEvaluacion` | mAP macro + `ResultadoClase` por especie |
| | `bootstrap_map()` | IC del mAP@0.5 remuestreando **imágenes** |
| | `correlacion_spearman()` | Correlación de rangos entre soporte y AP |
| [`predecir.py`](predecir.py) | `predecir()` | Corre el modelo sobre una partición y vuelca un JSONL |
| | `leer_verdades()` | Lee un `.txt` de YOLO y lo devuelve en píxeles `xyxy` |
| | `cargar_predicciones()` | Lee el JSONL: cabecera `_meta` + una línea por imagen |
| [`evaluar.py`](evaluar.py) | `a_muestras()` | JSONL → objetos de `metricas` |
| | `map_ponderado()` | mAP pesado por soporte, para contrastarlo con el macro |
| | `analizar()` | Todo el entregable #2 para una partición |
| | `tabla_por_especie()` | La tabla markdown del informe |
| [`contrastar_con_ultralytics.py`](contrastar_con_ultralytics.py) | `main()` | Script ejecutable: contrasta las métricas **propias** contra las de Ultralytics sobre la misma partición. Es la verificación de que el cálculo escrito a mano mide lo que dice medir |

## Comandos

```bash
# 1. volcar las predicciones (una vez; lo reusan los pasos 5 y 6)
.venv/Scripts/python.exe pregunta3/paso4_evaluacion/predecir.py

# solo una particion, o todas las cinco
.venv/Scripts/python.exe pregunta3/paso4_evaluacion/predecir.py --particion cis_test
.venv/Scripts/python.exe pregunta3/paso4_evaluacion/predecir.py --todas

# 2. calcular las metricas
.venv/Scripts/python.exe pregunta3/paso4_evaluacion/evaluar.py
.venv/Scripts/python.exe pregunta3/paso4_evaluacion/evaluar.py --sin-bootstrap   # mas rapido
```

Produce:

```
predicciones/<particion>.jsonl      una linea por imagen: predicciones + verdades
informes/metricas.json              mAP macro, ponderado, por clase, bootstrap
informes/<particion>.md             la tabla por especie, lista para el informe
```

### Usar las clases desde Python

```python
from pregunta3.paso4_evaluacion.evaluar import a_muestras
from pregunta3.paso4_evaluacion.metricas import evaluar

nombres, muestras = a_muestras("pregunta3/paso4_evaluacion/predicciones/cis_test.jsonl")
resultado = evaluar(muestras, nombres)

print(resultado.map50, resultado.map50_95)
print(resultado.ausentes)              # especies sin una sola caja: AP indefinida
for clase in resultado.presentes:
    print(clase.nombre, clase.soporte, clase.ap50)
```

---

## Lo que se hizo

### 1. Por qué no alcanzan las métricas de Ultralytics

**Clases sin positivos.** `deer` tiene 44 cajas en entrenamiento y **cero** en
`trans_test`. Su AP ahí no es 0.0: no está definida, igual que un AUC sin
positivos. Promediarla como cero bajaría el mAP con una especie que era
imposible de detectar porque no estaba. `average_precision()` devuelve `None` y
`ResultadoEvaluacion.map50` promedia solo las presentes, listando las ausentes
aparte.

**Subconjuntos arbitrarios.** El entregable #4 pide partir la evaluación por
tamaño del objeto, y el #3 por ubicación de cámara. Con las predicciones
guardadas, cada subconjunto es una lista filtrada y el mAP se recalcula en
segundos. Volver a correr `model.val()` por cada corte costaría una pasada
completa cada vez.

**Intervalos de confianza.** Un mAP de 0.87 contra uno de 0.85 no dice nada sin
incertidumbre, y el entregable #3 compara dos particiones.

### 2. La definición de AP que se usa

Área bajo la curva precisión-recall con **envolvente monótona** e interpolación
de **todos los puntos** — la definición de VOC2010 en adelante, que es la que usa
COCO. No la de 11 puntos de VOC2007, que sobreestima.

El emparejamiento es **codicioso por confianza**, el criterio de COCO: las
predicciones se recorren de mayor a menor confianza y cada una toma la caja
anotada libre con la que más se solapa. Una caja ya tomada no se reusa, así que
una segunda detección del mismo animal cuenta como falso positivo — está fijado
en [`tests/test_p3_metricas.py`](../../tests/test_p3_metricas.py) porque permitir
el reuso infla el recall sin que nada avise.

Para mAP@0.5:0.95 se promedian los diez umbrales de COCO (0.50 a 0.95 de 0.05 en
0.05), cada uno con su propio conjunto de cajas tomadas: un IoU de 0.6 empareja
a 0.5 pero no a 0.75, de modo que el estado no se puede compartir entre umbrales.

### 3. El umbral de confianza de la predicción es 0.001, no 0.25

La AP integra la curva precisión-recall **completa**. Recortar las detecciones de
baja confianza trunca la curva por el lado del recall alto y **infla** la
métrica. Por eso `predecir.py` usa `conf=0.001` con `max_det=300`, los valores
con los que se reportan los resultados de COCO.

El 0.25 habitual es un umbral de **decisión**, no de medición, y aparece en el
paso 6, donde el reporte ecológico sí tiene que decidir qué contar.

### 4. El bootstrap remuestrea imágenes, no cajas

Las cajas de una misma imagen no son independientes: comparten cámara,
iluminación y, muchas veces, al mismo animal fotografiado en ráfaga.
Remuestrearlas por separado daría un intervalo demasiado angosto, que es el error
que hace que dos modelos parezcan distintos cuando no lo están.

### 5. Dos mAP, no uno

- **Macro** — promedio simple sobre las especies presentes. Pesa a `fox`
  (5 cajas de entrenamiento) igual que a `opossum` (2,514). Es el que el
  entregable pide, porque es el que **expone** el desbalance.
- **Ponderado por soporte** — se reporta al lado a propósito. La diferencia entre
  los dos *es* la medida del desbalance: si el ponderado es mucho más alto, el
  modelo acierta donde hay datos y falla donde no los hay.

Y un número más: la **correlación de Spearman** entre soporte y AP por especie.
Se usa Spearman y no Pearson porque el soporte va de 5 a 4,800 cajas, y con esa
escala Pearson queda dominada por las dos especies más comunes.

### 6. Las predicciones se guardan una sola vez

`predecir.py` vuelca un JSONL con una cabecera `_meta` y una línea por imagen:
nombre, tamaño, **ubicación de cámara**, **ráfaga**, **fecha de captura**,
predicciones y cajas anotadas. Los tres campos del medio no los tienen las
etiquetas YOLO y salen de las anotaciones originales:

- `ubicacion` separa `cis` de `trans` (paso 5);
- `secuencia` identifica la ráfaga, que el paso 6 usa para no contar tres veces
  al mismo animal;
- `fecha` es la marca de tiempo sin la cual el reporte de actividad no existe.

La escritura es atómica (`.jsonl.parcial` + rename): una corrida interrumpida no
deja un JSONL a medias que la lectura siguiente tome por completo.

---

## Resultados

### Resumen por partición

| | `cis_test` | `trans_test` |
|---|---:|---:|
| Imágenes / cajas | 12,696 / 13,264 | 18,033 / 18,786 |
| Detecciones emitidas | 50,184 | 80,020 |
| **mAP@0.5 macro** | **0.7463** | **0.3978** |
| IC95% (bootstrap, 200 rep.) | [0.7236, 0.7563] | [0.3869, 0.4447] |
| mAP@0.5:0.95 | 0.5025 | 0.2724 |
| mAP@0.5 ponderado por soporte | 0.8415 | 0.6263 |
| Spearman soporte vs AP@0.5 | +0.293 | +0.763 |
| Especies sin cajas (AP indefinida) | — | `deer` |

El ponderado está 0.095 por encima del macro en `cis_test` y 0.229 en
`trans_test`: **esa distancia es la medida del desbalance**, y se duplica con
creces en ubicaciones nuevas.

### Contraste contra Ultralytics — la reimplementación es correcta

Reimplementar el mAP obliga a demostrar que coincide con la referencia sobre los
mismos datos. `contrastar_con_ultralytics.py`:

| | propio | Ultralytics | diferencia |
|---|---:|---:|---:|
| `cis_test` | 0.7463 | 0.7633 | −0.0170 |
| `trans_test` | 0.3978 | 0.4300 | −0.0322 |

Mismo número de clases promediadas en ambos casos, y ninguna clase con cajas
anotadas quedó sin detecciones. La diferencia es sistemática y del signo
esperado: Ultralytics interpola la curva P-R en 101 puntos (método de COCO) y
acá se integra el área con todos los puntos (VOC2010+), que es menos optimista.

Lo que importa: la **brecha** entre particiones da 0.333 con Ultralytics y 0.349
con la propia. La conclusión central no depende de la implementación.

### Por especie — `cis_test`

| Especie | Cajas | Detecciones | AP@0.5 | AP@0.5:0.95 |
|---|---:|---:|---:|---:|
| opossum | 4,010 | 8,956 | 0.9156 | 0.4996 |
| rabbit | 1,559 | 5,091 | 0.8436 | 0.5266 |
| cat | 1,528 | 3,425 | 0.8659 | 0.5796 |
| coyote | 1,182 | 3,614 | 0.8792 | 0.6700 |
| raccoon | 981 | 6,071 | 0.9057 | 0.5938 |
| car | 791 | 1,041 | 0.9999 | 0.9978 |
| bobcat | 772 | 2,723 | 0.7247 | 0.5212 |
| bird | 752 | 7,377 | 0.5592 | 0.2770 |
| dog | 725 | 3,335 | 0.6628 | 0.4600 |
| squirrel | 510 | 4,289 | 0.6700 | 0.3433 |
| skunk | 162 | 356 | 0.9430 | 0.6222 |
| rodent | 146 | 3,158 | 0.8089 | 0.4362 |
| deer | 140 | 595 | 0.4163 | 0.2607 |
| **badger** | **4** | 145 | 0.0000 | 0.0000 |
| **fox** | **2** | 8 | **1.0000** | 0.7500 |

Las dos últimas filas son la razón de la columna *Cajas*: `fox` da AP perfecta
con **dos** positivos y `badger` da cero con cuatro. Ninguna de las dos cifras
describe una capacidad del modelo.

La columna *Detecciones* es la que convierte un AP bajo en diagnóstico: `bird`
con 7,377 detecciones para 752 cajas tiene un problema de **precisión**;
`deer` con 595 para 140, también. Un AP bajo con pocas detecciones sería lo
contrario, falta de recall, y se arregla distinto.

### Por especie — `trans_test`

| Especie | Cajas | Detecciones | AP@0.5 |
|---|---:|---:|---:|
| raccoon | 4,803 | 7,793 | 0.7062 |
| opossum | 4,663 | 14,337 | 0.8623 |
| bobcat | 1,979 | 3,858 | 0.4536 |
| coyote | 1,902 | 6,421 | 0.4255 |
| cat | 1,366 | 5,920 | 0.2359 |
| car | 1,113 | 1,488 | **0.9950** |
| dog | 961 | 5,116 | 0.6860 |
| squirrel | 782 | 7,756 | 0.1719 |
| rabbit | 722 | 4,968 | 0.3919 |
| skunk | 273 | 3,639 | 0.4930 |
| bird | 180 | **17,086** | 0.0132 |
| rodent | 26 | 939 | 0.0018 |
| badger | 15 | 9 | 0.1333 |
| fox | 1 | 44 | 0.0000 |
| **deer** | **0** | **646** | **indefinida** |

Dos filas que sostienen todo el análisis del [paso 5](../paso5_generalizacion/):

- **`car` es la única clase que no se degrada** (0.9999 → 0.9950). Objeto
  rígido, grande, de alto contraste, sin variación de postura. Es el control que
  demuestra que la caída de las demás no es un efecto general de cambio de
  dominio.
- **`deer` tiene cero cajas y 646 detecciones.** Su AP es indefinida —y ese es
  exactamente el caso que motivó distinguirla de cero. Pero vale decir el límite:
  ningún valor de AP comunica lo que de verdad pasa, que el modelo inventa
  ciervos.

---

## Pendiente

| | Qué falta | Por qué importa |
|---|---|---|
| 1 | Matriz de confusión entre especies | El mAP no dice **con qué** se confunde cada especie; `bobcat`/`cat` y `coyote`/`dog` son los pares esperables |
| 2 | Curvas P-R por especie en el informe | La tabla da un número; la curva muestra si el problema es precisión o recall |
| 3 | Bootstrap pareado entre `cis` y `trans` | Hoy se comparan dos intervalos; el pareado sobre las mismas especies tiene más potencia |
