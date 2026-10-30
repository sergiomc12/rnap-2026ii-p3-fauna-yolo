# Paso 3 — Ajuste fino de YOLO · entregable #1

> Entregable #1: *"modelo YOLO ajustado, con decisiones de entrenamiento
> completamente documentadas (resolución de entrada, estrategia de aumento de
> datos, manejo explícito de las imágenes vacías como clase de fondo)"*.

Las tres decisiones que el enunciado nombra están abajo con su razón, y las tres
son argumentos del script, no constantes escondidas: la ablación del paso 5
necesita poder cambiar una sin tocar las otras.

## Clases y módulos de este paso

| Archivo | Clase / función | Qué hace |
|---|---|---|
| [`entrenar.py`](entrenar.py) | `ConfigEntrenamiento` | Dataclass con **todas** las decisiones de entrenamiento y su justificación en el comentario |
| | `ConfigEntrenamiento.a_kwargs()` | Traduce la configuración a los argumentos de Ultralytics |
| | `resumen_metricas()` | Extrae mAP@0.5, mAP@0.5:0.95 y el desglose por clase del objeto de métricas |
| | `main()` | Script: entrena, valida con los mejores pesos y escribe `resumen.json` |

## Comandos

```bash
# entrenamiento completo (60 epocas, ~50 min en RTX PRO 5000)
.venv/Scripts/python.exe pregunta3/paso3_modelo/entrenar.py --epocas 60 --etiqueta yolo11n_640

# continuar una corrida cortada, desde last.pt
.venv/Scripts/python.exe pregunta3/paso3_modelo/entrenar.py --epocas 60 --etiqueta yolo11n_640 --reanudar

# variantes para la ablacion del paso 5
.venv/Scripts/python.exe pregunta3/paso3_modelo/entrenar.py --resolucion 960 --etiqueta alta_res
.venv/Scripts/python.exe pregunta3/paso3_modelo/entrenar.py --sin-mosaic --etiqueta sin_mosaic
.venv/Scripts/python.exe pregunta3/paso3_modelo/entrenar.py --modelo yolo11s.pt --etiqueta yolo11s_640
```

Produce en `corridas/<etiqueta>/`:

```
corridas/yolo11n_640/
├── weights/best.pt      mejores pesos por fitness (0.1*mAP50 + 0.9*mAP50-95)
├── weights/last.pt      ultima epoca, lo que --reanudar necesita
├── results.csv          una fila por epoca: perdidas y metricas
├── args.yaml            la configuracion EFECTIVA, incluidos los defaults
├── labels.jpg           distribucion de clases y tamanos del conjunto
├── train_batch*.jpg     lotes reales con el aumento aplicado
└── resumen.json         metricas finales + la config de este paso
```

### Usar las clases desde Python

```python
from pregunta3.paso3_modelo.entrenar import ConfigEntrenamiento

config = ConfigEntrenamiento(epocas=60, resolucion=640)
print(config.a_kwargs("dataset/data_base.yaml", "corridas", "mi_corrida"))
# {'data': ..., 'epochs': 60, 'imgsz': 640, 'mosaic': 1.0, 'close_mosaic': 10, ...}
```

---

## Lo que se hizo

### 1. Resolución de entrada: 640 px

Las imágenes originales son **2048×1494**. A 640 px —el estándar de YOLO— se
reducen 3.2 veces, y un animal que ya ocupaba pocos píxeles queda diminuto. En
este conjunto eso no es un caso raro: es el caso dominante, porque las cámaras
disparan a animales que pasan a varios metros.

Se eligió 640 de todas formas, por dos razones: es la resolución con la que los
pesos preentrenados de COCO fueron ajustados, y permite un lote de 16 en una
sola GPU. Pero **no se dio por buena**: el entregable #4 mide exactamente ese
costo (ver [paso 5](../paso5_generalizacion/)) y `--resolucion` queda expuesto
para poder entrenar a 960 y comparar.

### 2. Aumento de datos: lo que se activó y lo que se dejó en cero

El criterio fue uno solo: **un aumento vale si produce imágenes que la cámara
podría haber tomado de verdad.** La cámara está atada a un poste y no gira.

| Aumento | Valor | Por qué |
|---|---|---|
| Volteo horizontal | 0.5 | Un animal puede cruzar en cualquier dirección; la especie no cambia al reflejarla |
| Volteo **vertical** | **0.0** | Pondría a los animales patas arriba — una vista que no existe en el despliegue |
| Mosaic | 1.0 | Pega cuatro imágenes en una: ayuda con objetos chicos y **rompe el fondo**, que es justo lo que el modelo tiende a memorizar |
| `close_mosaic` | 10 | Mosaic se **apaga** en las últimas diez épocas: las imágenes compuestas no son fotos reales, y terminar con ellas ajusta el modelo a una distribución que no es la del despliegue |
| HSV (h/s/v) | 0.015 / 0.5 / 0.4 | Las cámaras disparan de día, de noche con infrarrojo y con luz cambiante |
| Escala | ±50% | El aumento que más importa acá: simula al animal más cerca o más lejos, que es la variable del entregable #4 |
| Traslación | ±10% | El animal puede entrar al cuadro por cualquier lado |
| Rotación | **0.0** | La cámara está fija: el horizonte no gira |
| Perspectiva | **0.0** | Misma razón — inventarla agrega ruido sin realismo |

**Lo que NO se contó como aumento, aunque aparezca en el `args.yaml`.**
La corrida registra `erasing: 0.4` y `auto_augment: randaugment`, y los dos son
engañosos: Ultralytics los aplica **solo en clasificación**. En detección no
hacen nada. Darlos por buenos sería atribuirle al modelo un aumento por oclusión
que nunca recibió, y eso contaminaría justamente la conclusión del entregable #4
—donde se mide la robustez ante oclusión de un modelo que, se verificó, nunca
la vio durante el entrenamiento.

### 3. Imágenes vacías: no son una clase

El 70% del conjunto no tiene ningún animal, y la tentación es tratar `empty`
como una decimosexta clase. **Es un error de concepto, no una opción de diseño:**
en detección la ausencia de cajas en una imagen ya expresa "acá no hay nada".
Crear una clase de fondo le enseñaría al modelo a dibujar un bounding box
alrededor de la nada, y a esa caja habría que asignarle coordenadas que no
existen.

La decisión concreta, tomada en el [paso 2](../paso2_preprocesamiento/):

- `empty` se excluye del `MapaClases` → quedan **15 clases**;
- una imagen cuya única anotación es `empty` no genera archivo de etiqueta;
- el entrenamiento usa las **12,099 imágenes con caja** de `train`, no las
  13,553 de la partición completa.

Las imágenes vacías se podrían sumar como **negativos explícitos** (Ultralytics
las admite: una imagen sin `.txt` cuenta como fondo). No se hizo en esta corrida,
y la razón es honesta: habría cambiado dos cosas a la vez (el tamaño del conjunto
y la proporción fondo/objeto) y no se podría atribuir el efecto. Queda anotado en
*Pendiente* como ablación.

### 4. La corrida sobrevive a los cortes

Sesenta épocas no caben en una sesión de trabajo cualquiera. Ultralytics guarda
`last.pt` **con el estado del optimizador** en cada época, así que un corte
pierde como máximo la época en curso.

`--reanudar` carga `corridas/<etiqueta>/weights/last.pt` y llama a
`train(resume=True)`. Esta corrida se cortó una vez a la época 30 y se retomó
sin pérdida: `results.csv` continúa la numeración, y la columna de tiempo se
reinicia (es tiempo de proceso, no acumulado) — es la única señal de que hubo
dos procesos.

### 5. Hiperparámetros, tal como quedaron

No se ajustaron: son los defaults de Ultralytics para detección, y cambiarlos
sin un conjunto de validación dedicado sería ajustar a `cis_val`.

| | |
|---|---|
| Optimizador | MuSGD, `lr0=0.01`, `lrf=0.01`, momento 0.937 |
| Decaimiento de pesos | 0.0005 |
| Calentamiento | 3 épocas |
| Pesos de la pérdida | caja 7.5, clase 0.5, DFL 1.5 |
| Lote | 16 |
| Paciencia | 15 épocas sin mejorar |
| Semilla | 42, con `deterministic: true` |
| Precisión mixta | activada (`amp`) |

**Sobre el determinismo.** `deterministic: true` le pide a Ultralytics fijar las
semillas y los algoritmos de cuDNN. En la Pregunta 1 se verificó que ese tipo de
bandera **no alcanza** para reproducir un entrenamiento bit a bit en GPU (ver el
pendiente crítico de [pregunta1](../../pregunta1/README.md)), así que acá se
reporta como lo que es: reduce la varianza, no la elimina. Las conclusiones del
paso 5 se sostienen con intervalos de confianza, no con una sola corrida.

---

## Resultados

Corrida `yolo11n_640`: **60 épocas completas**, ~61 minutos de GPU repartidos en
tres procesos (dos cortes, dos reanudaciones). Validación final con `best.pt`
sobre `cis_val` — 1,590 imágenes, 1,686 cajas:

| | |
|---|---|
| **mAP@0.5** | **0.9342** |
| **mAP@0.5:0.95** | **0.6118** |
| Precisión media | 0.9278 |
| Recall medio | 0.8174 |

**Ese 0.9342 es el de la mejor época, no el de la última, y la diferencia no es
chica.** `best.pt` son los pesos de la **época 54**, donde el mAP@0.5 durante el
entrenamiento marcó 0.9346 (el 0.9342 es la revalidación posterior, que coincide).
La **época 60**, la última, da **0.8562**. O sea que el 0.9342 es un **pico**: las
épocas vecinas están en 0.8954 (53) y 0.8958 (55), y el promedio de las últimas
diez ronda 0.88.

Reportar el mejor de 60 valores ruidosos sobre el mismo conjunto de validación es
elegir el máximo de una muestra, y eso sesga hacia arriba — es el mismo motivo por
el que la Pregunta 1 reporta validación cruzada y no una partición suelta. **La
cifra que se compara entre experimentos en el paso 4 es la de `cis_test` y
`trans_test`, no esta.**

### Por especie

| Especie | AP@0.5 | AP@0.5:0.95 | P | R |
|---|---:|---:|---:|---:|
| car | 0.9950 | 0.9950 | 0.988 | 1.000 |
| badger ⚠ | 0.9950 | 0.0995 | 1.000 | **0.000** |
| opossum | 0.9800 | 0.5762 | 0.979 | 0.946 |
| raccoon | 0.9769 | 0.7084 | 0.884 | 0.965 |
| rabbit | 0.9648 | 0.6527 | 0.984 | 0.926 |
| cat | 0.9593 | 0.6695 | 0.957 | 0.927 |
| coyote | 0.9437 | 0.7149 | 0.931 | 0.890 |
| dog | 0.9423 | 0.6804 | 0.892 | 0.914 |
| skunk | 0.9214 | 0.6901 | 0.885 | 0.900 |
| rodent | 0.9211 | 0.5352 | 0.927 | 0.833 |
| squirrel | 0.8924 | 0.5299 | 0.904 | 0.836 |
| bobcat | 0.8821 | 0.6483 | 0.945 | 0.833 |
| bird | 0.7708 | 0.4540 | 0.785 | 0.656 |

**`deer` y `fox` no aparecen**: no tienen ninguna caja en `cis_val`, así que su
AP no existe. Son 13 de 15 clases evaluadas, y el mAP de arriba promedia esas 13
— tratarlas como cero habría bajado el mAP con dos especies que era imposible
detectar porque no estaban. El paso 4 lo reporta explícitamente como indefinido.

### La fila de `badger` es un artefacto, no un resultado

`badger` tiene **AP@0.5 = 0.995 con recall 0.000**. No es una contradicción: en
`cis_val` tiene **una sola caja, en una sola imagen**, y con un único positivo la
curva precisión-recall se reduce a un par de puntos donde el área queda cerca
de 1. El AP@0.5:0.95 de 0.0995 lo delata — es exactamente 1/10 del AP@0.5, es
decir que solo el umbral 0.50 aporta algo y los otros nueve dan cero. El modelo
no detecta tejones; la métrica no tiene con qué decirlo.

**Corrección.** Acá decía *"en `cis_val` tiene cuatro cajas"*. Son **cuatro en
`cis_test`** y **una** en `cis_val`; el reparto real de las 29 cajas de `badger`
del dataset es: 9 en `train`, 1 en `cis_val`, 4 en `cis_test`, 0 en `trans_val`
y 15 en `trans_test`. Con un positivo en vez de cuatro, el argumento de arriba
no se debilita: se vuelve más extremo.

Con nueve cajas de entrenamiento no podía ser de otro modo, y es la razón por la
que el paso 4 reporta el **soporte al lado de cada AP**: una AP sin su soporte es
un número sin unidades.

---

## Hallazgos

### Mosaic apagado se nota en la pérdida **de clasificación**, y el mAP sí salta

En la época 51, cuando `close_mosaic` desactiva el mosaic, la pérdida de
**clasificación** cae de 0.571 a 0.492 de una época a la otra — un salto mucho
mayor que el de cualquier época anterior. Las otras dos pérdidas van en la
dirección contraria: box sube 1.044 → 1.073 y **DFL sube** 1.118 → 1.137.

| época | box | cls | dfl | mAP@0.5 |
|---:|---:|---:|---:|---:|
| 50 (último con mosaic) | 1.044 | 0.571 | 1.118 | 0.8492 |
| 51 (sin mosaic) | 1.073 | **0.492** | 1.137 | 0.8510 |
| 52 | 1.062 | 0.474 | 1.127 | 0.8809 |
| 53 | 1.056 | 0.463 | 1.119 | 0.8954 |
| **54** | 1.046 | 0.457 | 1.118 | **0.9346** |

**Y el mAP sí salta: +0.085 entre la época 50 y la 54** (0.8492 → 0.9346), que es
el salto más grande de toda la corrida después de las primeras épocas.

La lectura, entonces, es la inversa de la que estaba escrita: lo que mosaic le
costaba al modelo era **clasificación**, no regresión de caja. Tiene sentido con
lo que mosaic hace: pega cuatro fotos en una, así que el contexto de la escena
—que en cámaras trampa es justamente lo que distingue un mapache de un gato— deja
de ser coherente. Las pérdidas de caja empeoran un poco porque desaparece una
fuente de variedad geométrica que las ayudaba.

**Corrección.** Una versión anterior de este README atribuía la caída
0.571 → 0.492 a la **pérdida DFL** y concluía que el costo era *"de regresión de
caja, no de clasificación"*. En `corridas/yolo11n_640/results.csv` esa caída es
de `train/cls_loss`; `train/dfl_loss` **sube** en esa misma época. Decía además
que *"el mAP no salta igual"*, cuando salta +0.085. Las dos afirmaciones estaban
invertidas respecto de los datos de la propia corrida.

### El `args.yaml` es la única fuente confiable de la configuración

Los valores que no se pasan explícitamente **no son los del código**: son los
defaults de Ultralytics, y algunos están activos sin que uno lo pida (escala
±50%, traslación ±10%). Documentar el aumento leyendo solo el script habría
dejado afuera el aumento que más pesa en este problema. Por eso
`ConfigEntrenamiento` ahora declara esos dos de forma explícita, con los mismos
valores, para que la configuración documentada y la efectiva coincidan.

---

## Pendiente

| | Qué falta | Por qué importa |
|---|---|---|
| 1 | Ablación con **negativos explícitos** (las 70% de imágenes vacías como fondo) | Es la decisión del entregable #1 que quedó argumentada pero no medida |
| 2 | Ablación de **resolución** (640 contra 960) | El entregable #4 pide el efecto del tamaño del objeto; subir la resolución es la mitigación obvia y hay que cuantificarla |
| 3 | Ablación `--sin-mosaic` | Mosaic es la decisión más discutible: rompe el fondo, que acá juega a favor, pero también destruye el contexto |
| 4 | Un modelo más grande (`yolo11s`) | Para separar "el problema es el modelo" de "el problema son los datos" |
