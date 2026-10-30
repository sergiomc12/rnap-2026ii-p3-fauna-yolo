# Paso 5 — Generalización, Grad-CAM y robustez · entregables #3 y #4

> Entregable #3: *"generalización entre ubicaciones de cámara usando
> `eccv_train`/`eccv_val`, con Grad-CAM sobre el backbone"*.
> Entregable #4: *"robustez ante oclusión y tamaño del objeto"*.

## Clases y módulos de este paso

| Archivo | Clase / función | Qué hace |
|---|---|---|
| [`generalizacion.py`](generalizacion.py) | `BrechaClase` | AP de una especie en ubicaciones vistas contra nuevas, con su cambio de prevalencia |
| | `brecha_por_clase()` | Empareja las dos evaluaciones especie por especie |
| | `mapa_por_ubicacion()` | mAP@0.5 de **cada cámara** por separado |
| | `dispersion()` | Mínimo, máximo, media y sigma de esos mAP |
| | `remuestrear_a_distribucion()` | Recorta una partición para imitar la mezcla de especies de otra |
| | `distribucion_de()` | `{clase: cajas}` de una evaluación |
| [`robustez.py`](robustez.py) | `TRAMOS_TAMANO` | Los cuatro tramos por fracción del cuadro |
| | `muestras_en_tramo()` | Imágenes cuyas cajas caen **todas** en un rango |
| | `barrido_por_tamano()` → `ResultadoTramo` | mAP por tramo, marcando los de muestra chica |
| | `ocluir_cajas()` | Tapa una fracción del **área** de cada animal con un parche gris |
| | `area_tapada()` | Cuánto debería tapar, para verificar el parche |
| [`gradcam_yolo.py`](gradcam_yolo.py) | `GradCAMDeteccion` | Grad-CAM sobre una capa del backbone, como context manager |
| | `letterbox()`, `a_tensor()` | El preprocesamiento de Ultralytics, reproducido con gradientes |
| | `mapa_en_imagen()` | Lleva el mapa del espacio del letterbox al de la foto |
| | `masa_dentro_de_cajas()` | Qué fracción de la activación cae sobre el animal |
| | `superponer()` | Pinta el mapa como mapa de calor |
| [`comparar_ubicaciones.py`](comparar_ubicaciones.py) | `main()` | Script del entregable #3 (parte cuantitativa) |
| [`generar_gradcam.py`](generar_gradcam.py) | `razon_de_masa()`, `barrido_de_capas()`, `guardar_figura()` | Script del entregable #3 (parte interpretativa) |
| [`evaluar_robustez.py`](evaluar_robustez.py) | `barrido_oclusion()` | Script del entregable #4 |

## Comandos

Los tres necesitan las predicciones del [paso 4](../paso4_evaluacion/).

```bash
# entregable #3, parte cuantitativa: cis vs trans, por camara, con control de distribucion
.venv/Scripts/python.exe pregunta3/paso5_generalizacion/comparar_ubicaciones.py

# entregable #3, parte interpretativa: Grad-CAM + barrido de capas
.venv/Scripts/python.exe pregunta3/paso5_generalizacion/generar_gradcam.py --barrido

# entregable #4: tamano (barato) + oclusion (vuelve a correr el modelo)
.venv/Scripts/python.exe pregunta3/paso5_generalizacion/evaluar_robustez.py
.venv/Scripts/python.exe pregunta3/paso5_generalizacion/evaluar_robustez.py --solo-tamano
```

Produce:

```
informes/generalizacion.json        cis vs trans, por clase, por camara, controlado
informes/brechas_por_especie.md     la tabla del informe
informes/robustez.json              mAP por tramo de tamano y por nivel de oclusion
informes/robustez_tamano.md         la tabla de tamanos
informes/gradcam.json               razon de masa por particion + barrido de capas
figuras/<particion>_<caso>.jpg      original con cajas | mapa de calor
```

---

## Lo que se hizo

### 1. La comparación cis/trans sola estaría mal hecha

`cis_test` son 10 ubicaciones vistas en el entrenamiento; `trans_test`, 9 nunca
vistas. La diferencia de mAP entre las dos es la respuesta obvia al entregable
#3 — y **confunde dos efectos**:

1. el cambio de fondo, que es lo que se quiere medir;
2. el cambio de distribución de especies: `raccoon` pasa de 981 cajas en
   `cis_test` a 4,803 en `trans_test`, y `rabbit` cae de 1,559 a 722.

El mAP macro no se deja arrastrar por la prevalencia al **promediar**, pero la AP
de cada clase sí depende de cuántos positivos tiene: con pocos positivos, un
puñado de falsos positivos hunde la precisión.

`remuestrear_a_distribucion()` recorta `trans_test` para que su mezcla de
especies imite la de `cis_test`, y la brecha que queda después de ese control es
la atribuible al fondo. El muestreo es **por imagen**, no por caja: sacar cajas
de a una rompería la escena y el modelo detectaría un animal que ya no está
anotado, que contaría como falso positivo. Y se recorre **la clase más escasa
primero**: empezando por la más común, al llegar a las raras ya no quedarían
imágenes y la mezcla volvería a parecerse a la original.

### 2. El mAP por cámara, no solo por partición

Una sola cifra para nueve ubicaciones nuevas esconde que el desempeño varía entre
cámaras, y esa dispersión es lo que alguien necesita saber antes de instalar una.
`mapa_por_ubicacion()` reporta cada cámara por separado y marca las que tienen
menos de 50 cajas: un mAP sobre veinte cajas es ruido, y mezclarlo con el de una
cámara con tres mil daría una dispersión inventada.

### 3. Grad-CAM: elegir qué derivar, y elegir la capa midiendo

Grad-CAM se definió para clasificadores, donde hay un escalar por clase. Un
detector tiene miles de anclas, cada una con su caja y sus quince puntajes. Hay
que elegir qué derivar, y la elección cambia lo que el mapa significa:

- derivar el puntaje de **una ancla** muestra qué miró el modelo para esa
  detección puntual;
- derivar la **suma de los puntajes de una clase** sobre las `k` anclas más
  confiadas muestra dónde está la evidencia de que *hay* un animal de esa
  especie.

Se usa la segunda con `k=10`, porque es la que responde la pregunta del
entregable: si el modelo mira al animal o mira el fondo de la cámara.

**La capa se eligió con una medición, y el resultado contradice la convención.**
Lo habitual es tomar la última capa convolucional, que acá sería el C2PSA del
índice 10. Con 640 px de entrada esa capa tiene una rejilla de 20×20 — 32 px por
celda — y un animal que ocupa el 2% del cuadro cae dentro de una sola celda.
Midiendo la **razón de masa** (fracción de la activación dentro de las cajas
anotadas, dividida por el área que esas cajas ocupan; 1.0 = igual que un mapa
uniforme, o sea ninguna localización) sobre 30 imágenes de `cis_test`:

| Capa | Bloque | Rejilla | Razón media | Mediana | Mapas vacíos |
|---|---|---|---:|---:|---:|
| **4** | C3k2 | 80×80 | 1.59 | **1.09** | **0/30** |
| 6 | C3k2 | 40×40 | 1.38 | 1.01 | 0/30 |
| 8 | C3k2 | 20×20 | 1.32 | 0.66 | 9/30 |
| 9 | SPPF | 20×20 | 1.56 | 0.75 | 3/30 |
| 10 | C2PSA | 20×20 | 0.95 | 0.79 | 5/30 |

La media **no** separa a la capa 4 (1.59) de la capa 9 (1.56), y decir que sí lo
hace sería forzar el dato. Lo que las separa son las otras dos columnas: la
mediana (1.09 contra 0.75) y los mapas idénticamente nulos dentro del animal
(0 de 30 contra 3 de 30). La media de la capa 9 está sostenida por unos pocos
casos buenos; su mediana muestra que en la mitad de las imágenes no localiza
mejor que un mapa uniforme.

Las tres capas de 20×20 producen mapas **idénticamente nulos** dentro del animal
en 3 a 9 de cada 30 imágenes. No es que miren el fondo: no tienen resolución para
decir nada. El default quedó en la **capa 4**, reproducible con `--barrido`.

### 4. La oclusión es sintética, y se dice por qué

El conjunto no trae etiquetas de oclusión. La alternativa a la oclusión
sintética sería no responder el entregable #4.

`ocluir_cajas()` tapa con un parche gris (valor 114, el mismo relleno del
letterbox de Ultralytics) una fracción del **área** de cada caja anotada — el
lado del parche escala con la raíz de la fracción, así que "tapar la mitad" tapa
la mitad del área y no la mitad del lado. Está verificado contra `area_tapada()`
en los tests, porque confundir área con lado reportaría un 25% de oclusión donde
hay un 50%.

Dos modos, que no miden lo mismo:

- **centro** — deja la silueta completa y quita la textura interior: separa
  "reconocer la forma" de "reconocer el pelaje";
- **inferior** — tapa desde el borde de abajo, como hace la vegetación real.
  Rompe también la silueta.

Al diseñarlo se anotó acá que `inferior` sería "el caso más duro" porque rompe
la silueta. **Medido, es al revés** (ver *Resultados*): tapar el 25% del centro
cuesta más que tapar el 25% desde abajo. El modelo depende más de la textura
interior que del contorno, que es lo contrario de lo que la intuición sugería.

El barrido corre sobre una submuestra fija por semilla (1,500 imágenes), no sobre
las 12,696 de `cis_test`: cada nivel y cada modo es una pasada completa del
modelo. Fija por semilla importa — todos los niveles se comparan sobre
exactamente las mismas imágenes, así que la caída es atribuible a la oclusión y
no al cambio de muestra. El nivel 0% se mide una sola vez y se reusa como línea
base de los dos modos.

Y un dato que viene del [paso 3](../paso3_modelo/) y que hace válido el
experimento: el modelo **nunca vio oclusión durante el entrenamiento**. Las
banderas `erasing`/`auto_augment` que aparecen en su `args.yaml` solo se aplican
en clasificación, no en detección.

### 5. Los tramos de tamaño no mezclan

Una imagen entra en un tramo solo si **todas** sus cajas anotadas caen en ese
rango. Filtrando caja por caja, las descartadas seguirían en la imagen y el
modelo las detectaría: cada detección correcta de un animal fuera del rango
contaría como falso positivo y el mAP del tramo saldría artificialmente bajo.

Los cortes son por fracción del cuadro, no en píxeles, porque en este conjunto
conviven varias resoluciones: diminuto (<0.2%), chico (0.2–1%), mediano (1–5%),
grande (>5%).

---

## Resultados

### Generalización: la brecha es real y no es por el cambio de especies

| | `cis_test` | `trans_test` |
|---|---:|---:|
| Imágenes | 12,696 | 18,033 |
| **mAP@0.5** | **0.7463** | **0.3978** |
| IC95% | [0.7236, 0.7563] | [0.3869, 0.4447] |
| mAP@0.5:0.95 | 0.5025 | 0.2724 |
| mAP@0.5 ponderado por soporte | 0.8415 | 0.6263 |
| Spearman soporte vs AP | +0.293 | **+0.763** |

**Brecha: 0.3485 en mAP@0.5, con intervalos disjuntos.** Controlando la mezcla de
especies (recortando `trans_test` a 11,000 imágenes con la distribución de
`cis_test`), el mAP sube solo a 0.4308 y la brecha queda en 0.3155: el control
explica 0.033 de los 0.3485, así que el **91% de la brecha es atribuible al
cambio de ubicación**, no a la prevalencia.

**El desbalance pesa mucho más en terreno desconocido.** Spearman sube de +0.293
a +0.763, y la distancia entre macro y ponderado pasa de 0.095 a 0.229. En las
cámaras conocidas el modelo compensa la falta de datos con el fondo memorizado;
sin ese apoyo, lo único que queda es cuántos ejemplos vio de cada especie.

**El colapso es específico de los animales.** `car` mantiene AP@0.5 = 0.9950 en
ubicaciones nuevas (0.9999 en las conocidas), mientras `bird` cae de 0.5592 a
0.0132, `squirrel` de 0.6700 a 0.1719 y `cat` de 0.8659 a 0.2359. La imagen, la
cámara y la iluminación cambian igual para el auto: lo que cambia es que la
evidencia que sostenía a los animales estaba en el fondo.

Dos casos que la cifra agregada esconde: `bird` emite **17,086 detecciones para
180 cajas reales** en `trans_test`, y `deer`, que **no tiene ninguna caja** en
esa partición, recibe **646 detecciones**. Su AP es indefinida, no cero — y lo
que realmente pasa, que inventa ciervos, no es algo que la AP pueda expresar.

**Dispersión entre cámaras**, que el promedio por partición esconde:

| | Cámaras | Media | Mín. | Máx. | σ |
|---|---:|---:|---:|---:|---:|
| `cis` | 10 | 0.6797 | 0.5036 | 0.7901 | 0.092 |
| `trans` | 9 | 0.4092 | 0.2218 | 0.6717 | **0.158** |

La dispersión casi se duplica: en terreno desconocido no solo baja el promedio,
se vuelve mucho menos predecible qué dará una cámara nueva.

### Grad-CAM: la evidencia no está sobre el animal

Sobre 120 imágenes por partición, con la capa 4:

| | `cis_test` | `trans_test` |
|---|---:|---:|
| Razón de masa media | 1.46 | 1.25 |
| Mediana | 1.12 | **1.00** |
| Imágenes con razón ≤ 1.0 | 52/120 | 62/120 |
| Animales chicos (<1% del cuadro) | 1.89 | **0.89** |
| Animales grandes | 1.43 | 1.27 |

La mediana de `trans_test` es exactamente 1.00: en la mitad de las imágenes de
ubicaciones nuevas la evidencia del modelo no está más concentrada sobre el
animal que si estuviera repartida al azar. Y para animales chicos en ubicaciones
nuevas la razón baja a **0.89**, es decir *menos* concentrada que un mapa
uniforme. La degradación se concentra justo donde el modelo ya era más débil.

### Tamaño del objeto: la relación no es monótona

| Tramo | Fracción del cuadro | `cis_test` | `trans_test` |
|---|---|---:|---:|
| diminuto ⚠ | <0.2% | 0.1602 (18 cajas) | 0.0913 (16 cajas) |
| chico | 0.2–1% | 0.5131 | 0.2310 |
| mediano | 1–5% | **0.8033** | **0.4555** |
| grande | >5% | 0.7341 | 0.2852 |

El tramo **grande rinde peor que el mediano** en las dos particiones, y no era lo
esperado. La explicación plausible es que un animal que ocupa más del 5% del
cuadro está muy cerca del lente: entra parcialmente fuera de cuadro, suele tener
desenfoque de movimiento y, de noche, el infrarrojo lo sobreexpone. No se
verificó cuál de esos factores domina, así que queda como observación y no como
conclusión.

Lo que sí es claro y monótono es el extremo chico: de 0.8033 a 0.1602 al pasar
de mediano a diminuto en `cis_test`. A 640 px de entrada, un animal que ocupa
menos del 0.2% del cuadro queda en unos pocos píxeles.

### Oclusión: el modelo no la resiste en absoluto

mAP@0.5 sobre 1,500 imágenes fijas por semilla, de `cis_test`:

| Área tapada de cada caja | 0% | 25% | 50% | 75% |
|---|---:|---:|---:|---:|
| `centro` | 0.7652 | **0.2375** | 0.0584 | 0.0044 |
| `inferior` | 0.7652 | 0.3699 | 0.0647 | 0.0234 |

**Tapar apenas el 25% del área cuesta el 69% del mAP.** Al 75%, el modelo está
en 0.004: prácticamente ciego. Es coherente con lo verificado en el
[paso 3](../paso3_modelo/): el modelo **nunca vio oclusión durante el
entrenamiento** —las banderas `erasing`/`auto_augment` de su `args.yaml` solo se
aplican en clasificación.

**Y la predicción de diseño salió al revés.** Se había anotado que `inferior`
sería el caso más duro por romper la silueta; al 25% resulta claramente más
benigno (0.3699 contra 0.2375). Tapar el centro —quitar la textura interior
dejando el contorno— duele más que romper el contorno dejando la textura. El
modelo se apoya más en el pelaje que en la forma, al menos a este nivel de
oclusión; al 50% la diferencia se desvanece (0.058 contra 0.065), porque ahí ya
no queda ni una cosa ni la otra.

---

## Hallazgos

### El mapa de Grad-CAM vive en el espacio del letterbox, no en el de la foto

Primer intento: la masa de activación dentro de las cajas anotadas daba 0.0001
donde el animal ocupaba el 9% del cuadro. No era un hallazgo sobre el modelo, era
un error de geometría: una imagen de 2048×1494 entra a la red como 640×467 con
86 píxeles de gris arriba y abajo, y estirar el mapa de 20×20 directamente al
tamaño original corre todo verticalmente. Con `mapa_en_imagen()` —que recorta el
relleno antes de reescalar— la misma imagen dio 0.337.

Vale anotarlo porque el error era **invisible en la figura**: el mapa de calor se
veía plausible, con una mancha sobre vegetación, y solo el número lo delató.

### `predict` de Ultralytics y los hooks de Grad-CAM no conviven

`predict` clona el modelo internamente. Con los hooks puestos, el clon arrastra
los tensores intermedios que el hook guardó, que no son hojas del grafo y no se
pueden copiar: `RuntimeError` de `deepcopy`. La solución fue de orden —
clasificar los casos **antes** de registrar los hooks — y `cerrar()` ahora suelta
los tensores además de quitar los hooks.

---

## Pendiente

| | Qué falta | Por qué importa |
|---|---|---|
| 1 | Grad-CAM sobre un **falso positivo** de `trans_test` en el informe | Es el caso que mejor muestra el sesgo de fondo: el modelo detecta un animal donde solo hay un arbusto conocido |
| 2 | Oclusión con parches de **textura natural** en vez de gris | Un rectángulo gris uniforme es más fácil de ignorar que una rama real |
| 3 | Reentrenar a 960 px y repetir el barrido de tamaño | Para cerrar el entregable #4 con una mitigación medida, no solo con el diagnóstico |
