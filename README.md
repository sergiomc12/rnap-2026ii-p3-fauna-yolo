# Pregunta 3 — Monitoreo de Fauna Silvestre con Cámaras Trampa (YOLO)

Examen Parcial de Redes Neuronales y Aprendizaje Profundo · UNI FIIS 2026-II

Entrega: **31 de octubre de 2026**. Penalización 10% por día.

---

## El problema en una frase

Detectar y clasificar la especie de los animales en fotografías de cámaras
trampa, sabiendo que **el 70% de las imágenes están vacías**, que algunas
especies tienen **cinco ejemplos de entrenamiento**, y que los modelos de este
dominio tienden a memorizar el fondo de cada cámara en lugar de aprender al
animal.

## Los seis entregables del examen

| # | Entregable | Paso que lo resuelve |
|---|---|---|
| 1 | Modelo YOLO ajustado, con decisiones de entrenamiento documentadas (resolución, aumento de datos, manejo de imágenes vacías) | [paso3](paso3_modelo/) |
| 2 | mAP@0.5 y mAP@0.5:0.95 con **desglose por especie**, señalando el efecto del desbalance | [paso4](paso4_evaluacion/) |
| 3 | Generalización **entre ubicaciones de cámara** usando `eccv_train`/`eccv_val`, con Grad-CAM sobre el backbone | [paso5](paso5_generalizacion/) |
| 4 | Robustez ante **oclusión y tamaño del objeto** | [paso5](paso5_generalizacion/) |
| 5 | **Módulo de monitoreo ecológico**: reporte de actividad por especie y periodo, demostrado sobre una cámara **no vista** en entrenamiento (`trans_test`) | [paso6](paso6_monitoreo/) |
| 6 | Media página sobre **uso responsable de IA en conservación** | [paso7](paso7_informe/) |

Más los requisitos transversales: informe de 6–10 páginas en formato NeurIPS con
mínimo 5 referencias, repositorio reproducible, y ejecución de extremo a extremo
con **un único comando**.

## Los siete pasos

```
paso1_datos/            anotaciones COCO Camera Traps + descarga de imagenes
         ↓
paso2_preprocesamiento/ COCO -> formato YOLO, los cuatro data.yaml
         ↓
paso3_modelo/           fine-tuning de YOLO                    entregable #1
         ↓
paso4_evaluacion/       mAP por especie                        entregable #2
         ↓
paso5_generalizacion/   cis vs trans, Grad-CAM, robustez       entregables #3 y #4
         ↓
paso6_monitoreo/        reporte de actividad ecologica          entregable #5
         ↓
paso7_informe/          informe NeurIPS + comando unico         entregable #6
```

| Paso | Qué produce | Estado |
|---|---|---|
| [1 — Datos](paso1_datos/) | 46,026 imágenes (18 GB), 0 corruptas | 🟢 completo |
| [2 — Preprocesamiento](paso2_preprocesamiento/) | Etiquetas YOLO + 4 `data.yaml`, fuga de ráfagas filtrada | 🟢 completo |
| [3 — Modelo](paso3_modelo/) | YOLO11n a 640 px, 60 épocas, mAP@0.5 = 0.9342 en `cis_val` — **pico de la época 54**; la última época da 0.8562 | 🟢 completo |
| [4 — Evaluación](paso4_evaluacion/) | mAP por especie con IC, contrastado contra Ultralytics | 🟢 completo |
| [5 — Generalización](paso5_generalizacion/) | Brecha cis/trans, Grad-CAM, robustez | 🟢 completo |
| [6 — Monitoreo](paso6_monitoreo/) | Reporte de actividad validado contra las anotaciones | 🟢 completo |
| [7 — Informe](paso7_informe/) | PDF NeurIPS de 10 páginas + comando único | 🟢 completo |

**Los seis entregables están cubiertos.** El trabajo se reproduce entero con un
único comando, tras crear el entorno e instalar las dependencias:

```bash
python -m venv .venv
.venv/Scripts/activate                       # Windows; source .venv/bin/activate en Unix
pip install -r pregunta3/requirements.txt
python pregunta3/paso7_informe/ejecutar_todo.py
```

Trece etapas, idempotentes: cada una se salta si su artefacto ya existe. El paso
6 corre el módulo de monitoreo dos veces, sobre cámaras vistas (`cis_test`) y
sobre cámaras nunca vistas (`trans_test`, la demostración que pide el entregable
#5). **289 pruebas** pasando en todo el repositorio, 113 propias de esta pregunta.

### Los resultados, en una tabla

| | `cis_test` (vistas) | `trans_test` (nuevas) |
|---|---:|---:|
| mAP@0.5 | 0.7463 [0.7236, 0.7563] | 0.3978 [0.3869, 0.4447] |
| mAP@0.5:0.95 | 0.5025 | 0.2724 |
| Ponderado por soporte | 0.8415 | 0.6263 |
| Spearman soporte vs AP | +0.293 | +0.763 |

**Brecha de 0.3485 con intervalos disjuntos**, de la cual el 91% sobrevive al
control del cambio de prevalencia de especies. El colapso es específico de los
animales: `car` conserva 0.9950 mientras `bird` cae de 0.5592 a 0.0132.

Tapar el 25% del área de cada animal cuesta el 69% del mAP. El índice de
nocturnidad del reporte ecológico reproduce la historia natural de las catorce
especies detectadas, y a la vez sobreestima el conteo de eventos en todas ellas.

Cada paso tiene su `README.md` con **qué se hizo, qué se midió, qué se descartó y
qué quedó pendiente** — misma convención que la [Pregunta 1](../pregunta1/).

---

## El conjunto de datos

**Caltech Camera Traps (CCT-20)**, vía LILA BC. 243,100 imágenes de 140
ubicaciones de cámara en el suroeste de Estados Unidos, con 21 categorías de
animales más la clase vacía.

De esas, **63,025 tienen bounding box**. Un detector solo puede entrenarse con
esas, de modo que el resto se ignora: **46,026 imágenes únicas**, 20.5 GB.

### La partición oficial resuelve el entregable #3

El conjunto trae la partición de ECCV 2018, que separa por **tipo de ubicación**:

| Partición | Imágenes | Con caja | Cajas | Ubicaciones | Qué es |
|---|---|---|---|---|---|
| `train` | 13,553 | 12,099 | 12,617 | 10 | entrenamiento |
| `cis_val` | 3,484 | 1,665 | 1,763 | 10 | **mismas** ubicaciones |
| `cis_test` | 15,827 | 12,696 | 13,264 | 10 | **mismas** ubicaciones |
| `trans_val` | 1,725 | 1,533 | 1,673 | **1** | ubicaciones **nuevas** |
| `trans_test` | 23,275 | 18,033 | 18,786 | 9 | ubicaciones **nuevas** |

*cis* = misma ubicación de cámara que el entrenamiento; *trans* = ubicación nunca
vista. La brecha entre `cis_test` y `trans_test` **es** el experimento de
generalización geográfica que pide el entregable #3, medido sobre la partición
que los propios autores del conjunto diseñaron para eso.

### El desbalance, medido

Cajas por especie en cada partición:

| Especie | train | cis_test | trans_test |
|---|---|---|---|
| opossum | 2,514 | 4,010 | 4,663 |
| rabbit | 2,278 | 1,559 | 722 |
| coyote | 1,371 | 1,182 | 1,902 |
| cat | 1,170 | 1,528 | 1,366 |
| squirrel | 1,037 | 510 | 782 |
| raccoon | 1,030 | 981 | 4,803 |
| dog | 769 | 725 | 961 |
| bobcat | 684 | 772 | 1,979 |
| car | 668 | 791 | 1,113 |
| bird | 560 | 752 | 180 |
| rodent | 264 | 146 | 26 |
| skunk | 214 | 162 | 273 |
| **deer** | **44** | 140 | **0** |
| **badger** | **9** | 4 | 15 |
| **fox** | **5** | 2 | 1 |

**Factor 500 entre la más común y la más rara.** `fox` tiene **cinco** cajas de
entrenamiento y `badger` nueve: con eso no hay nada que aprender, y decirlo con
evidencia es justamente lo que pide el entregable #2.

Dos observaciones que condicionan la evaluación:

- **`deer` tiene 44 cajas en train y cero en `trans_test`.** Su mAP en ese
  conjunto no existe, igual que un AUC sin positivos. Hay que reportarlo como
  indefinido, no como cero.
- **Las proporciones cambian entre particiones.** `raccoon` pasa de 1,030 cajas
  en train a 4,803 en `trans_test`, mientras `rabbit` cae de 2,278 a 722. La
  brecha de generalización no se deberá solo al cambio de fondo, sino también al
  cambio de distribución de especies. Separar ambos efectos es parte del análisis.

## Dónde están los datos

| Qué | Dónde | Tamaño |
|---|---|---|
| Anotaciones de bounding box | `pregunta3/paso1_datos/raw/caltech_bboxes_20200316.json` | 37 MB |
| Particiones ECCV | `pregunta3/paso1_datos/raw/eccv_18_annotation_files/` | 27 MB |
| Imágenes | `pregunta3/paso1_datos/imagenes/` | 20.5 GB |
| Etiquetas YOLO + yaml | `pregunta3/paso2_preprocesamiento/dataset/` | ~15 MB |

Todo está en `.gitignore` y se regenera con los comandos de cada paso.

## Qué se reutiliza de la Pregunta 1

- **Grad-CAM** (`pregunta1/paso5_ablacion_gradcam/gradcam.py`), que el
  entregable #3 pide aplicar sobre el backbone de detección
- El **bootstrap** y los intervalos de confianza
- La plantilla `neurips.sty` y la bibliografía
- La convención de pasos, READMEs de bitácora y comando único

## Material de clase disponible

En [`doc/yolo/`](../doc/yolo/):

| | |
|---|---|
| `Yolo_paper_original.pdf` | Redmon et al. 2016 |
| `YOLO_Notas_de_Clase.pdf` | Cuadrícula, IoU, NMS, pérdida multi-término, limitaciones, evolución v2–v8 |
| `Clase_6_DeepLearning_Yolo.pdf` | 17 diapositivas |
| `Lab07_YOLO_Roboflow.ipynb` | **53 celdas**: YOLO v1 desde cero (20 funciones) + pipeline de `ultralytics` |

El lab adelanta buena parte de los entregables #1 y #2: ya trae `evaluate_map`,
`nms`, la pérdida de YOLO v1, el fine-tuning con `ultralytics` y la evaluación
con desglose por clase.

## Integridad académica

El examen es explícito:

> El plagio de cualquier tipo — incluyendo el de cuadernos públicos — resultará
> en la reprobación automática del curso.

El lab del curso es material propio y sirve de base. Cualquier cuaderno público
de Kaggle o Roboflow se lee y se cita en trabajos relacionados, nunca se copia.

## Pesos preentrenados

El peso del YOLO entrenado esta publicado como *release*:

**https://github.com/sergiomc12/rnap-2026ii-p3-fauna-yolo/releases/tag/v1.0-pesos**

El asset `yolo11n_640_best.pt` va en `pregunta3/paso3_modelo/corridas/yolo11n_640/weights/best.pt`. No hace falta para leer el informe ni correr el notebook; solo para reusar el detector sin reentrenar. El dataset CCT-20 se descarga con los scripts del repo.
