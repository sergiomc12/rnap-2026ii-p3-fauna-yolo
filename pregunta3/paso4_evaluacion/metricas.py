"""Métricas de detección calculadas acá, no delegadas a Ultralytics.

El entregable #2 pide mAP@0.5 y mAP@0.5:0.95 "con desglose por especie,
señalando el efecto del desbalance". Ultralytics ya reporta esos números, pero
se reimplementan por tres razones concretas:

* **Clases sin positivos.** Si una especie no aparece en el conjunto de
  evaluación, su AP no es cero: es *indefinido*. ``deer`` tiene 44 cajas en
  entrenamiento y **cero** en ``trans_test``; promediarla como 0.0 bajaría el
  mAP con una especie que nunca se podía detectar. Acá se devuelve ``None``.

* **Subconjuntos arbitrarios.** El entregable #4 pide particionar la evaluación
  por tamaño del objeto y por oclusión. Teniendo las predicciones guardadas, un
  subconjunto se evalúa recalculando sobre una lista filtrada, sin volver a
  correr el modelo.

* **Intervalos de confianza.** Un mAP de 0.87 contra otro de 0.85 no dice nada
  sin incertidumbre. El bootstrap sobre imágenes (no sobre cajas: las cajas de
  una misma imagen no son independientes) da ese intervalo.

La AP usa interpolación de **todos los puntos** del área bajo la curva
precisión-recall, con la envolvente monótona — la definición de VOC2010 en
adelante y la que usa COCO, no la de 11 puntos de VOC2007.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional, Sequence

# Los diez umbrales de IoU que COCO promedia para su métrica principal.
UMBRALES_COCO = tuple(round(0.50 + 0.05 * i, 2) for i in range(10))

Caja = tuple[float, float, float, float]   # x1, y1, x2, y2 en pixeles


@dataclass(frozen=True)
class Prediccion:
    """Una detección del modelo."""

    clase: int
    confianza: float
    caja: Caja


@dataclass(frozen=True)
class Verdad:
    """Una caja anotada a mano."""

    clase: int
    caja: Caja

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.caja
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@dataclass(frozen=True)
class Muestra:
    """Una imagen con lo que el modelo predijo y lo que realmente había."""

    archivo: str
    ancho: int
    alto: int
    predicciones: tuple[Prediccion, ...] = ()
    verdades: tuple[Verdad, ...] = ()
    ubicacion: Optional[int] = None
    secuencia: Optional[str] = None
    fecha: Optional[str] = None       # marca de tiempo de la cámara

    @property
    def area_imagen(self) -> float:
        return float(self.ancho * self.alto)

    def fraccion_mayor_caja(self) -> float:
        """Área de la caja anotada más grande, como fracción de la imagen.

        Es la medida de "qué tan lejos está el animal" que el entregable #4
        necesita para partir la evaluación por tamaño.
        """
        if not self.verdades or not self.area_imagen:
            return 0.0
        return max(v.area for v in self.verdades) / self.area_imagen


def iou(a: Caja, b: Caja) -> float:
    """Intersección sobre unión de dos cajas en píxeles."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    interseccion = ix * iy
    if interseccion <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - interseccion
    return interseccion / union if union > 0 else 0.0


def ap_todos_los_puntos(recalls: Sequence[float],
                        precisiones: Sequence[float]) -> float:
    """Área bajo la curva precisión-recall, con envolvente monótona.

    La envolvente (recorrer de derecha a izquierda tomando el máximo) evita que
    un diente de sierra en la precisión castigue la métrica: a igual recall,
    siempre se puede elegir el umbral que da la precisión más alta.
    """
    if not recalls:
        return 0.0

    r = [0.0, *recalls, 1.0]
    p = [0.0, *precisiones, 0.0]
    for i in range(len(p) - 2, -1, -1):
        p[i] = max(p[i], p[i + 1])

    return sum((r[i + 1] - r[i]) * p[i + 1] for i in range(len(r) - 1))


def _aciertos_por_umbral(muestras: Sequence[Muestra], clase: int,
                         umbrales: Sequence[float]) -> tuple[list, int]:
    """Marca cada predicción de ``clase`` como acierto o error, por umbral.

    El emparejamiento es **codicioso por confianza**, el criterio de COCO: se
    recorren las predicciones de mayor a menor confianza y cada una toma la caja
    anotada libre con la que más se solapa. Una caja anotada ya tomada no puede
    volver a usarse, así que una segunda detección del mismo animal cuenta como
    falso positivo — que es exactamente lo que debe pasar.
    """
    positivos = 0
    entradas: list[tuple[float, list[bool]]] = []

    for muestra in muestras:
        verdades = [v for v in muestra.verdades if v.clase == clase]
        positivos += len(verdades)

        predicciones = sorted(
            (p for p in muestra.predicciones if p.clase == clase),
            key=lambda p: -p.confianza,
        )
        if not predicciones:
            continue

        # Un conjunto de cajas tomadas por umbral: un IoU de 0.6 empareja a 0.5
        # pero no a 0.75, de modo que el estado no se puede compartir.
        tomadas: list[set[int]] = [set() for _ in umbrales]
        solapamientos = [[iou(p.caja, v.caja) for v in verdades]
                         for p in predicciones]

        for indice_pred, prediccion in enumerate(predicciones):
            aciertos = []
            for indice_umbral, umbral in enumerate(umbrales):
                mejor, mejor_iou = -1, umbral
                for indice_verdad, valor in enumerate(solapamientos[indice_pred]):
                    if (indice_verdad not in tomadas[indice_umbral]
                            and valor >= mejor_iou):
                        mejor, mejor_iou = indice_verdad, valor
                if mejor >= 0:
                    tomadas[indice_umbral].add(mejor)
                    aciertos.append(True)
                else:
                    aciertos.append(False)
            entradas.append((prediccion.confianza, aciertos))

    return entradas, positivos


def average_precision(muestras: Sequence[Muestra], clase: int,
                      umbrales: Sequence[float] = (0.5,)) -> list[Optional[float]]:
    """AP de una clase, una por umbral de IoU.

    Devuelve ``None`` en vez de 0.0 cuando la clase no tiene ninguna caja
    anotada en ``muestras``: sin positivos la métrica no está definida, y
    promediar un cero inventado arrastra el mAP hacia abajo.
    """
    entradas, positivos = _aciertos_por_umbral(muestras, clase, umbrales)
    if positivos == 0:
        return [None] * len(umbrales)
    if not entradas:
        return [0.0] * len(umbrales)   # había animales, el modelo no vio ninguno

    entradas.sort(key=lambda e: -e[0])
    resultado: list[Optional[float]] = []
    for indice in range(len(umbrales)):
        verdaderos = falsos = 0
        recalls, precisiones = [], []
        for _, aciertos in entradas:
            if aciertos[indice]:
                verdaderos += 1
            else:
                falsos += 1
            recalls.append(verdaderos / positivos)
            precisiones.append(verdaderos / (verdaderos + falsos))
        resultado.append(ap_todos_los_puntos(recalls, precisiones))
    return resultado


@dataclass(frozen=True)
class ResultadoClase:
    """Lo medido para una especie."""

    nombre: str
    indice: int
    soporte: int                      # cajas anotadas en la evaluación
    predicciones: int
    ap50: Optional[float]
    ap50_95: Optional[float]

    @property
    def definida(self) -> bool:
        return self.ap50 is not None


@dataclass(frozen=True)
class ResultadoEvaluacion:
    """mAP macro y el desglose que pide el entregable #2."""

    conjunto: str
    imagenes: int
    clases: tuple[ResultadoClase, ...] = field(default_factory=tuple)

    @property
    def presentes(self) -> tuple[ResultadoClase, ...]:
        return tuple(c for c in self.clases if c.definida)

    @property
    def ausentes(self) -> tuple[str, ...]:
        return tuple(c.nombre for c in self.clases if not c.definida)

    @property
    def map50(self) -> Optional[float]:
        return _promedio([c.ap50 for c in self.presentes])

    @property
    def map50_95(self) -> Optional[float]:
        return _promedio([c.ap50_95 for c in self.presentes])

    def a_dict(self) -> dict:
        return {
            "conjunto": self.conjunto,
            "imagenes": self.imagenes,
            "map50": self.map50,
            "map50_95": self.map50_95,
            "clases_sin_positivos": list(self.ausentes),
            "por_clase": {
                c.nombre: {"soporte": c.soporte, "predicciones": c.predicciones,
                           "ap50": c.ap50, "ap50_95": c.ap50_95}
                for c in self.clases
            },
        }


def _promedio(valores: Sequence[Optional[float]]) -> Optional[float]:
    reales = [v for v in valores if v is not None]
    return sum(reales) / len(reales) if reales else None


def evaluar(muestras: Sequence[Muestra], nombres: Sequence[str],
            conjunto: str = "", completo: bool = True) -> ResultadoEvaluacion:
    """Evalúa todas las clases de ``nombres`` sobre ``muestras``.

    Con ``completo=False`` solo calcula AP@0.5, diez veces más rápido: útil para
    los barridos por subconjunto del entregable #4, donde interesa la tendencia
    y no el segundo decimal.
    """
    umbrales = UMBRALES_COCO if completo else (0.5,)
    clases = []
    for indice, nombre in enumerate(nombres):
        aps = average_precision(muestras, indice, umbrales)
        soporte = sum(1 for m in muestras for v in m.verdades if v.clase == indice)
        predicciones = sum(1 for m in muestras for p in m.predicciones
                           if p.clase == indice)
        clases.append(ResultadoClase(
            nombre=nombre, indice=indice, soporte=soporte,
            predicciones=predicciones,
            ap50=aps[0],
            ap50_95=_promedio(aps) if completo else None,
        ))
    return ResultadoEvaluacion(conjunto=conjunto, imagenes=len(muestras),
                               clases=tuple(clases))


def bootstrap_map(muestras: Sequence[Muestra], nombres: Sequence[str],
                  repeticiones: int = 200, semilla: int = 42,
                  confianza: float = 0.95) -> dict:
    """Intervalo de confianza del mAP@0.5, remuestreando **imágenes**.

    Se remuestrean imágenes y no cajas porque las cajas de una misma imagen no
    son independientes: comparten iluminación, cámara y, muchas veces, al mismo
    animal fotografiado en ráfaga. Remuestrear cajas daría un intervalo
    demasiado angosto.
    """
    azar = random.Random(semilla)
    n = len(muestras)
    valores = []
    for _ in range(repeticiones):
        remuestra = [muestras[azar.randrange(n)] for _ in range(n)]
        valor = evaluar(remuestra, nombres, completo=False).map50
        if valor is not None:
            valores.append(valor)

    valores.sort()
    alfa = (1.0 - confianza) / 2.0
    bajo = valores[max(0, int(alfa * len(valores)) - 1)] if valores else None
    alto = (valores[min(len(valores) - 1, int((1 - alfa) * len(valores)))]
            if valores else None)
    return {
        "map50": evaluar(muestras, nombres, completo=False).map50,
        "ic_bajo": bajo, "ic_alto": alto,
        "repeticiones": len(valores), "confianza": confianza,
    }


def correlacion_spearman(x: Sequence[float],
                         y: Sequence[float]) -> Optional[float]:
    """Correlación de rangos, para medir cuánto depende la AP del soporte.

    Se usa Spearman y no Pearson porque el soporte va de 5 a 4,800 cajas: una
    escala así hace que Pearson quede dominada por las dos especies más comunes.
    """
    if len(x) < 3 or len(x) != len(y):
        return None

    def rangos(valores: Sequence[float]) -> list[float]:
        orden = sorted(range(len(valores)), key=lambda i: valores[i])
        resultado = [0.0] * len(valores)
        i = 0
        while i < len(orden):
            j = i
            while (j + 1 < len(orden)
                   and valores[orden[j + 1]] == valores[orden[i]]):
                j += 1
            promedio = (i + j) / 2.0 + 1.0     # empates comparten el rango medio
            for k in range(i, j + 1):
                resultado[orden[k]] = promedio
            i = j + 1
        return resultado

    rx, ry = rangos(x), rangos(y)
    n = len(rx)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx)
                    * sum((b - my) ** 2 for b in ry))
    return num / den if den else None
