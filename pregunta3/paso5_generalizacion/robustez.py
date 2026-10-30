"""Entregable #4: robustez ante oclusión y tamaño del objeto.

Dos análisis distintos, con costos muy distintos:

* **Tamaño.** No hace falta volver a correr el modelo. Las predicciones ya
  guardadas se reparten en tramos según qué fracción de la imagen ocupa el
  animal, y el mAP se recalcula en cada tramo. Un animal a veinte metros de la
  cámara ocupa menos del 1% del cuadro y, reducido a 640 px, queda en unos
  pocos píxeles.

* **Oclusión.** Acá sí hay que volver a predecir: se tapa a propósito parte de
  cada animal anotado y se mide cuánto cae el mAP. El conjunto no trae etiquetas
  de oclusión, así que la alternativa sería no responder el entregable; la
  oclusión sintética lo responde de forma controlada, con el costo de que un
  parche gris no es exactamente un arbusto.

Criterio para los tramos de tamaño: una imagen entra en un tramo solo si
**todas** sus cajas anotadas caen en ese rango. Si se filtrara caja por caja,
las cajas descartadas seguirían en la imagen y el modelo las detectaría: cada
detección correcta de un animal fuera del rango contaría como falso positivo y
el mAP del tramo saldría artificialmente bajo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from pregunta3.paso4_evaluacion.metricas import Muestra, Verdad, evaluar

# Tramos por fracción del área de la imagen. Los cortes siguen el espíritu de
# COCO (small/medium/large sobre 1024 y 9216 px² en imágenes de ~640x480), pero
# expresados como fracción, porque en este conjunto conviven diez resoluciones.
TRAMOS_TAMANO = (
    ("diminuto", 0.0, 0.002),      # menos del 0.2% del cuadro
    ("chico", 0.002, 0.01),
    ("mediano", 0.01, 0.05),
    ("grande", 0.05, 1.01),
)

# Fracción del ÁREA de cada caja anotada que se tapa.
NIVELES_OCLUSION = (0.0, 0.25, 0.50, 0.75)

GRIS = 114   # el valor de relleno que usa Ultralytics al hacer letterbox


def fraccion_de(verdad: Verdad, muestra: Muestra) -> float:
    return verdad.area / muestra.area_imagen if muestra.area_imagen else 0.0


def muestras_en_tramo(muestras: Sequence[Muestra], minimo: float,
                      maximo: float) -> list[Muestra]:
    """Imágenes cuyas cajas anotadas caen **todas** en ``[minimo, maximo)``."""
    elegidas = []
    for muestra in muestras:
        if not muestra.verdades:
            continue
        if all(minimo <= fraccion_de(v, muestra) < maximo
               for v in muestra.verdades):
            elegidas.append(muestra)
    return elegidas


@dataclass(frozen=True)
class ResultadoTramo:
    """mAP de un tramo de tamaño, con cuántos datos lo sostienen."""

    tramo: str
    minimo: float
    maximo: float
    imagenes: int
    cajas: int
    map50: Optional[float]
    especies: int

    @property
    def confiable(self) -> bool:
        return self.cajas >= 50


def barrido_por_tamano(muestras: Sequence[Muestra], nombres: Sequence[str],
                       tramos=TRAMOS_TAMANO) -> list[ResultadoTramo]:
    """mAP@0.5 por tramo de tamaño del objeto."""
    salida = []
    for nombre, minimo, maximo in tramos:
        grupo = muestras_en_tramo(muestras, minimo, maximo)
        resultado = evaluar(grupo, nombres, completo=False)
        salida.append(ResultadoTramo(
            tramo=nombre, minimo=minimo, maximo=maximo,
            imagenes=len(grupo),
            cajas=sum(len(m.verdades) for m in grupo),
            map50=resultado.map50,
            especies=len(resultado.presentes),
        ))
    return salida


def ocluir_cajas(imagen: np.ndarray, cajas: Sequence[tuple],
                 fraccion: float, modo: str = "centro") -> np.ndarray:
    """Tapa con un parche gris una ``fraccion`` del área de cada caja.

    El lado del parche escala con la raíz de la fracción, de modo que tapar
    "la mitad" tape la mitad del **área** y no la mitad del lado.

    ``modo``:

    * ``centro`` — el parche va centrado en la caja. Deja la silueta completa y
      quita la textura interior: separa "reconocer la forma" de "reconocer el
      pelaje".
    * ``inferior`` — tapa desde el borde de abajo, que es lo que hace la
      vegetación real. Rompe también la silueta, y por eso es el caso más
      duro.
    """
    if fraccion <= 0:
        return imagen
    lado = min(1.0, float(fraccion)) ** 0.5
    salida = imagen.copy()
    alto_img, ancho_img = salida.shape[:2]

    for x1, y1, x2, y2 in cajas:
        ancho, alto = (x2 - x1), (y2 - y1)
        if ancho <= 0 or alto <= 0:
            continue
        pw, ph = ancho * lado, alto * lado
        if modo == "inferior":
            px1, py1 = x1 + (ancho - pw) / 2, y2 - ph
        else:
            px1, py1 = x1 + (ancho - pw) / 2, y1 + (alto - ph) / 2

        a = max(0, int(round(px1)))
        b = max(0, int(round(py1)))
        c = min(ancho_img, int(round(px1 + pw)))
        d = min(alto_img, int(round(py1 + ph)))
        if c > a and d > b:
            salida[b:d, a:c] = GRIS
    return salida


def area_tapada(caja: tuple, fraccion: float) -> float:
    """Área que ``ocluir_cajas`` taparía de una caja. Para verificar el parche."""
    x1, y1, x2, y2 = caja
    lado = min(1.0, max(0.0, float(fraccion))) ** 0.5
    return (x2 - x1) * lado * (y2 - y1) * lado
