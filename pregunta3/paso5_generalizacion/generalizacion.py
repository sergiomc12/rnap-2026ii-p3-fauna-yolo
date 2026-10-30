"""Entregable #3: generalización entre ubicaciones de cámara.

La comparación obvia es ``cis_test`` (10 ubicaciones vistas en el
entrenamiento) contra ``trans_test`` (9 ubicaciones nunca vistas). Pero esa
diferencia **confunde dos efectos**, y presentarla sola sería un error de
método:

1. el cambio de fondo — lo que el entregable quiere medir;
2. el cambio de distribución de especies — ``raccoon`` pasa de 981 cajas en
   ``cis_test`` a 4,803 en ``trans_test``, y ``rabbit`` cae de 1,559 a 722.

Este módulo separa los dos. El mAP **macro** ya no se deja arrastrar por la
prevalencia al promediar, pero la AP de cada clase sí depende de cuántos
positivos tiene: con pocos positivos, un puñado de falsos positivos hunde la
precisión. Para controlarlo se construye un subconjunto de la partición nueva
con la **misma distribución de especies** que la vista, y la brecha que queda
después de ese control es la que se puede atribuir al fondo.

También se reporta el mAP **por ubicación de cámara**: una sola cifra para las
nueve ubicaciones nuevas esconde que el desempeño varía mucho entre cámaras, y
esa dispersión es lo que un ecólogo necesita saber antes de instalar una.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional, Sequence

from pregunta3.paso4_evaluacion.metricas import (
    Muestra, ResultadoEvaluacion, evaluar,
)


@dataclass(frozen=True)
class BrechaClase:
    """Cuánto cayó la AP de una especie al cambiar de ubicación."""

    nombre: str
    ap_vista: Optional[float]
    ap_nueva: Optional[float]
    soporte_vista: int
    soporte_nueva: int

    @property
    def caida(self) -> Optional[float]:
        if self.ap_vista is None or self.ap_nueva is None:
            return None
        return self.ap_vista - self.ap_nueva

    @property
    def cambio_de_prevalencia(self) -> Optional[float]:
        """Cuántas veces más (o menos) frecuente es la especie en la nueva."""
        if not self.soporte_vista:
            return None
        return self.soporte_nueva / self.soporte_vista


def brecha_por_clase(vista: ResultadoEvaluacion,
                     nueva: ResultadoEvaluacion) -> list[BrechaClase]:
    """Empareja las dos evaluaciones clase por clase."""
    por_nombre = {c.nombre: c for c in nueva.clases}
    brechas = []
    for clase in vista.clases:
        otra = por_nombre.get(clase.nombre)
        brechas.append(BrechaClase(
            nombre=clase.nombre,
            ap_vista=clase.ap50,
            ap_nueva=otra.ap50 if otra else None,
            soporte_vista=clase.soporte,
            soporte_nueva=otra.soporte if otra else 0,
        ))
    return brechas


def mapa_por_ubicacion(muestras: Sequence[Muestra], nombres: Sequence[str],
                       minimo_cajas: int = 50) -> dict[str, dict]:
    """mAP@0.5 de cada ubicación de cámara por separado.

    Las ubicaciones con menos de ``minimo_cajas`` cajas anotadas se reportan
    igual, pero marcadas: un mAP calculado sobre veinte cajas es ruido, y
    mezclarlo con el de una cámara con tres mil daría una dispersión inventada.
    """
    por_ubicacion: dict[Optional[int], list[Muestra]] = {}
    for muestra in muestras:
        por_ubicacion.setdefault(muestra.ubicacion, []).append(muestra)

    salida = {}
    for ubicacion, grupo in sorted(por_ubicacion.items(),
                                   key=lambda kv: (kv[0] is None, kv[0])):
        resultado = evaluar(grupo, nombres, completo=False)
        cajas = sum(len(m.verdades) for m in grupo)
        salida[str(ubicacion)] = {
            "imagenes": len(grupo),
            "cajas": cajas,
            "map50": resultado.map50,
            "especies_presentes": len(resultado.presentes),
            "muestra_suficiente": cajas >= minimo_cajas,
        }
    return salida


def dispersion(valores: Sequence[float]) -> dict:
    """Mínimo, máximo, media y rango de una lista de mAP por ubicación."""
    reales = [v for v in valores if v is not None]
    if not reales:
        return {"n": 0}
    media = sum(reales) / len(reales)
    return {
        "n": len(reales), "media": media,
        "minimo": min(reales), "maximo": max(reales),
        "rango": max(reales) - min(reales),
        "desviacion": (sum((v - media) ** 2 for v in reales)
                       / len(reales)) ** 0.5,
    }


def remuestrear_a_distribucion(
    muestras: Sequence[Muestra], objetivo: dict[int, int],
    semilla: int = 42,
) -> list[Muestra]:
    """Toma un subconjunto cuya mezcla de especies imite ``objetivo``.

    ``objetivo`` es ``{clase: cajas}`` de la partición de referencia. El
    muestreo es **por imagen**, no por caja: una imagen no se puede partir, y
    sacar cajas de a una rompería la escena (el modelo detectaría un animal que
    ya no está anotado y contaría como falso positivo).

    Se recorre la clase más escasa primero. Si se empezara por la más común, al
    llegar a las raras ya no quedarían imágenes disponibles y la mezcla
    resultante volvería a parecerse a la original.
    """
    azar = random.Random(semilla)
    disponibles: dict[int, list[int]] = {}
    for indice, muestra in enumerate(muestras):
        for clase in {v.clase for v in muestra.verdades}:
            disponibles.setdefault(clase, []).append(indice)
    for lista in disponibles.values():
        azar.shuffle(lista)

    elegidas: set[int] = set()
    cuenta: dict[int, int] = {}

    def cajas_de(indice: int, clase: int) -> int:
        return sum(1 for v in muestras[indice].verdades if v.clase == clase)

    for clase in sorted(objetivo, key=lambda c: objetivo[c]):
        meta = objetivo[clase]
        cuenta.setdefault(clase, 0)
        # Las ya elegidas por otra clase también aportan cajas de esta.
        for indice in elegidas:
            cuenta[clase] += cajas_de(indice, clase)
        for indice in disponibles.get(clase, ()):
            if cuenta[clase] >= meta:
                break
            if indice in elegidas:
                continue
            elegidas.add(indice)
            for otra in {v.clase for v in muestras[indice].verdades}:
                cuenta[otra] = cuenta.get(otra, 0) + cajas_de(indice, otra)

    return [muestras[i] for i in sorted(elegidas)]


def distribucion_de(resultado: ResultadoEvaluacion) -> dict[int, int]:
    """``{indice de clase: cajas anotadas}`` de una evaluación."""
    return {c.indice: c.soporte for c in resultado.clases if c.soporte}
