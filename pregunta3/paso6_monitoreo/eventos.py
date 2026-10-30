"""De detecciones a eventos independientes, que es la unidad de la ecología.

El entregable #5 pide un "reporte de actividad por especie y periodo". Contar
detecciones no sirve para eso, y el error no es de implementación sino
conceptual: una cámara trampa dispara ráfagas de unas tres fotos por evento, y
un animal que se queda comiendo frente al lente dispara decenas. Contando
detecciones, un zorro curioso durante diez minutos pesaría igual que diez
zorros distintos.

La literatura de cámaras trampa resuelve esto con el **evento independiente**:
dos detecciones de la misma especie, en la misma cámara, separadas por menos de
un intervalo de silencio (30 minutos es el valor convencional) son el mismo
evento. Es el criterio que se implementa acá, y es un parámetro explícito
porque la elección cambia los números y conviene poder mostrarlo.

Las dos métricas que el módulo produce son las estándar del dominio:

* **tasa de detección** — eventos por cada 100 noches-cámara. Normaliza por
  esfuerzo de muestreo: una cámara que estuvo seis meses y otra que estuvo dos
  semanas no son comparables en conteo crudo.
* **ocupación naive** — en qué fracción de las cámaras apareció la especie al
  menos una vez. Es "naive" porque no corrige por detección imperfecta: una
  especie presente pero nunca fotografiada cuenta como ausente. Decirlo es
  parte del reporte, no una nota al pie.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

# Intervalo de silencio convencional en la literatura de cámaras trampa.
MINUTOS_INDEPENDENCIA = 30

# Franjas del día. El amanecer y el atardecer van aparte porque muchas especies
# son crepusculares, y meterlas en "día" o "noche" borraría justamente eso.
FRANJAS = (
    ("noche", 0, 5),
    ("amanecer", 5, 8),
    ("dia", 8, 17),
    ("atardecer", 17, 20),
    ("noche_tarde", 20, 24),
)


@dataclass(frozen=True)
class Avistamiento:
    """Una detección con su cuándo, su dónde y su qué."""

    especie: str
    fecha: dt.datetime
    ubicacion: Optional[int]
    confianza: float
    secuencia: Optional[str] = None

    @property
    def hora(self) -> int:
        return self.fecha.hour

    @property
    def franja(self) -> str:
        for nombre, desde, hasta in FRANJAS:
            if desde <= self.fecha.hour < hasta:
                return nombre
        return "desconocida"


@dataclass(frozen=True)
class Evento:
    """Un avistamiento independiente: una visita, no una foto."""

    especie: str
    ubicacion: Optional[int]
    inicio: dt.datetime
    fin: dt.datetime
    detecciones: int
    confianza_maxima: float

    @property
    def minutos(self) -> float:
        return (self.fin - self.inicio).total_seconds() / 60.0


def leer_fecha(texto: Optional[str]) -> Optional[dt.datetime]:
    """Interpreta la marca de tiempo de CCT, tolerando lo que venga mal.

    El conjunto trae ``date_captured`` como ``"2012-03-10 14:22:31"``, pero no
    todas las imágenes la tienen y algunas cámaras la escribieron mal. Una
    fecha ilegible se descarta del reporte temporal en vez de inventarse: un
    reporte de actividad con fechas adivinadas sería peor que uno más chico.
    """
    if not texto:
        return None
    for formato in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return dt.datetime.strptime(texto.strip(), formato)
        except (ValueError, AttributeError):
            continue
    return None


def agrupar_en_eventos(
    avistamientos: Sequence[Avistamiento],
    minutos: int = MINUTOS_INDEPENDENCIA,
) -> list[Evento]:
    """Colapsa avistamientos cercanos de la misma especie y cámara.

    Se agrupa por ``(especie, ubicación)`` y dentro de cada grupo se recorre en
    orden de tiempo, cerrando el evento cuando el salto supera ``minutos``.
    """
    por_grupo: dict[tuple[str, Optional[int]], list[Avistamiento]] = {}
    for avistamiento in avistamientos:
        por_grupo.setdefault((avistamiento.especie, avistamiento.ubicacion),
                             []).append(avistamiento)

    limite = dt.timedelta(minutes=minutos)
    eventos: list[Evento] = []

    for (especie, ubicacion), grupo in por_grupo.items():
        grupo.sort(key=lambda a: a.fecha)
        inicio = fin = grupo[0].fecha
        cuenta, confianza = 0, 0.0

        for avistamiento in grupo:
            if avistamiento.fecha - fin > limite:
                eventos.append(Evento(especie, ubicacion, inicio, fin,
                                      cuenta, confianza))
                inicio, cuenta, confianza = avistamiento.fecha, 0, 0.0
            fin = avistamiento.fecha
            cuenta += 1
            confianza = max(confianza, avistamiento.confianza)

        eventos.append(Evento(especie, ubicacion, inicio, fin, cuenta,
                              confianza))

    eventos.sort(key=lambda e: (e.inicio, e.especie))
    return eventos


@dataclass
class EsfuerzoMuestreo:
    """Cuántas noches-cámara sostiene el reporte.

    Una noche-cámara es un día de una cámara encendida. Sin esta normalización
    los conteos no se pueden comparar entre cámaras ni entre periodos, y es el
    error más común al reportar datos de cámaras trampa.
    """

    por_ubicacion: dict[Optional[int], set[dt.date]] = field(
        default_factory=dict)

    def registrar(self, ubicacion: Optional[int], fecha: dt.datetime) -> None:
        self.por_ubicacion.setdefault(ubicacion, set()).add(fecha.date())

    @property
    def noches_camara(self) -> int:
        return sum(len(d) for d in self.por_ubicacion.values())

    @property
    def camaras(self) -> int:
        return len(self.por_ubicacion)

    def noches_de(self, ubicacion: Optional[int]) -> int:
        return len(self.por_ubicacion.get(ubicacion, ()))


@dataclass(frozen=True)
class ActividadEspecie:
    """El renglón del reporte para una especie."""

    especie: str
    eventos: int
    detecciones: int
    camaras: int
    primera: Optional[dt.datetime]
    ultima: Optional[dt.datetime]
    por_hora: tuple[int, ...]            # 24 posiciones
    por_franja: dict[str, int]
    por_mes: dict[str, int]
    noches_camara: int
    camaras_totales: int

    @property
    def tasa_por_100_noches(self) -> Optional[float]:
        if not self.noches_camara:
            return None
        return 100.0 * self.eventos / self.noches_camara

    @property
    def ocupacion_naive(self) -> Optional[float]:
        if not self.camaras_totales:
            return None
        return self.camaras / self.camaras_totales

    @property
    def indice_nocturnidad(self) -> Optional[float]:
        """Fracción de eventos entre las 20:00 y las 05:00.

        Sirve de control de sanidad del modelo contra historia natural
        conocida: si el reporte dice que los mapaches son diurnos, el problema
        está en el modelo, no en los mapaches.
        """
        if not self.eventos:
            return None
        nocturnos = sum(self.por_hora[h] for h in list(range(20, 24)) + list(range(0, 5)))
        return nocturnos / self.eventos


def actividad_por_especie(eventos: Sequence[Evento],
                          esfuerzo: EsfuerzoMuestreo) -> list[ActividadEspecie]:
    """Agrega los eventos a un renglón por especie."""
    por_especie: dict[str, list[Evento]] = {}
    for evento in eventos:
        por_especie.setdefault(evento.especie, []).append(evento)

    filas = []
    for especie, grupo in por_especie.items():
        horas = [0] * 24
        franjas: dict[str, int] = {}
        meses: dict[str, int] = {}
        for evento in grupo:
            horas[evento.inicio.hour] += 1
            clave = f"{evento.inicio.year:04d}-{evento.inicio.month:02d}"
            meses[clave] = meses.get(clave, 0) + 1
            for nombre, desde, hasta in FRANJAS:
                if desde <= evento.inicio.hour < hasta:
                    franjas[nombre] = franjas.get(nombre, 0) + 1
                    break

        ubicaciones = {e.ubicacion for e in grupo}
        filas.append(ActividadEspecie(
            especie=especie,
            eventos=len(grupo),
            detecciones=sum(e.detecciones for e in grupo),
            camaras=len(ubicaciones),
            primera=min(e.inicio for e in grupo),
            ultima=max(e.fin for e in grupo),
            por_hora=tuple(horas),
            por_franja=dict(sorted(franjas.items())),
            por_mes=dict(sorted(meses.items())),
            noches_camara=sum(esfuerzo.noches_de(u) for u in ubicaciones),
            camaras_totales=esfuerzo.camaras,
        ))

    filas.sort(key=lambda f: -f.eventos)
    return filas


def avistamientos_desde(
    lineas: Iterable[dict], nombres: Sequence[str],
    confianza_minima: float = 0.25, campo: str = "predicciones",
) -> tuple[list[Avistamiento], EsfuerzoMuestreo, dict]:
    """Convierte el volcado de predicciones del paso 4 en avistamientos.

    El umbral por defecto es 0.25, no el 0.001 con el que se calcula la AP: un
    reporte ecológico es una decisión, y a 0.001 el listado se llenaría de
    detecciones que nadie creería. Se deja como parámetro para poder mostrar
    cuánto cambia el reporte según el umbral.

    Con ``campo="verdades"`` se construye el mismo reporte a partir de las
    cajas anotadas a mano. Eso es lo que permite **validar** el módulo: el
    reporte del modelo se compara contra el que habría escrito un humano con
    las mismas fotos, especie por especie.
    """
    avistamientos: list[Avistamiento] = []
    esfuerzo = EsfuerzoMuestreo()
    sin_fecha = con_fecha = 0

    for linea in lineas:
        fecha = leer_fecha(linea.get("fecha"))
        if fecha is None:
            sin_fecha += 1
            continue
        con_fecha += 1
        ubicacion = linea.get("ubicacion")
        esfuerzo.registrar(ubicacion, fecha)

        for prediccion in linea.get(campo, ()):
            # Las cajas anotadas no traen confianza: valen 1.0 por definición.
            confianza = float(prediccion.get("confianza", 1.0))
            if confianza < confianza_minima:
                continue
            indice = int(prediccion["clase"])
            if not 0 <= indice < len(nombres):
                continue
            avistamientos.append(Avistamiento(
                especie=nombres[indice], fecha=fecha, ubicacion=ubicacion,
                confianza=confianza,
                secuencia=linea.get("secuencia"),
            ))

    return avistamientos, esfuerzo, {
        "imagenes_con_fecha": con_fecha,
        "imagenes_sin_fecha_legible": sin_fecha,
        "confianza_minima": confianza_minima,
        "campo": campo,
    }
