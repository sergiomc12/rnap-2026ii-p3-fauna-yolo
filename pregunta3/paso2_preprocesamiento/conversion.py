"""Conversión de COCO Camera Traps al formato que YOLO espera.

Dos diferencias que hay que salvar, y confundir cualquiera de las dos arruina
el entrenamiento en silencio:

* **Coordenadas.** COCO guarda la esquina superior izquierda más el tamaño, en
  píxeles. YOLO guarda el centro más el tamaño, normalizado a [0, 1]. Usar uno
  por el otro desplaza cada caja media caja hacia arriba y a la izquierda, y el
  modelo entrena igual: aprende a predecir cajas corridas.

* **Identificadores de clase.** Las categorías de COCO Camera Traps no son
  contiguas ni empiezan en cero. YOLO exige 0..N-1 en ese orden exacto, porque
  el índice de clase es la posición en la lista ``names`` del ``data.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from ..paso1_datos.anotaciones import Imagen, Particion


@dataclass(frozen=True)
class MapaClases:
    """Traduce los identificadores de COCO a los índices contiguos de YOLO."""

    nombres: tuple[str, ...]
    de_coco: dict[int, int]

    @property
    def n(self) -> int:
        return len(self.nombres)

    def indice(self, categoria_coco: int) -> Optional[int]:
        return self.de_coco.get(categoria_coco)

    @classmethod
    def desde_particiones(
        cls, particiones: Sequence[Particion], excluir: Sequence[str] = ("empty",)
    ) -> "MapaClases":
        """Construye el mapa a partir de las clases que realmente aparecen.

        Se ordena por frecuencia descendente para que el índice 0 sea la clase
        más común: facilita leer una matriz de confusión, donde las clases con
        datos quedan juntas arriba a la izquierda.

        ``empty`` se excluye porque no es un objeto a detectar. En detección, la
        ausencia de cajas en una imagen ya expresa "acá no hay nada"; crear una
        clase para el fondo le enseñaría al modelo a dibujar una caja alrededor
        de la nada.
        """
        cuenta: dict[int, int] = {}
        nombres_coco: dict[int, str] = {}
        for particion in particiones:
            nombres_coco.update(particion.categorias)
            for imagen in particion.imagenes:
                for caja in imagen.cajas:
                    cuenta[caja.categoria] = cuenta.get(caja.categoria, 0) + 1

        excluidas = {e.lower() for e in excluir}
        ordenadas = [
            cid for cid, _ in sorted(cuenta.items(), key=lambda kv: -kv[1])
            if nombres_coco.get(cid, "").lower() not in excluidas
        ]
        return cls(
            nombres=tuple(nombres_coco[cid] for cid in ordenadas),
            de_coco={cid: i for i, cid in enumerate(ordenadas)},
        )


@dataclass(frozen=True)
class LineasEtiqueta:
    """Las líneas del archivo .txt de una imagen, y qué se descartó."""

    lineas: tuple[str, ...]
    descartadas_clase: int = 0
    descartadas_geometria: int = 0

    @property
    def vacia(self) -> bool:
        return not self.lineas


def convertir_imagen(imagen: Imagen, mapa: MapaClases) -> LineasEtiqueta:
    """Produce las líneas YOLO de una imagen: ``clase cx cy w h``."""
    lineas, sin_clase, mala_geometria = [], 0, 0

    for caja in imagen.cajas:
        indice = mapa.indice(caja.categoria)
        if indice is None:
            sin_clase += 1          # clase excluida, como "empty"
            continue

        cx, cy, ancho, alto = caja.a_yolo(imagen.ancho, imagen.alto)
        if not _geometria_valida(cx, cy, ancho, alto):
            mala_geometria += 1
            continue

        lineas.append(f"{indice} {cx:.6f} {cy:.6f} {ancho:.6f} {alto:.6f}")

    return LineasEtiqueta(tuple(lineas), sin_clase, mala_geometria)


def _geometria_valida(cx: float, cy: float, ancho: float, alto: float) -> bool:
    """Descarta cajas degeneradas o fuera de la imagen.

    Las anotaciones reales traen algunas con tamaño cero y otras que se salen
    del borde. Ultralytics las acepta sin quejarse y luego entrena con ellas,
    de modo que filtrarlas acá es la única oportunidad.
    """
    if ancho <= 0 or alto <= 0:
        return False
    if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
        return False
    return ancho <= 1.0 and alto <= 1.0


@dataclass(frozen=True)
class ResumenConversion:
    """Qué produjo la conversión de una partición."""

    particion: str
    imagenes: int
    imagenes_con_caja: int
    cajas: int
    descartadas_clase: int
    descartadas_geometria: int
    sin_imagen_en_disco: int
    por_fuga_de_rafaga: int = 0

    @property
    def tasa_descarte(self) -> float:
        total = self.cajas + self.descartadas_clase + self.descartadas_geometria
        return (self.descartadas_clase + self.descartadas_geometria) / total if total else 0.0


def secuencias_de(particion: Particion) -> frozenset[str]:
    """Identificadores de ráfaga presentes en una partición."""
    return frozenset(i.secuencia for i in particion.imagenes if i.secuencia)


def filtrar_fuga_de_rafagas(
    particion: Particion, secuencias_prohibidas: frozenset[str]
) -> tuple[Imagen, ...]:
    """Quita las imágenes que pertenecen a una ráfaga del entrenamiento.

    Las cámaras trampa disparan ráfagas de unas tres fotos casi idénticas. Si
    una ráfaga se reparte entre entrenamiento y evaluación, el modelo ve en la
    evaluación una imagen que prácticamente memorizó, y la métrica queda
    inflada sin que nada lo advierta.

    Medido en este conjunto: ``cis_val`` comparte 224 ráfagas con ``train``, lo
    que contamina 75 de sus 1,665 imágenes con caja (4.5%). Los dos conjuntos
    de prueba, en cambio, no comparten ninguna.
    """
    return tuple(i for i in particion.con_cajas
                 if i.secuencia not in secuencias_prohibidas)


def escribir_etiquetas(
    particion: Particion,
    mapa: MapaClases,
    destino: Path,
    imagenes_disponibles: Optional[set[str]] = None,
    secuencias_prohibidas: frozenset[str] = frozenset(),
) -> ResumenConversion:
    """Escribe un .txt por imagen en ``destino``.

    ``imagenes_disponibles`` limita la conversión a las imágenes ya
    descargadas: etiquetar una imagen ausente dejaría a Ultralytics buscando un
    archivo que no existe.
    """
    destino.mkdir(parents=True, exist_ok=True)
    con_caja = cajas = sin_clase = mala_geom = faltantes = contaminadas = 0

    for imagen in particion.con_cajas:
        if imagen.secuencia in secuencias_prohibidas:
            contaminadas += 1       # su ráfaga también está en entrenamiento
            continue
        if imagenes_disponibles is not None and imagen.archivo not in imagenes_disponibles:
            faltantes += 1
            continue

        etiqueta = convertir_imagen(imagen, mapa)
        sin_clase += etiqueta.descartadas_clase
        mala_geom += etiqueta.descartadas_geometria
        if etiqueta.vacia:
            continue

        archivo = destino / (Path(imagen.archivo).stem + ".txt")
        archivo.write_text("\n".join(etiqueta.lineas) + "\n", encoding="utf-8")
        con_caja += 1
        cajas += len(etiqueta.lineas)

    return ResumenConversion(
        particion=particion.nombre,
        imagenes=len(particion.imagenes),
        imagenes_con_caja=con_caja,
        cajas=cajas,
        descartadas_clase=sin_clase,
        descartadas_geometria=mala_geom,
        sin_imagen_en_disco=faltantes,
        por_fuga_de_rafaga=contaminadas,
    )


def escribir_data_yaml(
    ruta: Path, mapa: MapaClases, raiz: Path,
    train: str, val: str, test: Optional[str] = None,
) -> Path:
    """Escribe el ``data.yaml`` que Ultralytics lee.

    Los nombres van en el orden de ``mapa.nombres``: ese orden ES la numeración
    de clases.

    **No se escribe la clave ``path``, y es a propósito.** Antes se escribía
    ``path: D:/Backend/redesneuronales/pregunta3/...`` —la ruta absoluta de la
    máquina que generó el dataset—, así que el yaml no servía en ninguna otra:
    ``pregunta3/`` se comparte con el equipo y esa carpeta no existe del otro
    lado. Tampoco sirve poner ``path`` relativo: Ultralytics resuelve un ``path``
    relativo contra su ``DATASETS_DIR`` global (``check_det_dataset``), que es
    una configuración del usuario y no tiene por qué apuntar acá. Sin la clave,
    Ultralytics usa **el directorio del propio yaml**, que es exactamente lo que
    hace falta. ``raiz`` se sigue recibiendo porque las listas se escriben
    relativas a ella.
    """
    lineas = [
        "# Generado por pregunta3/paso2_preprocesamiento",
        "# El orden de 'names' define los indices de clase: no reordenar.",
        "# Sin 'path': Ultralytics usa el directorio de este yaml, para que el",
        "# dataset funcione en cualquier maquina (ver escribir_data_yaml).",
        f"train: {train}",
        f"val: {val}",
    ]
    if test:
        lineas.append(f"test: {test}")
    lineas += [f"nc: {mapa.n}", "names:"]
    lineas += [f"  {i}: {nombre}" for i, nombre in enumerate(mapa.nombres)]

    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    return ruta
