"""Lectura de las anotaciones de Caltech Camera Traps.

El conjunto se distribuye en formato COCO Camera Traps, con los archivos de
partición de ECCV 2018 separados por tipo de ubicación:

* ``cis``  -- las mismas ubicaciones de cámara que el entrenamiento
* ``trans`` -- ubicaciones completamente nuevas

Esa distinción es la que el entregable #3 necesita: comparar el desempeño en
ubicaciones vistas contra ubicaciones nunca vistas es exactamente la diferencia
entre ``cis_test`` y ``trans_test``.

De las 243,100 imágenes del conjunto, solo 63,025 tienen bounding box. Un
detector solo puede entrenarse con esas, así que todo lo demás se ignora.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

# Las rutas de abajo se anclan a ESTE archivo, no al directorio de
# trabajo: la pregunta se comparte sola y sus scripts se corren desde
# cualquier lado.
_RAIZ = Path(__file__).resolve().parents[2]

RAW = _RAIZ / "pregunta3/paso1_datos/raw"
ECCV = RAW / "eccv_18_annotation_files"

# Los cinco archivos de la partición oficial, con lo que significa cada uno.
PARTICIONES = {
    "train": ("train_annotations.json", "ubicaciones de entrenamiento"),
    "cis_val": ("cis_val_annotations.json", "validación, mismas ubicaciones"),
    "cis_test": ("cis_test_annotations.json", "prueba, mismas ubicaciones"),
    "trans_val": ("trans_val_annotations.json", "validación, ubicaciones nuevas"),
    "trans_test": ("trans_test_annotations.json", "prueba, ubicaciones nuevas"),
}


@dataclass(frozen=True)
class Caja:
    """Un bounding box en formato COCO: esquina superior izquierda y tamaño."""

    x: float
    y: float
    ancho: float
    alto: float
    categoria: int

    @property
    def area(self) -> float:
        return self.ancho * self.alto

    def a_yolo(self, ancho_img: int, alto_img: int) -> tuple[float, float, float, float]:
        """Convierte a formato YOLO: centro y tamaño, normalizados a [0, 1].

        YOLO usa el centro porque es lo que la red predice como desplazamiento
        dentro de su celda de la cuadrícula; COCO usa la esquina. Confundirlos
        desplaza cada caja media caja hacia arriba y a la izquierda.
        """
        return (
            (self.x + self.ancho / 2) / ancho_img,
            (self.y + self.alto / 2) / alto_img,
            self.ancho / ancho_img,
            self.alto / alto_img,
        )


@dataclass(frozen=True)
class Imagen:
    """Una fotografía, con la ubicación de cámara que la tomó."""

    id: str
    archivo: str
    ancho: int
    alto: int
    ubicacion: Optional[int] = None
    secuencia: Optional[str] = None
    fecha: Optional[str] = None
    cajas: tuple[Caja, ...] = ()

    @property
    def tiene_cajas(self) -> bool:
        return len(self.cajas) > 0


@dataclass(frozen=True)
class Particion:
    """Una de las cinco particiones oficiales."""

    nombre: str
    descripcion: str
    imagenes: tuple[Imagen, ...]
    categorias: dict[int, str]

    @property
    def con_cajas(self) -> tuple[Imagen, ...]:
        return tuple(i for i in self.imagenes if i.tiene_cajas)

    @property
    def n_cajas(self) -> int:
        return sum(len(i.cajas) for i in self.imagenes)

    @property
    def ubicaciones(self) -> frozenset[int]:
        return frozenset(i.ubicacion for i in self.imagenes if i.ubicacion is not None)

    def prevalencia(self) -> dict[str, int]:
        """Cuántas cajas hay de cada especie."""
        cuenta: dict[str, int] = {}
        for imagen in self.imagenes:
            for caja in imagen.cajas:
                nombre = self.categorias.get(caja.categoria, str(caja.categoria))
                cuenta[nombre] = cuenta.get(nombre, 0) + 1
        return dict(sorted(cuenta.items(), key=lambda kv: -kv[1]))


def cargar_particion(nombre: str, directorio: Path = ECCV) -> Particion:
    """Lee una partición desde su JSON en formato COCO."""
    if nombre not in PARTICIONES:
        raise ValueError(f"partición desconocida: {nombre}. "
                         f"Opciones: {', '.join(PARTICIONES)}")
    archivo, descripcion = PARTICIONES[nombre]
    datos = json.loads((directorio / archivo).read_text(encoding="utf-8"))

    categorias = {c["id"]: c["name"] for c in datos["categories"]}

    # Las anotaciones vienen en una lista aparte y se enlazan por image_id;
    # agruparlas primero evita recorrerla una vez por imagen.
    por_imagen: dict[str, list[Caja]] = {}
    for anotacion in datos["annotations"]:
        caja = anotacion.get("bbox")
        if not caja:
            continue    # anotación de solo-clase, sin localización
        por_imagen.setdefault(anotacion["image_id"], []).append(
            Caja(*caja, categoria=anotacion["category_id"])
        )

    imagenes = tuple(
        Imagen(
            id=i["id"],
            archivo=i["file_name"],
            ancho=int(i["width"]),
            alto=int(i["height"]),
            ubicacion=_entero_o_nulo(i.get("location")),
            secuencia=i.get("seq_id"),
            fecha=i.get("date_captured"),
            cajas=tuple(por_imagen.get(i["id"], ())),
        )
        for i in datos["images"]
    )
    return Particion(nombre=nombre, descripcion=descripcion,
                     imagenes=imagenes, categorias=categorias)


def cargar_todas(directorio: Path = ECCV) -> dict[str, Particion]:
    return {nombre: cargar_particion(nombre, directorio) for nombre in PARTICIONES}


def archivos_necesarios(particiones: Iterable[Particion]) -> list[str]:
    """Nombres de archivo de todas las imágenes con caja, sin repetir.

    Las particiones comparten imágenes en algunos casos, así que descargarlas
    por partición bajaría varias veces lo mismo.
    """
    vistos: dict[str, None] = {}
    for particion in particiones:
        for imagen in particion.con_cajas:
            vistos.setdefault(imagen.archivo, None)
    return list(vistos)


def _entero_o_nulo(valor) -> Optional[int]:
    try:
        return int(valor)
    except (TypeError, ValueError):
        return None
