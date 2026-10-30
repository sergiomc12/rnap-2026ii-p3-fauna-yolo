"""Descarga de las imágenes de Caltech Camera Traps desde LILA BC.

Las 243,100 imágenes del conjunto ocupan 105 GB, pero solo 63,025 tienen
bounding box y únicamente esas sirven para entrenar un detector. Este script
baja las que aparecen en las particiones oficiales con caja: unas 48,000
imágenes, 20.5 GB.

La descarga va en paralelo porque está limitada por latencia, no por ancho de
banda: medido, 0.47 MB/s en secuencial contra 1.78 MB/s con doce hilos.

Es **reanudable**: una imagen ya presente en disco se salta. La descarga
completa supera las tres horas, más de lo que suele durar una sesión, así que
la reanudación no es una comodidad sino un requisito.

Usage:
    python pregunta3/paso1_datos/descargar.py
    python pregunta3/paso1_datos/descargar.py --particiones train cis_val
    python pregunta3/paso1_datos/descargar.py --limite 100
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional, Sequence

_RAIZ = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(_RAIZ)]

from pregunta3.paso1_datos.anotaciones import (  # noqa: E402
    PARTICIONES, archivos_necesarios, cargar_particion,
)

BASE_URL = ("https://lilawildlife.blob.core.windows.net/lila-wildlife/"
            "caltech-unzipped/cct_images/")
DESTINO = _RAIZ / "pregunta3/paso1_datos/images"
HILOS = 12
REINTENTOS = 3
TIEMPO_LIMITE = 60


@dataclass
class Progreso:
    """Contadores de una corrida, compartidos entre hilos."""

    total: int
    bajadas: int = 0
    saltadas: int = 0
    fallidas: int = 0
    bytes: int = 0
    comenzado: float = 0.0

    def __post_init__(self) -> None:
        self.comenzado = time.time()
        self._candado = threading.Lock()

    def registrar(self, *, bajada: bool = False, saltada: bool = False,
                  fallida: bool = False, bytes_: int = 0) -> int:
        with self._candado:
            self.bajadas += bajada
            self.saltadas += saltada
            self.fallidas += fallida
            self.bytes += bytes_
            return self.hechas

    @property
    def hechas(self) -> int:
        return self.bajadas + self.saltadas + self.fallidas

    @property
    def mb_por_segundo(self) -> float:
        transcurrido = time.time() - self.comenzado
        return self.bytes / 1e6 / transcurrido if transcurrido else 0.0

    @property
    def minutos_restantes(self) -> float:
        if not self.hechas or self.hechas >= self.total:
            return 0.0
        por_item = (time.time() - self.comenzado) / self.hechas
        return (self.total - self.hechas) * por_item / 60


def bajar_una(nombre: str, destino: pathlib.Path, progreso: Progreso) -> None:
    """Descarga una imagen, saltándola si ya está completa en disco."""
    final = destino / nombre
    if final.is_file() and final.stat().st_size > 0:
        progreso.registrar(saltada=True)
        return

    # Archivo temporal + rename: una interrupción no deja un JPEG truncado
    # que la próxima corrida daría por bueno.
    parcial = final.with_suffix(final.suffix + ".parcial")
    for intento in range(1, REINTENTOS + 1):
        try:
            with urllib.request.urlopen(BASE_URL + nombre,
                                        timeout=TIEMPO_LIMITE) as respuesta:
                datos = respuesta.read()
            parcial.write_bytes(datos)
            parcial.replace(final)
            progreso.registrar(bajada=True, bytes_=len(datos))
            return
        except Exception:
            if intento == REINTENTOS:
                parcial.unlink(missing_ok=True)
                progreso.registrar(fallida=True)
                return
            time.sleep(2 * intento)


def descargar(nombres: Sequence[str], destino: pathlib.Path,
              hilos: int = HILOS, cada: int = 250) -> Progreso:
    destino.mkdir(parents=True, exist_ok=True)
    progreso = Progreso(total=len(nombres))

    with ThreadPoolExecutor(max_workers=hilos) as ejecutor:
        futuros = [ejecutor.submit(bajar_una, n, destino, progreso)
                   for n in nombres]
        for indice, _ in enumerate(as_completed(futuros), start=1):
            if indice % cada == 0:
                print(f"  {indice:>6,}/{len(nombres):,} | "
                      f"{progreso.bytes/1e9:5.2f} GB | "
                      f"{progreso.mb_por_segundo:5.2f} MB/s | "
                      f"ETA {progreso.minutos_restantes:5.0f} min | "
                      f"fallidas={progreso.fallidas}", flush=True)
    return progreso


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particiones", nargs="*", default=list(PARTICIONES),
                        choices=list(PARTICIONES))
    parser.add_argument("--destino", type=pathlib.Path, default=DESTINO)
    parser.add_argument("--hilos", type=int, default=HILOS)
    parser.add_argument("--limite", type=int, help="bajar solo las primeras N")
    args = parser.parse_args(argv)

    particiones = [cargar_particion(n) for n in args.particiones]
    for p in particiones:
        print(f"  {p.nombre:11} {len(p.imagenes):>7,} img  "
              f"{len(p.con_cajas):>7,} con caja  {p.n_cajas:>7,} cajas  "
              f"{len(p.ubicaciones):>3} ubicaciones")

    nombres = archivos_necesarios(particiones)
    if args.limite:
        nombres = nombres[:args.limite]

    presentes = sum(1 for n in nombres if (args.destino / n).is_file())
    print(f"\n{len(nombres):,} imágenes únicas con caja | "
          f"{presentes:,} ya en disco | {len(nombres)-presentes:,} por bajar\n")

    progreso = descargar(nombres, args.destino, args.hilos)

    minutos = (time.time() - progreso.comenzado) / 60
    print(f"\nbajadas={progreso.bajadas:,} saltadas={progreso.saltadas:,} "
          f"fallidas={progreso.fallidas:,} | {progreso.bytes/1e9:.2f} GB | "
          f"{minutos:.1f} min")
    return 1 if progreso.fallidas else 0


if __name__ == "__main__":
    raise SystemExit(main())
