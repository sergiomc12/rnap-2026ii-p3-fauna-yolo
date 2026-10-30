"""Corre el modelo sobre una partición y guarda las predicciones en disco.

Separar *predecir* de *medir* no es una formalidad. Las particiones de prueba
tienen 12,696 y 18,033 imágenes; una pasada completa del modelo toma varios
minutos, y los entregables #2, #3 y #4 piden medir **lo mismo** de maneras
distintas: por especie, por tipo de ubicación, por tamaño del objeto, con
oclusión sintética. Guardando las predicciones una vez, cada una de esas
preguntas se responde recalculando sobre una lista filtrada, en segundos.

El umbral de confianza es deliberadamente bajo (0.001). La AP integra la curva
precisión-recall completa: recortar las detecciones de baja confianza trunca la
curva por el lado del recall alto y **infla** la métrica.

Usage:
    python pregunta3/paso4_evaluacion/predecir.py --particion cis_test
    python pregunta3/paso4_evaluacion/predecir.py --particion trans_test
    python pregunta3/paso4_evaluacion/predecir.py --todas
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Iterator, Optional, Sequence

_RAIZ = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(_RAIZ)]

from pregunta3.paso1_datos.anotaciones import cargar_particion  # noqa: E402

DATASET = _RAIZ / "pregunta3/paso2_preprocesamiento/dataset"
ETIQUETAS = _RAIZ / "pregunta3/paso1_datos/labels"
PESOS = _RAIZ / "pregunta3/paso3_modelo/corridas/yolo11n_640/weights/best.pt"
SALIDA = _RAIZ / "pregunta3/paso4_evaluacion/predicciones"

# Umbral bajo a propósito: ver el docstring del módulo.
CONFIANZA_MINIMA = 0.001
MAX_DETECCIONES = 300
LOTE = 16


def leer_verdades(archivo: pathlib.Path, ancho: int,
                  alto: int) -> list[dict]:
    """Lee un .txt de YOLO y devuelve las cajas en píxeles ``xyxy``.

    Se parte de las etiquetas ya escritas por el paso 2, no de las anotaciones
    originales, para evaluar exactamente contra lo mismo que se entrenó: el
    paso 2 descarta la clase ``empty`` y las cajas degeneradas.
    """
    if not archivo.is_file():
        return []
    verdades = []
    for linea in archivo.read_text(encoding="utf-8").splitlines():
        partes = linea.split()
        if len(partes) != 5:
            continue
        clase = int(partes[0])
        cx, cy, ancho_n, alto_n = (float(p) for p in partes[1:])
        verdades.append({
            "clase": clase,
            "caja": [(cx - ancho_n / 2) * ancho, (cy - alto_n / 2) * alto,
                     (cx + ancho_n / 2) * ancho, (cy + alto_n / 2) * alto],
        })
    return verdades


def imagenes_de(particion: str, dataset: pathlib.Path) -> list[str]:
    """Las rutas que el paso 2 dejó en ``<particion>.txt``."""
    lista = dataset / f"{particion}.txt"
    if not lista.is_file():
        raise FileNotFoundError(
            f"falta {lista}. Corré primero "
            f"pregunta3/paso2_preprocesamiento/preparar.py")
    return [l for l in lista.read_text(encoding="utf-8").splitlines() if l.strip()]


def _metadatos(particion: str) -> dict[str, dict]:
    """Ubicación de cámara y ráfaga de cada imagen, por nombre de archivo.

    Tres campos que las etiquetas YOLO no tienen y que hacen falta después:

    * ``ubicacion`` separa ``cis`` de ``trans`` (paso 5);
    * ``secuencia`` identifica la ráfaga, que el paso 2 usa para detectar fuga
      y el paso 6 para no contar tres veces al mismo animal;
    * ``fecha`` es la marca de tiempo de la cámara, sin la cual el reporte de
      actividad por periodo del paso 6 no existe.
    """
    datos = cargar_particion(particion)
    return {i.archivo: {"ubicacion": i.ubicacion, "secuencia": i.secuencia,
                        "fecha": i.fecha, "ancho": i.ancho, "alto": i.alto}
            for i in datos.imagenes}


def predecir(particion: str, pesos: pathlib.Path, salida: pathlib.Path,
             dataset: pathlib.Path = DATASET,
             etiquetas: pathlib.Path = ETIQUETAS,
             resolucion: int = 640, limite: Optional[int] = None,
             cada: int = 500) -> pathlib.Path:
    """Guarda un JSONL con una línea por imagen: predicciones y verdades."""
    from ultralytics import YOLO

    rutas = imagenes_de(particion, dataset)
    if limite:
        rutas = rutas[:limite]
    meta = _metadatos(particion)

    salida.mkdir(parents=True, exist_ok=True)
    destino = salida / f"{particion}.jsonl"
    # Escritura atómica: una corrida interrumpida no debe dejar un JSONL a
    # medias que la próxima lectura tome por completo.
    parcial = destino.with_suffix(".jsonl.parcial")

    modelo = YOLO(str(pesos))
    nombres = [modelo.names[i] for i in sorted(modelo.names)]
    comenzado = time.time()
    escritas = 0

    with parcial.open("w", encoding="utf-8") as mano:
        mano.write(json.dumps({"_meta": {
            "particion": particion, "pesos": str(pesos),
            "clases": nombres, "resolucion": resolucion,
            "confianza_minima": CONFIANZA_MINIMA,
            "imagenes": len(rutas),
        }}, ensure_ascii=False) + "\n")

        for inicio in range(0, len(rutas), LOTE):
            lote = rutas[inicio:inicio + LOTE]
            resultados = modelo.predict(
                lote, imgsz=resolucion, conf=CONFIANZA_MINIMA,
                max_det=MAX_DETECCIONES, verbose=False, stream=False,
            )
            for ruta, resultado in zip(lote, resultados):
                nombre = pathlib.Path(ruta).name
                info = meta.get(nombre, {})
                alto, ancho = resultado.orig_shape
                cajas = resultado.boxes
                predicciones = [
                    {"clase": int(c), "confianza": round(float(f), 5),
                     "caja": [round(float(v), 2) for v in caja]}
                    for c, f, caja in zip(cajas.cls.tolist(),
                                          cajas.conf.tolist(),
                                          cajas.xyxy.tolist())
                ]
                mano.write(json.dumps({
                    "archivo": nombre,
                    "ancho": int(ancho), "alto": int(alto),
                    "ubicacion": info.get("ubicacion"),
                    "secuencia": info.get("secuencia"),
                    "fecha": info.get("fecha"),
                    "predicciones": predicciones,
                    "verdades": leer_verdades(
                        etiquetas / (pathlib.Path(nombre).stem + ".txt"),
                        int(ancho), int(alto)),
                }, ensure_ascii=False) + "\n")
                escritas += 1

            if escritas % cada < LOTE:
                transcurrido = time.time() - comenzado
                ritmo = escritas / transcurrido if transcurrido else 0
                print(f"  {escritas:>6,}/{len(rutas):,} | {ritmo:5.1f} img/s | "
                      f"ETA {(len(rutas)-escritas)/ritmo/60 if ritmo else 0:4.1f} min",
                      flush=True)

    parcial.replace(destino)
    print(f"{particion}: {escritas:,} imágenes en "
          f"{(time.time()-comenzado)/60:.1f} min -> {destino}")
    return destino


def cargar_predicciones(ruta: pathlib.Path) -> tuple[dict, Iterator[dict]]:
    """Lee el JSONL: devuelve la cabecera ``_meta`` y las líneas restantes."""
    lineas = ruta.read_text(encoding="utf-8").splitlines()
    cabecera = json.loads(lineas[0])["_meta"]
    return cabecera, (json.loads(l) for l in lineas[1:] if l.strip())


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particion", action="append", dest="particiones",
                        help="repetible; por defecto cis_test y trans_test")
    parser.add_argument("--todas", action="store_true",
                        help="las cinco particiones oficiales")
    parser.add_argument("--pesos", type=pathlib.Path, default=PESOS)
    parser.add_argument("--dataset", type=pathlib.Path, default=DATASET)
    parser.add_argument("--etiquetas", type=pathlib.Path, default=ETIQUETAS)
    parser.add_argument("--salida", type=pathlib.Path, default=SALIDA)
    parser.add_argument("--resolucion", type=int, default=640)
    parser.add_argument("--limite", type=int, help="solo las primeras N imágenes")
    args = parser.parse_args(argv)

    if not args.pesos.is_file():
        print(f"faltan los pesos {args.pesos}. Corré primero "
              f"pregunta3/paso3_modelo/entrenar.py")
        return 1

    if args.todas:
        particiones = ["train", "cis_val", "cis_test", "trans_val", "trans_test"]
    else:
        particiones = args.particiones or ["cis_test", "trans_test"]

    for particion in particiones:
        print(f"\n== {particion}")
        predecir(particion, args.pesos, args.salida, args.dataset,
                 args.etiquetas, args.resolucion, args.limite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
