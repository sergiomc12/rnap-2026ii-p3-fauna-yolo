"""Contrasta las métricas propias contra las de Ultralytics, misma partición.

Reimplementar el mAP tiene una obligación: demostrar que el número coincide con
la implementación de referencia cuando se evalúa lo mismo. Sin esta
comprobación, cualquier diferencia entre particiones podría ser un error de la
reimplementación disfrazado de hallazgo.

No se espera coincidencia exacta y conviene saber por qué:

* Ultralytics interpola la curva precisión-recall en 101 puntos (el método de
  COCO); acá se integra el área con todos los puntos y envolvente monótona (el
  de VOC2010 en adelante). Las dos son aceptadas y difieren en el tercer
  decimal.
* Ultralytics omite del promedio las clases sin predicciones; acá una clase con
  cajas anotadas y sin ninguna detección cuenta 0.0, que es lo correcto —el
  modelo falló— y empuja el macro hacia abajo.
* Ultralytics escala las cajas al tamaño del lote con letterbox y compara en ese
  espacio; acá la comparación es en píxeles de la imagen original.

Usage:
    python pregunta3/paso4_evaluacion/contrastar_con_ultralytics.py
    python pregunta3/paso4_evaluacion/contrastar_con_ultralytics.py --particion trans_test
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Optional, Sequence

_RAIZ = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(_RAIZ)]

from pregunta3.paso4_evaluacion.evaluar import a_muestras  # noqa: E402
from pregunta3.paso4_evaluacion.metricas import evaluar  # noqa: E402
from pregunta3.paso4_evaluacion.predecir import PESOS, SALIDA  # noqa: E402

DATASET = _RAIZ / "pregunta3/paso2_preprocesamiento/dataset"
INFORMES = _RAIZ / "pregunta3/paso4_evaluacion/informes"

# Qué data.yaml evalúa cada partición (ver paso 2).
YAML_DE = {
    "cis_val": "data_base.yaml",
    "cis_test": "data_cis.yaml",
    "trans_val": "data_trans_val.yaml",
    "trans_test": "data_trans.yaml",
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particion", action="append", dest="particiones",
                        choices=sorted(YAML_DE))
    parser.add_argument("--pesos", type=pathlib.Path, default=PESOS)
    parser.add_argument("--predicciones", type=pathlib.Path, default=SALIDA)
    parser.add_argument("--dataset", type=pathlib.Path, default=DATASET)
    parser.add_argument("--informes", type=pathlib.Path, default=INFORMES)
    parser.add_argument("--resolucion", type=int, default=640)
    args = parser.parse_args(argv)

    particiones = args.particiones or ["cis_test", "trans_test"]
    from ultralytics import YOLO
    modelo = YOLO(str(args.pesos))

    comparacion = {}
    for particion in particiones:
        volcado = args.predicciones / f"{particion}.jsonl"
        if not volcado.is_file():
            print(f"falta {volcado}, se omite {particion}")
            continue

        nombres, muestras = a_muestras(volcado)
        propio = evaluar(muestras, nombres, conjunto=particion)

        referencia = modelo.val(
            data=str((args.dataset / YAML_DE[particion]).resolve()),
            imgsz=args.resolucion, verbose=False, plots=False)

        fila = {
            "imagenes": propio.imagenes,
            "propio_map50": propio.map50,
            "propio_map50_95": propio.map50_95,
            "ultralytics_map50": float(referencia.box.map50),
            "ultralytics_map50_95": float(referencia.box.map),
            "clases_propias": len(propio.presentes),
            "clases_ultralytics": len(referencia.box.ap_class_index),
            "clases_sin_positivos": list(propio.ausentes),
        }
        fila["diferencia_map50"] = fila["propio_map50"] - fila["ultralytics_map50"]
        fila["diferencia_map50_95"] = (fila["propio_map50_95"]
                                       - fila["ultralytics_map50_95"])

        # Las clases que el propio cuenta como 0.0 y Ultralytics no promedia:
        # tienen cajas anotadas pero el modelo no emitió ninguna detección.
        sin_detecciones = [c.nombre for c in propio.presentes
                           if c.predicciones == 0]
        fila["clases_con_cajas_y_sin_detecciones"] = sin_detecciones

        comparacion[particion] = fila
        print(f"\n== {particion} ({fila['imagenes']:,} imágenes)")
        print(f"  mAP@0.5       propio {fila['propio_map50']:.4f} | "
              f"ultralytics {fila['ultralytics_map50']:.4f} | "
              f"dif {fila['diferencia_map50']:+.4f}")
        print(f"  mAP@0.5:0.95  propio {fila['propio_map50_95']:.4f} | "
              f"ultralytics {fila['ultralytics_map50_95']:.4f} | "
              f"dif {fila['diferencia_map50_95']:+.4f}")
        print(f"  clases promediadas: propio {fila['clases_propias']} | "
              f"ultralytics {fila['clases_ultralytics']}")
        if sin_detecciones:
            print(f"  con cajas y sin ninguna detección (propio las cuenta "
                  f"0.0): {', '.join(sin_detecciones)}")

    if comparacion:
        args.informes.mkdir(parents=True, exist_ok=True)
        (args.informes / "contraste_ultralytics.json").write_text(
            json.dumps(comparacion, indent=2, ensure_ascii=False),
            encoding="utf-8")
        print(f"\nescrito en {args.informes / 'contraste_ultralytics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
