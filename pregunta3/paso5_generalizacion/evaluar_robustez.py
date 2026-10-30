"""Entregable #4: barrido por tamaño del objeto y por oclusión.

El barrido por tamaño sale de las predicciones ya guardadas. El de oclusión
vuelve a correr el modelo, porque hay que modificar la imagen antes de que la
vea: se tapa con un parche gris una fracción del área de cada animal anotado y
se mide cuánto cae el mAP.

La oclusión se evalúa sobre una **submuestra** (1,500 imágenes por defecto) y
no sobre las 12,696 de ``cis_test``: cada nivel y cada modo es una pasada
completa del modelo, y con cuatro niveles por dos modos serían ocho pasadas.
La submuestra es fija por semilla, así que todos los niveles se comparan sobre
exactamente las mismas imágenes — que es lo que importa para que la caída sea
atribuible a la oclusión y no al cambio de muestra.

Usage:
    python pregunta3/paso5_generalizacion/evaluar_robustez.py
    python pregunta3/paso5_generalizacion/evaluar_robustez.py --solo-tamano
    python pregunta3/paso5_generalizacion/evaluar_robustez.py --muestras 300
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
import time
from typing import Optional, Sequence

_RAIZ = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(_RAIZ)]

from pregunta3.paso4_evaluacion.evaluar import a_muestras  # noqa: E402
from pregunta3.paso4_evaluacion.metricas import (  # noqa: E402
    Muestra, Prediccion, evaluar,
)
from pregunta3.paso4_evaluacion.predecir import (  # noqa: E402
    CONFIANZA_MINIMA, MAX_DETECCIONES, PESOS, SALIDA,
)
from pregunta3.paso5_generalizacion.robustez import (  # noqa: E402
    NIVELES_OCLUSION, TRAMOS_TAMANO, barrido_por_tamano, ocluir_cajas,
)

IMAGENES = _RAIZ / "pregunta3/paso1_datos/images"
INFORMES = _RAIZ / "pregunta3/paso5_generalizacion/informes"
MODOS = ("centro", "inferior")
LOTE = 16


def tabla_tamano(resultados) -> str:
    filas = ["| Tramo | Fracción del cuadro | Imágenes | Cajas | Especies | mAP@0.5 |",
             "|---|---|---:|---:|---:|---:|"]
    for r in resultados:
        valor = f"{r.map50:.4f}" if r.map50 is not None else "n/d"
        nota = "" if r.confiable else " ⚠"
        filas.append(f"| {r.tramo}{nota} | {r.minimo:.3%}–{r.maximo:.1%} | "
                     f"{r.imagenes:,} | {r.cajas:,} | {r.especies} | {valor} |")
    return "\n".join(filas)


def _predecir_ocluidas(modelo, muestras: Sequence[Muestra], fraccion: float,
                       modo: str, carpeta: pathlib.Path) -> list[Muestra]:
    """Predice sobre las mismas imágenes, con los animales tapados."""
    import cv2

    salida = []
    for inicio in range(0, len(muestras), LOTE):
        lote = muestras[inicio:inicio + LOTE]
        imagenes, validas = [], []
        for muestra in lote:
            arreglo = cv2.imread(str(carpeta / muestra.archivo))
            if arreglo is None:
                continue
            imagenes.append(ocluir_cajas(
                arreglo, [v.caja for v in muestra.verdades], fraccion, modo))
            validas.append(muestra)
        if not imagenes:
            continue

        resultados = modelo.predict(imagenes, conf=CONFIANZA_MINIMA,
                                    max_det=MAX_DETECCIONES, verbose=False)
        for muestra, resultado in zip(validas, resultados):
            cajas = resultado.boxes
            salida.append(Muestra(
                archivo=muestra.archivo, ancho=muestra.ancho, alto=muestra.alto,
                ubicacion=muestra.ubicacion, secuencia=muestra.secuencia,
                fecha=muestra.fecha,
                verdades=muestra.verdades,
                predicciones=tuple(
                    Prediccion(int(c), float(f), tuple(caja))
                    for c, f, caja in zip(cajas.cls.tolist(), cajas.conf.tolist(),
                                          cajas.xyxy.tolist())),
            ))
    return salida


def barrido_oclusion(muestras: Sequence[Muestra], nombres: Sequence[str],
                     pesos: pathlib.Path, carpeta: pathlib.Path,
                     niveles=NIVELES_OCLUSION, modos=MODOS) -> dict:
    """mAP@0.5 para cada combinación de nivel de oclusión y modo."""
    from ultralytics import YOLO

    modelo = YOLO(str(pesos))

    def medir(fraccion: float, modo: str) -> dict:
        comenzado = time.time()
        ocluidas = _predecir_ocluidas(modelo, muestras, fraccion, modo, carpeta)
        resultado = evaluar(ocluidas, nombres, completo=False)
        valor = resultado.map50 if resultado.map50 is not None else float("nan")
        print(f"  {modo:9} tapado {fraccion:>5.0%} -> mAP@0.5 {valor:.4f} "
              f"({(time.time()-comenzado)/60:.1f} min)", flush=True)
        return {"map50": resultado.map50, "imagenes": resultado.imagenes,
                "especies_presentes": len(resultado.presentes)}

    # El nivel 0.0 deja la imagen intacta, así que no depende del modo: se mide
    # una vez y se repite como línea base de todos los modos.
    base = medir(0.0, modos[0])
    salida: dict[str, dict] = {}
    for modo in modos:
        salida[modo] = {"0.00": base}
        for fraccion in (f for f in niveles if f > 0.0):
            salida[modo][f"{fraccion:.2f}"] = medir(fraccion, modo)
    return salida


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particion", default="cis_test")
    parser.add_argument("--tambien", default="trans_test",
                        help="segunda partición, solo para el barrido de tamaño")
    parser.add_argument("--predicciones", type=pathlib.Path, default=SALIDA)
    parser.add_argument("--imagenes", type=pathlib.Path, default=IMAGENES)
    parser.add_argument("--pesos", type=pathlib.Path, default=PESOS)
    parser.add_argument("--informes", type=pathlib.Path, default=INFORMES)
    parser.add_argument("--muestras", type=int, default=1500,
                        help="imágenes del barrido de oclusión")
    parser.add_argument("--semilla", type=int, default=42)
    parser.add_argument("--solo-tamano", action="store_true")
    args = parser.parse_args(argv)

    ruta = args.predicciones / f"{args.particion}.jsonl"
    if not ruta.is_file():
        print(f"falta {ruta}. Corré pregunta3/paso4_evaluacion/predecir.py")
        return 1

    nombres, muestras = a_muestras(ruta)
    informe: dict = {"particion": args.particion, "tramos": list(TRAMOS_TAMANO)}

    print(f"== tamaño del objeto ({args.particion})")
    por_tamano = barrido_por_tamano(muestras, nombres)
    for r in por_tamano:
        valor = f"{r.map50:.4f}" if r.map50 is not None else "n/d"
        print(f"  {r.tramo:9} {r.minimo:.3%}-{r.maximo:.1%} | "
              f"{r.imagenes:>6,} img | {r.cajas:>6,} cajas | mAP@0.5 {valor}"
              + ("" if r.confiable else "  (muestra chica)"))
    informe["tamano"] = {args.particion: [r.__dict__ for r in por_tamano]}
    tablas = [f"## {args.particion}\n\n{tabla_tamano(por_tamano)}"]

    otra = args.predicciones / f"{args.tambien}.jsonl"
    if args.tambien and otra.is_file():
        print(f"\n== tamaño del objeto ({args.tambien})")
        nombres_otra, otras = a_muestras(otra)
        por_tamano_otra = barrido_por_tamano(otras, nombres_otra)
        for r in por_tamano_otra:
            valor = f"{r.map50:.4f}" if r.map50 is not None else "n/d"
            print(f"  {r.tramo:9} {r.imagenes:>6,} img | {r.cajas:>6,} cajas | "
                  f"mAP@0.5 {valor}")
        informe["tamano"][args.tambien] = [r.__dict__ for r in por_tamano_otra]
        tablas.append(f"## {args.tambien}\n\n{tabla_tamano(por_tamano_otra)}")

    if not args.solo_tamano:
        azar = random.Random(args.semilla)
        con_cajas = [m for m in muestras if m.verdades]
        submuestra = azar.sample(con_cajas, min(args.muestras, len(con_cajas)))
        print(f"\n== oclusión sintética ({len(submuestra):,} imágenes, "
              f"semilla {args.semilla})")
        informe["oclusion"] = barrido_oclusion(
            submuestra, nombres, args.pesos, args.imagenes)
        informe["oclusion_submuestra"] = len(submuestra)

    args.informes.mkdir(parents=True, exist_ok=True)
    (args.informes / "robustez.json").write_text(
        json.dumps(informe, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.informes / "robustez_tamano.md").write_text(
        "# Robustez ante el tamaño del objeto\n\n" + "\n\n".join(tablas) + "\n",
        encoding="utf-8")
    print(f"\nescrito en {args.informes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
