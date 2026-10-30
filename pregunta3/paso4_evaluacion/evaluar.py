"""Entregable #2: mAP@0.5 y mAP@0.5:0.95 con desglose por especie.

Lee las predicciones que dejó ``predecir.py`` y produce, por partición:

* mAP@0.5 y mAP@0.5:0.95 macro, con intervalo de confianza por bootstrap
* la AP de **cada especie**, junto a su soporte y a cuántas detecciones emitió
  el modelo
* las especies con **cero cajas** en el conjunto, reportadas como indefinidas y
  excluidas del promedio
* la correlación de rangos entre soporte y AP, que es la forma de responder con
  un número la pregunta por "el efecto del desbalance"

El mAP macro pesa a ``fox`` (5 cajas de entrenamiento) igual que a ``opossum``
(2,514). Eso es deliberado: un promedio ponderado por frecuencia escondería el
desbalance justamente en la métrica que debe exponerlo. Ambos se reportan.

Usage:
    python pregunta3/paso4_evaluacion/evaluar.py
    python pregunta3/paso4_evaluacion/evaluar.py --particion cis_test
    python pregunta3/paso4_evaluacion/evaluar.py --sin-bootstrap
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Optional, Sequence

_RAIZ = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(_RAIZ)]

from pregunta3.paso4_evaluacion.metricas import (  # noqa: E402
    Muestra, Prediccion, ResultadoEvaluacion, Verdad, bootstrap_map,
    correlacion_spearman, evaluar,
)
from pregunta3.paso4_evaluacion.predecir import (  # noqa: E402
    SALIDA, cargar_predicciones,
)

INFORMES = _RAIZ / "pregunta3/paso4_evaluacion/informes"


def a_muestras(ruta: pathlib.Path) -> tuple[list[str], list[Muestra]]:
    """Convierte el JSONL de predicciones en objetos de ``metricas``."""
    cabecera, lineas = cargar_predicciones(ruta)
    muestras = [
        Muestra(
            archivo=linea["archivo"], ancho=linea["ancho"], alto=linea["alto"],
            ubicacion=linea.get("ubicacion"), secuencia=linea.get("secuencia"),
            fecha=linea.get("fecha"),
            predicciones=tuple(
                Prediccion(p["clase"], p["confianza"], tuple(p["caja"]))
                for p in linea["predicciones"]),
            verdades=tuple(Verdad(v["clase"], tuple(v["caja"]))
                           for v in linea["verdades"]),
        )
        for linea in lineas
    ]
    return cabecera["clases"], muestras


def map_ponderado(resultado: ResultadoEvaluacion) -> Optional[float]:
    """mAP pesado por soporte, para contrastarlo con el macro.

    La diferencia entre los dos **es** la medida del desbalance: si el
    ponderado es mucho más alto, el modelo acierta donde hay datos y falla
    donde no los hay.
    """
    presentes = resultado.presentes
    total = sum(c.soporte for c in presentes)
    if not total:
        return None
    return sum(c.ap50 * c.soporte for c in presentes) / total


def tabla_por_especie(resultado: ResultadoEvaluacion) -> str:
    """Tabla markdown ordenada por soporte descendente."""
    filas = ["| Especie | Cajas anotadas | Detecciones | AP@0.5 | AP@0.5:0.95 |",
             "|---|---:|---:|---:|---:|"]
    for clase in sorted(resultado.clases, key=lambda c: -c.soporte):
        ap50 = f"{clase.ap50:.4f}" if clase.ap50 is not None else "indefinida"
        ap = f"{clase.ap50_95:.4f}" if clase.ap50_95 is not None else "indefinida"
        filas.append(f"| {clase.nombre} | {clase.soporte:,} | "
                     f"{clase.predicciones:,} | {ap50} | {ap} |")
    return "\n".join(filas)


def analizar(ruta: pathlib.Path,
             repeticiones: int = 200) -> tuple[dict, ResultadoEvaluacion]:
    """Todo lo que el entregable #2 pide, para una partición.

    Devuelve el informe serializable y el ``ResultadoEvaluacion`` que lo
    produjo: la tabla por especie se arma con el segundo, para no repetir
    una evaluación de diez umbrales que ya se hizo.
    """
    nombres, muestras = a_muestras(ruta)
    conjunto = ruta.stem
    resultado = evaluar(muestras, nombres, conjunto=conjunto)

    presentes = resultado.presentes
    correlacion = correlacion_spearman(
        [float(c.soporte) for c in presentes],
        [float(c.ap50) for c in presentes],
    )

    informe = resultado.a_dict()
    informe["map50_ponderado_por_soporte"] = map_ponderado(resultado)
    informe["spearman_soporte_vs_ap50"] = correlacion
    informe["cajas_anotadas"] = sum(c.soporte for c in resultado.clases)
    informe["detecciones"] = sum(c.predicciones for c in resultado.clases)

    if repeticiones:
        informe["bootstrap"] = bootstrap_map(muestras, nombres,
                                             repeticiones=repeticiones)
    return informe, resultado


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particion", action="append", dest="particiones",
                        help="repetible; por defecto, todo lo predicho")
    parser.add_argument("--predicciones", type=pathlib.Path, default=SALIDA)
    parser.add_argument("--informes", type=pathlib.Path, default=INFORMES)
    parser.add_argument("--repeticiones", type=int, default=200)
    parser.add_argument("--sin-bootstrap", action="store_true",
                        help="omite los intervalos de confianza (más rápido)")
    args = parser.parse_args(argv)

    rutas = ([args.predicciones / f"{p}.jsonl" for p in args.particiones]
             if args.particiones
             else sorted(args.predicciones.glob("*.jsonl")))
    rutas = [r for r in rutas if r.is_file()]
    if not rutas:
        print(f"no hay predicciones en {args.predicciones}. Corré primero "
              f"pregunta3/paso4_evaluacion/predecir.py")
        return 1

    args.informes.mkdir(parents=True, exist_ok=True)
    todo = {}
    for ruta in rutas:
        print(f"\n== {ruta.stem}")
        informe, resultado = analizar(
            ruta, 0 if args.sin_bootstrap else args.repeticiones)
        todo[ruta.stem] = informe

        print(f"  imágenes {informe['imagenes']:,} | "
              f"cajas {informe['cajas_anotadas']:,} | "
              f"detecciones {informe['detecciones']:,}")
        print(f"  mAP@0.5      {informe['map50']:.4f} (macro, "
              f"{len(informe['por_clase']) - len(informe['clases_sin_positivos'])} "
              f"especies presentes)")
        print(f"  mAP@0.5:0.95 {informe['map50_95']:.4f}")
        print(f"  mAP@0.5 ponderado por soporte "
              f"{informe['map50_ponderado_por_soporte']:.4f}")
        if informe["clases_sin_positivos"]:
            print(f"  sin cajas en este conjunto (AP indefinida): "
                  f"{', '.join(informe['clases_sin_positivos'])}")
        if informe["spearman_soporte_vs_ap50"] is not None:
            print(f"  Spearman soporte vs AP@0.5: "
                  f"{informe['spearman_soporte_vs_ap50']:+.3f}")
        if "bootstrap" in informe:
            b = informe["bootstrap"]
            print(f"  IC95% del mAP@0.5: [{b['ic_bajo']:.4f}, {b['ic_alto']:.4f}]")

        (args.informes / f"{ruta.stem}.md").write_text(
            f"# {ruta.stem}\n\n" + tabla_por_especie(resultado) + "\n",
            encoding="utf-8")

    (args.informes / "metricas.json").write_text(
        json.dumps(todo, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nescrito en {args.informes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
