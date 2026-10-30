"""Entregable #3: cis contra trans, con el control de distribución.

Produce cuatro cosas:

1. mAP@0.5 y mAP@0.5:0.95 en ``cis_test`` (ubicaciones vistas) y en
   ``trans_test`` (ubicaciones nuevas), con intervalos por bootstrap;
2. la brecha por especie, junto al cambio de prevalencia de cada una;
3. la misma comparación **controlando la mezcla de especies**, para separar el
   efecto del fondo del efecto de la distribución;
4. el mAP de **cada ubicación de cámara** por separado, con su dispersión.

Usage:
    python pregunta3/paso5_generalizacion/comparar_ubicaciones.py
    python pregunta3/paso5_generalizacion/comparar_ubicaciones.py --sin-bootstrap
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
from pregunta3.paso4_evaluacion.metricas import (  # noqa: E402
    bootstrap_map, evaluar,
)
from pregunta3.paso4_evaluacion.predecir import SALIDA  # noqa: E402
from pregunta3.paso5_generalizacion.generalizacion import (  # noqa: E402
    brecha_por_clase, dispersion, distribucion_de, mapa_por_ubicacion,
    remuestrear_a_distribucion,
)

INFORMES = _RAIZ / "pregunta3/paso5_generalizacion/informes"


def tabla_brechas(brechas) -> str:
    filas = ["| Especie | AP cis | AP trans | Caída | Cajas cis | Cajas trans | Prevalencia |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    ordenadas = sorted(brechas,
                       key=lambda b: (b.caida is None, -(b.caida or 0.0)))
    for b in ordenadas:
        def fmt(v):
            return f"{v:.4f}" if v is not None else "n/d"
        prevalencia = (f"×{b.cambio_de_prevalencia:.2f}"
                       if b.cambio_de_prevalencia is not None else "n/d")
        filas.append(f"| {b.nombre} | {fmt(b.ap_vista)} | {fmt(b.ap_nueva)} | "
                     f"{fmt(b.caida)} | {b.soporte_vista:,} | "
                     f"{b.soporte_nueva:,} | {prevalencia} |")
    return "\n".join(filas)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vista", default="cis_test")
    parser.add_argument("--nueva", default="trans_test")
    parser.add_argument("--predicciones", type=pathlib.Path, default=SALIDA)
    parser.add_argument("--informes", type=pathlib.Path, default=INFORMES)
    parser.add_argument("--repeticiones", type=int, default=200)
    parser.add_argument("--sin-bootstrap", action="store_true")
    args = parser.parse_args(argv)

    rutas = {n: args.predicciones / f"{n}.jsonl" for n in (args.vista, args.nueva)}
    faltan = [str(r) for r in rutas.values() if not r.is_file()]
    if faltan:
        print("faltan predicciones: " + ", ".join(faltan))
        print("corré pregunta3/paso4_evaluacion/predecir.py --particion "
              + " --particion ".join(rutas))
        return 1

    nombres, vistas = a_muestras(rutas[args.vista])
    _, nuevas = a_muestras(rutas[args.nueva])

    resultado_vista = evaluar(vistas, nombres, conjunto=args.vista)
    resultado_nueva = evaluar(nuevas, nombres, conjunto=args.nueva)

    print(f"{args.vista:11} mAP@0.5 {resultado_vista.map50:.4f} | "
          f"mAP@0.5:0.95 {resultado_vista.map50_95:.4f} | "
          f"{resultado_vista.imagenes:,} img")
    print(f"{args.nueva:11} mAP@0.5 {resultado_nueva.map50:.4f} | "
          f"mAP@0.5:0.95 {resultado_nueva.map50_95:.4f} | "
          f"{resultado_nueva.imagenes:,} img")
    print(f"brecha absoluta en mAP@0.5: "
          f"{resultado_vista.map50 - resultado_nueva.map50:+.4f}")

    informe = {
        "vista": resultado_vista.a_dict(),
        "nueva": resultado_nueva.a_dict(),
        "brecha_map50": resultado_vista.map50 - resultado_nueva.map50,
        "brecha_map50_95": resultado_vista.map50_95 - resultado_nueva.map50_95,
    }

    if not args.sin_bootstrap:
        informe["bootstrap_vista"] = bootstrap_map(
            vistas, nombres, repeticiones=args.repeticiones)
        informe["bootstrap_nueva"] = bootstrap_map(
            nuevas, nombres, repeticiones=args.repeticiones)
        bv, bn = informe["bootstrap_vista"], informe["bootstrap_nueva"]
        print(f"  IC95% {args.vista}: [{bv['ic_bajo']:.4f}, {bv['ic_alto']:.4f}]")
        print(f"  IC95% {args.nueva}: [{bn['ic_bajo']:.4f}, {bn['ic_alto']:.4f}]")
        # Si los intervalos no se solapan, la brecha no es ruido de muestreo.
        informe["intervalos_disjuntos"] = (bv["ic_bajo"] > bn["ic_alto"]
                                           or bn["ic_bajo"] > bv["ic_alto"])

    brechas = brecha_por_clase(resultado_vista, resultado_nueva)
    informe["brechas"] = [b.__dict__ | {"caida": b.caida,
                                        "prevalencia": b.cambio_de_prevalencia}
                          for b in brechas]

    # --- Control de distribución -------------------------------------------
    # Se recorta la partición nueva para que su mezcla de especies imite la de
    # la vista. Lo que quede de brecha ya no se explica por la prevalencia.
    objetivo = distribucion_de(resultado_vista)
    recortada = remuestrear_a_distribucion(nuevas, objetivo)
    resultado_recortado = evaluar(recortada, nombres,
                                  conjunto=f"{args.nueva}_controlado")
    informe["controlado"] = resultado_recortado.a_dict()
    informe["brecha_map50_controlada"] = (resultado_vista.map50
                                          - resultado_recortado.map50)
    print(f"\ncontrolando la mezcla de especies "
          f"({resultado_recortado.imagenes:,} img de {len(nuevas):,}):")
    print(f"  mAP@0.5 {resultado_recortado.map50:.4f} | brecha "
          f"{informe['brecha_map50_controlada']:+.4f}")

    # --- Por ubicación de cámara -------------------------------------------
    for clave, muestras in (("vista", vistas), ("nueva", nuevas)):
        por_ubicacion = mapa_por_ubicacion(muestras, nombres)
        informe[f"ubicaciones_{clave}"] = por_ubicacion
        valores = [v["map50"] for v in por_ubicacion.values()
                   if v["muestra_suficiente"] and v["map50"] is not None]
        informe[f"dispersion_{clave}"] = dispersion(valores)
        d = informe[f"dispersion_{clave}"]
        if d["n"]:
            print(f"{clave:6} {d['n']} ubicaciones con muestra suficiente | "
                  f"mAP@0.5 de {d['minimo']:.4f} a {d['maximo']:.4f} "
                  f"(rango {d['rango']:.4f}, sigma {d['desviacion']:.4f})")

    args.informes.mkdir(parents=True, exist_ok=True)
    (args.informes / "generalizacion.json").write_text(
        json.dumps(informe, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.informes / "brechas_por_especie.md").write_text(
        f"# {args.vista} vs {args.nueva}\n\n" + tabla_brechas(brechas) + "\n",
        encoding="utf-8")
    print(f"\nescrito en {args.informes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
