"""Entregable #5: reporte de actividad por especie y periodo.

Lee el volcado de predicciones del paso 4 y produce el reporte que usaría quien
monitorea la fauna de un sitio:

* eventos independientes por especie, con tasa por 100 noches-cámara y
  ocupación naive;
* actividad por hora del día y por mes;
* índice de nocturnidad, que se contrasta contra la historia natural conocida
  de cada especie como control de sanidad;
* el **mismo reporte calculado desde las cajas anotadas a mano**, y el error
  del modelo contra él, especie por especie.

Esa última parte es lo que separa un reporte de un adorno. Un módulo que
imprime números sin decir cuánto se equivoca no sirve para decidir nada, y en
este dominio el error no es uniforme: las especies con pocos datos de
entrenamiento son justamente las que el reporte va a subestimar.

Usage:
    python pregunta3/paso6_monitoreo/generar_reporte.py
    python pregunta3/paso6_monitoreo/generar_reporte.py --particion trans_test
    python pregunta3/paso6_monitoreo/generar_reporte.py --confianza 0.5
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Optional, Sequence

_RAIZ = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(_RAIZ)]

from pregunta3.paso4_evaluacion.predecir import (  # noqa: E402
    SALIDA, cargar_predicciones,
)
from pregunta3.paso6_monitoreo.eventos import (  # noqa: E402
    MINUTOS_INDEPENDENCIA, ActividadEspecie, actividad_por_especie,
    agrupar_en_eventos, avistamientos_desde,
)

INFORMES = _RAIZ / "pregunta3/paso6_monitoreo/informes"

# Historia natural conocida, para el control de sanidad. No se usa para
# corregir nada: solo para señalar cuándo el reporte contradice lo esperado.
NOCTURNIDAD_ESPERADA = {
    "opossum": "nocturna", "raccoon": "nocturna", "skunk": "nocturna",
    "bobcat": "crepuscular", "coyote": "crepuscular", "fox": "nocturna",
    "badger": "nocturna", "deer": "crepuscular", "rabbit": "crepuscular",
    "rodent": "nocturna", "squirrel": "diurna", "bird": "diurna",
    "cat": "variable", "dog": "variable", "car": "diurna",
}


def tabla_actividad(filas: Sequence[ActividadEspecie]) -> str:
    cabecera = ("| Especie | Eventos | Detecciones | Cámaras | "
                "Eventos/100 noches | Ocupación naive | Nocturnidad | "
                "Esperado |")
    lineas = [cabecera, "|---|---:|---:|---:|---:|---:|---:|---|"]
    for fila in filas:
        tasa = (f"{fila.tasa_por_100_noches:.2f}"
                if fila.tasa_por_100_noches is not None else "n/d")
        ocupacion = (f"{fila.ocupacion_naive:.0%}"
                     if fila.ocupacion_naive is not None else "n/d")
        nocturnidad = (f"{fila.indice_nocturnidad:.0%}"
                       if fila.indice_nocturnidad is not None else "n/d")
        lineas.append(
            f"| {fila.especie} | {fila.eventos:,} | {fila.detecciones:,} | "
            f"{fila.camaras} | {tasa} | {ocupacion} | {nocturnidad} | "
            f"{NOCTURNIDAD_ESPERADA.get(fila.especie, '—')} |")
    return "\n".join(lineas)


def tabla_horaria(filas: Sequence[ActividadEspecie]) -> str:
    """Eventos por hora del día, una fila por especie."""
    lineas = ["especie," + ",".join(f"h{h:02d}" for h in range(24))]
    for fila in filas:
        lineas.append(fila.especie + "," + ",".join(str(v) for v in fila.por_hora))
    return "\n".join(lineas)


def comparar_con_anotaciones(
    del_modelo: Sequence[ActividadEspecie],
    de_anotaciones: Sequence[ActividadEspecie],
) -> list[dict]:
    """Error del reporte del modelo contra el de las cajas anotadas."""
    por_especie = {f.especie: f for f in del_modelo}
    salida = []
    for referencia in de_anotaciones:
        estimado = por_especie.get(referencia.especie)
        eventos_modelo = estimado.eventos if estimado else 0
        error = eventos_modelo - referencia.eventos
        salida.append({
            "especie": referencia.especie,
            "eventos_anotados": referencia.eventos,
            "eventos_modelo": eventos_modelo,
            "error_absoluto": error,
            "error_relativo": (error / referencia.eventos
                               if referencia.eventos else None),
            "camaras_anotadas": referencia.camaras,
            "camaras_modelo": estimado.camaras if estimado else 0,
        })
    # Especies que el modelo reporta y la anotación no: falsos positivos puros.
    reales = {f.especie for f in de_anotaciones}
    for estimado in del_modelo:
        if estimado.especie not in reales:
            salida.append({
                "especie": estimado.especie,
                "eventos_anotados": 0,
                "eventos_modelo": estimado.eventos,
                "error_absoluto": estimado.eventos,
                "error_relativo": None,
                "camaras_anotadas": 0,
                "camaras_modelo": estimado.camaras,
            })
    salida.sort(key=lambda d: -abs(d["error_absoluto"]))
    return salida


def tabla_comparacion(comparacion: Sequence[dict]) -> str:
    lineas = ["| Especie | Eventos anotados | Eventos del modelo | Error | "
              "Error relativo |", "|---|---:|---:|---:|---:|"]
    for fila in comparacion:
        relativo = (f"{fila['error_relativo']:+.0%}"
                    if fila["error_relativo"] is not None else "n/d")
        lineas.append(f"| {fila['especie']} | {fila['eventos_anotados']:,} | "
                      f"{fila['eventos_modelo']:,} | "
                      f"{fila['error_absoluto']:+,} | {relativo} |")
    return "\n".join(lineas)


def _reporte(lineas, nombres, confianza: float, minutos: int, campo: str):
    avistamientos, esfuerzo, meta = avistamientos_desde(
        lineas, nombres, confianza_minima=confianza, campo=campo)
    eventos = agrupar_en_eventos(avistamientos, minutos=minutos)
    return actividad_por_especie(eventos, esfuerzo), esfuerzo, meta, len(avistamientos)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particion", default="cis_test")
    parser.add_argument("--predicciones", type=pathlib.Path, default=SALIDA)
    parser.add_argument("--informes", type=pathlib.Path, default=INFORMES)
    parser.add_argument("--confianza", type=float, default=0.25)
    parser.add_argument("--minutos", type=int, default=MINUTOS_INDEPENDENCIA,
                        help="intervalo de independencia entre eventos")
    args = parser.parse_args(argv)

    ruta = args.predicciones / f"{args.particion}.jsonl"
    if not ruta.is_file():
        print(f"falta {ruta}. Corré pregunta3/paso4_evaluacion/predecir.py")
        return 1

    cabecera, _ = cargar_predicciones(ruta)
    nombres = cabecera["clases"]

    # El generador se consume una sola vez: se materializa porque el reporte se
    # calcula dos veces, una desde las predicciones y otra desde las
    # anotaciones.
    _, generador = cargar_predicciones(ruta)
    lineas = list(generador)

    filas, esfuerzo, meta, avistados = _reporte(
        lineas, nombres, args.confianza, args.minutos, "predicciones")
    referencia, _, _, anotados = _reporte(
        lineas, nombres, 0.0, args.minutos, "verdades")

    print(f"{args.particion}: {meta['imagenes_con_fecha']:,} imágenes con fecha, "
          f"{meta['imagenes_sin_fecha_legible']:,} sin fecha legible")
    print(f"esfuerzo: {esfuerzo.camaras} cámaras, "
          f"{esfuerzo.noches_camara:,} noches-cámara")
    print(f"detecciones sobre {args.confianza:.2f}: {avistados:,} -> "
          f"{sum(f.eventos for f in filas):,} eventos independientes "
          f"(ventana de {args.minutos} min)")
    print(f"cajas anotadas: {anotados:,} -> "
          f"{sum(f.eventos for f in referencia):,} eventos\n")

    print(f"{'especie':10} {'eventos':>8} {'/100 noches':>12} "
          f"{'ocupación':>10} {'nocturn.':>9}  esperado")
    for fila in filas:
        print(f"{fila.especie:10} {fila.eventos:>8,} "
              f"{fila.tasa_por_100_noches or 0:>12.2f} "
              f"{fila.ocupacion_naive or 0:>9.0%} "
              f"{fila.indice_nocturnidad or 0:>8.0%}  "
              f"{NOCTURNIDAD_ESPERADA.get(fila.especie, '—')}")

    comparacion = comparar_con_anotaciones(filas, referencia)
    print("\nerror del reporte contra las cajas anotadas:")
    for fila in comparacion[:8]:
        relativo = (f"{fila['error_relativo']:+.0%}"
                    if fila["error_relativo"] is not None else "n/d")
        print(f"  {fila['especie']:10} anotados {fila['eventos_anotados']:>6,} | "
              f"modelo {fila['eventos_modelo']:>6,} | {relativo}")

    # El nombre del archivo lleva la partición salvo para cis_test, que
    # conserva el nombre histórico (actividad.json) del que dependen las tablas
    # del informe. Así el reporte sobre cámaras nunca vistas (trans_test) no
    # pisa el de cámaras vistas, y ejecutar_todo puede generar ambos.
    sufijo = "" if args.particion == "cis_test" else f"_{args.particion}"
    args.informes.mkdir(parents=True, exist_ok=True)
    (args.informes / f"actividad{sufijo}.json").write_text(json.dumps({
        "particion": args.particion,
        "parametros": {"confianza_minima": args.confianza,
                       "minutos_independencia": args.minutos},
        "cobertura": meta,
        "esfuerzo": {"camaras": esfuerzo.camaras,
                     "noches_camara": esfuerzo.noches_camara},
        "especies": [f.__dict__ | {
            "tasa_por_100_noches": f.tasa_por_100_noches,
            "ocupacion_naive": f.ocupacion_naive,
            "indice_nocturnidad": f.indice_nocturnidad,
            "primera": f.primera.isoformat() if f.primera else None,
            "ultima": f.ultima.isoformat() if f.ultima else None,
        } for f in filas],
        "comparacion_con_anotaciones": comparacion,
    }, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    (args.informes / f"actividad{sufijo}.md").write_text(
        f"# Actividad de fauna — {args.particion}\n\n"
        f"Confianza mínima {args.confianza:.2f}, eventos independientes con "
        f"ventana de {args.minutos} minutos. "
        f"{esfuerzo.camaras} cámaras, {esfuerzo.noches_camara:,} "
        f"noches-cámara.\n\n"
        f"## Actividad por especie\n\n{tabla_actividad(filas)}\n\n"
        f"## Error contra las cajas anotadas\n\n"
        f"{tabla_comparacion(comparacion)}\n",
        encoding="utf-8")

    (args.informes / f"actividad_por_hora{sufijo}.csv").write_text(
        tabla_horaria(filas) + "\n", encoding="utf-8")

    print(f"\nescrito en {args.informes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
