"""Corre la Pregunta 3 de extremo a extremo, con un único comando.

El examen lo exige: "El código debe ejecutarse de extremo a extremo con un
único comando documentado en el README".

Cada etapa se salta si su artefacto ya existe, así que volver a correrlo solo
hace lo que falta. ``--rehacer`` fuerza todo. ``--rapido`` recorta las etapas
caras (pocas épocas, pocas imágenes) para verificar el circuito completo en
minutos en vez de horas.

La descarga de las 46,026 imágenes (20.5 GB, más de tres horas) es la primera
etapa y es reanudable: si se corta, volver a lanzar el comando sigue donde
quedó.

Usage:
    python pregunta3/paso7_informe/ejecutar_todo.py
    python pregunta3/paso7_informe/ejecutar_todo.py --rapido
    python pregunta3/paso7_informe/ejecutar_todo.py --desde 4
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Optional, Sequence

_RAIZ = pathlib.Path(__file__).resolve().parents[2]
PYTHON = sys.executable

CORRIDA = "pregunta3/paso3_modelo/corridas/yolo11n_640"


@dataclass(frozen=True)
class Etapa:
    """Una etapa del pipeline y cómo saber que ya corrió."""

    paso: int
    nombre: str
    script: str
    argumentos: tuple[str, ...] = ()
    artefacto: Optional[str] = None
    # Argumentos que reemplazan a los normales en modo rápido. ``None`` quiere
    # decir que la etapa es barata y corre igual.
    rapidos: Optional[tuple[str, ...]] = None

    @property
    def ya_corrio(self) -> bool:
        if self.artefacto is None:
            return False
        ruta = _RAIZ / self.artefacto
        if ruta.is_dir():
            return any(ruta.iterdir())
        return ruta.is_file()


ETAPAS = (
    Etapa(1, "Descargar las imágenes con bounding box (reanudable)",
          "pregunta3/paso1_datos/descargar.py",
          artefacto="pregunta3/paso1_datos/images",
          rapidos=("--limite", "300")),
    Etapa(2, "Convertir COCO a YOLO y armar los cuatro data.yaml",
          "pregunta3/paso2_preprocesamiento/preparar.py",
          artefacto="pregunta3/paso2_preprocesamiento/dataset/resumen.json",
          rapidos=("--solo-presentes",)),
    Etapa(3, "Ajustar YOLO11n a 640 px",
          "pregunta3/paso3_modelo/entrenar.py",
          ("--epocas", "60", "--etiqueta", "yolo11n_640"),
          f"{CORRIDA}/weights/best.pt",
          ("--epocas", "3", "--etiqueta", "yolo11n_640")),
    Etapa(4, "Predecir sobre cis_test y trans_test",
          "pregunta3/paso4_evaluacion/predecir.py",
          artefacto="pregunta3/paso4_evaluacion/predicciones/trans_test.jsonl",
          rapidos=("--limite", "300")),
    Etapa(4, "mAP por especie, con intervalos",
          "pregunta3/paso4_evaluacion/evaluar.py",
          artefacto="pregunta3/paso4_evaluacion/informes/metricas.json",
          rapidos=("--sin-bootstrap",)),
    Etapa(4, "Contrastar las métricas propias contra Ultralytics",
          "pregunta3/paso4_evaluacion/contrastar_con_ultralytics.py",
          artefacto="pregunta3/paso4_evaluacion/informes/contraste_ultralytics.json"),
    Etapa(5, "Generalización cis contra trans",
          "pregunta3/paso5_generalizacion/comparar_ubicaciones.py",
          artefacto="pregunta3/paso5_generalizacion/informes/generalizacion.json",
          rapidos=("--sin-bootstrap",)),
    Etapa(5, "Robustez ante tamaño y oclusión",
          "pregunta3/paso5_generalizacion/evaluar_robustez.py",
          artefacto="pregunta3/paso5_generalizacion/informes/robustez.json",
          rapidos=("--muestras", "100")),
    Etapa(5, "Grad-CAM sobre el backbone",
          "pregunta3/paso5_generalizacion/generar_gradcam.py",
          ("--barrido",),
          "pregunta3/paso5_generalizacion/informes/gradcam.json",
          ("--muestras", "20")),
    Etapa(6, "Reporte de actividad ecológica en cámaras vistas (cis_test)",
          "pregunta3/paso6_monitoreo/generar_reporte.py",
          artefacto="pregunta3/paso6_monitoreo/informes/actividad.json"),
    # El entregable #5 pide la demostración sobre una cámara NO vista en
    # entrenamiento: esta etapa corre el mismo módulo sobre trans_test.
    Etapa(6, "Reporte de actividad en cámaras nunca vistas (trans_test)",
          "pregunta3/paso6_monitoreo/generar_reporte.py",
          ("--particion", "trans_test"),
          artefacto="pregunta3/paso6_monitoreo/informes/actividad_trans_test.json"),
    # Las tablas SIEMPRE se regeneran: son la traducción de los JSON a LaTeX, y
    # si un informe quedó con números de una corrida anterior, el error es
    # invisible. Por eso no declara artefacto.
    Etapa(7, "Generar las tablas y figuras del informe desde los JSON",
          "pregunta3/paso7_informe/generar_tablas.py"),
    Etapa(7, "Compilar el informe NeurIPS",
          "pregunta3/paso7_informe/compilar_informe.py"),
    Etapa(7, "Exportar el informe a Word",
          "pregunta3/paso7_informe/exportar_word.py"),
)


def correr(etapa: Etapa, rapido: bool) -> int:
    argumentos = (etapa.rapidos if rapido and etapa.rapidos is not None
                  else etapa.argumentos)
    orden = [PYTHON, str(_RAIZ / etapa.script), *argumentos]
    return subprocess.run(orden, cwd=_RAIZ).returncode


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rehacer", action="store_true",
                        help="rehacer aunque el artefacto exista")
    parser.add_argument("--rapido", action="store_true",
                        help="recorta las etapas caras, para probar el circuito")
    parser.add_argument("--desde", type=int, default=1)
    parser.add_argument("--hasta", type=int, default=7)
    parser.add_argument("--listar", action="store_true",
                        help="mostrar las etapas y su estado, sin correr nada")
    args = parser.parse_args(argv)

    etapas = [e for e in ETAPAS if args.desde <= e.paso <= args.hasta]

    if args.listar:
        print(f"{'paso':>5}  {'estado':<12} etapa")
        print("-" * 78)
        for e in etapas:
            estado = "ya corrió" if e.ya_corrio else "pendiente"
            print(f"{e.paso:>5}  {estado:<12} {e.nombre}")
        return 0
    print(f"PREGUNTA 3 · {len(etapas)} etapas"
          + (" · modo rápido" if args.rapido else "") + "\n")

    comenzado = time.time()
    for indice, etapa in enumerate(etapas, start=1):
        encabezado = f"[{indice}/{len(etapas)}] paso {etapa.paso} — {etapa.nombre}"
        if etapa.ya_corrio and not args.rehacer:
            print(f"{encabezado}: ya existe, se salta")
            continue

        print(f"\n{'=' * 72}\n{encabezado}\n{'=' * 72}", flush=True)
        codigo = correr(etapa, args.rapido)
        if codigo != 0:
            print(f"\nFALLO en: {etapa.nombre} (código {codigo})")
            return codigo

    print(f"\n{'=' * 72}")
    print(f"PIPELINE COMPLETO en {(time.time() - comenzado) / 60:.1f} min")
    print("métricas  : pregunta3/paso4_evaluacion/informes/")
    print("robustez  : pregunta3/paso5_generalizacion/informes/")
    print("figuras   : pregunta3/paso5_generalizacion/figuras/")
    print("actividad : pregunta3/paso6_monitoreo/informes/")
    print("informe   : pregunta3/paso7_informe/informe/informe.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
