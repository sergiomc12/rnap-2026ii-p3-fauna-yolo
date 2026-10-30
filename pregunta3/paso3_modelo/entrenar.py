"""Ajuste fino de YOLO sobre Caltech Camera Traps.

Resuelve el entregable #1, que pide el modelo ajustado "con decisiones de
entrenamiento completamente documentadas (resolución de entrada, estrategia de
aumento de datos, manejo explícito de las imágenes vacías como clase de fondo)".
Cada una de esas tres decisiones está justificada abajo y expuesta como
argumento, de modo que la ablación pueda cambiarlas de a una.

Usage:
    python pregunta3/paso3_modelo/entrenar.py
    python pregunta3/paso3_modelo/entrenar.py --modelo yolo11s --epocas 80
    python pregunta3/paso3_modelo/entrenar.py --resolucion 960 --etiqueta alta_res
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from dataclasses import asdict, dataclass
from typing import Optional, Sequence

_RAIZ = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(_RAIZ)]

DATASET = _RAIZ / "pregunta3/paso2_preprocesamiento/dataset"
SALIDA = _RAIZ / "pregunta3/paso3_modelo/corridas"

# Las imágenes originales son 2048x1494. A 640 px, el estándar de YOLO, un
# animal lejano que ya ocupaba pocos píxeles queda diminuto: reducir más de
# tres veces es justamente lo que el entregable #4 pide analizar. Se deja como
# argumento para poder medir ese efecto en vez de suponerlo.
RESOLUCION = 640


@dataclass(frozen=True)
class ConfigEntrenamiento:
    """Las decisiones que el entregable #1 exige documentar."""

    modelo: str = "yolo11n.pt"
    epocas: int = 60
    resolucion: int = RESOLUCION
    lote: int = 16
    paciencia: int = 15
    semilla: int = 42

    # --- Aumento de datos ---------------------------------------------------
    # El volteo horizontal es seguro: una cámara trampa capta al animal desde
    # cualquier lado y la especie no cambia al reflejarla.
    volteo_horizontal: float = 0.5
    # El volteo VERTICAL se deja en cero a propósito: pondría a los animales
    # patas arriba, una vista que no existe en el despliegue real.
    volteo_vertical: float = 0.0
    # Mosaic combina cuatro imágenes en una. Ayuda con objetos chicos, que es
    # el caso dominante acá, pero destruye el contexto de fondo; como el fondo
    # es justamente lo que el modelo tiende a memorizar, en este problema eso
    # juega a favor.
    mosaic: float = 1.0
    # Variación de color moderada: las cámaras disparan de día, de noche con
    # infrarrojo, y con iluminación cambiante.
    hsv_h: float = 0.015
    hsv_s: float = 0.5
    hsv_v: float = 0.4
    # Mosaic se APAGA en las últimas diez épocas. Las imágenes compuestas no
    # se parecen a una foto real de cámara trampa; dejarlo prendido hasta el
    # final hace que el modelo termine ajustándose a una distribución que no es
    # la del despliegue.
    cerrar_mosaic: int = 10
    # Traslación y escala son los valores por defecto de Ultralytics, se dejan
    # explícitos porque están activos y el entregable #1 pide documentar el
    # aumento de datos completo, no solo lo que se cambió. La escala de ±50% es
    # el aumento que más importa acá: simula al animal más cerca o más lejos de
    # la cámara, que es la variable del entregable #4.
    traslacion: float = 0.1
    escala: float = 0.5
    # Rotación y perspectiva en cero: la cámara está fija en un poste, así que
    # el horizonte no gira. Inventarlo agregaría ruido sin realismo.
    grados: float = 0.0
    perspectiva: float = 0.0
    # NO se tocan ``erasing`` ni ``auto_augment``, que aparecen con 0.4 y
    # "randaugment" en el args.yaml de la corrida: Ultralytics los aplica solo
    # en clasificación. En detección no hacen nada, y darlos por buenos sería
    # atribuirle al modelo un aumento por oclusión que nunca recibió.

    def a_kwargs(self, data: str, proyecto: str, nombre: str) -> dict:
        """Traduce la configuración a los argumentos de Ultralytics."""
        return {
            "data": data,
            "epochs": self.epocas,
            "imgsz": self.resolucion,
            "batch": self.lote,
            "patience": self.paciencia,
            "seed": self.semilla,
            "project": proyecto,
            "name": nombre,
            "exist_ok": True,
            "fliplr": self.volteo_horizontal,
            "flipud": self.volteo_vertical,
            "mosaic": self.mosaic,
            "close_mosaic": self.cerrar_mosaic,
            "translate": self.traslacion,
            "scale": self.escala,
            "hsv_h": self.hsv_h,
            "hsv_s": self.hsv_s,
            "hsv_v": self.hsv_v,
            "degrees": self.grados,
            "perspective": self.perspectiva,
            "deterministic": True,
            "plots": True,
            "verbose": True,
        }


def resumen_metricas(metricas) -> dict:
    """Extrae las métricas que el entregable #2 pide, con desglose por clase."""
    caja = metricas.box
    por_clase = {}
    for indice, clase in enumerate(caja.ap_class_index):
        por_clase[metricas.names[int(clase)]] = {
            "map50": float(caja.ap50[indice]),
            "map50_95": float(caja.ap[indice]),
            "precision": float(caja.p[indice]),
            "recall": float(caja.r[indice]),
        }
    return {
        "map50": float(caja.map50),
        "map50_95": float(caja.map),
        "precision": float(caja.mp),
        "recall": float(caja.mr),
        "por_clase": por_clase,
    }


def _epocas_registradas(carpeta: pathlib.Path) -> Optional[int]:
    """Cuántas épocas quedaron escritas en results.csv (cabecera aparte)."""
    archivo = carpeta / "results.csv"
    if not archivo.is_file():
        return None
    lineas = [l for l in archivo.read_text(encoding="utf-8").splitlines()
              if l.strip()]
    return max(0, len(lineas) - 1)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modelo", default="yolo11n.pt")
    parser.add_argument("--data", type=pathlib.Path,
                        default=DATASET / "data_base.yaml")
    parser.add_argument("--epocas", type=int, default=60)
    parser.add_argument("--resolucion", type=int, default=RESOLUCION)
    parser.add_argument("--lote", type=int, default=16)
    parser.add_argument("--semilla", type=int, default=42)
    parser.add_argument("--sin-mosaic", action="store_true",
                        help="desactiva mosaic, para la ablación")
    parser.add_argument("--etiqueta", help="nombre de la corrida")
    parser.add_argument("--reanudar", action="store_true",
                        help="continuar desde last.pt de la misma corrida")
    parser.add_argument("--salida", type=pathlib.Path, default=SALIDA)
    args = parser.parse_args(argv)

    if not args.data.is_file():
        print(f"falta {args.data}. Corré primero "
              f"pregunta3/paso2_preprocesamiento/preparar.py")
        return 1

    config = ConfigEntrenamiento(
        modelo=args.modelo, epocas=args.epocas, resolucion=args.resolucion,
        lote=args.lote, semilla=args.semilla,
        mosaic=0.0 if args.sin_mosaic else 1.0,
    )
    nombre = args.etiqueta or f"{pathlib.Path(args.modelo).stem}_{config.resolucion}"

    from ultralytics import YOLO

    print(f"modelo={config.modelo} resolucion={config.resolucion} "
          f"epocas={config.epocas} lote={config.lote} mosaic={config.mosaic}")
    print(f"data={args.data}\n")

    # Reanudar desde last.pt. Ultralytics guarda los pesos y el estado del
    # optimizador en cada epoca, asi que un corte no pierde mas que la epoca
    # en curso; con 60 epocas de ~1.5 min, una corrida supera cualquier limite
    # de sesion razonable.
    ultimos = args.salida.resolve() / nombre / "weights" / "last.pt"
    reanudado = bool(args.reanudar and ultimos.is_file())
    if reanudado:
        print(f"reanudando desde {ultimos}")
        modelo = YOLO(str(ultimos))
        comenzado = time.time()
        modelo.train(resume=True)
    else:
        modelo = YOLO(config.modelo)
        comenzado = time.time()
        # La ruta del proyecto va ABSOLUTA: con una relativa, Ultralytics la
        # cuelga de su propio runs_dir y los resultados terminan en
        # runs/detect/<ruta>.
        modelo.train(**config.a_kwargs(str(args.data.resolve()),
                                       str(args.salida.resolve()), nombre))
    minutos = (time.time() - comenzado) / 60

    # La validación final usa los mejores pesos, no los de la última época.
    metricas = modelo.val(data=str(args.data.resolve()), imgsz=config.resolucion,
                          verbose=False)
    resumen = resumen_metricas(metricas)

    # save_dir es la carpeta que Ultralytics realmente uso, que puede diferir
    # de la pedida si ya existia una con ese nombre.
    carpeta = pathlib.Path(getattr(modelo.trainer, "save_dir", args.salida / nombre))
    carpeta.mkdir(parents=True, exist_ok=True)
    # El tiempo medido es el de ESTE proceso. En una corrida reanudada no es el
    # costo total del entrenamiento, y reportarlo como tal sería engañoso: el
    # tiempo acumulado no se puede reconstruir porque la columna ``time`` de
    # results.csv se reinicia en cada proceso.
    (carpeta / "resumen.json").write_text(json.dumps({
        "config": asdict(config),
        "data": str(args.data),
        "minutos_de_este_proceso": round(minutos, 1),
        "reanudado": reanudado,
        "epocas_en_results_csv": _epocas_registradas(carpeta),
        "metricas": resumen,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nmAP@0.5      {resumen['map50']:.4f}")
    print(f"mAP@0.5:0.95 {resumen['map50_95']:.4f}")
    print(f"{minutos:.1f} min de este proceso"
          + (" (corrida reanudada: no es el total)" if reanudado else "")
          + f" | guardado en {carpeta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
