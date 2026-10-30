"""Grad-CAM sobre el backbone: la parte interpretativa del entregable #3.

Produce tres cosas:

1. el **barrido de capas**, que justifica con una medición cuál capa del
   backbone se usa (``--barrido``);
2. la **razón de masa** en ubicaciones vistas contra ubicaciones nuevas: qué
   fracción de la activación cae dentro del animal, dividida por el área que el
   animal ocupa. Vale 1.0 si el mapa es uniforme, es decir si el modelo no
   localiza nada; cuanto más alto, más mira al animal y menos al fondo;
3. **figuras** de casos concretos — un acierto, un animal que no detectó y un
   falso positivo — en cada tipo de ubicación.

La razón de masa es lo que convierte "el mapa se ve bien" en un número
comparable entre particiones, que es lo que el entregable pide: si el modelo
memorizó el fondo de las cámaras de entrenamiento, su evidencia debería estar
**menos** sobre el animal en ``cis`` que en ``trans``.

Usage:
    python pregunta3/paso5_generalizacion/generar_gradcam.py
    python pregunta3/paso5_generalizacion/generar_gradcam.py --barrido
    python pregunta3/paso5_generalizacion/generar_gradcam.py --muestras 200
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import statistics
import sys
from dataclasses import dataclass
from typing import Optional, Sequence

_RAIZ = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(_RAIZ)]

from pregunta3.paso4_evaluacion.metricas import iou  # noqa: E402
from pregunta3.paso4_evaluacion.predecir import PESOS  # noqa: E402
from pregunta3.paso5_generalizacion.gradcam_yolo import (  # noqa: E402
    CAPA_BACKBONE, CAPAS_CANDIDATAS, GradCAMDeteccion, a_tensor, letterbox,
    mapa_en_imagen, masa_dentro_de_cajas, superponer,
)

DATASET = _RAIZ / "pregunta3/paso2_preprocesamiento/dataset"
ETIQUETAS = _RAIZ / "pregunta3/paso1_datos/labels"
FIGURAS = _RAIZ / "pregunta3/paso5_generalizacion/figuras"
INFORMES = _RAIZ / "pregunta3/paso5_generalizacion/informes"

RESOLUCION = 640
CONFIANZA_FIGURA = 0.25     # el umbral de uso real, no el de la métrica


@dataclass
class Caso:
    """Una imagen con su geometría, sus cajas anotadas y su clase."""

    ruta: pathlib.Path
    ancho: int
    alto: int
    cajas: tuple[tuple[float, float, float, float], ...]
    clases: tuple[int, ...]

    @property
    def area_relativa(self) -> float:
        total = sum((x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in self.cajas)
        return total / (self.ancho * self.alto) if self.ancho and self.alto else 0.0


def leer_caso(ruta: str, etiquetas: pathlib.Path) -> Optional[Caso]:
    """Lee una imagen y sus cajas anotadas, en píxeles ``xyxy``."""
    import cv2

    camino = pathlib.Path(ruta)
    imagen = cv2.imread(str(camino))
    if imagen is None:
        return None
    alto, ancho = imagen.shape[:2]

    archivo = etiquetas / (camino.stem + ".txt")
    if not archivo.is_file():
        return None

    cajas, clases = [], []
    for linea in archivo.read_text(encoding="utf-8").splitlines():
        partes = linea.split()
        if len(partes) != 5:
            continue
        clase = int(partes[0])
        cx, cy, w, h = (float(p) for p in partes[1:])
        cx, cy, w, h = cx * ancho, cy * alto, w * ancho, h * alto
        cajas.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
        clases.append(clase)
    if not cajas:
        return None
    return Caso(camino, ancho, alto, tuple(cajas), tuple(clases))


def _muestrear(particion: str, dataset: pathlib.Path, etiquetas: pathlib.Path,
               cantidad: int, semilla: int) -> list[Caso]:
    rutas = (dataset / f"{particion}.txt").read_text(encoding="utf-8").splitlines()
    random.Random(semilla).shuffle(rutas)
    casos = []
    for ruta in rutas:
        if len(casos) >= cantidad:
            break
        caso = leer_caso(ruta, etiquetas)
        if caso is not None:
            casos.append(caso)
    return casos


def razon_de_masa(cam: GradCAMDeteccion, caso: Caso,
                  resolucion: int = RESOLUCION, k: int = 10) -> Optional[float]:
    """Masa de activación dentro del animal, dividida por su área relativa."""
    import cv2

    imagen = cv2.imread(str(caso.ruta))
    if imagen is None:
        return None
    recuadro, escala, desplazamiento = letterbox(imagen, resolucion)
    tensor = a_tensor(recuadro)
    tensor.requires_grad_(True)

    resultado = cam.calcular(tensor, clase=caso.clases[0], k=k)
    mapa = mapa_en_imagen(resultado.mapa, resolucion, escala, desplazamiento,
                          caso.ancho, caso.alto)
    masa = masa_dentro_de_cajas(mapa, caso.cajas, caso.ancho, caso.alto)
    return masa / caso.area_relativa if caso.area_relativa else None


def barrido_de_capas(modelo, casos: Sequence[Caso],
                     capas=CAPAS_CANDIDATAS) -> dict:
    """Razón de masa de cada capa candidata, sobre las mismas imágenes."""
    salida = {}
    for capa in capas:
        razones, vacios = [], 0
        with GradCAMDeteccion(modelo, capa=capa) as cam:
            for caso in casos:
                razon = razon_de_masa(cam, caso)
                if razon is None:
                    continue
                razones.append(razon)
                vacios += razon <= 1e-6
        salida[str(capa)] = {
            "bloque": type(modelo.model.model[capa]).__name__,
            "media": statistics.mean(razones) if razones else None,
            "mediana": statistics.median(razones) if razones else None,
            "mapas_vacios": vacios, "imagenes": len(razones),
        }
        print(f"  capa {capa:2} ({salida[str(capa)]['bloque']:6}) "
              f"media {salida[str(capa)]['media']:.2f} "
              f"mediana {salida[str(capa)]['mediana']:.2f} "
              f"vacíos {vacios}/{len(razones)}", flush=True)
    return salida


def _clasificar_caso(modelo, caso: Caso) -> str:
    """¿La predicción del modelo es un acierto, un olvido o una invención?"""
    resultado = modelo.predict(str(caso.ruta), conf=CONFIANZA_FIGURA,
                               verbose=False)[0]
    cajas = resultado.boxes
    predicciones = list(zip(cajas.cls.tolist(), cajas.xyxy.tolist()))

    emparejadas = set()
    aciertos = 0
    for clase, caja in predicciones:
        for indice, (verdad, clase_verdad) in enumerate(zip(caso.cajas,
                                                            caso.clases)):
            if (indice not in emparejadas and int(clase) == clase_verdad
                    and iou(tuple(caja), verdad) >= 0.5):
                emparejadas.add(indice)
                aciertos += 1
                break

    if not predicciones:
        return "sin_deteccion"
    if aciertos == len(caso.cajas) and len(predicciones) == aciertos:
        return "acierto"
    if aciertos == 0:
        return "falso_positivo"
    return "parcial"


def guardar_figura(cam: GradCAMDeteccion, modelo, caso: Caso,
                   destino: pathlib.Path, resolucion: int = RESOLUCION) -> None:
    """Imagen original con las cajas anotadas, al lado del mapa de calor."""
    import cv2

    imagen = cv2.imread(str(caso.ruta))
    recuadro, escala, desplazamiento = letterbox(imagen, resolucion)
    tensor = a_tensor(recuadro)
    tensor.requires_grad_(True)
    resultado = cam.calcular(tensor, clase=caso.clases[0], k=10)
    mapa = mapa_en_imagen(resultado.mapa, resolucion, escala, desplazamiento,
                          caso.ancho, caso.alto)

    izquierda = imagen.copy()
    for (x1, y1, x2, y2), clase in zip(caso.cajas, caso.clases):
        cv2.rectangle(izquierda, (int(x1), int(y1)), (int(x2), int(y2)),
                      (0, 255, 0), 3)
        cv2.putText(izquierda, modelo.names[clase], (int(x1), max(20, int(y1) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    destino.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destino), cv2.hconcat([izquierda, superponer(imagen, mapa)]))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pesos", type=pathlib.Path, default=PESOS)
    parser.add_argument("--dataset", type=pathlib.Path, default=DATASET)
    parser.add_argument("--etiquetas", type=pathlib.Path, default=ETIQUETAS)
    parser.add_argument("--figuras", type=pathlib.Path, default=FIGURAS)
    parser.add_argument("--informes", type=pathlib.Path, default=INFORMES)
    parser.add_argument("--capa", type=int, default=CAPA_BACKBONE)
    parser.add_argument("--muestras", type=int, default=120,
                        help="imágenes por partición para la razón de masa")
    parser.add_argument("--semilla", type=int, default=42)
    parser.add_argument("--barrido", action="store_true",
                        help="repetir el barrido de capas del docstring")
    args = parser.parse_args(argv)

    if not args.pesos.is_file():
        print(f"faltan los pesos {args.pesos}")
        return 1

    from ultralytics import YOLO
    modelo = YOLO(str(args.pesos))

    informe: dict = {"capa": args.capa, "muestras_por_particion": args.muestras}

    casos = {p: _muestrear(p, args.dataset, args.etiquetas, args.muestras,
                           args.semilla)
             for p in ("cis_test", "trans_test")}

    if args.barrido:
        print("== barrido de capas (sobre cis_test)")
        informe["barrido"] = barrido_de_capas(modelo, casos["cis_test"][:30])

    print(f"\n== razón de masa con la capa {args.capa}")
    with GradCAMDeteccion(modelo, capa=args.capa) as cam:
        for particion, lote in casos.items():
            razones, chicos, grandes = [], [], []
            for caso in lote:
                razon = razon_de_masa(cam, caso)
                if razon is None:
                    continue
                razones.append(razon)
                (chicos if caso.area_relativa < 0.01 else grandes).append(razon)
            informe[particion] = {
                "imagenes": len(razones),
                "razon_media": statistics.mean(razones) if razones else None,
                "razon_mediana": statistics.median(razones) if razones else None,
                "razon_desviacion": (statistics.pstdev(razones)
                                     if len(razones) > 1 else None),
                "sin_localizacion": sum(1 for r in razones if r <= 1.0),
                "razon_animales_chicos": (statistics.mean(chicos)
                                          if chicos else None),
                "razon_animales_grandes": (statistics.mean(grandes)
                                           if grandes else None),
            }
            datos = informe[particion]
            print(f"  {particion:11} n={datos['imagenes']:>4} | "
                  f"media {datos['razon_media']:.2f} | "
                  f"mediana {datos['razon_mediana']:.2f} | "
                  f"sin localizar (<=1.0) {datos['sin_localizacion']}")
            print(f"              animal chico (<1% del cuadro) "
                  f"{datos['razon_animales_chicos'] or float('nan'):.2f} | "
                  f"animal grande "
                  f"{datos['razon_animales_grandes'] or float('nan'):.2f}")

    # --- figuras de casos concretos ----------------------------------------
    # La clasificación va FUERA del contexto de Grad-CAM: ``predict`` de
    # Ultralytics clona el modelo, y con los hooks puestos el clon arrastra los
    # tensores intermedios que el hook guardó, que no son hojas del grafo y no
    # se pueden copiar (RuntimeError de deepcopy).
    print("\n== figuras")
    elegidos: dict[str, dict[str, Caso]] = {}
    for particion, lote in casos.items():
        por_tipo: dict[str, Caso] = {}
        for caso in lote:
            por_tipo.setdefault(_clasificar_caso(modelo, caso), caso)
            if len(por_tipo) >= 4:
                break
        elegidos[particion] = por_tipo
        informe.setdefault("figuras", {})[particion] = {
            tipo: caso.ruta.name for tipo, caso in por_tipo.items()}

    with GradCAMDeteccion(modelo, capa=args.capa) as cam:
        for particion, por_tipo in elegidos.items():
            for tipo, caso in por_tipo.items():
                destino = args.figuras / f"{particion}_{tipo}.jpg"
                guardar_figura(cam, modelo, caso, destino)
                print(f"  {destino.name:34} {caso.ruta.name}")

    args.informes.mkdir(parents=True, exist_ok=True)
    (args.informes / "gradcam.json").write_text(
        json.dumps(informe, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nescrito en {args.informes} y {args.figuras}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
