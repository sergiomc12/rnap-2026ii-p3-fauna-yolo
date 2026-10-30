"""Arma el conjunto en el formato que Ultralytics espera.

Ultralytics deduce la ruta de las etiquetas reemplazando ``/images/`` por
``/labels/`` en la ruta de cada imagen, así que esos dos nombres de carpeta son
obligatorios y deben estar al mismo nivel.

Las 46,000 imágenes NO se copian por partición. Viven una sola vez en
``images/`` y cada partición es un archivo de texto con las rutas que le
corresponden: duplicar 20 GB cinco veces no tendría sentido, y además una misma
imagen aparece en más de una partición.

Se generan varios ``data.yaml``, uno por experimento, porque el entregable #3
necesita evaluar el mismo modelo sobre ubicaciones vistas (``cis``) y nunca
vistas (``trans``) por separado.

Usage:
    python pregunta3/paso2_preprocesamiento/preparar.py
    python pregunta3/paso2_preprocesamiento/preparar.py --solo-presentes
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Optional, Sequence

_RAIZ = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(_RAIZ)]

from pregunta3.paso1_datos.anotaciones import cargar_todas  # noqa: E402
from pregunta3.paso2_preprocesamiento.conversion import (  # noqa: E402
    MapaClases, escribir_data_yaml, escribir_etiquetas, secuencias_de,
)

IMAGENES = _RAIZ / "pregunta3/paso1_datos/images"
ETIQUETAS = _RAIZ / "pregunta3/paso1_datos/labels"
DATASET = _RAIZ / "pregunta3/paso2_preprocesamiento/dataset"

# Cada yaml responde una pregunta distinta del enunciado.
EXPERIMENTOS = {
    "base": ("train", "cis_val",
             "entrenamiento y validación en las mismas ubicaciones"),
    "cis": ("train", "cis_test",
            "prueba en ubicaciones VISTAS durante el entrenamiento"),
    "trans": ("train", "trans_test",
              "prueba en ubicaciones NUNCA VISTAS — el experimento del entregable #3"),
    "trans_val": ("train", "trans_val",
                  "validación en ubicaciones nuevas (una sola cámara)"),
}


# Ultralytics deduce la ruta de cada etiqueta reemplazando el segmento
# "/images/" por "/labels/" en la ruta de la imagen. No es configurable, asi
# que ambas carpetas deben llamarse exactamente asi y estar al mismo nivel.
# Un primer intento con enlaces de directorio fallo en este sistema de
# archivos, y copiar 20 GB para renombrar una carpeta no tenia sentido.


def escribir_lista(ruta: pathlib.Path, imagenes: Sequence[str],
                   carpeta: pathlib.Path) -> int:
    """Un archivo de texto con una ruta de imagen por línea, **relativa**.

    Las líneas se escriben en la forma ``./<relativa>`` porque es el mecanismo
    que Ultralytics documenta para listas portables: al leer el ``.txt``
    sustituye el ``./`` inicial por el directorio del propio ``.txt``. Antes se
    escribían rutas **absolutas** (``D:\\Backend\\redesneuronales\\...``), que
    convertían el dataset en inservible en cualquier otra máquina — y el modo en
    que falla es el peor: Ultralytics no encuentra ninguna imagen y entrena
    sobre un conjunto vacío en vez de avisar.

    Si la carpeta de imágenes no está bajo el mismo árbol que la lista (por
    ejemplo si se pasa ``--imagenes`` a otro disco), no hay relativa posible y
    se cae a la ruta absoluta, que es lo correcto en ese caso.
    """
    base = ruta.parent.resolve()
    lineas = []
    for nombre in imagenes:
        destino = (carpeta / nombre).resolve()
        try:
            lineas.append("./" + destino.relative_to(base).as_posix())
        except ValueError:
            # Distinto disco o fuera del arbol: relative_to no aplica.
            try:
                import os
                lineas.append("./" + pathlib.PurePosixPath(
                    os.path.relpath(destino, base).replace("\\", "/")).as_posix())
            except ValueError:
                lineas.append(str(destino))
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    return len(lineas)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imagenes", type=pathlib.Path, default=IMAGENES)
    parser.add_argument("--dataset", type=pathlib.Path, default=DATASET)
    parser.add_argument("--etiquetas", type=pathlib.Path, default=ETIQUETAS)
    parser.add_argument("--solo-presentes", action="store_true",
                        help="incluir solo las imágenes ya descargadas")
    parser.add_argument("--permitir-fuga", action="store_true",
                        help="no filtrar las ráfagas compartidas con train "
                             "(solo para medir cuánto infla la métrica)")
    args = parser.parse_args(argv)

    particiones = cargar_todas()
    mapa = MapaClases.desde_particiones(list(particiones.values()))
    print(f"{mapa.n} clases (sin 'empty'), ordenadas por frecuencia:")
    print("  " + ", ".join(f"{i}:{n}" for i, n in enumerate(mapa.nombres)) + "\n")

    disponibles = None
    if args.solo_presentes:
        disponibles = {p.name for p in args.imagenes.glob("*.jpg")}
        print(f"imágenes presentes en disco: {len(disponibles):,}\n")

    carpeta_img = args.imagenes.resolve()
    etiquetas = args.etiquetas

    # Las camaras disparan rafagas de ~3 fotos casi identicas. Una rafaga
    # repartida entre train y evaluacion infla la metrica sin avisar: medido,
    # cis_val comparte 224 rafagas con train.
    prohibidas = frozenset() if args.permitir_fuga else secuencias_de(particiones["train"])
    if prohibidas:
        print(f"filtrando {len(prohibidas):,} rafagas de entrenamiento "
              f"de las particiones de evaluacion" + chr(10))

    print(f"{'partición':12} {'con caja':>9} {'cajas':>8} {'desc.clase':>11} "
          f"{'desc.geom':>10} {'por fuga':>9} {'sin imagen':>11}")
    print("-" * 78)

    resumenes = {}
    for nombre, particion in particiones.items():
        # El entrenamiento no se filtra contra si mismo.
        veto = frozenset() if nombre == "train" else prohibidas
        resumen = escribir_etiquetas(particion, mapa, etiquetas, disponibles, veto)
        resumenes[nombre] = resumen
        print(f"{nombre:12} {resumen.imagenes_con_caja:>9,} {resumen.cajas:>8,} "
              f"{resumen.descartadas_clase:>11,} {resumen.descartadas_geometria:>10,} "
              f"{resumen.por_fuga_de_rafaga:>9,} {resumen.sin_imagen_en_disco:>11,}")

        incluidas = [i.archivo for i in particion.con_cajas
                     if i.secuencia not in veto
                     and (disponibles is None or i.archivo in disponibles)]
        escribir_lista(args.dataset / f"{nombre}.txt", incluidas, carpeta_img)

    print()
    for clave, (train, evaluacion, descripcion) in EXPERIMENTOS.items():
        ruta = escribir_data_yaml(
            args.dataset / f"data_{clave}.yaml", mapa, args.dataset,
            train=f"{train}.txt", val=f"{evaluacion}.txt",
        )
        print(f"  {ruta.name:22} {descripcion}")

    (args.dataset / "resumen.json").write_text(json.dumps({
        "clases": list(mapa.nombres),
        "particiones": {k: v.__dict__ | {"tasa_descarte": v.tasa_descarte}
                        for k, v in resumenes.items()},
        "experimentos": {k: {"train": v[0], "val": v[1], "descripcion": v[2]}
                         for k, v in EXPERIMENTOS.items()},
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\ndataset en {args.dataset}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
