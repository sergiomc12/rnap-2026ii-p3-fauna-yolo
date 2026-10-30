"""Genera las tablas del informe a partir de los JSON de los pasos 3 a 6.

Transcribir números a mano al ``.tex`` es la forma más fácil de publicar un
resultado que no existe: basta una cifra copiada de una corrida anterior. Acá
cada tabla se construye leyendo el JSON que la produjo, y el ``.tex`` las
incluye con ``\\input``. Si un número del informe está mal, está mal en el JSON,
y eso se puede rastrear.

Usage:
    python pregunta3/paso7_informe/generar_tablas.py
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Optional, Sequence

_RAIZ = pathlib.Path(__file__).resolve().parents[2]

ENTRADAS = {
    "entrenamiento": "pregunta3/paso3_modelo/corridas/yolo11n_640/resumen.json",
    "metricas": "pregunta3/paso4_evaluacion/informes/metricas.json",
    "generalizacion": "pregunta3/paso5_generalizacion/informes/generalizacion.json",
    "robustez": "pregunta3/paso5_generalizacion/informes/robustez.json",
    "gradcam": "pregunta3/paso5_generalizacion/informes/gradcam.json",
    "actividad": "pregunta3/paso6_monitoreo/informes/actividad.json",
    "actividad_trans": "pregunta3/paso6_monitoreo/informes/actividad_trans_test.json",
}
SALIDA = _RAIZ / "pregunta3/paso7_informe/informe/tablas"


def _num(valor, decimales: int = 4, guion: str = "---") -> str:
    """Formatea un número, o un guión si es ``None``.

    El guión no es cosmético: distingue "no está definido" de cero, que es la
    distinción central del entregable #2.
    """
    if valor is None:
        return guion
    return f"{valor:.{decimales}f}"


def _entero(valor) -> str:
    return f"{valor:,}".replace(",", "\\,") if valor is not None else "---"


def _tt(texto) -> str:
    """Nombre en ancho fijo, con los caracteres especiales de LaTeX escapados.

    Los nombres de partición traen guion bajo (``cis_test``), que en LaTeX es
    el operador de subíndice y fuera de modo matemático aborta la compilación.
    Escaparlo acá, en el único lugar que emite nombres, evita tener que
    acordarse en cada tabla.
    """
    escapado = str(texto)
    for crudo, seguro in (("\\", "\\textbackslash{}"), ("_", "\\_"),
                          ("&", "\\&"), ("%", "\\%"), ("#", "\\#"),
                          ("$", "\\$"), ("{", "\\{"), ("}", "\\}")):
        escapado = escapado.replace(crudo, seguro)
    return f"\\texttt{{{escapado}}}"


# "chico" es coloquial en una etiqueta de tabla; en el texto se mantiene.
_NOMBRE_TRAMO = {"chico": "pequeño"}


def _rango_area(minimo: float, maximo: float) -> str:
    """Tramo de tamaño como porcentaje del área del cuadro.

    Los límites se guardan como fracción del área (0.002 = 0.2\\%). Mostrarlos
    como fracción con dos decimales colapsaba ``diminuto`` (0.000--0.00) y hacía
    que ``grande`` mostrara 1.01, un valor mayor que 1 que no es un área real
    sino el tope abierto del último tramo. Se emiten porcentajes y los tramos
    extremos quedan abiertos (``< 0.2\\%`` y ``> 5\\%``).
    """
    def pct(valor: float) -> str:
        return f"{valor * 100:.1f}".rstrip("0").rstrip(".")

    if minimo <= 0.0:
        return f"$<$ {pct(maximo)}\\%"
    if maximo > 1.0:
        return f"$>$ {pct(minimo)}\\%"
    return f"{pct(minimo)}--{pct(maximo)}\\%"


def _tabular(columnas: str, cabecera: Sequence[str],
             filas: Sequence[Sequence[str]], leyenda: str,
             etiqueta: str, tamano: str = "small") -> str:
    """``tamano`` permite \\footnotesize en las tablas largas.

    Las de desglose por especie tienen 9 columnas y una fila por especie;
    con \\small ocupan casi una pagina cada una y el cuerpo del informe se
    pasa del limite de 10 paginas. Es el tamano habitual para tablas densas
    y no afecta al texto.
    """
    lineas = [
        "\\begin{table}[t]", "  \\centering", f"  \\{tamano}",
        f"  \\begin{{tabular}}{{{columnas}}}", "    \\toprule",
        "    " + " & ".join(cabecera) + " \\\\", "    \\midrule",
    ]
    lineas += ["    " + " & ".join(f) + " \\\\" for f in filas]
    lineas += [
        "    \\bottomrule", "  \\end{tabular}",
        f"  \\caption{{{leyenda}}}", f"  \\label{{{etiqueta}}}",
        "\\end{table}", "",
    ]
    return "\n".join(lineas)


def tabla_por_especie(metricas: dict, conjunto: str) -> str:
    """Entregable #2: AP por especie con su soporte, ordenada por soporte."""
    datos = metricas[conjunto]
    filas = []
    for nombre, valores in sorted(datos["por_clase"].items(),
                                  key=lambda kv: -kv[1]["soporte"]):
        filas.append([
            _tt(nombre),
            _entero(valores["soporte"]),
            _entero(valores["predicciones"]),
            _num(valores["ap50"]),
            _num(valores["ap50_95"]),
        ])
    ausentes = datos["clases_sin_positivos"]
    nota = ""
    if ausentes:
        nombres = ", ".join(_tt(a) for a in ausentes)
        nota = (f" Las especies {nombres} no tienen ninguna caja anotada en "
                f"este conjunto: su AP es indefinida (---), no cero, y queda "
                f"fuera del promedio macro.")
    return _tabular(
        "lrrrr",
        ["Especie", "Cajas", "Detecciones", "AP@0.5", "AP@0.5:0.95"],
        filas,
        f"Desglose por especie en {_tt(conjunto)}. "
        f"mAP@0.5 macro {_num(datos['map50'])}, "
        f"ponderado por soporte {_num(datos['map50_ponderado_por_soporte'])}."
        + nota,
        f"tab:especies_{conjunto}")


def tabla_resumen_particiones(metricas: dict) -> str:
    """mAP macro y ponderado de cada partición, con su IC."""
    filas = []
    for conjunto, datos in metricas.items():
        bootstrap = datos.get("bootstrap") or {}
        intervalo = "---"
        if bootstrap.get("ic_bajo") is not None:
            intervalo = (f"[{_num(bootstrap['ic_bajo'])}, "
                         f"{_num(bootstrap['ic_alto'])}]")
        filas.append([
            _tt(conjunto),
            _entero(datos["imagenes"]), _entero(datos["cajas_anotadas"]),
            _num(datos["map50"]), intervalo,
            _num(datos["map50_95"]),
            _num(datos["map50_ponderado_por_soporte"]),
            _num(datos["spearman_soporte_vs_ap50"], 3),
        ])
    return _tabular(
        "lrrrcrrr",
        ["Partición", "Imág.", "Cajas", "mAP@0.5", "IC 95\\%",
         "mAP@0.5:0.95", "Ponderado", "$\\rho_s$"],
        filas,
        "Resumen por partición. $\\rho_s$ es la correlación de rangos de "
        "Spearman entre el soporte de cada especie y su AP@0.5. El intervalo "
        "sale de un bootstrap sobre imágenes, no sobre cajas: las cajas de una "
        "misma imagen no son independientes.",
        "tab:resumen")


def tabla_especies_comparada(metricas: dict, generalizacion: dict) -> str:
    """Una sola tabla con las dos particiones, en vez de dos tablas.

    La tabla por especie de cada partición y la de brechas comparten casi todas
    las columnas. Separarlas repetía el soporte y la AP de cada especie tres
    veces y empujaba el informe por encima del límite de páginas del enunciado.
    """
    cis = metricas.get("cis_test", {}).get("por_clase", {})
    trans = metricas.get("trans_test", {}).get("por_clase", {})

    filas = []
    for nombre, valores in sorted(cis.items(),
                                  key=lambda kv: -kv[1]["soporte"]):
        otra = trans.get(nombre, {})
        filas.append([
            _tt(nombre),
            _entero(valores["soporte"]), _entero(valores["predicciones"]),
            _num(valores["ap50"]), _num(valores["ap50_95"]),
            _entero(otra.get("soporte")), _entero(otra.get("predicciones")),
            _num(otra.get("ap50")), _num(otra.get("ap50_95")),
        ])

    ausentes = (list(metricas.get("cis_test", {}).get("clases_sin_positivos", ()))
                + list(metricas.get("trans_test", {})
                       .get("clases_sin_positivos", ())))
    nota = ""
    if ausentes:
        nota = (f" Las especies {', '.join(_tt(a) for a in set(ausentes))} no "
                f"tienen ninguna caja anotada en alguna de las dos "
                f"particiones: su AP es indefinida (---), no cero, y queda "
                f"fuera del promedio macro.")
    return _tabular(
        "lrrrr|rrrr",
        ["Especie", "Cajas", "Det.", "AP@0.5", "AP@.5:.95",
         "Cajas", "Det.", "AP@0.5", "AP@.5:.95"],
        filas,
        "Desglose por especie: a la izquierda ubicaciones vistas "
        "(\\texttt{cis\\_test}), a la derecha nunca vistas "
        "(\\texttt{trans\\_test}). `Det.' es cuántas detecciones emitió el "
        "modelo: un AP bajo con pocas detecciones es un problema de recall, y "
        "con muchas, de precisión. Esa distinción no se puede hacer con la AP "
        "sola, y se arreglan de manera distinta."
        + nota,
        "tab:especies", tamano="scriptsize")


def tabla_brechas(generalizacion: dict) -> str:
    """Entregable #3: caída de AP por especie, con el cambio de prevalencia."""
    filas = []
    for brecha in sorted(generalizacion["brechas"],
                         key=lambda b: (b["caida"] is None,
                                        -(b["caida"] or 0.0))):
        prevalencia = brecha.get("prevalencia")
        filas.append([
            _tt(brecha['nombre']),
            _num(brecha["ap_vista"]), _num(brecha["ap_nueva"]),
            _num(brecha["caida"]),
            _entero(brecha["soporte_vista"]), _entero(brecha["soporte_nueva"]),
            f"$\\times${prevalencia:.2f}" if prevalencia is not None else "---",
        ])
    return _tabular(
        "lrrrrrr",
        ["Especie", "AP cis", "AP trans", "Caída", "Cajas cis", "Cajas trans",
         "Prev."],
        filas,
        "Brecha de generalización por especie. La última columna es el cambio "
        "de prevalencia entre particiones: una caída de AP acompañada de un "
        "cambio grande de prevalencia no se puede atribuir solo al cambio de "
        "fondo.",
        "tab:brechas")


def tabla_ubicaciones(generalizacion: dict) -> str:
    """mAP de cada cámara, que la cifra por partición esconde."""
    filas = []
    for clave, titulo in (("vista", "cis"), ("nueva", "trans")):
        dispersion = generalizacion.get(f"dispersion_{clave}", {})
        if not dispersion.get("n"):
            continue
        filas.append([
            _tt(titulo), _entero(dispersion["n"]),
            _num(dispersion["media"]), _num(dispersion["minimo"]),
            _num(dispersion["maximo"]), _num(dispersion["rango"]),
            _num(dispersion["desviacion"]),
        ])
    return _tabular(
        "lrrrrrr",
        ["Tipo", "Cámaras", "Media", "Mín.", "Máx.", "Rango", "$\\sigma$"],
        filas,
        "Dispersión del mAP@0.5 entre ubicaciones de cámara. Solo se incluyen "
        "las cámaras con al menos 50 cajas anotadas: un mAP sobre veinte cajas "
        "es ruido y produciría una dispersión inventada.",
        "tab:ubicaciones")


def tabla_tamano(robustez: dict) -> str:
    """Entregable #4: mAP por tramo de tamaño del objeto."""
    filas = []
    for conjunto, tramos in robustez["tamano"].items():
        for tramo in tramos:
            marca = "" if tramo["cajas"] >= 50 else "$^{\\dagger}$"
            nombre = _NOMBRE_TRAMO.get(tramo["tramo"], tramo["tramo"])
            filas.append([
                _tt(conjunto), nombre + marca,
                _rango_area(tramo["minimo"], tramo["maximo"]),
                _entero(tramo["imagenes"]), _entero(tramo["cajas"]),
                _num(tramo["map50"]),
            ])
    return _tabular(
        "llcrrr",
        ["Partición", "Tramo", "Área del cuadro", "Imág.", "Cajas",
         "mAP@0.5"],
        filas,
        "Robustez ante el tamaño del objeto. Una imagen entra en un tramo solo "
        "si \\emph{todas} sus cajas caen en el rango; filtrando caja por caja, "
        "las descartadas seguirían en la imagen y cada detección correcta de un "
        "animal fuera del rango contaría como falso positivo. "
        "$^{\\dagger}$ menos de 50 cajas.",
        "tab:tamano")


def tabla_oclusion(robustez: dict) -> str:
    """Entregable #4: mAP por nivel de oclusión sintética."""
    oclusion = robustez.get("oclusion") or {}
    if not oclusion:
        return ""
    niveles = sorted({n for modo in oclusion.values() for n in modo},
                     key=float)
    filas = []
    for modo, valores in oclusion.items():
        fila = [_tt(modo)]
        for nivel in niveles:
            fila.append(_num((valores.get(nivel) or {}).get("map50")))
        filas.append(fila)
    cabecera = ["Modo"] + [f"{float(n):.0%}".replace("%", "\\%")
                           for n in niveles]
    return _tabular(
        "l" + "r" * len(niveles), cabecera, filas,
        f"Robustez ante oclusión sintética sobre "
        f"{_entero(robustez.get('oclusion_submuestra'))} imágenes fijas por "
        f"semilla. El parche tapa la fracción indicada del \\emph{{área}} de "
        f"cada caja anotada. \\texttt{{centro}} deja la silueta y quita la "
        f"textura; \\texttt{{inferior}} tapa desde el borde de abajo, como la "
        f"vegetación real, y rompe también la silueta.",
        "tab:oclusion")


def tabla_gradcam(gradcam: dict) -> str:
    """Entregable #3: razón de masa de activación, y el barrido de capas."""
    filas = []
    for particion in ("cis_test", "trans_test"):
        datos = gradcam.get(particion)
        if not datos:
            continue
        filas.append([
            _tt(particion), _entero(datos["imagenes"]),
            _num(datos["razon_media"], 2), _num(datos["razon_mediana"], 2),
            _num(datos["razon_desviacion"], 2),
            _entero(datos["sin_localizacion"]),
            _num(datos["razon_animales_chicos"], 2),
            _num(datos["razon_animales_grandes"], 2),
        ])
    return _tabular(
        "lrrrrrrr",
        ["Partición", "Imág.", "Media", "Mediana", "$\\sigma$",
         "$\\leq 1.0$", "Chicos", "Grandes"],
        filas,
        "Razón de masa de Grad-CAM: fracción de la activación que cae dentro "
        "de las cajas anotadas, dividida por el área que esas cajas ocupan. "
        "Vale 1.0 si el mapa es uniforme, es decir si el modelo no localiza "
        "nada. La columna $\\leq 1.0$ cuenta las imágenes en que el modelo no "
        "localiza mejor que el azar. `Chicos' son los animales que ocupan "
        "menos del 1\\% del cuadro.",
        "tab:gradcam")


def tabla_capas(gradcam: dict) -> str:
    """La medición que decidió qué capa del backbone se usa."""
    barrido = gradcam.get("barrido")
    if not barrido:
        return ""
    filas = []
    for capa, datos in sorted(barrido.items(), key=lambda kv: int(kv[0])):
        filas.append([
            capa, _tt(datos['bloque']),
            _num(datos["media"], 2), _num(datos["mediana"], 2),
            f"{datos['mapas_vacios']}/{datos['imagenes']}",
        ])
    return _tabular(
        "llrrc",
        ["Capa", "Bloque", "Razón media", "Mediana", "Mapas vacíos"],
        filas,
        "Barrido de capas del backbone para Grad-CAM. La convención es usar la "
        "última capa convolucional; acá esa capa tiene una rejilla de "
        "$20\\times20$ y produce mapas idénticamente nulos dentro del animal "
        "en buena parte de las imágenes. La capa elegida es la de mejor "
        "localización medida, no la convencional.",
        "tab:capas")


def tabla_robustez(robustez: dict) -> str:
    """Tamaño y oclusión en una sola tabla: las dos miden lo mismo.

    El entregable \\#4 pide dos barridos y las dos mitades caben juntas: una
    tabla por cada una gastaba dos flotantes y dos leyendas para decir lo
    mismo, y el informe tiene un tope de diez páginas.
    """
    filas = []
    for conjunto, tramos in robustez["tamano"].items():
        for tramo in tramos:
            marca = "" if tramo["cajas"] >= 50 else "$^{\\dagger}$"
            nombre = _NOMBRE_TRAMO.get(tramo["tramo"], tramo["tramo"])
            filas.append([
                "tamaño", _tt(conjunto),
                f"{nombre}{marca} "
                f"({_rango_area(tramo['minimo'], tramo['maximo'])})",
                _entero(tramo["cajas"]), _num(tramo["map50"]),
            ])

    oclusion = robustez.get("oclusion") or {}
    submuestra = _entero(robustez.get("oclusion_submuestra"))
    for modo, valores in oclusion.items():
        for nivel in sorted(valores, key=float):
            if float(nivel) == 0.0 and modo != list(oclusion)[0]:
                continue
            etiqueta = ("sin tapar" if float(nivel) == 0.0
                        else f"{modo}, {float(nivel):.0%}".replace("%", "\\%"))
            filas.append(["oclusión", _tt("cis_test"),
                          etiqueta, submuestra,
                          _num((valores.get(nivel) or {}).get("map50"))])

    return _tabular(
        "lllrr",
        ["Barrido", "Partición", "Nivel", "Cajas / imág.", "mAP@0.5"],
        filas,
        "Robustez ante tamaño del objeto y ante oclusión sintética. En el "
        "barrido de tamaño, una imagen entra en un tramo solo si "
        "\\emph{todas} sus cajas caen en el rango —filtrando caja por caja, "
        "las descartadas seguirían en la imagen y cada detección correcta de "
        "un animal fuera del rango contaría como falso positivo; "
        "$^{\\dagger}$ menos de 50 cajas. En el de oclusión, el parche tapa la "
        "fracción indicada del \\emph{área} de cada caja anotada sobre una "
        "submuestra fija por semilla: \\texttt{centro} deja la silueta y quita "
        "la textura, \\texttt{inferior} tapa desde el borde de abajo, como la "
        "vegetación real, y rompe también la silueta.",
        "tab:robustez", tamano="footnotesize")


def tabla_gradcam_completa(gradcam: dict) -> str:
    """Barrido de capas y razón de masa por partición, en una sola tabla."""
    filas = []
    barrido = gradcam.get("barrido") or {}
    for capa, datos in sorted(barrido.items(), key=lambda kv: int(kv[0])):
        filas.append([
            f"capa {capa} ({_tt(datos['bloque'])})",
            _entero(datos["imagenes"]),
            _num(datos["media"], 2), _num(datos["mediana"], 2),
            f"{datos['mapas_vacios']}/{datos['imagenes']}", "---", "---",
        ])
    for particion in ("cis_test", "trans_test"):
        datos = gradcam.get(particion)
        if not datos:
            continue
        filas.append([
            _tt(particion), _entero(datos["imagenes"]),
            _num(datos["razon_media"], 2), _num(datos["razon_mediana"], 2),
            f"{datos['sin_localizacion']}/{datos['imagenes']}",
            _num(datos["razon_animales_chicos"], 2),
            _num(datos["razon_animales_grandes"], 2),
        ])
    return _tabular(
        "lrrrcrr",
        ["Qué", "Imág.", "Media", "Mediana", "Sin localizar", "Chicos",
         "Grandes"],
        filas,
        "Razón de masa de Grad-CAM: fracción de la activación que cae dentro "
        "de las cajas anotadas, dividida por el área que esas cajas ocupan; "
        "vale 1.0 si el mapa es uniforme, es decir si el modelo no localiza "
        "nada. Arriba, el barrido que decidió qué capa del backbone se usa "
        "(`sin localizar' cuenta los mapas idénticamente nulos dentro del "
        "animal). Abajo, la razón por partición con la capa elegida (`sin "
        "localizar' cuenta las imágenes con razón $\\leq 1.0$). `Chicos' son "
        "los animales que ocupan menos del 1\\% del cuadro.",
        "tab:gradcam")


def tabla_actividad(actividad: dict, etiqueta: str = "tab:actividad",
                    tamano: str = "small") -> str:
    """Entregable #5: actividad por especie y error contra las anotaciones."""
    error = {f["especie"]: f for f in actividad["comparacion_con_anotaciones"]}
    filas = []
    for especie in actividad["especies"]:
        nombre = especie["especie"]
        comparacion = error.get(nombre, {})
        relativo = comparacion.get("error_relativo")
        filas.append([
            _tt(nombre),
            _entero(especie["eventos"]),
            _entero(comparacion.get("eventos_anotados")),
            f"{relativo:+.0%}".replace("%", "\\%") if relativo is not None else "---",
            _num(especie["tasa_por_100_noches"], 2),
            f"{especie['ocupacion_naive']:.0%}".replace("%", "\\%")
            if especie["ocupacion_naive"] is not None else "---",
            f"{especie['indice_nocturnidad']:.0%}".replace("%", "\\%")
            if especie["indice_nocturnidad"] is not None else "---",
        ])
    esfuerzo = actividad["esfuerzo"]
    return _tabular(
        "lrrrrrr",
        ["Especie", "Eventos", "Anotados", "Error", "Ev./100 n.",
         "Ocupación", "Nocturn."],
        filas,
        f"Reporte de actividad sobre {_tt(actividad['particion'])}: "
        f"{esfuerzo['camaras']} cámaras, "
        f"{_entero(esfuerzo['noches_camara'])} noches-cámara. Un evento "
        f"independiente agrupa detecciones de la misma especie y cámara "
        f"separadas por menos de "
        f"{actividad['parametros']['minutos_independencia']} minutos. "
        f"`Anotados' es el mismo reporte calculado desde las cajas anotadas a "
        f"mano, y `Error' la diferencia relativa, que valida el módulo.",
        etiqueta, tamano=tamano)


def tabla_actividad_comparada(cis: dict, trans: dict) -> str:
    """Entregable #5: actividad y error en las dos particiones, en una tabla.

    La demostración principal es sobre \\texttt{trans\\_test} (cámaras nunca
    vistas, que es el caso de despliegue); \\texttt{cis\\_test} queda al lado como
    el caso favorable. Juntar las dos en una tabla, como ya se hace con el
    desglose por especie, evita gastar dos flotantes y pasar el límite de
    páginas del enunciado.
    """
    def indice(act):
        err = {f["especie"]: f for f in act["comparacion_con_anotaciones"]}
        return {e["especie"]: (e, err.get(e["especie"], {}))
                for e in act["especies"]}

    icis, itrans = indice(cis), indice(trans)

    def celdas(par):
        if par is None:
            return ["---", "---", "---"]
        especie, comp = par
        rel = comp.get("error_relativo")
        noct = especie["indice_nocturnidad"]
        return [
            _entero(especie["eventos"]),
            f"{rel:+.0%}".replace("%", "\\%") if rel is not None else "---",
            f"{noct:.0%}".replace("%", "\\%") if noct is not None else "---",
        ]

    # Orden por eventos en trans_test (la partición principal), con las especies
    # que solo aparecen en cis al final.
    orden = sorted(itrans, key=lambda n: -itrans[n][0]["eventos"])
    orden += [n for n in sorted(icis, key=lambda n: -icis[n][0]["eventos"])
              if n not in itrans]

    filas = [[_tt(n)] + celdas(itrans.get(n)) + celdas(icis.get(n))
             for n in orden]

    ec, et = cis["esfuerzo"], trans["esfuerzo"]
    return _tabular(
        "lrrr|rrr",
        ["Especie", "Eventos", "Error", "Noct.", "Eventos", "Error", "Noct."],
        filas,
        f"Reporte de actividad por especie. A la izquierda la demostración "
        f"principal, cámaras nunca vistas (\\texttt{{trans\\_test}}: "
        f"{et['camaras']} cámaras, {_entero(et['noches_camara'])} "
        f"noches-cámara); a la derecha el caso favorable, cámaras vistas en el "
        f"entrenamiento (\\texttt{{cis\\_test}}: {ec['camaras']} cámaras, "
        f"{_entero(ec['noches_camara'])} noches-cámara). Un evento independiente "
        f"agrupa detecciones de la misma especie y cámara separadas por menos de "
        f"{trans['parametros']['minutos_independencia']} minutos. `Error' es la "
        f"diferencia relativa de los eventos del modelo contra el mismo reporte "
        f"calculado desde las cajas anotadas a mano, que valida el módulo. "
        f"`Noct.' es el índice de nocturnidad, un control "
        f"contra la historia natural conocida de cada especie.",
        "tab:actividad", tamano="scriptsize")


def tabla_actividad_franjas(actividad: dict, minimo_eventos: int = 300) -> str:
    """Entregable #5: reporte de actividad por periodo del día.

    El enunciado pide un reporte tabular por especie \\emph{y por periodo}. Las
    franjas son las de ``paso6_monitoreo/eventos.py``, y los conteos salen del
    campo ``por_franja`` del mismo JSON que produce el resto del reporte.
    """
    franjas = ("noche", "amanecer", "dia", "atardecer", "noche_tarde")
    titulos = {"noche": "Noche", "amanecer": "Amanecer", "dia": "Día",
               "atardecer": "Atardecer", "noche_tarde": "Noche tarde"}
    especies = [e for e in sorted(actividad["especies"],
                                  key=lambda e: -e["eventos"])
                if e["eventos"] >= minimo_eventos]
    cabecera = ["Especie"] + [titulos[f] for f in franjas]
    filas = [[_tt(e["especie"])]
             + [_entero(e.get("por_franja", {}).get(f, 0)) for f in franjas]
             for e in especies]
    # Tabla NO flotante (en línea): el cuerpo del informe ya usa siete flotantes
    # y agregar uno más empujaba el texto por encima del límite de diez páginas.
    # Al emitirla en el punto donde se la cita, no entra en la cola de flotantes.
    lineas = [
        "\\begin{center}", "  \\footnotesize",
        "  \\captionof{table}{Actividad por periodo del día en "
        f"{_tt(actividad['particion'])} (entregable \\#5): eventos "
        f"independientes por franja horaria, para las especies con al menos "
        f"{minimo_eventos} eventos. Franjas: noche (0--5\\,h), amanecer (5--8), "
        f"día (8--17), atardecer (17--20), noche tarde (20--24).}}",
        "  \\label{tab:actividad_franjas}",
        "  \\begin{tabular}{lrrrrr}", "    \\toprule",
        "    " + " & ".join(cabecera) + " \\\\", "    \\midrule",
    ]
    lineas += ["    " + " & ".join(f) + " \\\\" for f in filas]
    lineas += ["    \\bottomrule", "  \\end{tabular}", "\\end{center}", ""]
    return "\n".join(lineas)


def tabla_entrenamiento(entrenamiento: dict) -> str:
    """Entregable #1: las decisiones de entrenamiento, en una tabla."""
    config = entrenamiento["config"]
    filas = [
        ["Modelo base", _tt(config['modelo'])],
        ["Resolución de entrada", f"{config['resolucion']} px"],
        ["Épocas / lote", f"{config['epocas']} / {config['lote']}"],
        ["Paciencia", str(config["paciencia"])],
        ["Semilla", str(config["semilla"])],
        ["Volteo horizontal / vertical",
         f"{config['volteo_horizontal']} / {config['volteo_vertical']}"],
        ["Mosaic (apagado en las últimas)",
         f"{config['mosaic']} ({config['cerrar_mosaic']} épocas)"],
        ["Escala / traslación",
         f"$\\pm${config['escala']:.0%} / $\\pm${config['traslacion']:.0%}".replace("%", "\\%")],
        ["HSV (h/s/v)",
         f"{config['hsv_h']} / {config['hsv_s']} / {config['hsv_v']}"],
        ["Rotación / perspectiva",
         f"{config['grados']} / {config['perspectiva']}"],
    ]
    # Dos pares por fila: son diez pares nombre-valor, y en una sola columna
    # ocupaban un tercio de pagina para lo que es una lista. El informe tiene
    # un limite de 10 paginas de cuerpo.
    mitad = (len(filas) + 1) // 2
    emparejadas = [a + (filas[mitad + i] if mitad + i < len(filas) else ["", ""])
                   for i, a in enumerate(filas[:mitad])]
    return _tabular(
        "ll@{\\qquad}ll",
        ["Decisión", "Valor", "Decisión", "Valor"], emparejadas,
        "Decisiones de entrenamiento (entregable \\#1). El volteo vertical, la "
        "rotación y la perspectiva van en cero por la misma razón: la cámara "
        "está fija en un poste y esas vistas no existen en el despliegue. "
        "Escala y traslación son los valores por defecto de la "
        "implementación, declarados explícitamente porque están activos y "
        "pesan: la escala de $\\pm$50\\% es el aumento más relevante para el "
        "entregable \\#4.",
        "tab:entrenamiento", tamano="footnotesize")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--salida", type=pathlib.Path, default=SALIDA)
    args = parser.parse_args(argv)

    datos = {}
    faltan = []
    for clave, ruta in ENTRADAS.items():
        archivo = _RAIZ / ruta
        if archivo.is_file():
            datos[clave] = json.loads(archivo.read_text(encoding="utf-8"))
        else:
            faltan.append(ruta)

    if faltan:
        print("faltan estos informes (se omiten sus tablas):")
        for ruta in faltan:
            print(f"  {ruta}")

    # Las figuras de Grad-CAM se copian junto a las tablas: el .tex las
    # referencia con una ruta relativa corta y el informe queda autocontenido,
    # sin depender de la ubicación del paso 5.
    figuras_origen = _RAIZ / "pregunta3/paso5_generalizacion/figuras"
    figuras_destino = args.salida.parent / "figuras"
    if figuras_origen.is_dir():
        figuras_destino.mkdir(parents=True, exist_ok=True)
        copiadas = 0
        for figura in figuras_origen.glob("*.jpg"):
            destino = figuras_destino / figura.name
            if (not destino.is_file()
                    or destino.stat().st_mtime < figura.stat().st_mtime):
                destino.write_bytes(figura.read_bytes())
                copiadas += 1
        print(f"figuras copiadas: {copiadas}")

    args.salida.mkdir(parents=True, exist_ok=True)
    escritas = 0

    def escribir(nombre: str, contenido: str) -> None:
        nonlocal escritas
        if not contenido:
            return
        (args.salida / f"{nombre}.tex").write_text(contenido, encoding="utf-8")
        print(f"  {nombre}.tex")
        escritas += 1

    if "entrenamiento" in datos:
        escribir("entrenamiento", tabla_entrenamiento(datos["entrenamiento"]))
    if "metricas" in datos:
        escribir("resumen", tabla_resumen_particiones(datos["metricas"]))
        for conjunto in datos["metricas"]:
            escribir(f"especies_{conjunto}",
                     tabla_por_especie(datos["metricas"], conjunto))
    if "generalizacion" in datos:
        escribir("brechas", tabla_brechas(datos["generalizacion"]))
        escribir("ubicaciones", tabla_ubicaciones(datos["generalizacion"]))
    # La tabla que el informe usa de verdad: las dos particiones juntas. Las
    # individuales se siguen generando porque los README de los pasos 4 y 5 las
    # citan por separado.
    if "metricas" in datos and "generalizacion" in datos:
        escribir("especies", tabla_especies_comparada(datos["metricas"],
                                                      datos["generalizacion"]))
    if "robustez" in datos:
        escribir("tamano", tabla_tamano(datos["robustez"]))
        escribir("oclusion", tabla_oclusion(datos["robustez"]))
        escribir("robustez", tabla_robustez(datos["robustez"]))
    if "gradcam" in datos:
        escribir("gradcam_particiones", tabla_gradcam(datos["gradcam"]))
        escribir("capas", tabla_capas(datos["gradcam"]))
        escribir("gradcam", tabla_gradcam_completa(datos["gradcam"]))
    # Tablas por partición (las citan los README de los pasos) más la combinada,
    # que es la que el informe usa de verdad para no gastar dos flotantes.
    if "actividad" in datos:
        escribir("actividad", tabla_actividad(datos["actividad"],
                                              "tab:actividad_cis", tamano="footnotesize"))
    if "actividad_trans" in datos:
        escribir("actividad_trans",
                 tabla_actividad(datos["actividad_trans"],
                                 "tab:actividad_trans", tamano="footnotesize"))
        escribir("actividad_franjas",
                 tabla_actividad_franjas(datos["actividad_trans"]))
    if "actividad" in datos and "actividad_trans" in datos:
        escribir("actividad_comparada",
                 tabla_actividad_comparada(datos["actividad"],
                                           datos["actividad_trans"]))

    print(f"\n{escritas} tablas en {args.salida}")
    return 0 if escritas else 1


if __name__ == "__main__":
    raise SystemExit(main())
