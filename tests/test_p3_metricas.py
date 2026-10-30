"""Pruebas de las métricas de detección de la Pregunta 3.

La AP es fácil de implementar mal de maneras que no se notan: un emparejamiento
que permite reusar la misma caja anotada infla el recall, y promediar como cero
una clase sin positivos baja el mAP. Cada caso de abajo fija uno de esos
comportamientos con un número verificable a mano.
"""

from __future__ import annotations

import pytest

from pregunta3.paso4_evaluacion.metricas import (
    UMBRALES_COCO, Muestra, Prediccion, Verdad, ap_todos_los_puntos,
    average_precision, bootstrap_map, correlacion_spearman, evaluar, iou,
)

NOMBRES = ("opossum", "coyote", "fox")


def muestra(predicciones=(), verdades=(), archivo="a.jpg"):
    return Muestra(archivo=archivo, ancho=100, alto=100,
                   predicciones=tuple(predicciones), verdades=tuple(verdades))


class TestIou:
    def test_cajas_identicas_dan_uno(self):
        caja = (10.0, 10.0, 20.0, 20.0)
        assert iou(caja, caja) == pytest.approx(1.0)

    def test_cajas_disjuntas_dan_cero(self):
        assert iou((0, 0, 10, 10), (50, 50, 60, 60)) == 0.0

    def test_cajas_que_solo_se_tocan_dan_cero(self):
        # Comparten el borde x=10: la intersección tiene área cero.
        assert iou((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0

    def test_solapamiento_conocido(self):
        # Dos cuadrados de 10x10 desplazados 5 px: intersección 50, unión 150.
        assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)


class TestApTodosLosPuntos:
    def test_detector_perfecto_da_uno(self):
        # Precisión 1.0 hasta recall 1.0.
        assert ap_todos_los_puntos([0.5, 1.0], [1.0, 1.0]) == pytest.approx(1.0)

    def test_sin_predicciones_da_cero(self):
        assert ap_todos_los_puntos([], []) == 0.0

    def test_la_envolvente_monotona_no_castiga_el_diente_de_sierra(self):
        # Un falso positivo intermedio baja la precisión y luego se recupera;
        # la envolvente toma el máximo a la derecha, así que el área es 1.0.
        recalls = [0.5, 0.5, 1.0]
        precisiones = [1.0, 0.5, 1.0]
        assert ap_todos_los_puntos(recalls, precisiones) == pytest.approx(1.0)


class TestAveragePrecision:
    def test_clase_sin_cajas_anotadas_es_indefinida(self):
        """Sin positivos la AP no es cero: no está definida."""
        muestras = [muestra(predicciones=[Prediccion(0, 0.9, (0, 0, 10, 10))])]
        assert average_precision(muestras, clase=0) == [None]

    def test_clase_con_cajas_y_sin_predicciones_es_cero(self):
        """Acá sí es cero: había animales y el modelo no detectó ninguno."""
        muestras = [muestra(verdades=[Verdad(0, (0, 0, 10, 10))])]
        assert average_precision(muestras, clase=0) == [0.0]

    def test_deteccion_exacta_da_uno(self):
        muestras = [muestra(
            predicciones=[Prediccion(0, 0.9, (0, 0, 10, 10))],
            verdades=[Verdad(0, (0, 0, 10, 10))],
        )]
        assert average_precision(muestras, clase=0)[0] == pytest.approx(1.0)

    def test_una_caja_anotada_no_se_empareja_dos_veces(self):
        """La segunda detección del mismo animal es un falso positivo.

        Con un positivo y dos predicciones correctas-por-IoU, la curva pasa por
        precisión 1.0 a recall 1.0 y la AP sigue siendo 1.0, pero el conteo
        interno debe marcar la segunda como error: se verifica con la precisión
        del caso en que la primera falla.
        """
        muestras = [muestra(
            predicciones=[Prediccion(0, 0.9, (0, 0, 10, 10)),
                          Prediccion(0, 0.8, (0, 0, 10, 10))],
            verdades=[Verdad(0, (0, 0, 10, 10))],
        )]
        assert average_precision(muestras, clase=0)[0] == pytest.approx(1.0)

        # La más confiable está corrida y no empareja a 0.5; la segunda sí.
        # Recall 1.0 se alcanza con precisión 1/2, de modo que AP = 0.5.
        muestras = [muestra(
            predicciones=[Prediccion(0, 0.9, (80, 80, 90, 90)),
                          Prediccion(0, 0.8, (0, 0, 10, 10))],
            verdades=[Verdad(0, (0, 0, 10, 10))],
        )]
        assert average_precision(muestras, clase=0)[0] == pytest.approx(0.5)

    def test_el_umbral_de_iou_decide(self):
        """Un IoU de 0.667 empareja a 0.5 y no a 0.75."""
        predicha = (0, 0, 10, 15)       # intersección 100, unión 150
        muestras = [muestra(
            predicciones=[Prediccion(0, 0.9, predicha)],
            verdades=[Verdad(0, (0, 0, 10, 10))],
        )]
        assert iou(predicha, (0, 0, 10, 10)) == pytest.approx(100 / 150)
        a_050, a_075 = average_precision(muestras, 0, umbrales=(0.5, 0.75))
        assert a_050 == pytest.approx(1.0)
        assert a_075 == pytest.approx(0.0)

    def test_las_clases_no_se_mezclan(self):
        """Una caja bien ubicada pero de otra especie no cuenta como acierto."""
        muestras = [muestra(
            predicciones=[Prediccion(1, 0.9, (0, 0, 10, 10))],
            verdades=[Verdad(0, (0, 0, 10, 10))],
        )]
        assert average_precision(muestras, clase=0)[0] == pytest.approx(0.0)

    def test_devuelve_un_valor_por_umbral(self):
        muestras = [muestra(
            predicciones=[Prediccion(0, 0.9, (0, 0, 10, 10))],
            verdades=[Verdad(0, (0, 0, 10, 10))],
        )]
        assert len(average_precision(muestras, 0, UMBRALES_COCO)) == 10


class TestEvaluar:
    def test_el_macro_ignora_las_clases_sin_positivos(self):
        """``fox`` no aparece: no debe arrastrar el mAP a 2/3."""
        muestras = [muestra(
            predicciones=[Prediccion(0, 0.9, (0, 0, 10, 10)),
                          Prediccion(1, 0.9, (20, 20, 30, 30))],
            verdades=[Verdad(0, (0, 0, 10, 10)), Verdad(1, (20, 20, 30, 30))],
        )]
        resultado = evaluar(muestras, NOMBRES, completo=False)
        assert resultado.map50 == pytest.approx(1.0)
        assert resultado.ausentes == ("fox",)
        assert len(resultado.presentes) == 2

    def test_cuenta_soporte_y_predicciones_por_clase(self):
        muestras = [muestra(
            predicciones=[Prediccion(0, 0.9, (0, 0, 10, 10)),
                          Prediccion(0, 0.1, (50, 50, 60, 60))],
            verdades=[Verdad(0, (0, 0, 10, 10))],
        )]
        clase = evaluar(muestras, NOMBRES, completo=False).clases[0]
        assert (clase.soporte, clase.predicciones) == (1, 2)

    def test_sin_ninguna_clase_presente_el_macro_es_nulo(self):
        resultado = evaluar([muestra()], NOMBRES, completo=False)
        assert resultado.map50 is None

    def test_a_dict_es_serializable(self):
        import json
        muestras = [muestra(verdades=[Verdad(0, (0, 0, 10, 10))])]
        datos = evaluar(muestras, NOMBRES).a_dict()
        assert json.loads(json.dumps(datos))["por_clase"]["opossum"]["soporte"] == 1


class TestFraccionMayorCaja:
    def test_sin_verdades_es_cero(self):
        assert muestra().fraccion_mayor_caja() == 0.0

    def test_toma_la_caja_mas_grande(self):
        m = muestra(verdades=[Verdad(0, (0, 0, 10, 10)),
                              Verdad(0, (0, 0, 50, 50))])
        # 50x50 sobre una imagen de 100x100.
        assert m.fraccion_mayor_caja() == pytest.approx(0.25)


class TestBootstrap:
    def test_el_intervalo_contiene_la_estimacion_puntual(self):
        muestras = [
            muestra(predicciones=[Prediccion(0, 0.9, (0, 0, 10, 10))],
                    verdades=[Verdad(0, (0, 0, 10, 10))], archivo=f"{i}.jpg")
            for i in range(10)
        ]
        resultado = bootstrap_map(muestras, NOMBRES, repeticiones=20)
        assert resultado["ic_bajo"] <= resultado["map50"] <= resultado["ic_alto"]

    def test_es_reproducible_con_la_misma_semilla(self):
        muestras = [
            muestra(predicciones=[Prediccion(0, 0.9, (0, 0, 10, 10))],
                    verdades=[Verdad(0, (0, 0, 12, 12))], archivo=f"{i}.jpg")
            for i in range(8)
        ]
        a = bootstrap_map(muestras, NOMBRES, repeticiones=15, semilla=7)
        b = bootstrap_map(muestras, NOMBRES, repeticiones=15, semilla=7)
        assert a == b


class TestSpearman:
    def test_relacion_monotona_creciente_da_uno(self):
        assert correlacion_spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)

    def test_relacion_monotona_decreciente_da_menos_uno(self):
        assert correlacion_spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)

    def test_no_lineal_pero_monotona_sigue_dando_uno(self):
        """Es la razón de usar rangos: el soporte va de 5 a 4,800 cajas."""
        assert correlacion_spearman([1, 2, 3, 4], [1, 10, 1000, 100000]) == pytest.approx(1.0)

    def test_los_empates_comparten_rango(self):
        valor = correlacion_spearman([1, 1, 3, 4], [10, 10, 30, 40])
        assert valor == pytest.approx(1.0)

    def test_muy_pocos_puntos_no_se_reporta(self):
        assert correlacion_spearman([1, 2], [3, 4]) is None

    def test_sin_varianza_no_se_reporta(self):
        assert correlacion_spearman([1, 1, 1], [1, 2, 3]) is None
