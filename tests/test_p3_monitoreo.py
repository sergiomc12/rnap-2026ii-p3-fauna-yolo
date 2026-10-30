"""Pruebas del paso 6: eventos independientes y reporte de actividad.

El riesgo de este módulo no es que se caiga, es que entregue números creíbles y
equivocados. Una ráfaga de tres fotos contada como tres visitas triplica la
actividad reportada de una especie, y nada en la salida lo delataría.
"""

from __future__ import annotations

import datetime as dt

import pytest

from pregunta3.paso6_monitoreo.eventos import (
    Avistamiento, EsfuerzoMuestreo, actividad_por_especie, agrupar_en_eventos,
    avistamientos_desde, leer_fecha,
)
from pregunta3.paso6_monitoreo.generar_reporte import comparar_con_anotaciones

NOMBRES = ("opossum", "coyote", "fox")


def avistamiento(especie="opossum", momento="2012-03-10 22:00:00",
                 ubicacion=1, confianza=0.9):
    return Avistamiento(especie=especie, fecha=leer_fecha(momento),
                        ubicacion=ubicacion, confianza=confianza)


class TestLeerFecha:
    def test_formato_de_cct(self):
        assert leer_fecha("2012-03-10 14:22:31") == dt.datetime(2012, 3, 10, 14, 22, 31)

    def test_formato_iso(self):
        assert leer_fecha("2012-03-10T14:22:31") == dt.datetime(2012, 3, 10, 14, 22, 31)

    def test_solo_fecha(self):
        assert leer_fecha("2012-03-10") == dt.datetime(2012, 3, 10, 0, 0)

    @pytest.mark.parametrize("valor", [None, "", "no es una fecha", "0000-00-00"])
    def test_lo_ilegible_se_descarta_en_vez_de_inventarse(self, valor):
        assert leer_fecha(valor) is None


class TestFranjas:
    @pytest.mark.parametrize("hora,franja", [
        (2, "noche"), (6, "amanecer"), (12, "dia"),
        (18, "atardecer"), (22, "noche_tarde"),
    ])
    def test_asigna_la_franja(self, hora, franja):
        a = avistamiento(momento=f"2012-03-10 {hora:02d}:00:00")
        assert a.franja == franja
        assert a.hora == hora


class TestAgruparEnEventos:
    def test_una_rafaga_es_un_solo_evento(self):
        """Tres fotos en seis segundos son una visita, no tres."""
        rafaga = [avistamiento(momento=f"2012-03-10 22:00:0{s}")
                  for s in (0, 3, 6)]
        eventos = agrupar_en_eventos(rafaga)
        assert len(eventos) == 1
        assert eventos[0].detecciones == 3

    def test_separadas_por_mas_que_la_ventana_son_dos_eventos(self):
        lejanas = [avistamiento(momento="2012-03-10 22:00:00"),
                   avistamiento(momento="2012-03-10 23:00:00")]
        assert len(agrupar_en_eventos(lejanas, minutos=30)) == 2

    def test_justo_dentro_de_la_ventana_sigue_siendo_un_evento(self):
        cercanas = [avistamiento(momento="2012-03-10 22:00:00"),
                    avistamiento(momento="2012-03-10 22:30:00")]
        assert len(agrupar_en_eventos(cercanas, minutos=30)) == 1

    def test_la_ventana_se_mide_contra_la_ultima_deteccion(self):
        """Un animal que se queda es un evento, aunque dure horas."""
        quedandose = [avistamiento(momento="2012-03-10 20:00:00"),
                      avistamiento(momento="2012-03-10 20:20:00"),
                      avistamiento(momento="2012-03-10 20:40:00"),
                      avistamiento(momento="2012-03-10 21:00:00")]
        eventos = agrupar_en_eventos(quedandose, minutos=30)
        assert len(eventos) == 1
        assert eventos[0].minutos == pytest.approx(60.0)

    def test_especies_distintas_no_se_mezclan(self):
        mezcla = [avistamiento(especie="opossum", momento="2012-03-10 22:00:00"),
                  avistamiento(especie="coyote", momento="2012-03-10 22:00:10")]
        assert len(agrupar_en_eventos(mezcla)) == 2

    def test_camaras_distintas_no_se_mezclan(self):
        """La misma especie a la misma hora en dos cámaras son dos eventos."""
        mezcla = [avistamiento(ubicacion=1, momento="2012-03-10 22:00:00"),
                  avistamiento(ubicacion=2, momento="2012-03-10 22:00:10")]
        assert len(agrupar_en_eventos(mezcla)) == 2

    def test_guarda_la_confianza_maxima_del_evento(self):
        grupo = [avistamiento(confianza=0.4, momento="2012-03-10 22:00:00"),
                 avistamiento(confianza=0.95, momento="2012-03-10 22:00:05")]
        assert agrupar_en_eventos(grupo)[0].confianza_maxima == pytest.approx(0.95)

    def test_sin_avistamientos_no_hay_eventos(self):
        assert agrupar_en_eventos([]) == []


class TestEsfuerzoMuestreo:
    def test_el_mismo_dia_cuenta_una_vez(self):
        esfuerzo = EsfuerzoMuestreo()
        for hora in (8, 14, 23):
            esfuerzo.registrar(1, dt.datetime(2012, 3, 10, hora))
        assert esfuerzo.noches_camara == 1
        assert esfuerzo.camaras == 1

    def test_suma_dias_y_camaras(self):
        esfuerzo = EsfuerzoMuestreo()
        esfuerzo.registrar(1, dt.datetime(2012, 3, 10))
        esfuerzo.registrar(1, dt.datetime(2012, 3, 11))
        esfuerzo.registrar(2, dt.datetime(2012, 3, 10))
        assert (esfuerzo.noches_camara, esfuerzo.camaras) == (3, 2)
        assert esfuerzo.noches_de(2) == 1

    def test_una_camara_desconocida_no_rompe_el_conteo(self):
        esfuerzo = EsfuerzoMuestreo()
        esfuerzo.registrar(None, dt.datetime(2012, 3, 10))
        assert esfuerzo.noches_de(None) == 1


class TestActividadPorEspecie:
    def _esfuerzo(self, camaras=2, dias=50):
        esfuerzo = EsfuerzoMuestreo()
        for camara in range(1, camaras + 1):
            for dia in range(dias):
                esfuerzo.registrar(camara,
                                   dt.datetime(2012, 1, 1) + dt.timedelta(days=dia))
        return esfuerzo

    def test_tasa_por_cien_noches(self):
        eventos = agrupar_en_eventos([
            avistamiento(momento=f"2012-01-{d:02d} 22:00:00") for d in (1, 5, 9)
        ])
        fila = actividad_por_especie(eventos, self._esfuerzo(camaras=1, dias=50))[0]
        assert fila.eventos == 3
        # 3 eventos sobre 50 noches-cámara -> 6 por cada 100.
        assert fila.tasa_por_100_noches == pytest.approx(6.0)

    def test_ocupacion_naive_es_la_fraccion_de_camaras(self):
        eventos = agrupar_en_eventos([avistamiento(ubicacion=1)])
        fila = actividad_por_especie(eventos, self._esfuerzo(camaras=4))[0]
        assert fila.ocupacion_naive == pytest.approx(0.25)

    def test_indice_de_nocturnidad(self):
        momentos = ["2012-01-01 22:00:00", "2012-01-02 02:00:00",
                    "2012-01-03 13:00:00", "2012-01-04 14:00:00"]
        eventos = agrupar_en_eventos([avistamiento(momento=m) for m in momentos])
        fila = actividad_por_especie(eventos, self._esfuerzo())[0]
        assert fila.eventos == 4
        assert fila.indice_nocturnidad == pytest.approx(0.5)

    def test_agrupa_por_hora_y_por_mes(self):
        eventos = agrupar_en_eventos([
            avistamiento(momento="2012-01-01 22:00:00"),
            avistamiento(momento="2012-02-01 22:00:00"),
        ])
        fila = actividad_por_especie(eventos, self._esfuerzo())[0]
        assert fila.por_hora[22] == 2
        assert fila.por_mes == {"2012-01": 1, "2012-02": 1}

    def test_ordena_por_cantidad_de_eventos(self):
        avistamientos = [avistamiento(especie="opossum",
                                      momento=f"2012-01-{d:02d} 22:00:00")
                         for d in (1, 5, 9)]
        avistamientos.append(avistamiento(especie="fox",
                                          momento="2012-01-02 22:00:00"))
        filas = actividad_por_especie(agrupar_en_eventos(avistamientos),
                                      self._esfuerzo())
        assert [f.especie for f in filas] == ["opossum", "fox"]


class TestAvistamientosDesde:
    def _linea(self, predicciones=(), verdades=(), fecha="2012-03-10 22:00:00"):
        return {"archivo": "a.jpg", "ancho": 100, "alto": 100, "ubicacion": 1,
                "secuencia": "s1", "fecha": fecha,
                "predicciones": list(predicciones), "verdades": list(verdades)}

    def test_filtra_por_confianza(self):
        linea = self._linea(predicciones=[
            {"clase": 0, "confianza": 0.9, "caja": [0, 0, 10, 10]},
            {"clase": 1, "confianza": 0.1, "caja": [0, 0, 10, 10]},
        ])
        avistamientos, _, _ = avistamientos_desde([linea], NOMBRES,
                                                  confianza_minima=0.25)
        assert [a.especie for a in avistamientos] == ["opossum"]

    def test_las_imagenes_sin_fecha_no_entran_al_reporte(self):
        linea = self._linea(predicciones=[
            {"clase": 0, "confianza": 0.9, "caja": [0, 0, 10, 10]}], fecha=None)
        avistamientos, esfuerzo, meta = avistamientos_desde([linea], NOMBRES)
        assert avistamientos == []
        assert esfuerzo.noches_camara == 0
        assert meta["imagenes_sin_fecha_legible"] == 1

    def test_desde_las_cajas_anotadas_sin_confianza(self):
        """Las verdades no traen confianza y valen 1.0."""
        linea = self._linea(verdades=[{"clase": 2, "caja": [0, 0, 10, 10]}])
        avistamientos, _, meta = avistamientos_desde(
            [linea], NOMBRES, confianza_minima=0.0, campo="verdades")
        assert [a.especie for a in avistamientos] == ["fox"]
        assert avistamientos[0].confianza == 1.0
        assert meta["campo"] == "verdades"

    def test_ignora_una_clase_fuera_de_rango(self):
        linea = self._linea(predicciones=[
            {"clase": 99, "confianza": 0.9, "caja": [0, 0, 10, 10]}])
        avistamientos, _, _ = avistamientos_desde([linea], NOMBRES)
        assert avistamientos == []

    def test_el_esfuerzo_cuenta_la_imagen_aunque_no_detecte_nada(self):
        """Una cámara que no vio nada igual estuvo encendida ese día."""
        avistamientos, esfuerzo, _ = avistamientos_desde([self._linea()], NOMBRES)
        assert avistamientos == []
        assert esfuerzo.noches_camara == 1


class TestCompararConAnotaciones:
    def _fila(self, especie, eventos, camaras=1):
        esfuerzo = EsfuerzoMuestreo()
        esfuerzo.registrar(1, dt.datetime(2012, 1, 1))
        avistamientos = [
            avistamiento(especie=especie,
                         momento=f"2012-01-{d:02d} 22:00:00")
            for d in range(1, eventos + 1)
        ]
        return actividad_por_especie(agrupar_en_eventos(avistamientos), esfuerzo)[0]

    def test_mide_la_subestimacion(self):
        comparacion = comparar_con_anotaciones(
            [self._fila("opossum", 2)], [self._fila("opossum", 4)])
        assert comparacion[0]["error_absoluto"] == -2
        assert comparacion[0]["error_relativo"] == pytest.approx(-0.5)

    def test_una_especie_que_el_modelo_nunca_vio_cuenta_como_cero(self):
        comparacion = comparar_con_anotaciones([], [self._fila("fox", 3)])
        assert comparacion[0]["eventos_modelo"] == 0
        assert comparacion[0]["error_relativo"] == pytest.approx(-1.0)

    def test_una_especie_inventada_por_el_modelo_aparece_igual(self):
        comparacion = comparar_con_anotaciones([self._fila("fox", 2)], [])
        assert comparacion[0]["especie"] == "fox"
        assert comparacion[0]["eventos_anotados"] == 0
        assert comparacion[0]["error_relativo"] is None

    def test_ordena_por_magnitud_del_error(self):
        comparacion = comparar_con_anotaciones(
            [self._fila("opossum", 1), self._fila("coyote", 1)],
            [self._fila("opossum", 2), self._fila("coyote", 9)])
        assert [f["especie"] for f in comparacion] == ["coyote", "opossum"]
