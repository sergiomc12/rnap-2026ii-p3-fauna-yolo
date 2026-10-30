"""Pruebas del paso 5: generalización entre ubicaciones y robustez.

Dos cosas se fijan acá porque equivocarlas pasaría inadvertida en los números
finales: que el parche de oclusión tape la fracción del **área** que dice tapar
(y no la del lado), y que los tramos de tamaño no mezclen imágenes con cajas
fuera del rango, lo que metería falsos positivos inevitables en el tramo.
"""

from __future__ import annotations

import numpy as np
import pytest

from pregunta3.paso4_evaluacion.metricas import (
    Muestra, Prediccion, ResultadoClase, ResultadoEvaluacion, Verdad,
)
from pregunta3.paso5_generalizacion.generalizacion import (
    brecha_por_clase, dispersion, distribucion_de, mapa_por_ubicacion,
    remuestrear_a_distribucion,
)
from pregunta3.paso5_generalizacion.robustez import (
    GRIS, area_tapada, barrido_por_tamano, muestras_en_tramo, ocluir_cajas,
)

NOMBRES = ("opossum", "coyote", "fox")


def muestra(verdades=(), predicciones=(), archivo="a.jpg", ancho=100, alto=100,
            ubicacion=None):
    return Muestra(archivo=archivo, ancho=ancho, alto=alto,
                   predicciones=tuple(predicciones), verdades=tuple(verdades),
                   ubicacion=ubicacion)


class TestOcluirCajas:
    def test_fraccion_cero_no_toca_la_imagen(self):
        imagen = np.full((50, 50, 3), 200, dtype=np.uint8)
        salida = ocluir_cajas(imagen, [(0, 0, 50, 50)], 0.0)
        assert np.array_equal(salida, imagen)

    def test_no_modifica_la_imagen_original(self):
        """El barrido corre varios niveles sobre la misma imagen leída."""
        imagen = np.full((50, 50, 3), 200, dtype=np.uint8)
        ocluir_cajas(imagen, [(0, 0, 50, 50)], 0.5)
        assert imagen.min() == 200

    def test_tapa_la_fraccion_del_area_no_del_lado(self):
        """Tapar 'la mitad' debe tapar la mitad del área: lado = sqrt(0.5)."""
        imagen = np.zeros((100, 100, 3), dtype=np.uint8)
        salida = ocluir_cajas(imagen, [(0, 0, 100, 100)], 0.5)
        tapados = int((salida[:, :, 0] == GRIS).sum())
        assert tapados == pytest.approx(5000, rel=0.02)

    def test_el_area_declarada_coincide_con_la_tapada(self):
        imagen = np.zeros((200, 200, 3), dtype=np.uint8)
        caja = (20.0, 20.0, 120.0, 180.0)
        for fraccion in (0.25, 0.5, 0.75):
            salida = ocluir_cajas(imagen, [caja], fraccion)
            tapados = int((salida[:, :, 0] == GRIS).sum())
            assert tapados == pytest.approx(area_tapada(caja, fraccion), rel=0.03)

    def test_modo_inferior_tapa_abajo(self):
        imagen = np.zeros((100, 100, 3), dtype=np.uint8)
        salida = ocluir_cajas(imagen, [(0, 0, 100, 100)], 0.25, modo="inferior")
        filas = np.where((salida[:, :, 0] == GRIS).any(axis=1))[0]
        # Lado = 0.5 -> 50 px de alto, anclado al borde inferior.
        assert filas.min() == pytest.approx(50, abs=1)
        assert filas.max() == 99

    def test_modo_centro_tapa_al_medio(self):
        imagen = np.zeros((100, 100, 3), dtype=np.uint8)
        salida = ocluir_cajas(imagen, [(0, 0, 100, 100)], 0.25, modo="centro")
        filas = np.where((salida[:, :, 0] == GRIS).any(axis=1))[0]
        assert filas.min() == pytest.approx(25, abs=1)
        assert filas.max() == pytest.approx(74, abs=1)

    def test_recorta_contra_el_borde_de_la_imagen(self):
        """Una caja anotada que se pasa del borde no debe romper el índice."""
        imagen = np.zeros((40, 40, 3), dtype=np.uint8)
        salida = ocluir_cajas(imagen, [(30, 30, 90, 90)], 1.0)
        assert salida.shape == imagen.shape
        assert (salida[:, :, 0] == GRIS).any()

    def test_ignora_cajas_degeneradas(self):
        imagen = np.zeros((40, 40, 3), dtype=np.uint8)
        salida = ocluir_cajas(imagen, [(10, 10, 10, 10)], 0.5)
        assert not (salida[:, :, 0] == GRIS).any()


class TestTramosDeTamano:
    def test_solo_entran_las_imagenes_con_todas_las_cajas_en_rango(self):
        chica = muestra(verdades=[Verdad(0, (0, 0, 2, 2))], archivo="chica.jpg")
        grande = muestra(verdades=[Verdad(0, (0, 0, 80, 80))], archivo="grande.jpg")
        mixta = muestra(verdades=[Verdad(0, (0, 0, 2, 2)),
                                  Verdad(0, (0, 0, 80, 80))], archivo="mixta.jpg")
        # 2x2 sobre 100x100 = 0.04%; el tramo "diminuto" va hasta 0.2%.
        elegidas = muestras_en_tramo([chica, grande, mixta], 0.0, 0.002)
        assert [m.archivo for m in elegidas] == ["chica.jpg"]

    def test_las_imagenes_sin_cajas_no_entran_a_ningun_tramo(self):
        assert muestras_en_tramo([muestra()], 0.0, 1.01) == []

    def test_el_barrido_reparte_sin_perder_imagenes(self):
        muestras = [
            muestra(verdades=[Verdad(0, (0, 0, 2, 2))], archivo="a.jpg"),
            muestra(verdades=[Verdad(0, (0, 0, 10, 10))], archivo="b.jpg"),
            muestra(verdades=[Verdad(0, (0, 0, 30, 30))], archivo="c.jpg"),
            muestra(verdades=[Verdad(0, (0, 0, 90, 90))], archivo="d.jpg"),
        ]
        resultados = barrido_por_tamano(muestras, NOMBRES)
        assert sum(r.imagenes for r in resultados) == 4
        assert [r.tramo for r in resultados] == ["diminuto", "chico",
                                                 "mediano", "grande"]

    def test_marca_los_tramos_con_muestra_chica(self):
        muestras = [muestra(verdades=[Verdad(0, (0, 0, 90, 90))])]
        grande = [r for r in barrido_por_tamano(muestras, NOMBRES)
                  if r.tramo == "grande"][0]
        assert grande.cajas == 1
        assert not grande.confiable


class TestMapaPorUbicacion:
    def test_agrupa_por_ubicacion(self):
        muestras = [
            muestra(verdades=[Verdad(0, (0, 0, 10, 10))],
                    predicciones=[Prediccion(0, 0.9, (0, 0, 10, 10))],
                    archivo=f"{i}.jpg", ubicacion=i % 2)
            for i in range(6)
        ]
        salida = mapa_por_ubicacion(muestras, NOMBRES, minimo_cajas=2)
        assert set(salida) == {"0", "1"}
        assert salida["0"]["imagenes"] == 3
        assert salida["0"]["map50"] == pytest.approx(1.0)
        assert salida["0"]["muestra_suficiente"] is True

    def test_marca_la_ubicacion_con_pocas_cajas(self):
        muestras = [muestra(verdades=[Verdad(0, (0, 0, 10, 10))], ubicacion=7)]
        assert mapa_por_ubicacion(muestras, NOMBRES)["7"]["muestra_suficiente"] is False

    def test_las_imagenes_sin_ubicacion_se_agrupan_aparte(self):
        muestras = [muestra(verdades=[Verdad(0, (0, 0, 10, 10))], ubicacion=None)]
        assert "None" in mapa_por_ubicacion(muestras, NOMBRES)


class TestDispersion:
    def test_resume_el_rango(self):
        d = dispersion([0.2, 0.5, 0.8])
        assert d["n"] == 3
        assert d["minimo"] == 0.2 and d["maximo"] == 0.8
        assert d["rango"] == pytest.approx(0.6)
        assert d["media"] == pytest.approx(0.5)

    def test_lista_vacia_no_inventa_estadisticos(self):
        assert dispersion([]) == {"n": 0}

    def test_ignora_los_nulos(self):
        assert dispersion([None, 0.4, None])["n"] == 1


class TestRemuestrearADistribucion:
    def _muestras(self):
        # Diez imágenes de la clase 0 y dos de la clase 1.
        comunes = [muestra(verdades=[Verdad(0, (0, 0, 10, 10))],
                           archivo=f"c{i}.jpg") for i in range(10)]
        raras = [muestra(verdades=[Verdad(1, (0, 0, 10, 10))],
                         archivo=f"r{i}.jpg") for i in range(2)]
        return comunes + raras

    def test_respeta_el_objetivo_cuando_alcanza(self):
        elegidas = remuestrear_a_distribucion(self._muestras(), {0: 3, 1: 1})
        cuenta = {}
        for m in elegidas:
            for v in m.verdades:
                cuenta[v.clase] = cuenta.get(v.clase, 0) + 1
        assert cuenta[0] == 3
        assert cuenta[1] == 1

    def test_no_pide_mas_de_lo_que_hay(self):
        elegidas = remuestrear_a_distribucion(self._muestras(), {0: 5, 1: 99})
        raras = sum(1 for m in elegidas for v in m.verdades if v.clase == 1)
        assert raras == 2          # solo había dos

    def test_es_reproducible_con_la_misma_semilla(self):
        muestras = self._muestras()
        a = remuestrear_a_distribucion(muestras, {0: 4, 1: 1}, semilla=3)
        b = remuestrear_a_distribucion(muestras, {0: 4, 1: 1}, semilla=3)
        assert [m.archivo for m in a] == [m.archivo for m in b]

    def test_objetivo_vacio_no_elige_nada(self):
        assert remuestrear_a_distribucion(self._muestras(), {}) == []


class TestBrechaPorClase:
    def _resultado(self, conjunto, aps, soportes):
        clases = tuple(
            ResultadoClase(nombre=n, indice=i, soporte=s, predicciones=s,
                           ap50=a, ap50_95=a)
            for i, (n, a, s) in enumerate(zip(NOMBRES, aps, soportes))
        )
        return ResultadoEvaluacion(conjunto=conjunto, imagenes=10, clases=clases)

    def test_calcula_la_caida_y_el_cambio_de_prevalencia(self):
        vista = self._resultado("cis", [0.9, 0.8, 0.5], [100, 50, 10])
        nueva = self._resultado("trans", [0.7, 0.8, 0.5], [200, 25, 10])
        brechas = {b.nombre: b for b in brecha_por_clase(vista, nueva)}
        assert brechas["opossum"].caida == pytest.approx(0.2)
        assert brechas["opossum"].cambio_de_prevalencia == pytest.approx(2.0)
        assert brechas["coyote"].caida == pytest.approx(0.0)

    def test_clase_indefinida_en_una_no_produce_caida(self):
        vista = self._resultado("cis", [0.9, 0.8, 0.5], [100, 50, 10])
        nueva = self._resultado("trans", [0.7, 0.8, None], [200, 25, 0])
        brechas = {b.nombre: b for b in brecha_por_clase(vista, nueva)}
        assert brechas["fox"].caida is None

    def test_distribucion_de_ignora_las_clases_sin_cajas(self):
        resultado = self._resultado("cis", [0.9, 0.8, None], [100, 50, 0])
        assert distribucion_de(resultado) == {0: 100, 1: 50}
