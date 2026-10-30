"""Pruebas de la conversión COCO -> YOLO de la Pregunta 3.

Las dos cosas que acá se fijan son las que el README del paso 2 identifica como
peligrosas: la conversión de coordenadas (COCO da esquina + tamaño en píxeles,
YOLO centro + tamaño normalizado) y el remapeo de clases a índices contiguos.
Equivocar cualquiera de las dos entrena igual, sin ningún error visible.
"""

from __future__ import annotations

import pytest

from pregunta3.paso1_datos.anotaciones import Caja, Imagen, Particion
from pregunta3.paso2_preprocesamiento.conversion import (
    MapaClases, convertir_imagen, escribir_data_yaml, escribir_etiquetas,
    filtrar_fuga_de_rafagas, secuencias_de,
)

CATEGORIAS = {1: "opossum", 3: "coyote", 7: "fox", 30: "empty"}


def imagen(cajas=(), archivo="a.jpg", ancho=2048, alto=1494, secuencia="s1"):
    return Imagen(id=archivo, archivo=archivo, ancho=ancho, alto=alto,
                  ubicacion=1, secuencia=secuencia, fecha="2012-03-10 22:00:00",
                  cajas=tuple(cajas))


def particion(imagenes, nombre="train"):
    return Particion(nombre=nombre, descripcion="", imagenes=tuple(imagenes),
                     categorias=CATEGORIAS)


class TestCajaAYolo:
    def test_conversion_verificada_a_mano(self):
        """El caso real que documenta el README del paso 2."""
        caja = Caja(x=1730.56, y=737.28, ancho=299.52, alto=279.04,
                    categoria=1)
        cx, cy, w, h = caja.a_yolo(2048, 1494)
        assert cx == pytest.approx(0.918125, abs=1e-6)
        assert cy == pytest.approx(0.586881, abs=1e-6)
        assert w == pytest.approx(0.146250, abs=1e-6)
        assert h == pytest.approx(0.186774, abs=1e-6)

    def test_una_caja_que_cubre_todo_queda_centrada(self):
        caja = Caja(x=0, y=0, ancho=100, alto=50, categoria=1)
        assert caja.a_yolo(100, 50) == (0.5, 0.5, 1.0, 1.0)

    def test_devuelve_el_centro_no_la_esquina(self):
        """Confundirlos corre cada caja media caja: es el error silencioso."""
        caja = Caja(x=0, y=0, ancho=50, alto=50, categoria=1)
        cx, cy, _, _ = caja.a_yolo(100, 100)
        assert (cx, cy) == (0.25, 0.25)        # no (0.0, 0.0)


class TestMapaClases:
    def test_ordena_por_frecuencia_descendente(self):
        datos = particion([
            imagen([Caja(0, 0, 10, 10, 3)] * 3, archivo="a.jpg"),
            imagen([Caja(0, 0, 10, 10, 1)] * 5, archivo="b.jpg"),
            imagen([Caja(0, 0, 10, 10, 7)], archivo="c.jpg"),
        ])
        mapa = MapaClases.desde_particiones([datos])
        assert mapa.nombres == ("opossum", "coyote", "fox")
        assert mapa.indice(1) == 0        # la más común queda en el índice 0

    def test_los_indices_son_contiguos_desde_cero(self):
        """YOLO exige 0..N-1: los ids de COCO no son contiguos."""
        datos = particion([imagen([Caja(0, 0, 10, 10, 7),
                                   Caja(0, 0, 10, 10, 30)])])
        mapa = MapaClases.desde_particiones([datos])
        assert sorted(mapa.de_coco.values()) == list(range(mapa.n))

    def test_excluye_empty(self):
        """En detección, la ausencia de cajas ya dice "acá no hay nada"."""
        datos = particion([imagen([Caja(0, 0, 10, 10, 1),
                                   Caja(0, 0, 10, 10, 30)])])
        mapa = MapaClases.desde_particiones([datos])
        assert "empty" not in mapa.nombres
        assert mapa.indice(30) is None

    def test_una_clase_desconocida_no_tiene_indice(self):
        datos = particion([imagen([Caja(0, 0, 10, 10, 1)])])
        assert MapaClases.desde_particiones([datos]).indice(999) is None


class TestConvertirImagen:
    def _mapa(self):
        datos = particion([imagen([Caja(0, 0, 10, 10, 1),
                                   Caja(0, 0, 10, 10, 3)])])
        return MapaClases.desde_particiones([datos])

    def test_una_linea_por_caja(self):
        etiqueta = convertir_imagen(
            imagen([Caja(0, 0, 100, 100, 1), Caja(100, 100, 100, 100, 3)],
                   ancho=1000, alto=1000),
            self._mapa())
        assert len(etiqueta.lineas) == 2
        assert all(len(l.split()) == 5 for l in etiqueta.lineas)

    def test_descarta_la_clase_excluida_y_lo_informa(self):
        etiqueta = convertir_imagen(
            imagen([Caja(0, 0, 100, 100, 30)], ancho=1000, alto=1000),
            self._mapa())
        assert etiqueta.vacia
        assert etiqueta.descartadas_clase == 1

    @pytest.mark.parametrize("caja", [
        Caja(0, 0, 0, 100, 1),              # ancho cero
        Caja(0, 0, 100, 0, 1),              # alto cero
        Caja(2000, 0, 100, 100, 1),         # centro fuera de la imagen
    ])
    def test_descarta_la_geometria_invalida(self, caja):
        """Ultralytics las acepta y entrena con ellas: filtrarlas es acá o nunca."""
        etiqueta = convertir_imagen(imagen([caja], ancho=1000, alto=1000),
                                    self._mapa())
        assert etiqueta.vacia
        assert etiqueta.descartadas_geometria == 1

    def test_el_indice_escrito_es_el_del_mapa_no_el_de_coco(self):
        etiqueta = convertir_imagen(
            imagen([Caja(0, 0, 100, 100, 3)], ancho=1000, alto=1000),
            self._mapa())
        assert etiqueta.lineas[0].split()[0] in {"0", "1"}
        assert etiqueta.lineas[0].split()[0] != "3"


class TestEscribirEtiquetas:
    def _mapa(self, datos):
        return MapaClases.desde_particiones([datos])

    def test_escribe_un_txt_por_imagen(self, tmp_path):
        datos = particion([imagen([Caja(0, 0, 100, 100, 1)], archivo="uno.jpg"),
                           imagen([Caja(0, 0, 100, 100, 3)], archivo="dos.jpg")])
        resumen = escribir_etiquetas(datos, self._mapa(datos), tmp_path)
        assert resumen.imagenes_con_caja == 2
        assert (tmp_path / "uno.txt").is_file()
        assert (tmp_path / "dos.txt").is_file()

    def test_salta_las_imagenes_que_no_estan_en_disco(self, tmp_path):
        datos = particion([imagen([Caja(0, 0, 100, 100, 1)], archivo="uno.jpg"),
                           imagen([Caja(0, 0, 100, 100, 1)], archivo="dos.jpg")])
        resumen = escribir_etiquetas(datos, self._mapa(datos), tmp_path,
                                     imagenes_disponibles={"uno.jpg"})
        assert resumen.imagenes_con_caja == 1
        assert resumen.sin_imagen_en_disco == 1
        assert not (tmp_path / "dos.txt").exists()

    def test_filtra_las_rafagas_prohibidas(self, tmp_path):
        datos = particion([
            imagen([Caja(0, 0, 100, 100, 1)], archivo="a.jpg", secuencia="sA"),
            imagen([Caja(0, 0, 100, 100, 1)], archivo="b.jpg", secuencia="sB"),
        ])
        resumen = escribir_etiquetas(datos, self._mapa(datos), tmp_path,
                                     secuencias_prohibidas=frozenset({"sA"}))
        assert resumen.por_fuga_de_rafaga == 1
        assert resumen.imagenes_con_caja == 1

    def test_la_tasa_de_descarte_es_cero_sin_descartes(self, tmp_path):
        datos = particion([imagen([Caja(0, 0, 100, 100, 1)])])
        resumen = escribir_etiquetas(datos, self._mapa(datos), tmp_path)
        assert resumen.tasa_descarte == 0.0

    def test_no_escribe_txt_vacios(self, tmp_path):
        """Una imagen cuya única caja es 'empty' no debe dejar archivo."""
        datos = particion([imagen([Caja(0, 0, 100, 100, 1)], archivo="ok.jpg"),
                           imagen([Caja(0, 0, 100, 100, 30)], archivo="no.jpg")])
        escribir_etiquetas(datos, self._mapa(datos), tmp_path)
        assert not (tmp_path / "no.txt").exists()


class TestFugaDeRafagas:
    def test_secuencias_de_lista_las_rafagas(self):
        datos = particion([imagen([], archivo="a.jpg", secuencia="s1"),
                           imagen([], archivo="b.jpg", secuencia="s2"),
                           imagen([], archivo="c.jpg", secuencia="s1")])
        assert secuencias_de(datos) == frozenset({"s1", "s2"})

    def test_ignora_las_imagenes_sin_secuencia(self):
        datos = particion([imagen([], secuencia=None)])
        assert secuencias_de(datos) == frozenset()

    def test_filtrar_quita_las_contaminadas(self):
        evaluacion = particion([
            imagen([Caja(0, 0, 10, 10, 1)], archivo="a.jpg", secuencia="sA"),
            imagen([Caja(0, 0, 10, 10, 1)], archivo="b.jpg", secuencia="sB"),
        ], nombre="cis_val")
        quedan = filtrar_fuga_de_rafagas(evaluacion, frozenset({"sA"}))
        assert [i.archivo for i in quedan] == ["b.jpg"]


class TestEscribirDataYaml:
    def test_los_nombres_van_en_el_orden_del_mapa(self, tmp_path):
        datos = particion([imagen([Caja(0, 0, 10, 10, 1)] * 2),
                           imagen([Caja(0, 0, 10, 10, 3)], archivo="b.jpg")])
        mapa = MapaClases.desde_particiones([datos])
        ruta = escribir_data_yaml(tmp_path / "data.yaml", mapa, tmp_path,
                                  train="train.txt", val="val.txt")
        texto = ruta.read_text(encoding="utf-8")
        assert "nc: 2" in texto
        assert "  0: opossum" in texto
        assert "  1: coyote" in texto

    def test_incluye_test_solo_si_se_pide(self, tmp_path):
        datos = particion([imagen([Caja(0, 0, 10, 10, 1)])])
        mapa = MapaClases.desde_particiones([datos])
        sin = escribir_data_yaml(tmp_path / "a.yaml", mapa, tmp_path,
                                 train="t.txt", val="v.txt")
        con = escribir_data_yaml(tmp_path / "b.yaml", mapa, tmp_path,
                                 train="t.txt", val="v.txt", test="e.txt")
        assert "test:" not in sin.read_text(encoding="utf-8")
        assert "test: e.txt" in con.read_text(encoding="utf-8")
