# rnap-2026ii-p3-fauna-yolo

**Monitoreo de fauna con cámaras trampa (YOLO sobre CCT-20)**

Entrega del Examen Parcial de Redes Neuronales y Aprendizaje Profundo (2026-II, Sección A) — UNI FIIS.

## Integrantes

- Flavio César Amayo Gamboa — flavio.amayo.g@uni.pe
- Gustavo Adolfo Huarcaya Delgado — gustavo.huarcaya.d@uni.pe
- Jhordy Joel Maximiliano Cusi — jmaximilianoc@uni.pe
- Sergio Mondragon Contreras — sergio.mondragon.c@uni.pe
## Informe

`pregunta3/paso7_informe/informe/informe.pdf`

## Reproducir (un comando)

```bash
python -m venv .venv
. .venv/Scripts/activate   # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python pregunta3/paso7_informe/ejecutar_todo.py
```

Detalle completo del pipeline: [pregunta3/README.md](pregunta3/README.md)

## Pesos preentrenados

Publicados como release (no entran al repo por el límite de 100 MB/archivo): **https://github.com/sergiomc12/rnap-2026ii-p3-fauna-yolo/releases/tag/v1.0-pesos**

## Datos

No se redistribuyen (tamaño y reglas de la fuente). Se descargan con los scripts del repo; ver [pregunta3/README.md](pregunta3/README.md).

## Pruebas

```bash
pytest tests/
```

> Algunas pruebas requieren artefactos descargados; las de lógica pura corren sin datos.

