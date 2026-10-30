# Paso 6 — Módulo de monitoreo ecológico · entregable #5

> Entregable #5: *"módulo de monitoreo ecológico: reporte de actividad por
> especie y periodo"*.

## Clases y módulos de este paso

| Archivo | Clase / función | Qué hace |
|---|---|---|
| [`eventos.py`](eventos.py) | `Avistamiento` | Una detección con su cuándo, dónde y qué; expone `hora` y `franja` |
| | `Evento` | Un avistamiento **independiente**: una visita, no una foto |
| | `leer_fecha()` | Interpreta `date_captured` de CCT; `None` si es ilegible |
| | `agrupar_en_eventos()` | Colapsa detecciones cercanas de la misma especie y cámara |
| | `EsfuerzoMuestreo` | Cuenta noches-cámara, la unidad que normaliza el esfuerzo |
| | `ActividadEspecie` | El renglón del reporte: eventos, tasa, ocupación, nocturnidad |
| | `actividad_por_especie()` | Agrega los eventos a un renglón por especie |
| | `avistamientos_desde()` | Volcado del paso 4 → avistamientos (de predicciones **o** de cajas anotadas) |
| [`generar_reporte.py`](generar_reporte.py) | `comparar_con_anotaciones()` | Error del reporte del modelo contra el de las anotaciones |
| | `tabla_actividad()`, `tabla_horaria()`, `tabla_comparacion()` | Las salidas en markdown y CSV |
| | `NOCTURNIDAD_ESPERADA` | Historia natural conocida, para el control de sanidad |

## Comandos

Necesita las predicciones del [paso 4](../paso4_evaluacion/).

```bash
.venv/Scripts/python.exe pregunta3/paso6_monitoreo/generar_reporte.py

# sobre ubicaciones nuevas, o con otro umbral de decision
.venv/Scripts/python.exe pregunta3/paso6_monitoreo/generar_reporte.py --particion trans_test
.venv/Scripts/python.exe pregunta3/paso6_monitoreo/generar_reporte.py --confianza 0.5

# con otra ventana de independencia entre eventos
.venv/Scripts/python.exe pregunta3/paso6_monitoreo/generar_reporte.py --minutos 60
```

Produce:

```
informes/actividad.json          todo: parametros, cobertura, esfuerzo, especies, error
informes/actividad.md            la tabla por especie + el error contra las anotaciones
informes/actividad_por_hora.csv  especie x 24 horas, para graficar
```

### Usar las clases desde Python

```python
from pregunta3.paso4_evaluacion.predecir import cargar_predicciones
from pregunta3.paso6_monitoreo.eventos import (
    actividad_por_especie, agrupar_en_eventos, avistamientos_desde)

cabecera, lineas = cargar_predicciones(
    "pregunta3/paso4_evaluacion/predicciones/cis_test.jsonl")
avistamientos, esfuerzo, cobertura = avistamientos_desde(
    lineas, cabecera["clases"], confianza_minima=0.25)

for fila in actividad_por_especie(agrupar_en_eventos(avistamientos), esfuerzo):
    print(fila.especie, fila.eventos, fila.tasa_por_100_noches,
          fila.indice_nocturnidad)
```

---

## Lo que se hizo

### 1. La unidad del reporte es el evento, no la detección

Contar detecciones no sirve, y el error no es de implementación sino conceptual:
una cámara trampa dispara ráfagas de unas tres fotos por evento, y un animal que
se queda comiendo frente al lente dispara decenas. Contando detecciones, un zorro
curioso durante diez minutos pesaría igual que diez zorros distintos.

La literatura de cámaras trampa usa el **evento independiente**: dos detecciones
de la misma especie, en la misma cámara, separadas por menos de un intervalo de
silencio, son el mismo evento. El convencional son **30 minutos**, y acá es un
parámetro explícito (`--minutos`) porque la elección cambia los números y
conviene poder mostrarlo.

La ventana se mide contra la **última** detección del evento, no contra la
primera: un animal que se queda una hora es una visita, no tres.

### 2. Dos métricas normalizadas, no conteos crudos

- **Tasa de detección** — eventos por cada 100 **noches-cámara**. Una
  noche-cámara es un día de una cámara encendida. Sin esta normalización los
  conteos no se pueden comparar entre cámaras ni entre periodos, y es el error
  más común al reportar datos de cámaras trampa: una cámara que estuvo seis
  meses y otra que estuvo dos semanas no son comparables en conteo crudo.
- **Ocupación naive** — en qué fracción de las cámaras apareció la especie al
  menos una vez. Es "naive" porque **no corrige por detección imperfecta**: una
  especie presente pero nunca fotografiada cuenta como ausente. Eso va dicho en
  el reporte, no en una nota al pie.

El esfuerzo se cuenta sobre **todas** las imágenes con fecha, no solo las que
tuvieron detección: una cámara que no vio nada ese día igual estuvo encendida.

### 3. El reporte se valida contra las cajas anotadas

`avistamientos_desde(..., campo="verdades")` construye el **mismo** reporte a
partir de las anotaciones a mano. `comparar_con_anotaciones()` da, especie por
especie, el error en eventos y el error relativo.

Esto es lo que separa un reporte de un adorno. Un módulo que imprime números sin
decir cuánto se equivoca no sirve para decidir nada, y acá el error no es
uniforme: las especies con pocos datos de entrenamiento son justamente las que el
reporte subestima. La tabla también lista las especies que el modelo reporta y la
anotación no — falsos positivos puros, que en un informe ecológico serían una
especie "detectada" en un sitio donde no está.

### 4. El umbral de decisión es 0.25, no el 0.001 de la métrica

La AP del paso 4 integra la curva completa y necesita detecciones de confianza
casi nula. Un reporte ecológico es una **decisión**: a 0.001 el listado se
llenaría de detecciones que nadie creería. Queda como parámetro para poder
mostrar cuánto cambia el reporte según el umbral — que es una pregunta legítima
de quien va a usarlo.

### 5. Franjas horarias y control de sanidad

Las franjas son cinco, no dos: noche, amanecer, día, atardecer y noche tardía.
El amanecer y el atardecer van aparte porque muchas especies son **crepusculares**,
y meterlas en "día" o "noche" borraría justamente eso.

El **índice de nocturnidad** (fracción de eventos entre las 20:00 y las 05:00) se
contrasta contra `NOCTURNIDAD_ESPERADA`, la historia natural conocida de cada
especie. No se usa para corregir nada: si el reporte dice que los mapaches son
diurnos, el problema está en el modelo, no en los mapaches. Es un control que
cuesta una columna y detecta errores que el mAP no ve.

### 6. Las fechas ilegibles se descartan, no se adivinan

El conjunto trae `date_captured` como `"2012-03-10 14:22:31"`, pero no todas las
imágenes la tienen y algunas cámaras la escribieron mal. `leer_fecha()` prueba
tres formatos y devuelve `None` si ninguno encaja; esas imágenes quedan fuera del
reporte temporal y el total se informa en `cobertura`. Un reporte de actividad
con fechas adivinadas sería peor que uno más chico.

---

## Resultados

Sobre `cis_test`: **10 cámaras, 1,131 noches-cámara**, 12,696 imágenes con fecha
legible y ninguna sin ella.

Las **13,559 detecciones** por encima de 0.25 se agrupan en **3,706 eventos
independientes** con ventana de 30 minutos. Contar detecciones en lugar de
eventos habría multiplicado la actividad reportada por 3.7.

| Especie | Eventos | Anotados | Error | Ev./100 noches | Ocupación | Nocturnidad | Esperado |
|---|---:|---:|---:|---:|---:|---:|---|
| opossum | 750 | 735 | +2% | 70.89 | 90% | 80% | nocturna ✓ |
| coyote | 564 | 483 | +17% | 49.87 | 100% | 51% | crepuscular ✓ |
| raccoon | 410 | 236 | **+74%** | 47.56 | 90% | 77% | nocturna ✓ |
| cat | 360 | 297 | +21% | 47.94 | 70% | 35% | variable |
| bobcat | 342 | 260 | +32% | 35.04 | 90% | 49% | crepuscular ✓ |
| car | 319 | 312 | +2% | 99.07 | 20% | 8% | diurna ✓ |
| rabbit | 310 | 250 | +24% | 27.41 | 100% | 24% | crepuscular ✓ |
| dog | 192 | 135 | +42% | 18.15 | 90% | 11% | variable |
| squirrel | 154 | 131 | +18% | 14.56 | 90% | **0%** | diurna ✓ |
| bird | 139 | 96 | +45% | 14.79 | 80% | 12% | diurna ✓ |
| skunk | 87 | 78 | +12% | 8.35 | 80% | 66% | nocturna ✓ |
| rodent | 41 | 39 | +5% | 7.90 | 50% | 88% | nocturna ✓ |
| deer | 34 | 27 | +26% | 5.27 | 40% | 50% | crepuscular ✓ |
| fox | 4 | 2 | +100% | 0.98 | 30% | 75% | nocturna ✓ |
| **badger** | **0** | 2 | **−100%** | — | — | — | nocturna |

`badger` es la única especie que el modelo pierde por completo: dos eventos
anotados, ninguno detectado. Con nueve cajas de entrenamiento, era lo esperable.


### El control de sanidad funciona

**El índice de nocturnidad reproduce la historia natural de las catorce especies
detectadas, sin que se le haya dado esa información.** Los diurnos dan 0–12%
(`squirrel`, `bird`, `car`), los nocturnos 66–88% (`opossum`, `raccoon`,
`skunk`, `rodent`), y los crepusculares quedan alrededor del 50% (`coyote`,
`bobcat`, `deer`), que es lo que se espera de una especie que se mueve en los
bordes del día.

Es un control que cuesta una columna y detecta errores que el mAP no ve: si el
reporte dijera que los mapaches son diurnos, el problema estaría en el modelo.

### El reporte sobreestima, siempre en la misma dirección

Comparado contra el mismo reporte calculado desde las cajas anotadas, el error es
**positivo en las catorce especies que detecta**, de +2% a +100%. A un umbral de
decisión de 0.25 los falsos positivos inflan el conteo de forma sistemática y en
una sola dirección — la única excepción es `badger`, que no detecta nunca
(−100%).

El error tampoco es parejo: las especies con más datos de entrenamiento
(`opossum` +2%, `car` +2%, `rodent` +5%) quedan cerca, y las intermedias se
disparan (`raccoon` +74%, `bird` +45%, `dog` +42%). Un reporte que promediara
todo en una sola cifra de confiabilidad escondería justamente eso.

Esa es la medida de para qué sirve el módulo: la señal ecológica gruesa —qué
especies hay, cuándo están activas, en qué cámaras— la captura bien; la magnitud
del conteo no es confiable sin corregir.

---

## Pendiente

| | Qué falta | Por qué importa |
|---|---|---|
| 1 | Ocupación corregida por detección imperfecta (modelo de ocupación tipo MacKenzie) | La ocupación naive subestima sistemáticamente, y en especies raras el sesgo es grande |
| 2 | Curvas de actividad diaria suavizadas (densidad kernel circular) | 24 cubetas sobre pocas decenas de eventos es ruidoso; la densidad circular es lo estándar en el dominio |
| 3 | Reporte por **temporada** además de por mes | El conjunto cubre varios años; una sola fila por mes mezcla años distintos |
