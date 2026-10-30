# Paso 7 — Informe NeurIPS y comando único · entregable #6

> Entregable #6: *"media página sobre uso responsable de IA en conservación"*,
> más los requisitos transversales: **informe de 6–10 páginas en formato
> NeurIPS con mínimo 5 referencias**, repositorio reproducible y **ejecución de
> extremo a extremo con un único comando**.

## Clases y módulos de este paso

| Archivo | Clase / función | Qué hace |
|---|---|---|
| [`ejecutar_todo.py`](ejecutar_todo.py) | `Etapa` | Una etapa del pipeline y cómo saber que ya corrió |
| | `Etapa.ya_corrio` | Chequea el artefacto: archivo existente, o directorio no vacío |
| | `ETAPAS` | Las diez etapas, del paso 1 al 7, en orden |
| | `correr()` | Lanza el script de una etapa, con sus argumentos rápidos si corresponde |
| [`compilar_informe.py`](compilar_informe.py) | `compilar()` | `latexmk`, o la secuencia pdflatex/bibtex/pdflatex×2 |
| | `paginas_del_log()` | Cuenta las páginas desde el log de LaTeX |
| | `referencias_del_bbl()` | Cuenta las referencias **citadas**, no las del `.bib` |
| | `paginas_de_cuerpo()` | **Las páginas de cuerpo**, que son las que limita el enunciado. Busca la página donde arranca la bibliografía y mide la altura de su título: contar páginas del PDF obligaría a suponer cuánto ocupan las referencias |
| | `_altura_relativa()` | Qué fracción de esa página ocupa el cuerpo. El corte es el **título** de la bibliografía, no su entrada `[1]`: si la página arranca con *References*, el cuerpo terminó en la anterior |
| | `errores_del_log()` | Las líneas de error, para no abrir el log |
| [`exportar_word.py`](exportar_word.py) | `main()` | Genera el `.docx` con pandoc desde el **mismo** `.tex` que el PDF |
| | `cuerpo_plano()` | Recorta el cuerpo y resuelve los `\input{}`. Pasarle el `.tex` entero a pandoc **se cuelga** con `neurips.sty`, y los `\input` no los sigue: sin esto las tablas faltarían en el Word sin ningún aviso |
| | `metadatos()` | Rescata título y autores del preámbulo descartado; sin esto el Word arrancaba directo en el resumen |
| [`generar_tablas.py`](generar_tablas.py) | `tabla_resumen_particiones()`<br>`tabla_por_especie()`<br>`tabla_brechas()`<br>`tabla_gradcam()`<br>`tabla_robustez()`<br>`tabla_actividad()`<br>(+7 más)<br>`main()` | Script ejecutable: arma las tablas LaTeX del informe desde los JSON de los pasos 3 a 6, para que el informe no tenga números copiados a mano |

## El comando único

```bash
.venv/Scripts/python.exe pregunta3/paso7_informe/ejecutar_todo.py
```

Eso es todo: descarga las imágenes, arma el conjunto, entrena, predice, evalúa,
corre generalización, robustez, Grad-CAM, el reporte ecológico y compila el PDF.

```bash
# verificar el circuito completo en minutos en vez de horas
.venv/Scripts/python.exe pregunta3/paso7_informe/ejecutar_todo.py --rapido

# retomar desde un paso, o rehacer todo
.venv/Scripts/python.exe pregunta3/paso7_informe/ejecutar_todo.py --desde 4
.venv/Scripts/python.exe pregunta3/paso7_informe/ejecutar_todo.py --rehacer
```

Y para compilar solo el informe:

```bash
.venv/Scripts/python.exe pregunta3/paso7_informe/compilar_informe.py
.venv/Scripts/python.exe pregunta3/paso7_informe/compilar_informe.py --limpiar
```

---

## Lo que se hizo

### 1. Idempotencia por artefacto, no por marca de tiempo

Cada etapa declara **qué archivo prueba que ya corrió**, y se salta si existe.
Es lo que hace que el comando único sea usable de verdad: la descarga sola supera
las tres horas, el entrenamiento otros cincuenta minutos, y volver a lanzar el
comando tras un corte sigue donde quedó en vez de empezar de nuevo.

No se usan marcas de tiempo a propósito. Comparar fechas de modificación obliga a
decidir qué pasa cuando un `.py` es más nuevo que su salida, y la respuesta
honesta —rehacer— convertiría cualquier edición de un comentario en una corrida
de cuatro horas. `--rehacer` existe para eso, explícito.

Para un directorio, "ya corrió" es **no vacío**, no "existe": un directorio
creado pero vacío es exactamente el estado que deja una corrida que falló al
empezar.

### 2. `--rapido` cambia argumentos, no saltea etapas

Cada etapa cara declara su variante rápida: `--limite 300` para la descarga,
`--epocas 3` para el entrenamiento, `--sin-bootstrap` para las métricas,
`--muestras 100` para la robustez. El circuito completo corre igual, de punta a
punta, con todos sus formatos de archivo y todas sus dependencias entre pasos.

La diferencia con saltear etapas importa: un pipeline que en modo rápido omite
pasos no prueba el pipeline, prueba otro.

### 3. El formato del informe se verifica, no se mira

`compilar_informe.py` comprueba las dos condiciones del enunciado:

- **páginas**, leídas del log de LaTeX (`Output written on ... (N pages`);
- **referencias citadas**, contadas en el `.bbl` ya resuelto — no en el `.bib`,
  que puede tener más entradas de las que el texto usa. Esa distinción es la que
  hace que el chequeo sirva: un `.bib` con quince entradas y un texto que cita
  tres cumpliría el chequeo ingenuo y no el del enunciado.

Se usa `latexmk` si está disponible porque resuelve por sí solo cuántas pasadas
hacen falta para que las citas converjan; sin él, cae a la secuencia clásica de
tres pasadas con `bibtex` en medio.

El tope se chequea contra 10 páginas **de cuerpo**: las referencias en una página
extra son la práctica estándar de NeurIPS y no cuentan como exceso. Es el mismo
criterio con el que quedó el informe de la [Pregunta 1](../../pregunta1/) (10 de
cuerpo, referencias en la 11), y el script se validó contra ese PDF real antes de
usarlo acá.

### 4. Lo que se reutiliza de la Pregunta 1

- `neurips.sty` (escrito a mano: el oficial devuelve 403) y la bibliografía;
- el bootstrap y los intervalos de confianza;
- la convención de un README por paso con *qué se hizo / qué se midió / qué se
  descartó / qué quedó pendiente*;
- el patrón de `ejecutar_todo.py`, con una diferencia: acá cada etapa declara sus
  argumentos de modo rápido en vez de reescribir `--epocas` sobre la línea de
  comando, que era frágil cuando una etapa no tiene ese argumento.

---

## Resultados

El informe compila y **cumple el formato que pide el enunciado**:

```
informe/informe.pdf (1.5 MB)
páginas: 11 (10 de cuerpo + referencias en la 11)
referencias citadas: 11 (mínimo 5)
```

Las 6 tablas y la figura de Grad-CAM se generan desde los JSON de las corridas
con `generar_tablas.py`; ningún número del documento se transcribió a mano.

### Lo que costó entrar en diez páginas

El primer borrador compiló en **12 páginas**. Las dos páginas se recuperaron sin
sacar ningún hallazgo:

1. **Fusionar tablas que medían lo mismo.** Las dos tablas por especie (una por
   partición) más la de brechas repetían el soporte y la AP de cada especie tres
   veces: se unieron en una sola con las dos particiones lado a lado. Igual con
   tamaño + oclusión (las dos mitades del entregable #4) y con el barrido de
   capas + la razón de masa por partición. De 10 flotantes a 6.
2. **La tabla de ubicaciones tenía dos filas** y la prosa de al lado ya las
   decía: pasó a prosa.
3. **Recortar prosa que duplicaba los README de cada paso.** El informe explica
   la decisión; el README del paso explica además cómo se implementó.

El contenido no se movió a un apéndice después de las referencias, que habría
sido la forma fácil de cumplir el límite sin cumplirlo.

---

## Pendiente

| | Qué falta | Por qué importa |
|---|---|---|
| 1 | Los nombres reales del grupo, hoy `Anonimo1`–`Anonimo4` | Es lo único del informe que no sale de una corrida |
| 2 | Que el grupo lea el informe antes de entregarlo | Está escrito con los números medidos, pero defenderlo exige conocerlos |
| 3 | Incluir una figura de Grad-CAM de un acierto, junto a la del fallo | Hoy el informe muestra solo el caso negativo; el contraste sería más fuerte, pero no entra sin sacar otra cosa |

## Estado del informe

| | |
|---|---:|
| cuerpo | **10.00 páginas** (límite 6–10) |
| páginas del PDF | 11, con las referencias aparte |
| referencias citadas | **11** (mínimo 5) |
| errores de LaTeX / citas sin resolver | 0 / 0 |
| tablas / figuras | 6 / 1 |
| Word | `informe.docx`, 6 909 palabras |

Todo esto lo verifica `compilar_informe.py` en cada corrida, contra el log de
LaTeX, el `.bbl` y el PDF —no contra la vista previa—. El límite de páginas es
sobre el **cuerpo**: se mide dónde arranca la bibliografía, no se supone cuánto
ocupa.
