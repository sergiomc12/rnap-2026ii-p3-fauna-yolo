# Actividad de fauna — cis_test

Confianza mínima 0.25, eventos independientes con ventana de 30 minutos. 10 cámaras, 1,131 noches-cámara.

## Actividad por especie

| Especie | Eventos | Detecciones | Cámaras | Eventos/100 noches | Ocupación naive | Nocturnidad | Esperado |
|---|---:|---:|---:|---:|---:|---:|---|
| opossum | 750 | 3,711 | 9 | 70.89 | 90% | 80% | nocturna |
| coyote | 564 | 1,262 | 10 | 49.87 | 100% | 51% | crepuscular |
| raccoon | 410 | 1,338 | 9 | 47.56 | 90% | 77% | nocturna |
| cat | 360 | 1,514 | 7 | 47.94 | 70% | 35% | variable |
| bobcat | 342 | 980 | 9 | 35.04 | 90% | 49% | crepuscular |
| car | 319 | 798 | 2 | 99.07 | 20% | 8% | diurna |
| rabbit | 310 | 1,577 | 10 | 27.41 | 100% | 24% | crepuscular |
| dog | 192 | 701 | 9 | 18.15 | 90% | 11% | variable |
| squirrel | 154 | 499 | 9 | 14.56 | 90% | 0% | diurna |
| bird | 139 | 685 | 8 | 14.79 | 80% | 12% | diurna |
| skunk | 87 | 164 | 8 | 8.35 | 80% | 66% | nocturna |
| rodent | 41 | 138 | 5 | 7.90 | 50% | 88% | nocturna |
| deer | 34 | 188 | 4 | 5.27 | 40% | 50% | crepuscular |
| fox | 4 | 4 | 3 | 0.98 | 30% | 75% | nocturna |

## Error contra las cajas anotadas

| Especie | Eventos anotados | Eventos del modelo | Error | Error relativo |
|---|---:|---:|---:|---:|
| raccoon | 236 | 410 | +174 | +74% |
| bobcat | 260 | 342 | +82 | +32% |
| coyote | 483 | 564 | +81 | +17% |
| cat | 297 | 360 | +63 | +21% |
| rabbit | 250 | 310 | +60 | +24% |
| dog | 135 | 192 | +57 | +42% |
| bird | 96 | 139 | +43 | +45% |
| squirrel | 131 | 154 | +23 | +18% |
| opossum | 735 | 750 | +15 | +2% |
| skunk | 78 | 87 | +9 | +12% |
| car | 312 | 319 | +7 | +2% |
| deer | 27 | 34 | +7 | +26% |
| rodent | 39 | 41 | +2 | +5% |
| badger | 2 | 0 | -2 | -100% |
| fox | 2 | 4 | +2 | +100% |
