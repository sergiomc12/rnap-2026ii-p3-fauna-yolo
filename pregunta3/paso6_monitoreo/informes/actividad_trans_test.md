# Actividad de fauna — trans_test

Confianza mínima 0.25, eventos independientes con ventana de 30 minutos. 9 cámaras, 2,090 noches-cámara.

## Actividad por especie

| Especie | Eventos | Detecciones | Cámaras | Eventos/100 noches | Ocupación naive | Nocturnidad | Esperado |
|---|---:|---:|---:|---:|---:|---:|---|
| opossum | 1,302 | 4,473 | 8 | 77.64 | 89% | 73% | nocturna |
| raccoon | 1,016 | 3,780 | 9 | 48.61 | 100% | 70% | nocturna |
| cat | 786 | 1,869 | 9 | 37.61 | 100% | 46% | variable |
| coyote | 724 | 2,271 | 9 | 34.64 | 100% | 58% | crepuscular |
| car | 679 | 1,201 | 2 | 89.93 | 22% | 4% | diurna |
| bobcat | 655 | 1,518 | 9 | 31.34 | 100% | 52% | crepuscular |
| rabbit | 403 | 973 | 9 | 19.28 | 100% | 28% | crepuscular |
| dog | 309 | 1,077 | 9 | 14.78 | 100% | 7% | variable |
| squirrel | 252 | 560 | 9 | 12.06 | 100% | 2% | diurna |
| bird | 244 | 550 | 9 | 11.67 | 100% | 20% | diurna |
| skunk | 226 | 511 | 9 | 10.81 | 100% | 71% | nocturna |
| rodent | 25 | 38 | 4 | 2.81 | 44% | 60% | nocturna |
| fox | 9 | 10 | 3 | 1.07 | 33% | 89% | nocturna |
| deer | 5 | 13 | 2 | 0.92 | 22% | 0% | crepuscular |
| badger | 2 | 2 | 1 | 0.48 | 11% | 100% | nocturna |

## Error contra las cajas anotadas

| Especie | Eventos anotados | Eventos del modelo | Error | Error relativo |
|---|---:|---:|---:|---:|
| cat | 235 | 786 | +551 | +234% |
| rabbit | 194 | 403 | +209 | +108% |
| bird | 64 | 244 | +180 | +281% |
| coyote | 557 | 724 | +167 | +30% |
| opossum | 1,142 | 1,302 | +160 | +14% |
| raccoon | 880 | 1,016 | +136 | +15% |
| dog | 178 | 309 | +131 | +74% |
| skunk | 125 | 226 | +101 | +81% |
| bobcat | 580 | 655 | +75 | +13% |
| car | 608 | 679 | +71 | +12% |
| squirrel | 237 | 252 | +15 | +6% |
| rodent | 10 | 25 | +15 | +150% |
| fox | 1 | 9 | +8 | +800% |
| badger | 8 | 2 | -6 | -75% |
| deer | 0 | 5 | +5 | n/d |
