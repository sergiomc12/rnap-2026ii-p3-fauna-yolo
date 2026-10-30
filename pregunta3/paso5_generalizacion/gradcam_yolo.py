"""Grad-CAM sobre el backbone de YOLO, para el entregable #3.

Grad-CAM se definió para clasificadores: hay un escalar por clase del que se
toma el gradiente. Un detector no tiene ese escalar — tiene miles de anclas,
cada una con su caja y sus quince puntajes de clase. Hay que elegir qué
derivar, y la elección cambia lo que el mapa significa:

* derivar el puntaje de **una ancla** muestra qué miró el modelo para esa
  detección puntual;
* derivar la **suma de los puntajes de una clase** sobre las anclas más
  confiadas muestra dónde está la evidencia de que *hay* un animal de esa
  especie en la imagen.

Acá se usa la segunda, con las ``k`` anclas de mayor puntaje: es la que
responde la pregunta del entregable — si el modelo mira al animal o mira el
fondo de la cámara. Con una sola ancla el mapa queda dominado por el ruido de
esa ancla en particular.

La capa se eligió **midiendo**, no por convención. Lo habitual en
clasificación es tomar la última capa convolucional, que acá sería el bloque
C2PSA del índice 10; con 640 px de entrada ese bloque tiene una rejilla de
20x20, es decir 32 px por celda. Un animal que ocupa el 2% del cuadro cae
dentro de una sola celda, y el mapa resultante no localiza nada.

Barrido sobre 30 imágenes de ``cis_test``, midiendo la razón entre la masa de
activación dentro de las cajas anotadas y el área que esas cajas ocupan (1.0 =
lo mismo que un mapa uniforme, es decir ninguna localización):

| Capa | Bloque | Rejilla | Razón media | Mediana | Mapas vacíos |
|---|---|---|---:|---:|---:|
| 4 | C3k2 | 80×80 | 1.59 | 1.09 | 0/30 |
| 6 | C3k2 | 40×40 | 1.38 | 1.01 | 0/30 |
| 8 | C3k2 | 20×20 | 1.32 | 0.66 | 9/30 |
| 9 | SPPF | 20×20 | 1.56 | 0.75 | 3/30 |
| 10 | C2PSA | 20×20 | 0.95 | 0.79 | 5/30 |

Por eso el default es la **capa 4**. La media no la separa de la capa 9 (1.59
contra 1.56); lo que las separa son la mediana (1.09 contra 0.75) y los mapas
idénticamente nulos dentro del animal (0/30 contra 3/30). Las tres capas de
20x20 producen esos mapas nulos en 3 a 9 de cada 30 imágenes: no es que miren
el fondo, es que no tienen resolución para decir nada.

Que incluso la mejor capa quede en 1.59 — poco más de un mapa uniforme —
es en sí un resultado, y no un defecto del método: coincide con lo que Beery
et al. (2018) describen para este dominio, donde el detector se apoya en el
fondo de cada cámara tanto como en el animal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch

# Capa del backbone con mejor localización medida (ver la tabla del docstring).
CAPA_BACKBONE = 4
# Las otras candidatas del barrido, para poder reproducirlo.
CAPAS_CANDIDATAS = (4, 6, 8, 9, 10)


def letterbox(imagen: np.ndarray, resolucion: int = 640,
              gris: int = 114) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Reescala conservando la proporción y rellena con gris hasta el cuadrado.

    Es el mismo preprocesamiento de Ultralytics. Reproducirlo acá, en vez de
    reusar su ``predict``, es inevitable: ``predict`` corre bajo ``no_grad`` y
    Grad-CAM necesita el grafo de gradientes.
    """
    import cv2

    alto, ancho = imagen.shape[:2]
    escala = min(resolucion / alto, resolucion / ancho)
    nuevo = (int(round(ancho * escala)), int(round(alto * escala)))
    reescalada = cv2.resize(imagen, nuevo, interpolation=cv2.INTER_LINEAR)

    lienzo = np.full((resolucion, resolucion, 3), gris, dtype=np.uint8)
    arriba = (resolucion - nuevo[1]) // 2
    izquierda = (resolucion - nuevo[0]) // 2
    lienzo[arriba:arriba + nuevo[1], izquierda:izquierda + nuevo[0]] = reescalada
    return lienzo, escala, (izquierda, arriba)


def a_tensor(imagen_bgr: np.ndarray, dispositivo: str = "cpu") -> torch.Tensor:
    """BGR uint8 -> tensor NCHW float en [0, 1], en RGB."""
    rgb = imagen_bgr[:, :, ::-1].copy()
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0)
    return tensor.unsqueeze(0).to(dispositivo)


@dataclass
class MapaActivacion:
    """El resultado de una pasada de Grad-CAM."""

    mapa: np.ndarray                 # [h, w] en [0, 1]
    clase: int
    puntaje: float                   # suma de puntajes derivada
    anclas: int


class GradCAMDeteccion:
    """Grad-CAM sobre una capa del backbone de un detector de Ultralytics."""

    def __init__(self, modelo, capa: int = CAPA_BACKBONE,
                 dispositivo: str = "cpu") -> None:
        self.red = modelo.model if hasattr(modelo, "model") else modelo
        self.red = self.red.to(dispositivo).float().eval()
        self.dispositivo = dispositivo
        self.capa = self.red.model[capa]
        self._activaciones: Optional[torch.Tensor] = None
        self._gradientes: Optional[torch.Tensor] = None

        # Los hooks quedan registrados mientras viva el objeto. El de backward
        # tiene que ser full_backward: el antiguo register_backward_hook no
        # recibe los gradientes de módulos con varias salidas.
        self._manijas = [
            self.capa.register_forward_hook(self._guardar_activacion),
            self.capa.register_full_backward_hook(self._guardar_gradiente),
        ]

    def _guardar_activacion(self, modulo, entrada, salida) -> None:
        self._activaciones = salida

    def _guardar_gradiente(self, modulo, grad_entrada, grad_salida) -> None:
        self._gradientes = grad_salida[0]

    def cerrar(self) -> None:
        """Quita los hooks y suelta los tensores intermedios.

        Soltarlos importa: mientras el hook siga apuntando a un tensor que
        no es hoja del grafo, cualquier ``deepcopy`` del modelo falla, y
        Ultralytics clona el modelo en cada ``predict``.
        """
        for manija in self._manijas:
            manija.remove()
        self._manijas = []
        self._activaciones = self._gradientes = None

    def __enter__(self) -> "GradCAMDeteccion":
        return self

    def __exit__(self, *_) -> None:
        self.cerrar()

    def calcular(self, tensor: torch.Tensor, clase: Optional[int] = None,
                 k: int = 10) -> MapaActivacion:
        """Mapa de activación para ``clase`` (la más confiada si es ``None``)."""
        self.red.zero_grad(set_to_none=True)
        salida = self.red(tensor)
        predicciones = salida[0] if isinstance(salida, (tuple, list)) else salida

        # [1, 4 + nc, anclas]: las primeras cuatro filas son la caja.
        puntajes = predicciones[0, 4:, :]
        if clase is None:
            clase = int(puntajes.max(dim=1).values.argmax().item())

        de_la_clase = puntajes[clase]
        k = max(1, min(k, de_la_clase.numel()))
        mejores = torch.topk(de_la_clase, k)
        objetivo = mejores.values.sum()
        objetivo.backward()

        if self._activaciones is None or self._gradientes is None:
            raise RuntimeError("los hooks no capturaron nada: ¿capa equivocada?")

        # Peso de cada canal: gradiente promediado sobre el plano espacial.
        pesos = self._gradientes.mean(dim=(2, 3), keepdim=True)
        mapa = torch.relu((pesos * self._activaciones).sum(dim=1))[0]

        arreglo = mapa.detach().cpu().numpy()
        maximo = float(arreglo.max())
        if maximo > 0:
            arreglo = arreglo / maximo
        return MapaActivacion(mapa=arreglo, clase=int(clase),
                              puntaje=float(objetivo.item()), anclas=k)


def mapa_en_imagen(mapa: np.ndarray, resolucion: int, escala: float,
                   desplazamiento: tuple[int, int],
                   ancho: int, alto: int) -> np.ndarray:
    """Lleva el mapa del espacio del letterbox al de la imagen original.

    El mapa que devuelve la red mide 20x20 y corresponde al **cuadrado
    rellenado**, no a la foto: una imagen de 2048x1494 entra como 640x467 con
    86 píxeles de gris arriba y abajo. Estirar el mapa de 20x20 directamente al
    tamaño original desplaza todo verticalmente y hace que la activación
    aparezca donde no está — se midió una masa dentro de las cajas anotadas de
    0.0001 por este error, antes de corregirlo.
    """
    import cv2

    grande = cv2.resize(mapa, (resolucion, resolucion),
                        interpolation=cv2.INTER_LINEAR)
    izquierda, arriba = desplazamiento
    alto_util = int(round(alto * escala))
    ancho_util = int(round(ancho * escala))
    recorte = grande[arriba:arriba + alto_util, izquierda:izquierda + ancho_util]
    return cv2.resize(recorte, (ancho, alto), interpolation=cv2.INTER_LINEAR)


def superponer(imagen_bgr: np.ndarray, mapa: np.ndarray,
               alfa: float = 0.5) -> np.ndarray:
    """Pinta el mapa como mapa de calor sobre la imagen."""
    import cv2

    alto, ancho = imagen_bgr.shape[:2]
    redimensionado = cv2.resize(mapa, (ancho, alto),
                                interpolation=cv2.INTER_LINEAR)
    calor = cv2.applyColorMap(np.uint8(255 * redimensionado), cv2.COLORMAP_JET)
    return cv2.addWeighted(imagen_bgr, 1 - alfa, calor, alfa, 0)


def masa_dentro_de_cajas(mapa: np.ndarray, cajas: Sequence[Sequence[float]],
                         ancho: int, alto: int) -> float:
    """Fracción de la activación que cae dentro de alguna caja anotada.

    Un valor alto significa que el modelo mira al animal; uno bajo, que la
    evidencia que usa está en el fondo — exactamente el sesgo que el entregable
    #3 pide investigar. Las cajas se unen antes de sumar, así que dos cajas
    solapadas no cuentan dos veces la misma activación.
    """
    import cv2

    if not len(cajas):
        return 0.0
    redimensionado = cv2.resize(mapa, (ancho, alto),
                                interpolation=cv2.INTER_LINEAR)
    total = float(redimensionado.sum())
    if total <= 0:
        return 0.0

    mascara = np.zeros((alto, ancho), dtype=bool)
    for x1, y1, x2, y2 in cajas:
        a, b = max(0, int(x1)), max(0, int(y1))
        c, d = min(ancho, int(x2)), min(alto, int(y2))
        if c > a and d > b:
            mascara[b:d, a:c] = True
    return float(redimensionado[mascara].sum()) / total
