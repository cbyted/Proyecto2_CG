import numpy as np
import matplotlib.pyplot as plt


"""
Objetivo: Obtener el canal especificado para el histograma.
Recive:
    - una imagen np.darray 
    - El nombre del color para el canal (Debe ser RGB o CMY) 
        * Rojo: "red"
        * Verde: "green"
        * Azul: "blue"
        * Cyan: "cyan"
        * Magenta: "magenta"
        * Amarillo. "yellow"
Devuelve:
    - El canal de la imagen 
"""


def get_color(image, channel=None, custom=None):
    channels = {
        "red": 0,
        "green": 1,
        "blue": 2,
        "cyan": 0,
        "magenta": 1,
        "yellow": 2,
    }

    if channel not in channels:
        raise ValueError(f"Canal desconocido: {channel}")

    # Convierte la imagen a np.darray si aún no lo es
    image = np.asarray(image)

    # convierte una imagen tipo float a uint8
    if image.max() <= 1.0:
        image = image * 255

    # Convertir a uint8 para mostrar los valores de [0, 255]
    image = np.clip(image, 0, 255).astype(np.uint8)

    dim = image.ndim
    if dim == 2:
        color = [image]
    elif dim == 3:
        color = image[:, :, channels[channel]]
        if channel in ("cyan", "magenta", "yellow"):
            color = 255 - color
    else:
        print("Image must be in 2D o 3D")

    return color


"""
Objetivo: Calcular el histograma de cada canal RGB y devolver los datos de estos.
Recive:
    - una imagen
    - El canal que se quiere mostrar 
      Soporta canales RGB y CMY:
        * Rojo: "red"
        * Verde: "green"
        * Azul: "blue"
        * Cyan: "cyan"
        * Magenta: "magenta"
        * Amarillo. "yellow"
Devuelve:
    - Los histogramas de la imagen 
"""


def show_histogram(image, channel):
    if image is None:
        print("Please enter a valid image.")
        return False

    if channel is None:
        print("Please specify a channel.")
        return False

    color = get_color(image, channel)

    plt.figure(figsize=(10, 6))
    plt.subplot(1, 1, 1)
    plt.hist(color.ravel(), bins=256, color=channel)
    plt.title(f"Histogram for color {channel}")
    plt.show()
