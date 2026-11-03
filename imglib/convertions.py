import numpy as np
from PIL import Image
from .validations import *


"""
Objetivo: Ajustar las dimensiones de una lista de imagenes a 
un mismo tamaño en común. La función asume una validación previa de las 
dimensiones de la imagen antes de ser llamada.
Recive:
    - La imagen np.darray
    - La altura de la imagen 
    - El ancho de la imagen
Devuelve:
    - La imagen con las dimensiones especificadas 
"""


def adjust_dimensions(image, width, height):
    valid_image(image)
    if width <= 0 or not isinstance(width, (int, np.integer)):
        raise ValueError("width debe ser un entero positivo.")
    if height <= 0 or not isinstance(height, (int, np.integer)):
        raise ValueError("height debe ser un entero positivo")

    image = Image.fromarray(image)
    resized = image.resize((int(width), int(height)))
    return np.array(resized)


"""
Objetivo: Descomposición RGB
Recibe:
    - Una imagen RGB de 3 canales
    - El color del canal: rojo, verde o azul
Devuelve:
    - Una imagen donde solo queda visible el canal especificado
Nota: La entrada debe estar en orden RGB, no BGR.
"""


def get_rgb_channels(image, channel):
    is_rgb_image(image)

    if not isinstance(channel, str):
        raise TypeError("Channel argument must be a string.")

    img_color = np.copy(image)
    channel = channel.lower()

    if channel == "red":
        img_color[:, :, 1] = 0
        img_color[:, :, 2] = 0
    elif channel == "green":
        img_color[:, :, 0] = 0
        img_color[:, :, 2] = 0
    elif channel == "blue":
        img_color[:, :, 0] = 0
        img_color[:, :, 1] = 0
    else:
        return None
    return img_color


"""
Objetivo: Descomposición CMY
Recibe:
    - Una imagen RGB de 3 canales
    - El color del canal: cyan, magenta o amarillo
Devuelve:
    - Una imagen donde quedan visibles los canales RGB que componen
      el color CMY especificado
Nota: La entrada debe estar en orden RGB, no BGR.
"""


def get_cmy_channels(image, channel):
    is_rgb_image(image)

    if not isinstance(channel, str):
        raise TypeError("Channel argument must be a string.")

    img_color = np.copy(image)
    channel = channel.lower()

    if channel == "cyan":
        img_color[:, :, 0] = 0
    elif channel == "magenta":
        img_color[:, :, 1] = 0
    elif channel == "yellow":
        img_color[:, :, 2] = 0
    else:
        return None
    return img_color


"""
Objetivo: Convertir una imagen a escala de grises
usando la técnica de promedio (average).
Recibe:
    - Una imagen RGB de 3 canales
Devuelve:
    - Imagen en escala de grises
Nota: La entrada debe estar en orden RGB, no BGR.
"""


def average(image):
    is_rgb_image(image)
    img_g = image.astype(np.float32)
    R = img_g[:, :, 0]
    G = img_g[:, :, 1]
    B = img_g[:, :, 2]
    gray = ((R + G + B) / 3).astype(np.uint8)
    return gray


"""
Objetivo: Convertir una imagen a escala de grises
usando la técnica de luminosidad (luminance).
Recibe:
    - Una imagen RGB de 3 canales
Devuelve:
    - Imagen en escala de grises
Nota: La entrada debe estar en orden RGB, no BGR.
"""


def luminity(image):
    is_rgb_image(image)
    img_g = image.astype(np.float32)
    R = img_g[:, :, 0]
    G = img_g[:, :, 1]
    B = img_g[:, :, 2]
    gray = 0.299 * R + 0.587 * G + 0.114 * B
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    return gray


"""
Objetivo: Convertir una imagen a escala de grises
usando la técnica de tonalidad (midgray).
Recibe:
    - Una imagen RGB de 3 canales
Devuelve:
    - Imagen en escala de grises
Nota: La entrada debe estar en orden RGB, no BGR.
"""


def midgray(image):
    is_rgb_image(image)
    img_g = image.astype(np.float32)
    mx = np.max(img_g, axis=2)
    mn = np.min(img_g, axis=2)
    gray = (mx + mn) / 2
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    return gray


"""
Objetivo: Aplicar binarización a una imagen. Es decir,
convertir una imagen en escala de grises a una imagen
en blanco y negro.
Recibe:
    - Una imagen 2D tipo np.ndarray en escala de grises
    - El umbral T, entre 0 y 255
Devuelve:
    - Una imagen binaria uint8 con valores 0 y 255
"""


def binarize(image, T):
    if not isinstance(T, (int, float, np.integer, np.floating)):
        raise TypeError("T must be a numeric value")
    if not 0 <= T <= 255:
        raise ValueError("T must be in the range 0 to 255")

    return np.where(image >= T, 255, 0).astype(np.uint8)
