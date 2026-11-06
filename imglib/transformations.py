import cv2
import numpy as np
from .validations import valid_image, is_numeric, is_integer

"""
Objetivo: Traslación de una imagen. Es decir, desplazar todos
los pixeles de una imágen una distancia fija del plano
Recive:
    - La imagen tipo np.darray
    - Desplazamiento Horizontal (X)
    - Desplazamiento Vertical (Y)
Devuelve:
    - La imagen trasladada
"""


def img_translation(image, dx, dy):
    valid_image(image)

    if not is_integer(dx):
        raise TypeError(
            f"Function expects integer value for 'dx' but received {dx} instead."
        )
    if not is_integer(dy):
        raise TypeError(
            f"Function expects integer value for 'dy' but received {dy} instead."
        )

    h, w = image.shape[:2]

    translated = np.zeros_like(image)

    # Coordenadas de la imágen destino
    x1d = max(0, dx)
    y1d = max(0, dy)
    x2d = min(w, w + dx)
    y2d = min(h, h + dy)

    if x2d > x1d and y2d > y1d:
        # Coordenadas en la imágen original
        x1s = max(0, -dx)
        y1s = max(0, -dy)
        x2s = x1s + (x2d - x1d)
        y2s = y1s + (y2d - y1d)
        translated[y1d:y2d, x1d:x2d] = image[y1s:y2s, x1s:x2s]

    return translated


"""
Objetivo: Recortar imagen 
Recive:
    - La imagen tipo np.darray
    - 
    - 
Devuelve:
    - La imagen recortada 
"""


def crop_image(image, x_start, x_end, y_start, y_end):
    valid_image(image)

    coords = (x_start, x_end, y_start, y_end)
    for coord in coords:
        if not is_integer(coord):
            raise TypeError("Image coordinates must be integers")

    height, width = image.shape[:2]

    if not (0 <= x_start < x_end <= width):
        raise ValueError("Horizontal coordinates exceed the image")
    if not (0 <= y_start < y_end <= height):
        raise ValueError("Vertical coordinates exceed the image")

    return image[y_start:y_end, x_start:x_end].copy()


"""
Objetivo: Reducir la resolución de una imagen
Recive:
    - La imagen tipo np.darray
    - El factor de reducción
Devuelve:
    - La imagen con resolución reducida
"""


def reduce_resolution(image, factor):
    valid_image(image)

    if not is_integer(factor) or factor <= 0:
        raise TypeError(
            f"Factor must be an positive integer value but received value {factor}."
        )

    return image[::factor, ::factor].copy()


"""
Objetivo: Rotar una imagen usando cv2.rotate 
Recive:
    - La imagen tipo np.darray
    - El angulo de rotación en grados
Devuelve:
    - La imagen rotada
"""


def rotate_image(image, angle):
    valid_image(image)

    if not is_numeric(angle):
        raise TypeError(
            f"Function expects a numeric value for 'angle' but received {angle} instead."
        )

    angle = angle % 360

    if angle == 0:
        return image.copy()
    elif angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    else:  # 270
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)


"""
Objetivo: Ampliar una zona  de la imagen (Zoom).
Recive:
    - La imagen tipo numpy array
    - El tamaño de la region 
    - El factor de ampliación
Devuelve:
    - La imagen ampliada 
"""


def zoom(image, area, factor):
    valid_image(image)

    if not isinstance(area, (int, np.integer)) or area <= 0:
        raise ValueError(
            f"Function expects positive integer for 'area' but received {area} instead."
        )

    if not is_numeric(factor) or factor <= 0:
        raise ValueError(
            f"Function expects positive numeric value for 'factor' but received {factor} instead."
        )

    height, width = image.shape[:2]

    if area > min(height, width):
        raise ValueError("Area cannot exceed the smaller side of the image")

    start_r = (height // 2) - (area // 2)
    end_r = (height // 2) + (area // 2)
    start_c = (width // 2) - (area // 2)
    end_c = (width // 2) + (area // 2)

    region = image[start_r:end_r, start_c:end_c]

    if region.ndim == 2:
        zoomed = np.kron(region, np.ones((factor, factor)))
    else:
        zoomed = np.kron(region, np.ones((factor, factor, 1)))
    return zoomed.astype(image.dtype)
