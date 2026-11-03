import numpy as np

"""
import cv2
Objetivo: Validar si la imagen cargada tiene un formato adecuado
Recive:
    - La imagen 
    - Los tipos de datos soportados  
Devuelve:
    - Nada
"""


def valid_image(image):
    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be of type numpy array.")
    if image.size == 0:
        raise ValueError("Image is empty. Please use a valid image.")
    if image.ndim not in (2, 3):
        raise ValueError("Invalid dimentions. Image must be 2D or 3D.")


"""
Objetivo: Validar una lista de imágenes.
Recibe:
    - Una lista de imagenes 
Devuelve:
    - La lista de imágenes
"""


def valid_images(images):
    try:
        images = list(images)
    except TypeError as error:
        raise TypeError(
            f"Function expects a list of images. Received {type(images)}"
        ) from error

    if len(images) == 0:
        raise ValueError(
            "List of images must contain at least one image. Received zero images."
        )

    for image in images:
        valid_image(image)


"""
Objetivo: Validar que un dato sea numérico (integer and float).
Recive:
    - Una variable
Devuelve:
    - True | False 
"""


def is_numeric(value):
    return isinstance(value, (int, float, np.integer, np.floating))


"""
Objetivo: Validar que un dato sea entero (integer).
Recive:
    - Una variable
Devuelve:
    - True | False 
"""


def is_integer(value):
    return isinstance(value, (int, np.integer))


"""
Objetivo: Validar que la imagen sea RGB de 3 canales y de 
tipo uint8.
Recive:
    - La imagen tipo numpy array
Devuelve:
    - True | False  
"""


def is_rgb_image(image):
    valid_image(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Image must be RGB with three channels.")
    return True


"""
Objetivo: Validar que la imagen sea 2D en escala de grises y de 
tipo uint8.
Recive:
    - La imagen tipo numpy array
Devuelve:
    - True | False  
"""


def is_gray_image(image):
    valid_image(image)
    return image.ndim == 2
