import numpy as np
from .convertions import adjust_dimensions
from .validations import valid_image, valid_images, is_numeric

"""
Objetivo: Ajustar el brillo de una imagen usando una
constante para cada pixel.
    Im = I + b
si b > 0 la imagen se aclara
si b < 0 la imagen se oscurece
Recibe:
    - La imagen tipo numpy array, con dtype uint8
    - La constante de ajuste
Devuelve:
    - La imagen con el brillo ajustado
"""


def adjust_brightness(image, factor):
    valid_image(image)
    if not is_numeric(factor):
        raise TypeError("factor must be a numeric value.")

    image = image.astype(np.float32)
    return np.clip(image + factor, 0, 255).astype(np.uint8)


"""
Objetivo: Ajustar el brillo de una imagen por medio de un canal
específico de una imagen.
    Im = I + b

Recibe:
    - La imagen tipo numpy array, con dtype uint8
    - El índice del canal a modificar
    - La constante de ajuste
Devuelve:
    - La imagen con el brillo ajustado en el canal especificado
"""


def adjust_channel_brightness(image, channel, factor):
    valid_image(image)

    if image.ndim != 3:
        raise ValueError(
            f"Function requires an image with 3 dimentions (3D). Received image with {image.ndim}."
        )

    if not isinstance(channel, (int, np.integer)):
        raise TypeError("Channel must be a numeric value")

    if channel < 0 or channel >= image.shape[2]:
        raise ValueError(f"Channel must be a number between 0 y {image.shape[2] - 1}")

    if not is_numeric(factor):
        raise TypeError(f"Factor must be a numeric value. Received value {factor}.")

    new_img = image.astype(np.float32).copy()
    new_img[:, :, channel] = np.clip(new_img[:, :, channel] + factor, 0, 255)
    return new_img.astype(np.uint8)


"""
Objetivo: Invertir colores de una imagen 
Soporta:
    - uint8:  [0, 255]
    - uint16: [0, 65535]
    - floats: [0.0, 1.0]
Por defecto, el canal alfa no se invierte
Recive:
    - una imagen tipo np.array
    - Booleano que determina si se invierte el canal alpha
Devuelve:
    - La imagen con los colores invertidos
"""


def inv_color(image):
    valid_image(image)

    if image.dtype == np.uint8:
        max_val = 255
    elif image.dtype == np.uint16:
        max_val = 65535
    else:
        if np.nanmin(image) < 0 or np.nanmax(image) > 1:
            raise ValueError("Floating images must be in range 0.0 to 1.0")
        max_val = 1.0

    inv_image = image.copy()

    if image.ndim == 3 and image.shape[2] == 4:  # Modo RGBA
        inv_image[:, :, :3] = max_val - image[:, :, :3]
    else:
        inv_image = max_val - image

    return inv_image


"""
Objetivo: Suma de dos imagenes.
Recive:
    - La lista de imagenes np.darray a sumar 
Devuelve:
    - La imagen resultante de la suma
"""


def sum_images(images):
    valid_images(images)

    if len(images) != 2:
        raise ValueError("Se deben proporcionar exactamente 2 imágenes.")

    image1, image2 = images

    if image1.ndim != image2.ndim:
        return None

    target_shape = image1.shape

    if image2.shape != target_shape:
        image2 = adjust_dimensions(image2, target_shape[1], target_shape[0])
        if image2.shape != target_shape:
            raise ValueError(
                f"La segunda imagen debe tener forma {target_shape}; "
                f"tiene forma {image2.shape} después del ajuste."
            )

    total = image1.astype(np.float32) + image2.astype(np.float32)
    result = total // 2
    return np.clip(result, 0, 255).astype(np.uint8)


"""
Objetivo: Suma de imagenes ponderada. Soporporta maximo dos imagenes.
Recive:
    - La lista de imagenes np.darray a sumar
    - La constante alpha para la suma 
Devuelve:
    - La imagen resultante de la suma ponderada 
"""


def weighted_image_sum(images, alpha):
    valid_images(images)

    if len(images) != 2:
        raise ValueError("Se deben proporcionar exactamente 2 imágenes.")

    if not is_numeric(alpha):
        raise TypeError(f"Alpha debe ser un valor numérico. Se recibió {type(alpha)}.")

    if not 0 <= alpha <= 1:
        raise ValueError(f"Alpha debe estar entre 0 y 1. Se recibió {alpha}.")

    image1, image2 = images

    if image1.ndim != image2.ndim:
        return None

    target_shape = image1.shape

    if image2.shape != target_shape:
        image2 = adjust_dimensions(image2, target_shape[1], target_shape[0])
        if image2.shape != target_shape:
            raise ValueError(
                f"La segunda imagen debe tener forma {target_shape}; "
                f"tiene forma {image2.shape} después del ajuste."
            )

    result = alpha * image1.astype(np.float32) + (1 - alpha) * image2.astype(np.float32)
    return np.clip(result, 0, 255).astype(np.uint8)
