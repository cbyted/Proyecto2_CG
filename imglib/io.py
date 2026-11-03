from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def load_image(path):
    image = plt.imread(path)

    # PNG suele llegar como float en [0, 1].
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0, 1)
        image = np.rint(image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)

    # Grises a RGB, para tener siempre tres canales de color.
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)

    # RGBA a RGB, mezclando la transparencia con fondo blanco.
    if image.shape[-1] == 4:
        rgb = image[:, :, :3].astype(np.float32)
        alpha = image[..., 3:].astype(np.float32) / 255.0
        image = rgb * alpha + (1.0 - alpha) * 255.0
        image = np.rint(image).astype(np.uint8)

    return image


def save_image(image, path, format):
    valid_formats = {
        "PNG",
        "JPG",
        "JPEG",
    }

    if format.upper() not in valid_formats:
        raise ValueError("El formato de la imagen debe ser PNG, JPG y JPEG")

    image = np.asarray(image)

    # Pillow espera valores de 8 bits para imágenes RGB/RGBA o escala de grises.
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    # Pillow usa "JPEG" como nombre de formato, aunque la extensión sea .jpg.
    pillow_format = "JPEG" if format in {"JPG", "JPEG"} else "PNG"

    path = Path(path)
    if not path.suffix:
        path = path.with_suffix(".jpg" if pillow_format == "JPEG" else ".png")

    Image.fromarray(image).save(path, format=pillow_format)
