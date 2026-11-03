import cv2
import numpy as np

"""
Objetivo: Crear la mascara binaria para la imagen.
Argumentos:
    - La imagen tipo numpy array 
"""


def binaryMask(img, lower_color, upper_color, start_x, end_x):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower_color), np.array(upper_color))

    # Limito la región donde se aplica la máscara porque algunos píxeles de las frutas
    # verdes y amarillas tienen valores HSV similares. Esto evita que se confundan
    # y ayuda a separarlas correctamente.

    region = np.zeros(mask.shape, dtype=np.uint8)
    region[:, start_x:end_x] = 255

    mask = cv2.bitwise_and(mask, region)
    return mask


"""
Objetivo: Segmentar la imagen imagen 
"""


def segment(img, mask):
    return cv2.bitwise_and(img, img, mask=mask)
