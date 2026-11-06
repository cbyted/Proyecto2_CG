from imglib.io import *
from imglib.histogram import *
from imglib.validations import *
from imglib.convertions import *
from imglib.operations import *
from imglib.segmentation import *
from imglib.transformations import *

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

from PyQt6.QtWidgets import (
    QFileDialog,
    QDialog,
    QVBoxLayout,
    QMessageBox,
)


class Controller:
    def __init__(self, window):
        """
        Objetivo: Inicializar el controlador y conectar los eventos de la interfaz.
        Recibe:
            - window: ventana principal de la aplicación.
        Devuelve:
            - No devuelve ningún valor.
        """

        self.window = window
        self.current_image = None
        self.original_image = None
        self.second_image = None

        # Brillo
        self.window.brightness_slider.valueChanged.connect(self.brightness)
        self.window.channel_brightness_selector.currentTextChanged.connect(
            self.channel_brightness
        )
        self.window.channel_brightness_slider.valueChanged.connect(
            self.channel_brightness
        )

        # Escala grises
        self.window.grayscale_selector.currentTextChanged.connect(self.grayscale)

        # Binarización
        self.window.binaryization_slider.valueChanged.connect(self.binarization)

        # Canales RGB y CMY
        self.window.channel_show_btn.clicked.connect(self.show_channel)

        # Sumar imagenes
        self.window.image_selector_btn.clicked.connect(self.select_second_image)
        self.window.add_btn.clicked.connect(self.add_images)

        # Reducir resolución
        self.window.factor_slider.valueChanged.connect(self.reduce)

        # Rotar imágen
        self.window.rotate_spinBox.valueChanged.connect(self.rotate)

        # Zoom
        self.window.zoom_slider.valueChanged.connect(self.zoom_image)

        # Invertir color
        self.window.inv_color_btn.clicked.connect(self.inv_image_color)

        # Histogramas
        self.window.show_hist_btn.clicked.connect(self.histograms)

        # Recorte
        self.window.crop_btn.clicked.connect(self.crop)

        # Restablecer imagen
        self.window.reset_btn.clicked.connect(self.reset_image)

    def set_image(self, image):
        """
        Objetivo: Guardar la imagen original y establecer la imagen de trabajo.
        Recibe:
            - Una imagen o None
        Devuelve:
            - No devuelve ningún valor.
        """
        self.original_image = image.copy() if image is not None else None
        self.current_image = image.copy() if image is not None else None
        self.set_crop_valid_ranges()

    def apply_result(self, result):
        """
        Objetivo: Guardar y mostrar el resultado de una transformación.
        Recibe:
            - result: imagen resultante tipo np.ndarray
        Realiza:
            - Actualiza la imagen de trabajo del controlador y de la ventana
            - Muestra el resultado
        """
        if result is None:
            return

        self.current_image = result
        self.window.image = result
        self.window.display_image(result)

    def brightness(self, value):
        """
        Objetivo: Ajustar el brillo de la imagen de trabajo.
        Recibe:
            - value: cantidad de brillo indicada por el usuario.
        Devuelve:
            - No devuelve ningún valor. Muestra la imagen modificada.
        """
        self.window.brightnessValue_label.setText(f"{value} %")

        if self.current_image is None:
            return

        result = adjust_brightness(self.current_image, value)
        self.window.display_image(result)

    def channel_brightness(self, value=None):
        """
        Objetivo: Ajustar el brillo de un canal RGB seleccionado.
        Recibe:
            - value: valor opcional enviado por la señal de un widget.
        Devuelve:
            - No devuelve ningún valor. Muestra la imagen modificada.
        """
        channel_values = {
            "Rojo": 0,
            "Verde": 1,
            "Azul": 2,
        }

        factor = self.window.channel_brightness_slider.value()
        value = f"{factor} %"
        self.window.channelBrightnessValue_label.setText(value)

        if (self.current_image is None) or (self.current_image.ndim != 3):
            return

        selected_channel = self.window.channel_brightness_selector.currentText()
        channel = channel_values[selected_channel]

        result = adjust_channel_brightness(self.current_image, channel, factor)
        self.window.display_image(result)

    def grayscale(self, method):
        """
        Objetivo: Convertir la imagen de trabajo a escala de grises
        usando el método seleccionado.
        Recibe:
            - method: método de conversión seleccionado en la interfaz.
        Devuelve:
            - No devuelve ningún valor. Muestra la imagen en escala de grises.
        """
        if self.current_image is None:
            return

        if method == "Original":
            result = self.original_image
        elif method == "Promedio":
            result = average(self.current_image)
        elif method == "Midgray":
            result = midgray(self.current_image)
        elif method == "Luminosidad":
            result = luminity(self.current_image)

        self.apply_result(result)

    def binarization(self, value):
        """
        Objetivo: Binarizar la imagen en escala de grises con un umbral.
        Recibe:
            - value: valor del umbral seleccionado.
        Devuelve:
            - No devuelve ningún valor. Muestra la imagen binarizada.
        """
        self.window.binaryization_label_2.setText(f"{value} %")

        if (self.current_image is None) or (is_gray_image(self.current_image) == False):
            return

        result = binarize(self.current_image, value)
        self.window.display_image(result)

    def show_channel(self, value=None):
        """
        Objetivo: Extraer y mostrar el canal seleccionado de la imagen.
        Recibe:
            - value: valor opcional enviado por la señal del botón.
        Devuelve:
            - No devuelve ningún valor. Muestra el canal seleccionado.
        """
        channel_values = {
            "Rojo": "red",
            "Verde": "green",
            "Azul": "blue",
            "Cian": "Cyan",
            "Magenta": "magenta",
            "Amarillo": "yellow",
        }

        if (self.current_image is None) or (self.current_image.ndim != 3):
            return

        selected_channel = self.window.channel_selector.currentText()
        channel = channel_values[selected_channel]

        result = get_rgb_channels(self.current_image, channel)
        if result is None:
            result = get_cmy_channels(self.current_image, channel)

        self.apply_result(result)

    def open_second_image(self):
        """
        Objetivo: Abrir un diálogo para seleccionar una segunda imagen.
        Recibe:
            - No recibe argumentos adicionales.
        Devuelve:
            - La imagen cargada o None si se cancela o ocurre un error.
        """
        path, _ = QFileDialog.getOpenFileName(
            self.window, "Abrir imagen", "", "Imágenes (*.png *.jpg *.jpeg)"
        )
        if not path:
            return

        try:
            image = load_image(path)
        except (OSError, ValueError) as error:
            QMessageBox.warning(
                self,
                "No se pudo abrir la imagen",
                str(error),
            )
            return None

        return image

    def select_second_image(self, checked=False):
        """
        Objetivo: Seleccionar y guardar la segunda imagen para una operación.
        Recibe:
            - checked: estado opcional enviado por la señal del botón.
        Devuelve:
            - No devuelve ningún valor.
        """
        image = self.open_second_image()

        if image is not None:
            self.second_image = image

    def add_images(self, value=None):
        """
        Objetivo: Sumar la imagen de trabajo y la segunda imagen seleccionada.
        Recibe:
            - value: valor opcional enviado por la señal del botón.
        Devuelve:
            - No devuelve ningún valor. Muestra la imagen resultante.
        """
        if (self.current_image is None) or (self.second_image is None):
            return

        images = [self.current_image, self.second_image]
        weight = self.window.weights_spinBox.value()
        if weight > 0:
            result = weighted_image_sum(images, weight)
        else:
            result = sum_images(images)

        self.apply_result(result)

    def reduce(self, factor):
        """
        Objetivo: Reducir la resolución de la imagen de trabajo.
        Recibe:
            - factor: factor de reducción seleccionado por el usuario.
        Devuelve:
            - No devuelve ningún valor. Muestra la imagen reducida.
        """

        if self.current_image is None:
            return

        result = reduce_resolution(self.current_image, factor)
        self.window.display_image(result)

    def rotate(self, angle):
        """
        Objetivo: Rotar la imagen de trabajo.
        Recibe:
            - angle: ángulo de rotación indicado por el usuario.
        Devuelve:
            - No devuelve ningún valor. Muestra la imagen rotada.
        """
        if self.current_image is None:
            return

        result = rotate_image(self.current_image, angle)
        self.window.display_image(result)

    def zoom_image(self, value):
        """
        Objetivo: Aplicar zoom a la imagen de trabajo.
        Recibe:
            - value: valor del zoom seleccionado en el slider.
        Devuelve:
            - No devuelve ningún valor. Muestra la imagen ampliada.
        """

        if self.current_image is None:
            return

        if value > 0:
            factor = value
        else:
            factor = 1
        min_side = min(self.current_image.shape[:2])
        area = max(1, min_side // factor)

        result = zoom(self.current_image, area, factor)
        self.window.display_image(result)

    def inv_image_color(self):
        """
        Objetivo: Invertir los colores de la imagen de trabajo.
        Recibe:
            - No recibe argumentos adicionales.
        Devuelve:
            - No devuelve ningún valor. Muestra la imagen con colores invertidos.
        """
        if self.current_image is None:
            return

        result = inv_color(self.current_image)
        self.apply_result(result)

    def histograms(self, checked=False):
        """
        Objetivo: Crear y mostrar el histograma del canal seleccionado.
        Recibe:
            - _checked: estado opcional enviado por la señal del botón.
        Devuelve:
            - No devuelve ningún valor. Abre una ventana con el histograma.
        """
        if (self.current_image is None) or (self.current_image.ndim != 3):
            return

        channel_names = {
            "Rojo": "red",
            "Verde": "green",
            "Azul": "blue",
            "Cian": "cyan",
            "Magenta": "magenta",
            "Amarillo": "yellow",
        }

        selected = self.window.select_color_hist.currentText()
        channel = channel_names.get(selected)

        if channel is None:
            return

        color = get_color(self.current_image, channel)

        dialog = QDialog(self.window)
        dialog.resize(700, 450)

        layout = QVBoxLayout(dialog)

        figure = Figure(figsize=(7, 4))
        canvas = FigureCanvasQTAgg(figure)
        layout.addWidget(canvas)

        ax = figure.add_subplot(111)
        ax.hist(
            color.ravel(),
            bins=256,
            range=(0, 256),
            color=channel,
            histtype="step",
            linewidth=1.5,
        )
        ax.set_xlabel("Intensidad del píxel")
        ax.set_ylabel("Número de píxeles")
        ax.set_xlim(0, 255)
        ax.set_title(f"Histograma: {selected}")

        figure.tight_layout()
        canvas.draw()
        dialog.show()

    def set_crop_valid_ranges(self):
        """
        Objetivo: Configurar los rangos de coordenadas del recorte
        según las dimensiones de la imagen original.
        Recibe:
            - No recibe argumentos adicionales.
        Devuelve:
            - No devuelve ningún valor. Actualiza los rangos de los SpinBox.
        """
        if self.current_image is None:
            return

        height, width = self.current_image.shape[:2]

        self.window.xi_spinBox.setRange(0, width - 1)
        self.window.xf_spinBox.setRange(1, width)
        self.window.yi_spinBox.setRange(0, height - 1)
        self.window.yf_spinBox.setRange(1, height)

        # Valores iniciales: recortar toda la imagen
        self.window.xi_spinBox.setValue(0)
        self.window.xf_spinBox.setValue(width)
        self.window.yi_spinBox.setValue(0)
        self.window.yf_spinBox.setValue(height)

    def crop(self, checked=False):
        """
        Objetivo: Recortar la imagen de trabajo usando las coordenadas de los SpinBox.
        Recibe:
            - El estado del botón, enviado por la señal clicked.
        Devuelve:
            - No devuelve ningún valor. Actualiza y muestra la imagen recortada.
        """
        if self.current_image is None:
            return

        xi = int(self.window.xi_spinBox.value())
        xf = int(self.window.xf_spinBox.value())
        yi = int(self.window.yi_spinBox.value())
        yf = int(self.window.yf_spinBox.value())

        try:
            cropped = crop_image(self.current_image, xi, xf, yi, yf)
        except (TypeError, ValueError) as error:
            QMessageBox.warning(self.window, "Recorte inválido", str(error))
            return

        self.current_image = cropped
        self.window.image = cropped
        self.apply_result(cropped)

    def reset_image(self):
        """
        Objetivo: Restaurar la imagen original y descartar las modificaciones.
        Recibe:
            - No recibe argumentos adicionales.
        Devuelve:
            - No devuelve ningún valor. Muestra la imagen original.
        """
        if self.original_image is None:
            return

        self.window.brightness_slider.setValue(0)
        self.window.channel_brightness_slider.setValue(0)
        self.window.grayscale_selector.setCurrentText("Original")
        self.window.binaryization_slider.setValue(0)
        self.window.factor_slider.setValue(1)
        self.window.rotate_spinBox.setValue(0)
        self.window.zoom_slider.setValue(1)
        self.window.weights_spinBox.setValue(0)

        self.current_image = self.original_image.copy()
        self.set_crop_valid_ranges()

        self.window.zoom_slider.setRange(1, 100)
        self.window.zoom_slider.setValue(1)

        self.window.image = self.current_image
        self.window.display_image(self.current_image)
