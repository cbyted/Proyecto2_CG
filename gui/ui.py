from pathlib import Path
from PyQt6 import uic
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QMainWindow,
    QFileDialog,
    QMessageBox,
)
from imglib.io import *
from controllers.controllers import *


class MainWindow(QMainWindow):
    def __init__(self):
        """
        Objetivo: Inicializar la ventana principal de la aplicación.
        Recibe:
            - Ningún argumento adicional
        Realiza:
            - Carga la interfaz y los estilos
            - Inicializa el estado de la imagen
            - Conecta los botones con sus funciones
            - Crea el controlador de la aplicación
        """
        super().__init__()

        # Cargar el diseño creado en Qt Designer
        ui_path = Path(__file__).resolve().parent / "ImageProcessing.ui"
        uic.loadUi(str(ui_path), self)

        # Aplicar los estilos de la aplicación
        ui_dir = Path(__file__).resolve().parent
        style_path = ui_dir / "styles" / "styles.qss"
        self.setStyleSheet(style_path.read_text(encoding="utf-8"))

        # Inicializar los atributos que guardan la imagen abierta
        self.image = None
        self.image_path = None
        self.pixmap = None

        # Conectar los botones del encabezado con sus funciones
        self.open_img.clicked.connect(self.open_image)
        self.close_img.clicked.connect(self.close_image)
        self.save_img.clicked.connect(self.save_image)

        # Crear el controlador de los controles de procesamiento
        self.controlls = Controller(self)

    def open_image(self):
        """
        Objetivo: Abrir una imagen seleccionada por el usuario y mostrarla.
        Recibe:
            - Ningún argumento adicional
        Realiza:
            - Abre un diálogo para seleccionar un archivo de imagen
            - Carga y normaliza la imagen mediante load_image()
            - Actualiza la vista, la información y el controlador
        """
        path, _ = QFileDialog.getOpenFileName(
            self, "Abrir imagen", "", "Imágenes (*.png *.jpg *.jpeg *.bmp)"
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
            return

        self.image = image
        self.image_path = path
        self.display_image(image)
        self.set_image_info(image, path)
        self.controlls.set_image(image)

    def display_image(self, image):
        """
        Objetivo: Mostrar una imagen en el visor de la interfaz.
        Recibe:
            - Una imagen tipo np.ndarray en escala de grises (2D)
            o RGB (alto × ancho × 3), con dtype uint8
        Realiza:
            - Convierte la imagen a QImage y después a QPixmap
            - Actualiza la vista previa manteniendo la proporción
        """
        if image is None:
            self.clear_image()
            return

        image = np.ascontiguousarray(image)
        height = image.shape[0]
        width = image.shape[1]

        if image.ndim == 2:
            height, width = image.shape
            qimage = QImage(
                image.data,
                width,
                height,
                image.strides[0],
                QImage.Format.Format_Grayscale8,
            ).copy()
        elif image.ndim == 3 and image.shape[2] == 3:
            # Imagen RGB: alto × ancho × 3 canales.
            height, width, _ = image.shape
            qimage = QImage(
                image.data,
                width,
                height,
                image.strides[0],
                QImage.Format.Format_RGB888,
            ).copy()

        else:
            raise ValueError(f"Formato de imagen no compatible: {image.shape}")

        self.pixmap = QPixmap.fromImage(qimage)
        self._update_preview()

    def _update_preview(self):
        """
        Objetivo: Actualizar la vista previa de la imagen en la interfaz.
        Recibe:
            - Ningún argumento adicional
        Realiza:
            - Ajusta la imagen al tamaño del widget image_label
            - Mantiene la proporción y suaviza el escalado
        """
        if self.pixmap is None:
            return

        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setPixmap(
            self.pixmap.scaled(
                self.image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def clear_image(self):
        """
        Objetivo: Eliminar la imagen actual y limpiar la vista y la información.
        Recibe:
            - Ningún argumento adicional
        Realiza:
            - Restablece los atributos de imagen a None
            - Limpia el visor y los labels informativos
        """
        self.image = None
        self.image_path = None
        self.pixmap = None
        self.image_label.clear()
        self.clear_info()

    def close_image(self):
        """
        Objetivo: Cerrar la imagen actual en la aplicación.
        Recibe:
            - Ningún argumento adicional
        Realiza:
            - Limpia la imagen y sus datos de la interfaz
            - Informa al controlador de que no hay una imagen cargada
        """
        self.clear_image()
        self.controlls.set_image(None)

    def save_image(self):
        """
        Objetivo: Guardar la imagen actual en un archivo.
        Recibe:
            - Ningún argumento adicional
        Realiza:
            - Abre un diálogo para elegir nombre, ubicación y formato
            - Guarda la imagen usando la función save_image de imglib.io
        """
        if self.image is None:
            QMessageBox.information(
                self,
                "No hay imagen",
                "Abre una imagen antes de guardarla.",
            )
            return

        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Guardar imagen",
            "",
            "PNG (*.png);;JPEG (*.jpg *.jpeg)",
        )

        if not path:
            return

        suffix = Path(path).suffix.lower()

        if suffix in {".jpg", ".jpeg"}:
            image_format = "JPEG"
        elif suffix == ".png":
            image_format = "PNG"
        elif "JPEG" in selected_filter:
            image_format = "JPEG"
            path += ".jpg"
        else:
            image_format = "PNG"
            path += ".png"

        try:
            save_image(self.image, path, image_format)
        except (OSError, ValueError) as error:
            QMessageBox.warning(
                self,
                "No se pudo guardar la imagen",
                str(error),
            )
            return

        QMessageBox.information(
            self,
            "Imagen guardada",
            f"La imagen se guardó en:\n{path}",
        )

    def set_image_info(self, image, path):
        """
        Objetivo: Mostrar información de la imagen en los labels del footer.
        Recibe:
            - image: imagen tipo np.ndarray con forma alto × ancho × canales
            - path: ruta del archivo de imagen
        Realiza:
            - Muestra el nombre, la ruta, el formato, el tamaño y las dimensiones
        """
        path = Path(path)
        size_kb = path.stat().st_size // 1024

        self.name_label.setText(f"Nombre: {path.name}")
        self.path_label.setText(f"Path: {path}")
        self.format_label.setText(f"Formato: {path.suffix.lstrip('.').upper()}")
        self.size_label.setText(f"Tamaño: {size_kb:,} KB")
        self.dim_label.setText(f"Dimensiones: {image.shape[1]} × {image.shape[0]}")

    def clear_info(self):
        """
        Objetivo: Restablecer los labels informativos después de cerrar una imagen.
        Recibe:
            - Ningún argumento adicional
        Realiza:
            - Sustituye la información de la imagen por valores vacíos
        """
        self.name_label.setText("Nombre: -")
        self.path_label.setText("Path: -")
        self.format_label.setText("Formato: -")
        self.size_label.setText("Tamaño: -")
        self.dim_label.setText("Dimensiones: -")
