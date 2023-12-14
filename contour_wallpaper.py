import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QFileDialog, QSlider, QToolBar, QAction, QComboBox, QLineEdit, QDesktopWidget
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import cv2

class WallpaperGenerator(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        # Add a toolbar
        toolbar = QToolBar("Main Toolbar", self)
        self.addToolBar(toolbar)

        # Add actions to the toolbar
        generate_action = QAction("Generate Wallpaper", self)
        generate_action.triggered.connect(self.generateWallpaper)
        toolbar.addAction(generate_action)

        load_action = QAction("Load Image", self)
        load_action.triggered.connect(self.loadImage)
        toolbar.addAction(load_action)

        save_action = QAction("Save As", self)
        save_action.triggered.connect(self.saveAs)
        toolbar.addAction(save_action)



        # Add a slider for edge detection sensitivity
        self.edge_slider = QSlider(Qt.Horizontal, self)
        self.edge_slider.setRange(0, 255)
        self.edge_slider.setValue(100)  # Initial sensitivity value
        self.edge_slider.valueChanged.connect(self.updateWallpaper)
        toolbar.addWidget(self.edge_slider)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout(self.central_widget)
        layout.addWidget(self.image_label)

        self.setWindowTitle("Wallpaper Generator")
        self.setGeometry(100, 100, 800, 600)

        self.current_wallpaper = None  # Add a new variable to store the current image

    def generateWallpaper(self):
        # Uzyskanie informacji o rozmiarze ekranu
        desktop = QDesktopWidget()
        screen_rect = desktop.availableGeometry(desktop.primaryScreen())

        # Stworzenie nowego obrazu o rozmiarze ekranu i czarnym tle
        black_background = Image.new('RGBA', (screen_rect.width(), screen_rect.height()), (0, 0, 0, 255))

        # Wywołanie funkcji aktualizującej tapetę, aby wyświetlić czarne tło
        self.current_wallpaper = black_background
        self.updateWallpaper()

    def loadImage(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Load Image", "", "Images (*.png *.jpg *.bmp);;All Files (*)", options=options)

        if file_path:
            base_image = Image.open(file_path)
            base_image = base_image.convert("RGBA")

            # Dostosowanie rozmiaru obrazu do wcześniej wygenerowanej tapety
            if self.current_wallpaper is not None:
                wallpaper_width, wallpaper_height = self.current_wallpaper.size
                base_width, base_height = base_image.size

                # Obliczenie pozycji, aby wczytany obraz był wysrodkowany
                x_offset = (wallpaper_width - base_width) // 2
                y_offset = (wallpaper_height - base_height) // 2    

                # Stworzenie nowego obrazu o rozmiarze tapety
                new_image = Image.new('RGBA', self.current_wallpaper.size, (0, 0, 0, 0))

                # Umieszczenie wczytanego obrazu na wcześniej wygenerowanej tapetze
                new_image.paste(base_image, (x_offset, y_offset))

                # Ustawienie nowego obrazu jako warstwa na wcześniej wygenerowanej tapetze
                self.current_wallpaper = Image.alpha_composite(self.current_wallpaper, new_image)
            else:
                self.current_wallpaper = base_image

            self.updateWallpaper()


    def updateWallpaper(self):
        if self.current_wallpaper is not None:
            sensitivity = self.edge_slider.value()
            edges = cv2.Canny(np.array(self.current_wallpaper), sensitivity, sensitivity * 2)

            edges_image = Image.fromarray(edges, 'L')
            edges_image = edges_image.convert("RGBA")

            wallpaper = Image.alpha_composite(self.current_wallpaper, edges_image)
            self.displayImage(wallpaper)

    def displayImage(self, image):
        q_image = QImage(image.tobytes(), image.width, image.height, image.width * 4, QImage.Format_RGBA8888)
        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap)
        self.image_label.setAlignment(Qt.AlignCenter)

    def saveAs(self):
        if self.current_wallpaper is not None:
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            file_dialog = QFileDialog()
            file_path, _ = file_dialog.getSaveFileName(self, "Save Wallpaper As", "", "PNG Images (*.png);;JPEG Images (*.jpg);;Bitmap Images (*.bmp)", options=options)

            if file_path:
                # Automatyczne dodanie rozszerzenia PNG, jeśli nie zostało podane przez użytkownika
                if not file_path.lower().endswith(('.png', '.jpg', '.bmp')):
                    file_path += ".png"

                # Pobierz widoczną tapetę z etykietą
                visible_wallpaper = self.getVisibleWallpaper()

                # Zapisz widoczną tapetę z efektem
                visible_wallpaper.save(file_path)
                print(f"Visible wallpaper saved as {file_path}")

    def getVisibleWallpaper(self):
        # Pobierz widoczną tapetę z etykiety
        pixmap = self.image_label.pixmap()
        if pixmap is not None:
            image = pixmap.toImage()
            image = image.convertToFormat(QImage.Format_RGBA8888)

            width = image.width()
            height = image.height()

            ptr = image.constBits()
            ptr.setsize(image.byteCount())
            arr = np.array(ptr).reshape(height, width, 4)  # Konwersja do tablicy NumPy

            visible_wallpaper = Image.fromarray(arr, 'RGBA')
            return visible_wallpaper
        else:
            return None


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = WallpaperGenerator()
    window.show()
    sys.exit(app.exec_())
