import sys
import time
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QFileDialog, QSlider, QToolBar, QAction, QComboBox, QLineEdit, QDesktopWidget, QProgressDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import cv2

class VideoGenerationThread(QThread):
    update_progress = pyqtSignal(int)

    def __init__(self, video_writer, current_wallpaper, parent=None):
        super().__init__(parent)
        self.video_writer = video_writer
        self.current_wallpaper = current_wallpaper
        self.stopped = False

    def stop(self):
        self.stopped = True

    def run(self):
        num_frames = 100  # Number of frames in the video
        for i in range(num_frames):
            if self.stopped:
                break

            revealed_image = self.revealPixels(self.current_wallpaper, i / num_frames)

            # Convert the PIL image to NumPy array
            frame_array = np.array(revealed_image)
            frame_array = cv2.cvtColor(frame_array, cv2.COLOR_RGBA2BGR)

            # Write the frame to the video file
            self.video_writer.write(frame_array)

            # Emit signal to update progress
            progress_percentage = int((i + 1) / num_frames * 100)
            self.update_progress.emit(progress_percentage)
            time.sleep(0.05)

        # Inform the main thread that the video generation is complete
        self.update_progress.emit(100)

    def revealPixels(self, image, fraction):
        # Copy the original image
        revealed_image = image.copy()

        # Get image dimensions
        width, height = image.size

        # Calculate the number of pixels to reveal based on the fraction
        num_pixels_to_reveal = int(fraction * width * height)

        # Generate random pixel coordinates
        coordinates = np.random.randint(0, width * height, num_pixels_to_reveal)
        coordinates = np.unravel_index(coordinates, (height, width))

        # Set the color of the revealed pixels to transparent
        for y, x in zip(*coordinates):
            revealed_image.putpixel((x, y), (0, 0, 0, 0))

        return revealed_image

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

        # Add a new action to save animated video
        save_video_action = QAction("Save Animated Video", self)
        save_video_action.triggered.connect(self.saveAnimatedVideo)
        toolbar.addAction(save_video_action)

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
        self.video_writer = None  # Variable to store VideoWriter instance

    def loadAudio(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Load Audio", "", "Audio Files (*.mp3);;All Files (*)", options=options)

        if file_path:
            self.audio_segment = AudioSegment.from_file(file_path, format="mp3")

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


    def generateWallpaper(self):
        # Uzyskanie informacji o rozmiarze ekranu
        desktop = QDesktopWidget()
        screen_rect = desktop.availableGeometry(desktop.primaryScreen())

        # Stworzenie nowego obrazu o rozmiarze ekranu i czarnym tle
        black_background = Image.new('RGBA', (screen_rect.width(), screen_rect.height()), (0, 0, 0, 255))

        # Wywołanie funkcji aktualizującej tapetę, aby wyświetlić czarne tło
        self.current_wallpaper = black_background
        self.updateWallpaper()

    def updateWallpaper(self):
        if self.current_wallpaper is not None:
            sensitivity = self.edge_slider.value()
            edges = cv2.Canny(np.array(self.current_wallpaper), sensitivity, sensitivity * 2)

            edges_image = Image.fromarray(edges, 'L')
            edges_image = edges_image.convert("RGBA")

            wallpaper = Image.alpha_composite(self.current_wallpaper, edges_image)
            self.displayImage(wallpaper)

            if self.video_writer is not None:
                # Convert the PIL image to NumPy array
                frame_array = np.array(wallpaper)
                frame_array = cv2.cvtColor(frame_array, cv2.COLOR_RGBA2BGR)

                # Write the frame to the video file
                self.video_writer.write(frame_array)

    def saveAnimatedVideo(self):
        if self.current_wallpaper is not None:
            # Utwórz nowego VideoWritera tylko w momencie zapisu animacji
            video_path, _ = QFileDialog.getSaveFileName(self, "Save Video As", "", "Video Files (*.mp4);;All Files (*)")
            if video_path:
                # Define the codec and create a VideoWriter object
                codec = cv2.VideoWriter_fourcc(*'XVID')
                fps = 24  # Frames per second
                duration_seconds = 3
                num_frames = int(fps * duration_seconds)
                self.video_writer = cv2.VideoWriter(video_path, codec, fps, (self.current_wallpaper.width, self.current_wallpaper.height))

                # Create and show progress dialog
                progress_dialog = QProgressDialog("Generating Video...", "Cancel", 0, 100, self)
                progress_dialog.setWindowTitle("Video Generation")
                progress_dialog.setWindowModality(Qt.WindowModal)
                progress_dialog.show()

                # Connect the progress dialog to the thread's signal
                video_thread = VideoGenerationThread(self.video_writer, self.current_wallpaper)
                video_thread.update_progress.connect(progress_dialog.setValue)
    
                # Start the thread
                video_thread.start()    

                # Connect the progress dialog's "canceled" signal to stop the thread
                progress_dialog.canceled.connect(video_thread.stop)

                # Connect the thread's finished signal to cleanup
                video_thread.finished.connect(self.cleanupVideoWriter)

                # Show the progress dialog and wait for it to finish
                result = progress_dialog.exec_()

                # If canceled, release the video writer
                if result == QProgressDialog.Rejected:
                    self.cleanupVideoWriter()
                    print("Video generation canceled.")
                else:
                    print("Animated video saved.")

    def cleanupVideoWriter(self):
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

    def revealPixels(self, image, fraction):
        # Copy the original image
        revealed_image = image.copy()

        # Get image dimensions
        width, height = image.size

        # Calculate the number of pixels to reveal based on the fraction
        num_pixels_to_reveal = int(fraction * width * height)

        # Generate random pixel coordinates
        coordinates = np.random.randint(0, width * height, num_pixels_to_reveal)
        coordinates = np.unravel_index(coordinates, (height, width))

        # Set the color of the revealed pixels to transparent
        for y, x in zip(*coordinates):
            revealed_image.putpixel((x, y), (0, 0, 0, 0))

        return revealed_image

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

                # Pobierz widoczną tapetę z etykiety
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
