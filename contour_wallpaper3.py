import sys
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget,
                           QPushButton, QFileDialog, QSlider, QToolBar, QAction,
                           QComboBox, QLineEdit, QDesktopWidget, QProgressDialog)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import cv2
import random
import librosa
import librosa.display

class AudioAnalyzer:
    def __init__(self, audio_path):
        # Load the audio file
        self.y, self.sr = librosa.load(audio_path)

        # Get tempo (BPM)
        self.tempo, self.beat_frames = librosa.beat.beat_track(y=self.y, sr=self.sr)

        # Convert beat frames to timestamps
        self.beat_times = librosa.frames_to_time(self.beat_frames, sr=self.sr)

        # Get amplitude envelope
        self.amplitude_env = librosa.onset.onset_strength(y=self.y, sr=self.sr)

        # Get spectral centroids for frequency analysis
        self.spectral_centroids = librosa.feature.spectral_centroid(y=self.y, sr=self.sr)[0]

        # Calculate frames per beat for video sync
        self.fps = 30  # Standard video fps
        self.frames_per_beat = int(60 * self.fps / self.tempo)

    def get_audio_features_at_time(self, time):
        """Get audio features for a specific timestamp"""
        frame_idx = int(time * self.sr / librosa.get_duration(y=self.y, sr=self.sr) * len(self.amplitude_env))
        if frame_idx >= len(self.amplitude_env):
            frame_idx = len(self.amplitude_env) - 1

        return {
            'amplitude': self.amplitude_env[frame_idx],
            'frequency': self.spectral_centroids[frame_idx] if frame_idx < len(self.spectral_centroids) else 0,
            'is_beat': any(abs(time - beat_time) < 0.05 for beat_time in self.beat_times)
        }

class VideoGenerationThread(QThread):
    update_progress = pyqtSignal(int)

    def __init__(self, video_writer, wallpaper_generator, audio_analyzer, parent=None):
        super().__init__(parent)
        self.video_writer = video_writer
        self.wallpaper_generator = wallpaper_generator
        self.audio_analyzer = audio_analyzer
        self.stopped = False

    def stop(self):
        self.stopped = True

    def generate_color_from_audio(self, audio_features):
        # Map frequency to hue
        hue = (audio_features['frequency'] % 1000) / 1000

        # Map amplitude to saturation and value
        amplitude_normalized = np.clip(audio_features['amplitude'] / np.max(self.audio_analyzer.amplitude_env), 0, 1)
        saturation = 0.5 + 0.5 * amplitude_normalized
        value = 0.5 + 0.5 * amplitude_normalized

        # Convert HSV to RGB
        h = hue * 6
        i = int(h)
        f = h - i
        p = value * (1 - saturation)
        q = value * (1 - f * saturation)
        t = value * (1 - (1 - f) * saturation)
        value = int(value * 255)

        if i == 0:
            return (value, int(t * 255), int(p * 255), 255)
        elif i == 1:
            return (int(q * 255), value, int(p * 255), 255)
        elif i == 2:
            return (int(p * 255), value, int(t * 255), 255)
        elif i == 3:
            return (int(p * 255), int(q * 255), value, 255)
        elif i == 4:
            return (int(t * 255), int(p * 255), value, 255)
        else:
            return (value, int(p * 255), int(q * 255), 255)

    def get_blend_mode_from_audio(self, audio_features):
        # Change blend mode based on beat and amplitude
        if audio_features['is_beat']:
            if audio_features['amplitude'] > np.mean(self.audio_analyzer.amplitude_env) * 1.5:
                return 'Screen'
            else:
                return 'Overlay'
        return 'Normal' if audio_features['amplitude'] < np.mean(self.audio_analyzer.amplitude_env) else 'Multiply'

    def run(self):
        base_image = self.wallpaper_generator.current_wallpaper
        sensitivity = self.wallpaper_generator.edge_slider.value()

        # Get edges
        edges = cv2.Canny(np.array(base_image), sensitivity, sensitivity * 2)
        continuous_edges = self.wallpaper_generator.edge_colorizer.get_continuous_edges(edges)

        # Calculate video duration based on audio length
        duration = librosa.get_duration(y=self.audio_analyzer.y, sr=self.audio_analyzer.sr)
        num_frames = int(duration * 30)  # 30 fps

        for frame in range(num_frames):
            if self.stopped:
                break

            current_time = frame / 30  # Convert frame to time
            audio_features = self.audio_analyzer.get_audio_features_at_time(current_time)

            # Create new image for colored edges
            edges_image = Image.new('RGBA', base_image.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(edges_image)

            # Generate color based on audio features
            for edge_group in continuous_edges:
                color = self.generate_color_from_audio(audio_features)
                for y, x in edge_group:
                    draw.point((x, y), fill=color)

            # Get blend mode based on audio features
            blend_mode = self.get_blend_mode_from_audio(audio_features)

            # Apply blending
            frame_image = self.wallpaper_generator.apply_blend_mode(
                base_image,
                edges_image,
                blend_mode
            )

            # Write frame
            frame_array = np.array(frame_image)
            frame_array = cv2.cvtColor(frame_array, cv2.COLOR_RGBA2BGR)
            self.video_writer.write(frame_array)

            # Update progress
            progress = int((frame + 1) / num_frames * 100)
            self.update_progress.emit(progress)

        self.update_progress.emit(100)

class EdgeColorizer:
    def __init__(self):
        self.visited = None
        self.current_edges = None

    def find_continuous_edge(self, start_point, edges):
        if self.visited[start_point[0], start_point[1]]:
            return []

        edge_pixels = []
        stack = [start_point]

        while stack:
            y, x = stack.pop()
            if self.visited[y, x]:
                continue

            self.visited[y, x] = True
            edge_pixels.append((y, x))

            # Check 8-connected neighbors
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    ny, nx = y + dy, x + dx
                    if (0 <= ny < edges.shape[0] and
                        0 <= nx < edges.shape[1] and
                        edges[ny, nx] > 0 and
                        not self.visited[ny, nx]):
                        stack.append((ny, nx))

        return edge_pixels

    def get_continuous_edges(self, edges):
        self.visited = np.zeros_like(edges, dtype=bool)
        self.current_edges = edges
        continuous_edges = []

        for y in range(edges.shape[0]):
            for x in range(edges.shape[1]):
                if edges[y, x] > 0 and not self.visited[y, x]:
                    edge_group = self.find_continuous_edge((y, x), edges)
                    if edge_group:
                        continuous_edges.append(edge_group)

        return continuous_edges

class WallpaperGenerator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.edge_colorizer = EdgeColorizer()
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

        load_action = QAction("Load Audio", self)
        load_action.triggered.connect(self.loadAudio)
        toolbar.addAction(load_action)

        save_action = QAction("Save As", self)
        save_action.triggered.connect(self.saveAs)
        toolbar.addAction(save_action)

        save_video_action = QAction("Save Animated Video", self)
        save_video_action.triggered.connect(self.saveAnimatedVideo)
        toolbar.addAction(save_video_action)

        # Add edge detection sensitivity slider
        self.edge_slider = QSlider(Qt.Horizontal, self)
        self.edge_slider.setRange(0, 255)
        self.edge_slider.setValue(100)
        self.edge_slider.valueChanged.connect(self.updateWallpaper)
        toolbar.addWidget(QLabel("Edge Sensitivity: "))
        toolbar.addWidget(self.edge_slider)

        # Add color randomness slider
        self.color_random_slider = QSlider(Qt.Horizontal, self)
        self.color_random_slider.setRange(0, 100)
        self.color_random_slider.setValue(50)
        self.color_random_slider.valueChanged.connect(self.updateWallpaper)
        toolbar.addWidget(QLabel("Color Randomness: "))
        toolbar.addWidget(self.color_random_slider)

        # Add blend mode combo box
        self.blend_mode = QComboBox(self)
        self.blend_mode.addItems(['Normal', 'Multiply', 'Screen', 'Overlay'])
        self.blend_mode.currentTextChanged.connect(self.updateWallpaper)
        toolbar.addWidget(QLabel("Blend Mode: "))
        toolbar.addWidget(self.blend_mode)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout(self.central_widget)
        layout.addWidget(self.image_label)

        self.setWindowTitle("Wallpaper Generator")
        self.setGeometry(100, 100, 800, 600)

        self.current_wallpaper = None
        self.video_writer = None

    # Replace the loadAudio method in the WallpaperGenerator class
    def loadAudio(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self,
            "Load Audio",
            "",
            "Audio Files (*.mp3 *.wav);;All Files (*)",
            options=options
        )

        if file_path:
            try:
                self.audio_analyzer = AudioAnalyzer(file_path)
                print(f"Audio loaded successfully. Tempo: {self.audio_analyzer.tempo} BPM")
            except Exception as e:
                print(f"Error loading audio: {str(e)}")

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

    def apply_blend_mode(self, base_image, overlay_image, mode):
        base_arr = np.array(base_image)
        overlay_arr = np.array(overlay_image)

        if mode == 'Normal':
            return Image.alpha_composite(base_image, overlay_image)
        elif mode == 'Multiply':
            result = (base_arr.astype(float) * overlay_arr.astype(float) / 255).astype(np.uint8)
        elif mode == 'Screen':
            result = 255 - ((255 - base_arr.astype(float)) * (255 - overlay_arr.astype(float)) / 255).astype(np.uint8)
        elif mode == 'Overlay':
            mask = base_arr >= 128
            result = np.zeros_like(base_arr)
            result[mask] = (255 - ((255 - 2 * (base_arr[mask] - 128)) * (255 - overlay_arr[mask])) / 255).astype(np.uint8)
            result[~mask] = ((2 * base_arr[~mask] * overlay_arr[~mask]) / 255).astype(np.uint8)

        return Image.fromarray(result)

    def generate_random_color(self, randomness):
        base_hue = random.random()
        hue_variation = (random.random() - 0.5) * 2 * (randomness / 100)
        hue = (base_hue + hue_variation) % 1.0

        # Convert HSV to RGB (simplified conversion)
        h = hue * 6
        i = int(h)
        f = h - i
        p = 0
        q = 1 - f
        t = f

        if i == 0:
            return (255, int(p * 255), int(p * 255), 255)
        elif i == 1:
            return (int(q * 255), 255, int(p * 255), 255)
        elif i == 2:
            return (int(p * 255), 255, int(t * 255), 255)
        elif i == 3:
            return (int(p * 255), int(q * 255), 255, 255)
        elif i == 4:
            return (int(t * 255), int(p * 255), 255, 255)
        else:
            return (255, int(p * 255), int(q * 255), 255)

    def updateWallpaper(self):
        if self.current_wallpaper is not None:
            sensitivity = self.edge_slider.value()
            randomness = self.color_random_slider.value()

            # Get edges using Canny
            edges = cv2.Canny(np.array(self.current_wallpaper), sensitivity, sensitivity * 2)

            # Find continuous edge groups
            continuous_edges = self.edge_colorizer.get_continuous_edges(edges)

            # Create a new image for colored edges
            edges_image = Image.new('RGBA', self.current_wallpaper.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(edges_image)

            # Color each continuous edge group
            for edge_group in continuous_edges:
                color = self.generate_random_color(randomness)
                for y, x in edge_group:
                    draw.point((x, y), fill=color)

            # Apply selected blend mode
            blend_mode = self.blend_mode.currentText()
            wallpaper = self.apply_blend_mode(self.current_wallpaper, edges_image, blend_mode)

            self.displayImage(wallpaper)

            if self.video_writer is not None:
                frame_array = np.array(wallpaper)
                frame_array = cv2.cvtColor(frame_array, cv2.COLOR_RGBA2BGR)
                self.video_writer.write(frame_array)

    def saveAnimatedVideo(self):
        if self.current_wallpaper is None:
            print("Please generate or load a wallpaper first")
            return

        if not hasattr(self, 'audio_analyzer'):
            print("Please load an audio file first")
            return

        video_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Video As",
            "",
            "Video Files (*.mp4);;All Files (*)"
        )

        if video_path:
            # Define the codec and create VideoWriter object
            codec = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 30  # Frames per second (matching AudioAnalyzer)
            self.video_writer = cv2.VideoWriter(
                video_path,
                codec,
                fps,
                (self.current_wallpaper.width, self.current_wallpaper.height)
            )

            # Create and show progress dialog
            progress_dialog = QProgressDialog("Generating Video...", "Cancel", 0, 100, self)
            progress_dialog.setWindowTitle("Video Generation")
            progress_dialog.setWindowModality(Qt.WindowModal)
            progress_dialog.show()

            # Create and start the video generation thread
            video_thread = VideoGenerationThread(
                self.video_writer,
                self,  # Pass self instead of just current_wallpaper
                self.audio_analyzer
            )
            video_thread.update_progress.connect(progress_dialog.setValue)
            video_thread.start()

            # Connect cancel button
            progress_dialog.canceled.connect(video_thread.stop)
            video_thread.finished.connect(self.cleanupVideoWriter)

            # Show progress dialog and wait for completion
            result = progress_dialog.exec_()

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
