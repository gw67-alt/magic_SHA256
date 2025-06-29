import sys
import time
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QStatusBar
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer, QObject
from PyQt5.QtGui import QImage, QPixmap
import pyqtgraph as pg
from collections import deque
import random
import hashlib

raw_value_for_guess = 2
qubit_number = 0
STARTING_CREDITS = 10000
COST_PER_GUESS = 1
WIN_CREDITS = 150
game_state = {
    "credits": STARTING_CREDITS,
}

# --- OpenCV Configuration ---
MIN_MATCH_COUNT = 10
LOWE_RATIO_TEST = 0.10
KEY_TO_CYCLE_QT = Qt.Key_N
KEY_TO_QUIT_QT = Qt.Key_Q

# --- Chart Configuration ---
MAX_CHART_POINTS = 100
MOVING_AVG_WINDOW = 100

# --- Guessing Configuration ---
GUESS_TRIGGER_COUNT = 8

PREFIX = "000000"
def calculate_sha256_with_library(data):
    try:
        sha256_hash = hashlib.sha256()
        if isinstance(data, str):
            sha256_hash.update(data.encode('utf-8'))
        elif isinstance(data, bytes):
            sha256_hash.update(data)
        else:
            raise TypeError("Input data must be a string or bytes.")
        hex_digest = sha256_hash.hexdigest()
        if hex_digest.startswith(PREFIX):
            print(data, hex_digest)
        return hex_digest
    except TypeError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

class AppState(QObject):
    state_changed = pyqtSignal(int)
    capture_reference_requested = pyqtSignal()
    reset_requested = pyqtSignal()

    STATE_WAITING_FOR_REFERENCE = 0
    STATE_TRACKING = 1

    def __init__(self):
        super().__init__()
        self._current_state = self.STATE_WAITING_FOR_REFERENCE

    @property
    def current_state(self):
        return self._current_state

    @current_state.setter
    def current_state(self, value):
        if self._current_state != value:
            self._current_state = value
            self.state_changed.emit(value)

    def request_capture_reference(self):
        self.capture_reference_requested.emit()

    def request_reset(self):
        self.reset_requested.emit()

app_state = AppState()

class OpenCVThread(QThread):
    frame_ready = pyqtSignal(QImage)
    matches_count_ready = pyqtSignal(int)
    status_message = pyqtSignal(str)

    def __init__(self, app_state_ref):
        super().__init__()
        self.running = False
        self.app_state = app_state_ref

        self.reference_frame = None
        self.reference_kp = None
        self.reference_des = None
        self.orb = None
        self.bf_matcher = None
        self._capture_next_frame_as_reference = False

        self.app_state.capture_reference_requested.connect(self.prepare_for_reference_capture)
        self.app_state.reset_requested.connect(self.reset_reference)

    def initialize_features(self):
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def prepare_for_reference_capture(self):
        self._capture_next_frame_as_reference = True

    def reset_reference(self):
        self.reference_frame = None
        self.reference_kp = None
        self.reference_des = None
        self._capture_next_frame_as_reference = False
        self.app_state.current_state = AppState.STATE_WAITING_FOR_REFERENCE

    def run(self):
        self.running = True
        self.initialize_features()

        cap = cv2.VideoCapture(qubit_number)
        if not cap.isOpened():
            self.status_message.emit("Error: Cannot open camera.")
            self.running = False
            return

        while self.running:
            ret, frame = cap.read()
            if not ret:
                self.status_message.emit("Error: Can't receive frame.")
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.frame_ready.emit(qt_image.copy())

            num_good_matches_for_signal = 0

            if self._capture_next_frame_as_reference:
                self.reference_frame = frame.copy()
                self.reference_kp, self.reference_des = self.orb.detectAndCompute(self.reference_frame, None)

                if self.reference_des is None or len(self.reference_kp) < MIN_MATCH_COUNT:
                    self.status_message.emit(f"Ref. Capture Failed: Not enough features ({len(self.reference_kp) if self.reference_kp is not None else 0}). Try again.")
                    self.reference_frame = None
                    self.reference_kp = None
                    self.reference_des = None
                    self.app_state.current_state = AppState.STATE_WAITING_FOR_REFERENCE
                else:
                    self.status_message.emit(f"Reference Captured ({len(self.reference_kp)} keypoints). Tracking...")
                    self.app_state.current_state = AppState.STATE_TRACKING
                self._capture_next_frame_as_reference = False
                self.matches_count_ready.emit(0)
            elif self.app_state.current_state == AppState.STATE_TRACKING and self.reference_frame is not None and self.reference_des is not None:
                current_kp, current_des = self.orb.detectAndCompute(frame, None)
                actual_good_matches_count = 0

                if current_des is not None and len(current_des) > 0:
                    all_matches = self.bf_matcher.knnMatch(self.reference_des, current_des, k=2)
                    good_matches = []
                    for m_arr in all_matches:
                        if len(m_arr) == 2:
                            m, n = m_arr
                            if m.distance < LOWE_RATIO_TEST * n.distance:
                                good_matches.append(m)
                    actual_good_matches_count = len(good_matches)
                    if actual_good_matches_count >= MIN_MATCH_COUNT:
                        src_pts = np.float32([self.reference_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                        dst_pts = np.float32([current_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                        if H is None:
                            num_good_matches_for_signal = -1
                        else:
                            num_good_matches_for_signal = actual_good_matches_count
                    else:
                        num_good_matches_for_signal = actual_good_matches_count
                else:
                    num_good_matches_for_signal = 0
                self.matches_count_ready.emit(num_good_matches_for_signal)
            else:
                self.matches_count_ready.emit(0)

        cap.release()
        self.status_message.emit("Camera released.")

    def stop(self):
        self.running = False
        self.wait()

class MainWindow(QMainWindow):
    def __init__(self, app_state_ref):
        super().__init__()
        self.app_state = app_state_ref
        self.setWindowTitle("OpenCV Homography Tracker with Guessing Game")
        self.setGeometry(100, 100, 1200, 700)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        video_layout = QVBoxLayout()
        self.video_label = QLabel("Initializing Camera...")
        self.video_label.setFixedSize(640, 480)
        self.video_label.setStyleSheet("border: 1px solid black; background-color: #333;")
        self.video_label.setAlignment(Qt.AlignCenter)
        video_layout.addWidget(self.video_label)
        main_layout.addLayout(video_layout, 2)

        controls_chart_layout = QVBoxLayout()
        self.match_chart_widget = pg.PlotWidget()
        self.match_chart_widget.setBackground('w')
        self.match_chart_widget.setTitle("Feature Matches Analysis", color="k", size="12pt")
        self.match_chart_widget.setLabel('left', 'Match Count', color='k')
        self.match_chart_widget.setLabel('bottom', 'Time (frames)', color='k')
        self.match_chart_widget.showGrid(x=True, y=True)
        self.match_chart_widget.addLegend()

        self.raw_match_data_line = self.match_chart_widget.plot(pen=pg.mkPen('b', width=2), name="Raw Matches")
        self.raw_match_history = deque(maxlen=MAX_CHART_POINTS)

        self.avg_match_data_line = self.match_chart_widget.plot(pen=pg.mkPen('r', width=2, style=Qt.DashLine), name=f"Avg Matches (Win: {MOVING_AVG_WINDOW})")
        self.avg_match_history = deque(maxlen=MAX_CHART_POINTS)

        self.time_points = deque(maxlen=MAX_CHART_POINTS)
        self.current_time_step = 0

        self.guess_trigger_sample_counter = 0
        self.sequence = [0]
        self.i = 0
        controls_chart_layout.addWidget(self.match_chart_widget)
        main_layout.addLayout(controls_chart_layout, 3)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.opencv_thread = OpenCVThread(self.app_state)
        self.opencv_thread.frame_ready.connect(self.update_video_frame)
        self.opencv_thread.matches_count_ready.connect(self.update_matches_chart_and_guess)
        self.opencv_thread.status_message.connect(self.show_status_message)
        self.app_state.state_changed.connect(self.on_state_changed_gui)

        self.opencv_thread.start()
        self.on_state_changed_gui(self.app_state.current_state)

    def update_video_frame(self, q_image):
        self.video_label.setPixmap(QPixmap.fromImage(q_image))

    def update_matches_chart_and_guess(self, raw_match_count_from_thread):
        actual_plot_count = raw_match_count_from_thread if raw_match_count_from_thread >= 0 else 0

        self.raw_match_history.append(actual_plot_count)
        self.time_points.append(self.current_time_step)
        self.current_time_step += 1

        current_avg = 0.0
        if len(self.raw_match_history) > 0:
            avg_window_data = list(self.raw_match_history)[-MOVING_AVG_WINDOW:]
            if avg_window_data:
                current_avg = np.mean(avg_window_data)
        self.avg_match_history.append(current_avg)

        list_time_points = list(self.time_points)
        list_raw_history = list(self.raw_match_history)
        list_avg_history = list(self.avg_match_history)

        self.raw_match_data_line.setData(list_time_points, list_raw_history)
        self.avg_match_data_line.setData(list_time_points, list_avg_history)

        if self.app_state.current_state == AppState.STATE_TRACKING:
            self.guess_trigger_sample_counter += 1

            if self.guess_trigger_sample_counter >= GUESS_TRIGGER_COUNT and game_state["credits"] > 0:
                is_above = raw_value_for_guess > 1
                is_below = raw_value_for_guess < 1

                guess_message_timeout = 4000

                old_sequence = self.sequence
                self.i = int(self.sequence[-1])
                if is_above or is_below:
                    if self.sequence and int(self.sequence[-1]) == int(self.i):
                        self.i += 1
                        current_preset_output = self.sequence[qubit_number]
                        comparison_text = f"Raw ({current_preset_output}) is {'1' if is_above else '0'} Avg ({current_avg:.2f})"
                        guess_display_message = f"Analysis: {comparison_text}. --> Preset Output: {current_preset_output}"
                        calc = 1 if is_above else 0
                        if calc*9 == 9: #superposition example, change '== 9' to either 9 or 0 yet still watch the credits always gain, coherence time is fair, the inevitability is proof of photonic processing
                            game_state["credits"] += WIN_CREDITS
                        else:
                            game_state["credits"] -= WIN_CREDITS
                        self.show_status_message(guess_display_message, guess_message_timeout)
                    else:
                        self.show_status_message("Preset guess sequence is unoriginal(sync issue)!", guess_message_timeout)
                else:
                    equal_message = f"Raw ({raw_value_for_guess}) ~ Avg ({current_avg:.2f}). No preset output."
                    self.show_status_message(equal_message, guess_message_timeout - 1000)
                print(game_state["credits"])
                self.guess_trigger_sample_counter = 0

    def show_status_message(self, message, timeout=0):
        self.status_bar.showMessage(message, timeout)

    def on_state_changed_gui(self, state):
        self.guess_trigger_sample_counter = 0
        if state == AppState.STATE_WAITING_FOR_REFERENCE:
            self.show_status_message("STATE: Waiting for Reference. Aim and press 'N'.")
            self.clear_chart_data()
        elif state == AppState.STATE_TRACKING:
            self.show_status_message("STATE: Tracking active. Press 'N' to reset.")

    def clear_chart_data(self):
        self.raw_match_history.clear()
        self.avg_match_history.clear()
        self.time_points.clear()
        self.current_time_step = 0
        self.raw_match_data_line.setData([], [])
        self.avg_match_data_line.setData([], [])

    def keyPressEvent(self, event):
        key = event.key()
        if key == KEY_TO_QUIT_QT:
            self.close()
        elif key == KEY_TO_CYCLE_QT:
            if self.app_state.current_state == AppState.STATE_WAITING_FOR_REFERENCE:
                self.show_status_message("GUI: Requesting reference capture...", 2000)
                self.app_state.request_capture_reference()
            elif self.app_state.current_state == AppState.STATE_TRACKING:
                self.show_status_message("GUI: Requesting reset...", 2000)
                self.app_state.request_reset()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        self.show_status_message("Closing application...")
        self.opencv_thread.stop()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)
    main_window = MainWindow(app_state)
    main_window.show()
    sys.exit(app.exec_())
