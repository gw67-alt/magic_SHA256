import sys
import time
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QStatusBar, QGridLayout
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer, QObject
from PyQt5.QtGui import QImage, QPixmap, QFont
import pyqtgraph as pg
from collections import deque
import os
import hashlib

PREFIX = "000000"

def calculate_sha256_with_library(data):
    """
    Calculates the SHA-256 hash of the given data using Python's hashlib library.

    Args:
        data (bytes or str): The data to hash. If it's a string, it will be encoded to UTF-8.

    Returns:
        str: The hexadecimal representation of the SHA-256 hash.
    """
    try:
        # Create a SHA-256 hash object
        sha256_hash = hashlib.sha256()

        # Update the hash object with the data
        if isinstance(data, str):
            sha256_hash.update(data.encode('utf-8'))
        elif isinstance(data, bytes):
            sha256_hash.update(data)
        else:
            raise TypeError("Input data must be a string or bytes.")

        # Get the hexadecimal representation of the hash digest
        hex_digest = sha256_hash.hexdigest()
        if hex_digest.startswith(PREFIX):
            print(data, hex_digest)
        return hex_digest

    except TypeError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"
        
        
        
# Adjustable game parameters
# Initial threshold - will be dynamically updated to use the average match count
MATCH_THRESHOLD_FOR_GUESS = 0.5  # Initial value, will be adjusted dynamically
STARTING_CREDITS = 1000000
COST_PER_GUESS = 1
WIN_CREDITS = 1

# Game state
game_state = {
    "credits": STARTING_CREDITS,
    "wins": 0,
    "losses": 0
}

# Load data file safely
data = []
try:
    with open("x.txt", 'r', encoding='utf-8') as file:
        data = [line.strip() for line in file.readlines() if line.strip()]
    if not data:
        print("Warning: x.txt file is empty. Using dummy data.")
        data = ["55", "55", "AA", "55", "BB"]  # Dummy data if file is empty
except FileNotFoundError:
    print("Warning: x.txt file not found. Creating with dummy data.")
    with open("x.txt", 'w', encoding='utf-8') as file:
        file.write("55\n55\nAA\n55\nBB\n")
    data = ["55", "55", "AA", "55", "BB"]  # Dummy data for new file

# --- OpenCV Configuration ---
MIN_MATCH_COUNT = 5  # Lowered from 10 to be more lenient
LOWE_RATIO_TEST = 0.999  # Increased from 0.10 to be less strict
KEY_TO_CYCLE_QT = Qt.Key_N
KEY_TO_QUIT_QT = Qt.Key_Q

# --- Chart Configuration ---
MAX_CHART_POINTS = 100  # Number of data points to display on the chart
MOVING_AVG_WINDOW = 150  # Window size for the moving average - reduced for quicker response

# --- Guessing Configuration ---
GUESS_TRIGGER_COUNT = 0  # Number of samples before attempting a guess

# --- Camera IDs ---
CAMERA_0_ID = 0
CAMERA_1_ID = 1  # Second camera ID (usually 1 for built-in + external)

# --- State Management Object ---
class AppState(QObject):
    state_changed = pyqtSignal(int, int)  # camera_id, state
    capture_reference_requested = pyqtSignal(int)  # camera_id
    reset_requested = pyqtSignal(int)  # camera_id
    game_state_updated = pyqtSignal(dict)  # New signal for game state updates

    STATE_WAITING_FOR_REFERENCE = 0
    STATE_TRACKING = 1

    def __init__(self):
        super().__init__()
        # Store state per camera
        self._camera_states = {
            CAMERA_0_ID: self.STATE_WAITING_FOR_REFERENCE,
            CAMERA_1_ID: self.STATE_WAITING_FOR_REFERENCE
        }

    def get_camera_state(self, camera_id):
        return self._camera_states.get(camera_id, self.STATE_WAITING_FOR_REFERENCE)

    def set_camera_state(self, camera_id, value):
        if self._camera_states.get(camera_id) != value:
            self._camera_states[camera_id] = value
            self.state_changed.emit(camera_id, value)

    def request_capture_reference(self, camera_id):
        self.capture_reference_requested.emit(camera_id)

    def request_reset(self, camera_id):
        self.reset_requested.emit(camera_id)
    
    def update_game_state(self, state_dict):
        self.game_state_updated.emit(state_dict)

app_state = AppState()

# --- OpenCV Processing Thread for a single camera ---
class OpenCVThread(QThread):
    frame_ready = pyqtSignal(QImage, int)  # QImage, camera_id
    matches_count_ready = pyqtSignal(int, int)  # matches, camera_id
    status_message = pyqtSignal(str, int)  # message, camera_id

    def __init__(self, app_state_ref, camera_id):
        super().__init__()
        self.running = False
        self.app_state = app_state_ref
        self.camera_id = camera_id

        self.reference_frame = None
        self.reference_kp = None
        self.reference_des = None
        self.orb = None
        self.bf_matcher = None
        self._capture_next_frame_as_reference = False

        # Connect to signals filtered by camera_id
        self.app_state.capture_reference_requested.connect(
            lambda cam_id: self.prepare_for_reference_capture() if cam_id == self.camera_id else None)
        self.app_state.reset_requested.connect(
            lambda cam_id: self.reset_reference() if cam_id == self.camera_id else None)

    def initialize_features(self):
        # Enhanced ORB parameters for better feature detection
        self.orb = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8)
        # BFMatcher with crossCheck=False is used for knnMatch
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def prepare_for_reference_capture(self):
        self._capture_next_frame_as_reference = True

    def reset_reference(self):
        self.reference_frame = None
        self.reference_kp = None
        self.reference_des = None
        self._capture_next_frame_as_reference = False
        # This will trigger state_changed signal handled by MainWindow
        self.app_state.set_camera_state(self.camera_id, AppState.STATE_WAITING_FOR_REFERENCE)
        
    def preprocess_frame(self, frame):
        """Enhance frame for better feature detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization to improve contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Optional: Apply slight Gaussian blur to reduce noise
        # enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        return enhanced

    def run(self):
        self.running = True
        self.initialize_features()

        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            self.status_message.emit(f"Error: Cannot open camera {self.camera_id}.", self.camera_id)
            self.running = False
            return

        while self.running:
            ret, frame = cap.read()
            if not ret:
                self.status_message.emit(f"Error: Can't receive frame from camera {self.camera_id}.", self.camera_id)
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.frame_ready.emit(qt_image.copy(), self.camera_id)  # Emit a copy with camera_id

            num_good_matches_for_signal = 0  # Default to 0 (no good matches / not tracking)

            if self._capture_next_frame_as_reference:
                self.reference_frame = frame.copy()  # Keep original for display
                processed_ref = self.preprocess_frame(self.reference_frame)
                self.reference_kp, self.reference_des = self.orb.detectAndCompute(processed_ref, None)

                if self.reference_des is None or len(self.reference_kp) < MIN_MATCH_COUNT:
                    self.status_message.emit(
                        f"Ref. Capture Failed (Cam {self.camera_id}): Not enough features " +
                        f"({len(self.reference_kp) if self.reference_kp is not None else 0}). Try again.",
                        self.camera_id
                    )
                    self.reference_frame = None  # Clear invalid reference
                    self.reference_kp = None
                    self.reference_des = None
                    self.app_state.set_camera_state(self.camera_id, AppState.STATE_WAITING_FOR_REFERENCE)
                else:
                    self.status_message.emit(
                        f"Reference Captured (Cam {self.camera_id}): {len(self.reference_kp)} keypoints. Tracking...",
                        self.camera_id
                    )
                    self.app_state.set_camera_state(self.camera_id, AppState.STATE_TRACKING)
                self._capture_next_frame_as_reference = False
                self.matches_count_ready.emit(0, self.camera_id)  # Emit 0 matches right after capture attempt

            elif self.app_state.get_camera_state(self.camera_id) == AppState.STATE_TRACKING and self.reference_frame is not None and self.reference_des is not None:
                processed_frame = self.preprocess_frame(frame)
                current_kp, current_des = self.orb.detectAndCompute(processed_frame, None)
                actual_good_matches_count = 0

                if current_des is not None and len(current_des) > 0:
                    try:
                        # Try to perform knnMatch
                        all_matches = self.bf_matcher.knnMatch(self.reference_des, current_des, k=2)
                        good_matches = []

                        for m_arr in all_matches:
                            if len(m_arr) == 2:
                                m, n = m_arr
                                if m.distance < LOWE_RATIO_TEST * n.distance:
                                    good_matches.append(m)

                        actual_good_matches_count = len(good_matches)

                        # Improved logging
                        if actual_good_matches_count < MIN_MATCH_COUNT:
                            self.status_message.emit(
                                f"Low matches (Cam {self.camera_id}): {actual_good_matches_count}/{MIN_MATCH_COUNT}",
                                self.camera_id
                            )
                        
                        if actual_good_matches_count >= MIN_MATCH_COUNT:
                            try:
                                src_pts = np.float32([self.reference_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                                dst_pts = np.float32([current_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                                # RANSAC threshold increased slightly for more tolerance
                                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10.0)

                                if H is None:
                                    num_good_matches_for_signal = actual_good_matches_count // 2  # Half the matches instead of -1
                                    self.status_message.emit(f"Homography calculation failed (Cam {self.camera_id})", self.camera_id)
                                else:
                                    num_good_matches_for_signal = actual_good_matches_count
                            except Exception as e:
                                self.status_message.emit(f"Homography error (Cam {self.camera_id}): {str(e)}", self.camera_id)
                                num_good_matches_for_signal = 0
                        else:
                            num_good_matches_for_signal = actual_good_matches_count
                    except Exception as e:
                        self.status_message.emit(f"Match error (Cam {self.camera_id}): {str(e)}", self.camera_id)
                        num_good_matches_for_signal = 0
                else:
                    num_good_matches_for_signal = 0
                    self.status_message.emit(f"No features detected in current frame (Cam {self.camera_id})", self.camera_id)

                self.matches_count_ready.emit(num_good_matches_for_signal, self.camera_id)

            else:  # Waiting for reference or reference invalid
                self.matches_count_ready.emit(0, self.camera_id)  # Emit 0 if not tracking

        cap.release()
        self.status_message.emit(f"Camera {self.camera_id} released.", self.camera_id)

    def stop(self):
        self.running = False
        self.wait()


# --- Camera Match Data Tracker Class ---
class CameraTracker:
    def __init__(self, camera_id, initial_threshold=MATCH_THRESHOLD_FOR_GUESS):
        self.camera_id = camera_id
        self.current_threshold = initial_threshold
        self.threshold_history = deque(maxlen=30)
        self.raw_match_history = deque(maxlen=MAX_CHART_POINTS)
        self.avg_match_history = deque(maxlen=MAX_CHART_POINTS)
        self.time_points = deque(maxlen=MAX_CHART_POINTS)
        self.current_time_step = 0
        self.guess_trigger_sample_counter = 0
        self.is_below_threshold = False  # Track if current matches are below threshold

    def update_match_data(self, raw_match_count):
        """Update match data and threshold calculations"""
        actual_plot_count = raw_match_count if raw_match_count >= 0 else 0

        self.raw_match_history.append(actual_plot_count)
        self.time_points.append(self.current_time_step)
        self.current_time_step += 1

        # Update threshold history and recalculate dynamic threshold
        self.threshold_history.append(actual_plot_count)
        
        # Calculate moving average for chart display
        current_avg = 0.0
        if len(self.raw_match_history) > 0:
            avg_window_data = list(self.raw_match_history)[-MOVING_AVG_WINDOW:]
            if avg_window_data:  # Ensure not empty
                current_avg = np.mean(avg_window_data)
        self.avg_match_history.append(current_avg)
        
        # Update dynamic threshold based on recent match history
        if len(self.threshold_history) > 5:  # Need at least a few data points
            self.current_threshold = np.mean(self.threshold_history)
            
        # Update is_below_threshold status
        self.is_below_threshold = actual_plot_count < self.current_threshold
        
        # Increment the guess trigger counter
        self.guess_trigger_sample_counter += 1
        
        return {
            'time_points': list(self.time_points),
            'raw_history': list(self.raw_match_history),
            'avg_history': list(self.avg_match_history),
            'current_threshold': self.current_threshold,
            'is_below_threshold': self.is_below_threshold,
            'should_guess': self.guess_trigger_sample_counter >= GUESS_TRIGGER_COUNT
        }
        
    def reset_guess_counter(self):
        """Reset the guess trigger counter after a guess is made"""
        self.guess_trigger_sample_counter = 0
        
    def clear_data(self):
        """Clear all tracked data"""
        self.raw_match_history.clear()
        self.avg_match_history.clear()
        self.time_points.clear()
        self.threshold_history.clear()
        self.current_time_step = 0
        self.current_threshold = MATCH_THRESHOLD_FOR_GUESS
        self.guess_trigger_sample_counter = 0
        self.is_below_threshold = False


# --- Main Application Window (Updated with Dual Camera Support) ---
class MainWindow(QMainWindow):
    def __init__(self, app_state_ref):
        super().__init__()
        self.app_state = app_state_ref
        self.setWindowTitle("Dual Camera Homography Tracker with Guessing Game")
        self.setGeometry(100, 100, 1400, 800)

        # Create camera trackers
        self.camera_trackers = {
            CAMERA_0_ID: CameraTracker(CAMERA_0_ID),
            CAMERA_1_ID: CameraTracker(CAMERA_1_ID)
        }
        self.last_alignment_alert_time = 0

        # Setup UI
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Top section: cameras grid
        cameras_layout = QGridLayout()
        
        # Video displays
        self.video_labels = {}
        for cam_id in [CAMERA_0_ID, CAMERA_1_ID]:
            label = QLabel(f"Initializing Camera {cam_id}...")
            label.setFixedSize(640, 360)
            label.setStyleSheet("border: 1px solid black; background-color: #333;")
            label.setAlignment(Qt.AlignCenter)
            self.video_labels[cam_id] = label
            row = cam_id // 2
            col = cam_id % 2
            cameras_layout.addWidget(label, row, col)
        
        main_layout.addLayout(cameras_layout)
        
        # Middle section: charts grid
        charts_layout = QGridLayout()
        
        # Create charts for each camera
        self.match_charts = {}
        self.raw_match_lines = {}
        self.avg_match_lines = {}
        self.threshold_lines = {}
        
        for cam_id in [CAMERA_0_ID, CAMERA_1_ID]:
            chart = pg.PlotWidget()
            chart.setBackground('w')
            chart.setTitle(f"Camera {cam_id} Feature Matches", color="k", size="12pt")
            chart.setLabel('left', 'Match Count', color='k')
            chart.setLabel('bottom', 'Time (frames)', color='k')
            chart.showGrid(x=True, y=True)
            chart.addLegend()
            
            # Add threshold line
            threshold_line = pg.InfiniteLine(
                pos=self.camera_trackers[cam_id].current_threshold, 
                angle=0, 
                pen=pg.mkPen('g', width=2, style=Qt.DashLine),
                label=f'Threshold ({self.camera_trackers[cam_id].current_threshold:.2f})'
            )
            chart.addItem(threshold_line)
            
            # Add data plots
            raw_line = chart.plot(pen=pg.mkPen('b', width=2), name="Raw Matches")
            avg_line = chart.plot(pen=pg.mkPen('r', width=2, style=Qt.DashLine), name=f"Avg ({MOVING_AVG_WINDOW})")
            
            # Store references
            self.match_charts[cam_id] = chart
            self.raw_match_lines[cam_id] = raw_line
            self.avg_match_lines[cam_id] = avg_line
            self.threshold_lines[cam_id] = threshold_line
            
            # Add to layout
            row = cam_id // 2
            col = cam_id % 2
            charts_layout.addWidget(chart, row, col)
            
        main_layout.addLayout(charts_layout)
        
        # Bottom section: game stats and instructions
        bottom_layout = QVBoxLayout()
        
        # Game stats panel
        stats_layout = QHBoxLayout()
        
        self.credits_label = QLabel(f"Credits: {game_state['credits']}")
        self.credits_label.setFont(QFont('Arial', 14))
        self.credits_label.setStyleSheet("color: green; font-weight: bold;")
        
        self.wins_label = QLabel(f"Success states: {game_state.get('wins', 0)}")
        self.wins_label.setFont(QFont('Arial', 14))
        self.wins_label.setStyleSheet("color: blue;")
        
        self.losses_label = QLabel(f"Ready states: {game_state.get('losses', 0)}")
        self.losses_label.setFont(QFont('Arial', 14))
        self.losses_label.setStyleSheet("color: red;")
        
        stats_layout.addWidget(self.credits_label)
        stats_layout.addWidget(self.wins_label)
        stats_layout.addWidget(self.losses_label)
        
        bottom_layout.addLayout(stats_layout)
        
        # Instructions
        instructions_label = QLabel("Instructions: Press 'N' to capture reference or reset for both cameras. Press 'Q' to quit.")
        instructions_label.setAlignment(Qt.AlignCenter)
        instructions_label.setFont(QFont('Arial', 10))
        bottom_layout.addWidget(instructions_label)
        
        main_layout.addLayout(bottom_layout)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Create OpenCV threads for each camera
        self.opencv_threads = {}
        for cam_id in [CAMERA_0_ID, CAMERA_1_ID]:
            thread = OpenCVThread(self.app_state, cam_id)
            thread.frame_ready.connect(self.update_video_frame)
            thread.matches_count_ready.connect(self.update_matches_chart)
            thread.status_message.connect(self.show_status_message)
            self.opencv_threads[cam_id] = thread
            
        # Connect app state signals
        self.app_state.state_changed.connect(self.on_state_changed_gui)
        self.app_state.game_state_updated.connect(self.update_game_stats)
        
        # Game state tracking variables
        self.data_index = 0
        self.ready_count = 0
        self.success_count = 0
        self.init_count = 0
        
        # Start the threads
        for thread in self.opencv_threads.values():
            thread.start()
            
        # Set initial UI state
        for cam_id in [CAMERA_0_ID, CAMERA_1_ID]:
            self.on_state_changed_gui(cam_id, self.app_state.get_camera_state(cam_id))

    def update_video_frame(self, q_image, camera_id):
        """Update the video frame for the specified camera"""
        if camera_id in self.video_labels:
            self.video_labels[camera_id].setPixmap(QPixmap.fromImage(q_image))
    def check_visual_alignment(self):
        """Check if the average bars from both cameras are at the same visual position"""
        # Only check if we have data for both cameras
        cam0_tracker = self.camera_trackers[CAMERA_0_ID]
        cam1_tracker = self.camera_trackers[CAMERA_1_ID]
        
        # Check if we're in cooldown period
        current_time = time.time()
        if hasattr(self, 'last_alignment_alert_time') and \
           (current_time - self.last_alignment_alert_time) < 500:  # 5 second cooldown
            return
        
        if not cam0_tracker.avg_match_history or not cam1_tracker.avg_match_history:
            return
        
        # Get the most recent average values
        cam0_avg = cam0_tracker.avg_match_history[-1]
        cam1_avg = cam1_tracker.avg_match_history[-1]
        
        # Get the current Y range for each chart
        cam0_y_range = self.match_charts[CAMERA_0_ID].getViewBox().viewRange()[1]
        cam1_y_range = self.match_charts[CAMERA_1_ID].getViewBox().viewRange()[1]
        
        # Normalize the values to the chart's Y range (0-1 scale)
        cam0_y_min, cam0_y_max = cam0_y_range
        cam1_y_min, cam1_y_max = cam1_y_range
        
        # Avoid division by zero
        if cam0_y_max == cam0_y_min or cam1_y_max == cam1_y_min:
            return
            
        cam0_normalized = (cam0_avg - cam0_y_min) / (cam0_y_max - cam0_y_min)
        cam1_normalized = (cam1_avg - cam1_y_min) / (cam1_y_max - cam1_y_min)
        
        # Define alignment threshold (how close the normalized positions need to be)
        # Adjust this value based on how precise you want the alignment to be
        alignment_threshold = 0.03  # 5% of the chart height
        
        # Check if normalized positions are within the threshold
        if abs(cam0_normalized - cam1_normalized) <= alignment_threshold:
            # Aligned! Print message and update status
            alignment_msg = (f"ALIGNMENT DETECTED! Both cameras' averages are visually aligned "
                            f"(Cam0: {cam0_avg:.2f} at {cam0_normalized:.2f}, "
                            f"Cam1: {cam1_avg:.2f} at {cam1_normalized:.2f})")
            print(alignment_msg)
            self.show_status_message(alignment_msg, 3000)
            
            # Update the last alert time
            self.last_alignment_alert_time = current_time
            
    def update_matches_chart(self, raw_match_count, camera_id):
        """Update the matches chart for the specified camera and check for guesses"""
        if camera_id not in self.camera_trackers:
            return
            
        # Update tracker data
        tracker = self.camera_trackers[camera_id]
        results = tracker.update_match_data(raw_match_count)
        
        # Update chart
        self.raw_match_lines[camera_id].setData(results['time_points'], results['raw_history'])
        self.avg_match_lines[camera_id].setData(results['time_points'], results['avg_history'])
        
        # Update threshold line
        self.threshold_lines[camera_id].setValue(results['current_threshold'])
        self.threshold_lines[camera_id].label.setText(f'Threshold ({results["current_threshold"]:.2f})')
        
        # Only process guesses when actively tracking both cameras
        cam0_state = self.app_state.get_camera_state(CAMERA_0_ID)
        cam1_state = self.app_state.get_camera_state(CAMERA_1_ID)
        
        both_tracking = (cam0_state == AppState.STATE_TRACKING and 
                         cam1_state == AppState.STATE_TRACKING)
        
        # Check for visual alignment of average bars (only when both cameras are tracking)
        if both_tracking and len(results['avg_history']) > 0:
            self.check_visual_alignment()
        
        if both_tracking and results['should_guess']:
            # Reset the guess counter
            tracker.reset_guess_counter()
            
            # If this is camera 0 and it's time to make a guess, check both cameras
            if camera_id == CAMERA_0_ID:
                self.process_dual_camera_guess()

    def process_dual_camera_guess(self):
        """Process a guess using the state of both cameras"""
        # Get the below threshold status for both cameras
        cam0_below = self.camera_trackers[CAMERA_0_ID].is_below_threshold
        cam1_below = self.camera_trackers[CAMERA_1_ID].is_below_threshold
        
        # Only proceed if we have credits
        if game_state["credits"] <= 0:
            self.show_status_message("No credits left! Game over.", 3000)
            return
            
        # Guard against empty data list
        if not data:
            self.show_status_message("Error: No data available for guessing!", 3000)
            return
            
        try:
            # Try to interpret the current data value as hex
            current_value = self.init_count
            hex_value = current_value
            
            # Build status message with camera states
            camera_status = f"Cam0: {'Below' if cam0_below else 'Above'}, " \
                           f"Cam1: {'Below' if cam1_below else 'Above'}"
            if cam0_below:
                # Win scenario
                game_state["credits"] += COST_PER_GUESS
                self.show_status_message(
                    f"Win! {camera_status} | {current_value} = 0x55. +{WIN_CREDITS} credits!", 2000)
            else:
                # Any other scenario is a loss
                game_state["credits"] -= COST_PER_GUESS
                self.show_status_message(
                    f"Lost! {camera_status} | {current_value} " + 
                    ("= 0x55" if hex_value == 0x55 else "≠ 0x55") + 
                    f". -{COST_PER_GUESS} credits.", 2000)
                    
            # Win condition: Both cameras are below threshold AND value is 0x55
            if cam1_below:
                # Win scenario
                game_state["credits"] += COST_PER_GUESS
                self.show_status_message(
                    f"Win! {camera_status} | {current_value} = 0x55. +{WIN_CREDITS} credits!", 2000)
            else:
                # Any other scenario is a loss
                game_state["credits"] -= COST_PER_GUESS
                self.show_status_message(
                    f"Lost! {camera_status} | {current_value} " + 
                    ("= 0x55" if hex_value == 0x55 else "≠ 0x55") + 
                    f". -{COST_PER_GUESS} credits.", 2000)
            
            # Win condition: Both cameras are below threshold AND value is 0x55
            for i in range(100000):
                if cam0_below and cam1_below:
                
                    if calculate_sha256_with_library("GeorgeW"+str(self.init_count+i)).startswith(PREFIX): # measure comparisons
                        # Win scenario
                        game_state["credits"] += COST_PER_GUESS
                        game_state["wins"] = game_state.get("wins", 0) + 1
                        print("Success @ ", game_state["losses"], " Ready states!")
                        self.show_status_message(
                            f"Win! {camera_status} | {current_value} = 0x55. +{WIN_CREDITS} credits!", 2000)
                    else:
                        game_state["losses"] = game_state.get("losses", 0) + 1
            else:

                # Any other scenario is a loss
                game_state["credits"] -= COST_PER_GUESS
                self.show_status_message(
                    f"Lost! {camera_status} | {current_value} " + 
                    ("= 0x55" if hex_value == 0x55 else "≠ 0x55") + 
                    f". -{COST_PER_GUESS} credits.", 2000)
            # Win condition: Both cameras are below threshold AND value is 0x55
           
                    
            # Log debug info
            print(f"Guess: {camera_status} | {current_value} @ index {self.init_count}, " 
                  f"Credits: {game_state['credits']}, " 
                  f"Thresholds: Cam0={self.camera_trackers[CAMERA_0_ID].current_threshold:.2f}, " 
                  f"Cam1={self.camera_trackers[CAMERA_1_ID].current_threshold:.2f}")
                  
        except (ValueError, IndexError) as e:
            print(f"Error processing data at index {self.data_index}: {e}")
            self.show_status_message(f"Data processing error: {str(e)}", 3000)
            
        finally:
            # Move to next data point, wrap around if needed
            self.init_count = (self.init_count + 100000) % 1000000000000
            # Update UI with new game state
            self.app_state.update_game_state(game_state)

    def update_game_stats(self, state_dict):
        """Update the game statistics display"""
        self.credits_label.setText(f"Credits: {state_dict['credits']}")
        self.wins_label.setText(f"Success states: {state_dict.get('wins', 0)}")
        self.losses_label.setText(f"Ready states: {state_dict.get('losses', 0)}")

    def show_status_message(self, message, timeout=0, camera_id=None):
        """Show a status message, optionally from a specific camera"""
        if camera_id is not None:
            message = f"Camera {camera_id}: {message}"
        self.status_bar.showMessage(message, timeout)

    def on_state_changed_gui(self, camera_id, state):
        """Handle UI state changes for a specific camera"""
        tracker = self.camera_trackers.get(camera_id)
        if tracker:
            if state == AppState.STATE_WAITING_FOR_REFERENCE:
                self.show_status_message(f"STATE (Cam {camera_id}): Waiting for Reference. Aim and press 'N'.", 0, camera_id)
                tracker.clear_data()  # Clear chart data
            elif state == AppState.STATE_TRACKING:
                self.show_status_message(f"STATE (Cam {camera_id}): Tracking active. Press 'N' to reset.", 0, camera_id)

    def keyPressEvent(self, event):
        key = event.key()
        if key == KEY_TO_QUIT_QT:
            self.close()
        elif key == KEY_TO_CYCLE_QT:
            # Request state change for both cameras
            for cam_id in [CAMERA_0_ID, CAMERA_1_ID]:
                current_state = self.app_state.get_camera_state(cam_id)
                if current_state == AppState.STATE_WAITING_FOR_REFERENCE:
                    self.show_status_message(f"GUI: Requesting reference capture for Camera {cam_id}...", 2000)
                    self.app_state.request_capture_reference(cam_id)
                elif current_state == AppState.STATE_TRACKING:
                    self.show_status_message(f"GUI: Requesting reset for Camera {cam_id}...", 2000)
                    self.app_state.request_reset(cam_id)
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        self.show_status_message("Closing application...")
        for thread in self.opencv_threads.values():
            thread.stop()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)
    main_window = MainWindow(app_state)
    main_window.show()
    sys.exit(app.exec_())
