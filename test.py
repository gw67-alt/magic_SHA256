import sys
import time
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QStatusBar, QGridLayout
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer, QObject
from PyQt5.QtGui import QImage, QPixmap, QFont
import pyqtgraph as pg
from collections import deque
import numpy as np
import hashlib

# Adjustable game parameters
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

# --- Camera ID ---
CAMERA_0_ID = 0

# --- Hashing Configuration ---
HASHING_WORK_ITEMS_PER_FRAME = 1000000  # Number of nonces to test per frame/batch


def calculate_sha256_with_hashlib(data, prefix="", work_items=1, nonce_offset=0):
    """
    Calculate SHA-256 hash of the given data using hashlib.
    Search for a hash that starts with the given prefix.
    
    Args:
        data (str or bytes): The data to hash
        prefix (str): The prefix to search for in the resulting hash (hex string)
        work_items (int): Number of nonces to test
        nonce_offset (int): The starting nonce value for this batch
        
    Returns:
        list: List of (nonce, hash) tuples for matches found
    """
    # Convert input to bytes if it's a string
    if isinstance(data, str):
        data_bytes = data.encode('utf-8')
    elif isinstance(data, bytes):
        data_bytes = data
    else:
        raise TypeError("Input data must be a string or bytes.")
    
    results = []
    
    for i in range(work_items):
        nonce = nonce_offset + i
        
        # Append nonce bytes (big-endian)
        nonce_bytes = nonce.to_bytes(4, byteorder='big')
        message = data_bytes + nonce_bytes
        
        # Calculate hash using hashlib
        sha256 = hashlib.sha256()
        sha256.update(message)
        hash_hex = sha256.hexdigest()
        
        # Check if hash starts with the desired prefix
        if hash_hex.startswith(prefix):
            results.append((nonce, hash_hex))
            print(f"Match found! {data} Nonce: {nonce}, Hash: {hash_hex}")
            
    return results

        
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
        # Store state for single camera
        self._camera_state = self.STATE_WAITING_FOR_REFERENCE

    def get_camera_state(self, camera_id):
        return self._camera_state

    def set_camera_state(self, camera_id, value):
        if self._camera_state != value:
            self._camera_state = value
            self.state_changed.emit(camera_id, value)

    def request_capture_reference(self, camera_id):
        self.capture_reference_requested.emit(camera_id)

    def request_reset(self, camera_id):
        self.reset_requested.emit(camera_id)
    
    def update_game_state(self, state_dict):
        self.game_state_updated.emit(state_dict)

app_state = AppState()

# --- OpenCV Processing Thread for single camera ---
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
            self.current_threshold = np.mean(self.avg_match_history)
            
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


# --- Main Application Window (Updated for Single Camera) ---
class MainWindow(QMainWindow):
    def __init__(self, app_state_ref):
        super().__init__()
        self.app_state = app_state_ref
        self.setWindowTitle("Single Camera Homography Tracker with Guessing Game")
        self.setGeometry(100, 100, 800, 600)

        # Create camera tracker
        self.camera_tracker = CameraTracker(CAMERA_0_ID)

        # Setup UI
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Top section: camera display
        self.video_label = QLabel(f"Initializing Camera {CAMERA_0_ID}...")
        self.video_label.setFixedSize(640, 360)
        self.video_label.setStyleSheet("border: 1px solid black; background-color: #333;")
        self.video_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.video_label)
        
        # Middle section: chart
        self.match_chart = pg.PlotWidget()
        self.match_chart.setBackground('w')
        self.match_chart.setTitle(f"Camera {CAMERA_0_ID} Feature Matches", color="k", size="12pt")
        self.match_chart.setLabel('left', 'Match Count', color='k')
        self.match_chart.setLabel('bottom', 'Time (frames)', color='k')
        self.match_chart.showGrid(x=True, y=True)
        self.match_chart.addLegend()
        
        # Add threshold line
        self.threshold_line = pg.InfiniteLine(
            pos=self.camera_tracker.current_threshold, 
            angle=0, 
            pen=pg.mkPen('g', width=2, style=Qt.DashLine),
            label=f'Threshold ({self.camera_tracker.current_threshold:.2f})'
        )
        self.match_chart.addItem(self.threshold_line)
        
        # Add data plots
        self.raw_match_line = self.match_chart.plot(pen=pg.mkPen('b', width=2), name="Raw Matches")
        self.avg_match_line = self.match_chart.plot(pen=pg.mkPen('r', width=2, style=Qt.DashLine), name=f"Avg ({MOVING_AVG_WINDOW})")
        
        main_layout.addWidget(self.match_chart)
        
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
        instructions_label = QLabel("Instructions: Press 'N' to capture reference or reset camera. Press 'Q' to quit.")
        instructions_label.setAlignment(Qt.AlignCenter)
        instructions_label.setFont(QFont('Arial', 10))
        bottom_layout.addWidget(instructions_label)
        
        main_layout.addLayout(bottom_layout)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Create OpenCV thread
        self.opencv_thread = OpenCVThread(self.app_state, CAMERA_0_ID)
        self.opencv_thread.frame_ready.connect(self.update_video_frame)
        self.opencv_thread.matches_count_ready.connect(self.update_matches_chart)
        self.opencv_thread.status_message.connect(self.show_status_message)
            
        # Connect app state signals
        self.app_state.state_changed.connect(self.on_state_changed_gui)
        self.app_state.game_state_updated.connect(self.update_game_stats)
        
        # Game state tracking variables
        self.data_index = 0
        self.ready_count = 0
        self.success_count = 0
        self.init_count = 0
        
        # Start the thread
        self.opencv_thread.start()
            
        # Set initial UI state
        self.on_state_changed_gui(CAMERA_0_ID, self.app_state.get_camera_state(CAMERA_0_ID))

    def update_video_frame(self, q_image, camera_id):
        """Update the video frame"""
        self.video_label.setPixmap(QPixmap.fromImage(q_image))
        
    def update_matches_chart(self, raw_match_count, camera_id):
        """Update the matches chart and check for guesses"""
        # Update tracker data
        results = self.camera_tracker.update_match_data(raw_match_count)
        
        # Update chart
        self.raw_match_line.setData(results['time_points'], results['raw_history'])
        self.avg_match_line.setData(results['time_points'], results['avg_history'])
        
        # Update threshold line
        self.threshold_line.setValue(results['current_threshold'])
        self.threshold_line.label.setText(f'Threshold ({results["current_threshold"]:.2f})')
        
        # Only process guesses when actively tracking
        cam_state = self.app_state.get_camera_state(CAMERA_0_ID)
        
        if cam_state == AppState.STATE_TRACKING and results['should_guess']:
            # Reset the guess counter
            self.camera_tracker.reset_guess_counter()
            self.process_single_camera_guess()

    def process_single_camera_guess(self):
        """Process a guess using the state of the single camera"""
        # Get the below threshold status
        cam_below = self.camera_tracker.is_below_threshold
        
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
            
            # Build status message with camera state
            camera_status = f"Cam0: {'Below' if cam_below else 'Above'}"
            
            # Win condition: Camera is below threshold
            if cam_below:
                # Check for hash match
                if calculate_sha256_with_hashlib("GeorgeW", prefix="0000", work_items=80000, nonce_offset=self.init_count):
                    # Win scenario
                    game_state["credits"] += WIN_CREDITS
                    game_state["wins"] = game_state.get("wins", 0) + 1
                    print("Success @ ", game_state["losses"], " Ready states!", "nonce:", str(self.init_count))
                    self.show_status_message(
                        f"Win! {camera_status} | Hash match found. +{WIN_CREDITS} credits!", 2000)
                else:
                    game_state["losses"] = game_state.get("losses", 0) + 1
                    game_state["credits"] -= COST_PER_GUESS
                    self.show_status_message(
                        f"Ready state! {camera_status} | No hash match. -{COST_PER_GUESS} credits.", 2000)
            else:
                # Any other scenario is a loss
                game_state["credits"] -= COST_PER_GUESS
                self.show_status_message(
                    f"Lost! {camera_status} | Above threshold. -{COST_PER_GUESS} credits.", 2000)
                    
            # Log debug info
            print(f"Guess: {camera_status} | {current_value} @ index {self.init_count}, " 
                  f"Credits: {game_state['credits']}, " 
                  f"Threshold: {self.camera_tracker.current_threshold:.2f}")
                  
        except (ValueError, IndexError) as e:
            print(f"Error processing data at index {self.data_index}: {e}")
            self.show_status_message(f"Data processing error: {str(e)}", 3000)
            
        finally:
            # Move to next data point, wrap around if needed
            self.init_count = (self.init_count + 80000) % 4294967294
            # Update UI with new game state
            self.app_state.update_game_state(game_state)

    def update_game_stats(self, state_dict):
        """Update the game statistics display"""
        self.credits_label.setText(f"Credits: {state_dict['credits']}")
        self.wins_label.setText(f"Success states: {state_dict.get('wins', 0)}")
        self.losses_label.setText(f"Ready states: {state_dict.get('losses', 0)}")

    def show_status_message(self, message, timeout=0, camera_id=None):
        """Show a status message"""
        if camera_id is not None:
            message = f"Camera {camera_id}: {message}"
        self.status_bar.showMessage(message, timeout)

    def on_state_changed_gui(self, camera_id, state):
        """Handle UI state changes"""
        if state == AppState.STATE_WAITING_FOR_REFERENCE:
            self.show_status_message(f"STATE: Waiting for Reference. Aim and press 'N'.", 0)
            self.camera_tracker.clear_data()  # Clear chart data
        elif state == AppState.STATE_TRACKING:
            self.show_status_message(f"STATE: Tracking active. Press 'N' to reset.", 0)

    def keyPressEvent(self, event):
        key = event.key()
        if key == KEY_TO_QUIT_QT:
            self.close()
        elif key == KEY_TO_CYCLE_QT:
            # Request state change for camera
            current_state = self.app_state.get_camera_state(CAMERA_0_ID)
            if current_state == AppState.STATE_WAITING_FOR_REFERENCE:
                self.show_status_message(f"GUI: Requesting reference capture...", 2000)
                self.app_state.request_capture_reference(CAMERA_0_ID)
            elif current_state == AppState.STATE_TRACKING:
                self.show_status_message(f"GUI: Requesting reset...", 2000)
                self.app_state.request_reset(CAMERA_0_ID)
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        self.show_status_message("Closing application...")
        self.opencv_thread.stop()
        event.accept()
        
if __name__ == '__main__':
    # Check if QApplication already exists
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)
    
    pg.setConfigOptions(antialias=True)
    main_window = MainWindow(app_state)
    main_window.show()
    
    # Only execute if not already running
    if not QApplication.instance().activeWindow():
        sys.exit(app.exec_())