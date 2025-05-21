import sys
import time
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QStatusBar, QGridLayout
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer, QObject
from PyQt5.QtGui import QImage, QPixmap, QFont
import pyqtgraph as pg
from collections import deque
import numpy as np
import pyopencl as cl
import os
import hashlib

# Set environment variable to avoid spurious OpenCL compiler output
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '0'

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

# --- Camera IDs ---
CAMERA_0_ID = 0
CAMERA_1_ID = 1  # Second camera ID (usually 1 for built-in + external)

# --- Hashing Configuration ---
HASHING_WORK_ITEMS_PER_FRAME = 1000000 # Number of nonces to test per frame/batch


class OpenCLSHA256:
    """
    A class to calculate SHA-256 hashes using OpenCL for GPU acceleration.
    """
    def __init__(self):
        # Initialize OpenCL environment with better error handling
        try:
            platforms = cl.get_platforms()
            if not platforms:
                raise RuntimeError("No OpenCL platforms found")

            # Automatically select the first platform/device (can be modified for user choice)
            self.ctx = None
            self.queue = None

            for platform in platforms:
                try:
                    devices = platform.get_devices(device_type=cl.device_type.GPU)
                    if not devices:
                        devices = platform.get_devices(device_type=cl.device_type.CPU)
                    if devices:
                        self.ctx = cl.Context(devices)
                        self.queue = cl.CommandQueue(self.ctx)
                        print(f"Using device: {devices[0].name} from platform: {platform.name}")
                        break # Found a suitable device, exit loop
                except cl.LogicError as e:
                    print(f"Could not create context for platform {platform.name}: {e}")
            
            if not self.ctx:
                raise RuntimeError("No suitable OpenCL device found on any platform.")

        except Exception as e:
            print(f"OpenCL initialization error: {e}")
            # Fallback to a simple context if specific device selection fails
            try:
                self.ctx = cl.create_some_context(interactive=False)
                self.queue = cl.CommandQueue(self.ctx)
                print(f"Fallback: Using default OpenCL context.")
            except Exception as fallback_e:
                print(f"FATAL: Could not create any OpenCL context: {fallback_e}")
                sys.exit(1) # Exit if OpenCL is not usable at all

        # SHA-256 OpenCL kernel code
        self.kernel_code = """
        // SHA-256 constants
        __constant uint k[64] = {
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
            0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
            0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
        };

        // SHA-256 helper macros
        #define ROTR(x, n) ((x >> n) | (x << (32 - n)))
        #define Ch(x, y, z) ((x & y) ^ (~x & z))
        #define Maj(x, y, z) ((x & y) ^ (x & z) ^ (y & z))
        #define Sigma0(x) (ROTR(x, 2) ^ ROTR(x, 13) ^ ROTR(x, 22))
        #define Sigma1(x) (ROTR(x, 6) ^ ROTR(x, 11) ^ ROTR(x, 25))
        #define sigma0(x) (ROTR(x, 7) ^ ROTR(x, 18) ^ (x >> 3))
        #define sigma1(x) (ROTR(x, 17) ^ ROTR(x, 19) ^ (x >> 10))

        // Convert a byte array to a uint array (big-endian)
        void bytes_to_uint(const uchar *input, uint *output, int length) {
            for (int i = 0; i < length / 4; i++) {
                output[i] = ((uint)input[i*4] << 24) | 
                            ((uint)input[i*4+1] << 16) | 
                            ((uint)input[i*4+2] << 8) | 
                            ((uint)input[i*4+3]);
            }
        }

        // Convert a uint array to a byte array (big-endian)
        void uint_to_bytes(const uint *input, uchar *output, int length) {
            for (int i = 0; i < length; i++) {
                output[i*4] = (input[i] >> 24) & 0xFF;
                output[i*4+1] = (input[i] >> 16) & 0xFF;
                output[i*4+2] = (input[i] >> 8) & 0xFF;
                output[i*4+3] = input[i] & 0xFF;
            }
        }

        // Main SHA-256 kernel
        __kernel void calculate_sha256(
            __global const uchar* input_data,
            uint input_length,
            __global uchar* output_hashes,
            __global const char* prefix,
            uint prefix_length,
            uint nonce_offset // Now explicitly the starting nonce for this batch
        ) {
            // Get global ID for thread, added to nonce_offset to form the actual nonce
            uint id = get_global_id(0);
            uint nonce = id + nonce_offset;
            
            // Allocate memory for the message. SHA-256 processes 64-byte blocks.
            // A message + nonce up to 55 bytes fits in one block after padding.
            // If it's longer, it will require a second block.
            // We assume input_length + 4 (for nonce) is small enough for single block processing.
            uchar message[64]; // One SHA-256 block

            // Hash state (initial H values)
            uint h0 = 0x6a09e667;
            uint h1 = 0xbb67ae85;
            uint h2 = 0x3c6ef372;
            uint h3 = 0xa54ff53a;
            uint h4 = 0x510e527f;
            uint h5 = 0x9b05688c;
            uint h6 = 0x1f83d9ab;
            uint h7 = 0x5be0cd19;
            
            // Copy input data to local message buffer
            for (uint i = 0; i < input_length; i++) {
                message[i] = input_data[i];
            }
            
            // Append the 4-byte nonce value (big-endian)
            // Ensure there's space for the nonce + padding
            if (input_length + 4 > 55) {
                // This scenario means multi-block hashing is needed, which this kernel doesn't support
                // For now, we'll just return early or produce incorrect results.
                // In a real miner, you'd handle this by processing more blocks.
                return; 
            }
            message[input_length] = (nonce >> 24) & 0xFF;
            message[input_length+1] = (nonce >> 16) & 0xFF;
            message[input_length+2] = (nonce >> 8) & 0xFF;
            message[input_length+3] = nonce & 0xFF;
            
            // Calculate the total message length including nonce
            uint msg_length = input_length + 4;
            
            // Add padding
            // 1. Append a single '1' bit (0x80 byte)
            message[msg_length] = 0x80;
            
            // 2. Fill with zeros until message length in bits is 448 (mod 512),
            // which means bytes up to index 55 (0-indexed).
            // This loop assumes msg_length is < 56.
            for (uint i = msg_length + 1; i < 56; i++) {
                message[i] = 0;
            }
            
            // 3. Append the original message length (before padding) in bits as a 64-bit big-endian integer
            ulong bit_length = (ulong)msg_length * 8;
            message[56] = (bit_length >> 56) & 0xFF;
            message[57] = (bit_length >> 48) & 0xFF;
            message[58] = (bit_length >> 40) & 0xFF;
            message[59] = (bit_length >> 32) & 0xFF;
            message[60] = (bit_length >> 24) & 0xFF;
            message[61] = (bit_length >> 16) & 0xFF;
            message[62] = (bit_length >> 8) & 0xFF;
            message[63] = bit_length & 0xFF;
            
            // Initialize message schedule array
            uint w[64];
            
            // Fill first 16 words from the padded message
            bytes_to_uint(message, w, 64);
            
            // Extend the first 16 words into the remaining 48 words
            for (uint i = 16; i < 64; i++) {
                w[i] = sigma1(w[i-2]) + w[i-7] + sigma0(w[i-15]) + w[i-16];
            }
            
            // Initialize working variables with current hash values
            uint a = h0;
            uint b = h1;
            uint c = h2;
            uint d = h3;
            uint e = h4;
            uint f = h5;
            uint g = h6;
            uint h = h7;
            
            // Main compression loop (64 rounds)
            for (uint i = 0; i < 64; i++) {
                uint temp1 = h + Sigma1(e) + Ch(e, f, g) + k[i] + w[i];
                uint temp2 = Sigma0(a) + Maj(a, b, c);
                
                h = g;
                g = f;
                f = e;
                e = d + temp1;
                d = c;
                c = b;
                b = a;
                a = temp1 + temp2;
            }
            
            // Add the compressed chunk to the current hash value
            h0 += a;
            h1 += b;
            h2 += c;
            h3 += d;
            h4 += e;
            h5 += f;
            h6 += g;
            h7 += h;
            
            // Store the final hash values (32-bit words)
            uint final_hash_words[8];
            final_hash_words[0] = h0;
            final_hash_words[1] = h1;
            final_hash_words[2] = h2;
            final_hash_words[3] = h3;
            final_hash_words[4] = h4;
            final_hash_words[5] = h5;
            final_hash_words[6] = h6;
            final_hash_words[7] = h7;
            
            // Convert final hash words to byte array (32 bytes)
            uchar hash_bytes[32];
            uint_to_bytes(final_hash_words, hash_bytes, 8); // 8 uints = 32 bytes
            
            // Convert hash bytes to hex string (64 characters + null terminator)
            char hex_hash[65];
            hex_hash[64] = '\0';
            
            for (int i = 0; i < 32; i++) {
                uchar byte = hash_bytes[i];
                char hex_chars[16] = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'};
                hex_hash[i*2] = hex_chars[(byte >> 4) & 0xF];
                hex_hash[i*2+1] = hex_chars[byte & 0xF];
            }
            
            // Check if hash starts with the desired prefix
            bool matches = true;
            for (uint i = 0; i < prefix_length && matches; i++) {
                if (hex_hash[i] != prefix[i]) {
                    matches = false;
                }
            }
            
            // If the hash matches the prefix, copy it and the nonce to the output buffer
            if (matches) {
                // Store the flag (1 for match)
                output_hashes[id*37] = 1; 
                
                // Store the nonce (4 bytes)
                output_hashes[id*37+1] = (nonce >> 24) & 0xFF;
                output_hashes[id*37+2] = (nonce >> 16) & 0xFF;
                output_hashes[id*37+3] = (nonce >> 8) & 0xFF;
                output_hashes[id*37+4] = nonce & 0xFF;
                
                // Store the hash bytes (32 bytes)
                for (int i = 0; i < 32; i++) {
                    output_hashes[id*37+5+i] = hash_bytes[i];
                }
            } else {
                // No match, clear the flag
                output_hashes[id*37] = 0;
            }
        }
        """
        
        # Build the program
        try:
            self.program = cl.Program(self.ctx, self.kernel_code).build()
        except cl.LogicError as e:
            print(f"OpenCL kernel compilation error: {e}")
            sys.exit(1) # Exit if kernel can't compile

    def calculate_hash(self, data, prefix="", num_work_items=1, nonce_offset=0):
        """
        Calculate SHA-256 hashes for the given data with different nonces,
        looking for a hash that starts with the specified prefix.
        
        Args:
            data (str or bytes): The input data to hash
            prefix (str): The prefix to match in the resulting hash (hex string)
            num_work_items (int): Number of parallel work items to use
            nonce_offset (int): Starting nonce value for the work group
            
        Returns:
            list: List of tuples (nonce, hash) for matches found
        """
        # Convert input to bytes if it's a string
        if isinstance(data, str):
            data_bytes = data.encode('utf-8')
        elif isinstance(data, bytes):
            data_bytes = data
        else:
            raise TypeError("Input data must be a string or bytes.")
        
        # Ensure input_length + 4 (for nonce) does not exceed 55 bytes
        # to fit in a single SHA-256 block with padding.
        if len(data_bytes) + 4 > 55:
            print(f"Warning: Input data length ({len(data_bytes)} bytes) + nonce (4 bytes) "
                  f"exceeds single SHA-256 block capacity (55 bytes before padding). "
                  f"The current kernel only supports single-block hashing. "
                  f"Results might be incorrect for longer inputs.")
            # For simplicity in this context, we will truncate or pad to fit,
            # but in a real scenario, you'd need a multi-block SHA-256 kernel.
            # Here, we'll just proceed, knowing it might be wrong.
            pass # Keep going for demonstration, but be aware of limitation

        # Create numpy arrays for input and output
        data_np = np.frombuffer(data_bytes, dtype=np.uint8)
        # Output buffer size: 1 byte flag + 4 byte nonce + 32 byte hash = 37 bytes per work item
        output_np = np.zeros(num_work_items * 37, dtype=np.uint8)
        
        # Create OpenCL buffers
        mf = cl.mem_flags
        data_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data_np)
        prefix_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, 
                               hostbuf=np.frombuffer(prefix.encode('ascii'), dtype=np.uint8))
        output_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, output_np.nbytes)
        
        # Execute the kernel
        self.program.calculate_sha256(
            self.queue, (num_work_items,), None,
            data_buf, np.uint32(len(data_bytes)),
            output_buf, prefix_buf, np.uint32(len(prefix)),
            np.uint32(nonce_offset) # Pass the explicit nonce_offset
        ).wait() # Wait for kernel execution to complete
        
        # Get the results back to host
        cl.enqueue_copy(self.queue, output_np, output_buf).wait()
        
        # Process the results
        results = []
        for i in range(num_work_items):
            if output_np[i*37] == 1:  # If a match was found (index changed to 37)
                # Extract the nonce
                nonce = (int(output_np[i*37+1]) << 24) | \
                        (int(output_np[i*37+2]) << 16) | \
                        (int(output_np[i*37+3]) << 8) | \
                        int(output_np[i*37+4])
                
                # Extract the hash bytes (index changed to 37, start at +5, take 32 bytes)
                hash_bytes = output_np[i*37+5:i*37+5+32].tobytes()
                
                # Convert hash bytes to hex string
                hash_hex = hash_bytes.hex()
                
                results.append((nonce, hash_hex))
        
        return results

    def verify_with_cpu(self, data, nonce):
        """
        Verify a hash using CPU-based computation for cross-checking.
        
        Args:
            data (str or bytes): The input data
            nonce (int): The nonce value
            
        Returns:
            str: The hexadecimal hash digest
        """
        # Convert data to bytes if necessary
        if isinstance(data, str):
            data_bytes = data.encode('utf-8')
        elif isinstance(data, bytes):
            data_bytes = data
        else:
            raise TypeError("Input must be string or bytes")
        
        # Append nonce bytes (big-endian)
        nonce_bytes = nonce.to_bytes(4, byteorder='big')
        message = data_bytes + nonce_bytes
        
        # Calculate hash using hashlib
        sha256 = hashlib.sha256()
        sha256.update(message)
        return sha256.hexdigest()

# Instantiate OpenCLSHA256 once globally (or as a member of MainWindow)
sha256_cl_instance = OpenCLSHA256()

def calculate_sha256_with_opencl(data, prefix="", work_items=1, nonce_offset=0):
    """
    Calculate SHA-256 hash of the given data using OpenCL acceleration.
    Search for a hash that starts with the given prefix.
    
    Args:
        data (str or bytes): The data to hash
        prefix (str): The prefix to search for in the resulting hash (hex string)
        work_items (int): Number of parallel work items to use
        nonce_offset (int): The starting nonce value for this batch of work_items.
        
    Returns:
        list: List of (nonce, hash) tuples for matches found
    """

    # Calculate hash with GPU acceleration
    results = sha256_cl_instance.calculate_hash(data, prefix, work_items, nonce_offset)
    
    # If we found any matches, verify them with CPU
    verified_results = []
    for nonce, gpu_hash_hex in results:
        cpu_hash = sha256_cl_instance.verify_with_cpu(data, nonce)
        if cpu_hash == gpu_hash_hex:
            verified_results.append((nonce, gpu_hash_hex))
            print(f"Match found!{data} Nonce: {nonce}, Hash: {gpu_hash_hex}") # Keep this for debugging
        else:
            print(f"WARNING: GPU hash verification failed for Nonce {nonce}. GPU: {gpu_hash_hex}, CPU: {cpu_hash}")
    
    return verified_results
         
        
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
                    ("= 0x55" if hex_value == 0x55 else " 0x55") + 
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
                    ("= 0x55" if hex_value == 0x55 else " 0x55") + 
                    f". -{COST_PER_GUESS} credits.", 2000)
            
            # Win condition: Both cameras are below threshold AND value is 0x55
            if cam0_below and cam1_below:
            
                if calculate_sha256_with_opencl("GeorgeW", prefix="0000", work_items=100000, nonce_offset=self.init_count): # measure comparisons
                    # Win scenario
                    game_state["credits"] += COST_PER_GUESS
                    game_state["wins"] = game_state.get("wins", 0) + 1
                    print("Success @ ", game_state["losses"], " Ready states!", "nonce:",str(self.init_count))
                    self.show_status_message(
                        f"Win! {camera_status} | {current_value} = 0x55. +{WIN_CREDITS} credits!", 2000)
                else:
                    game_state["losses"] = game_state.get("losses", 0) + 1
            else:

                # Any other scenario is a loss
                game_state["credits"] -= COST_PER_GUESS
                self.show_status_message(
                    f"Lost! {camera_status} | {current_value} " + 
                    ("= 0x55" if hex_value == 0x55 else " 0x55") + 
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
            self.init_count = (self.init_count + 100000) % 4294967294
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
