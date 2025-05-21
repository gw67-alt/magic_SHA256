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
data_list_from_file = [] # Renamed to avoid conflict with other 'data' variables
try:
    with open("x.txt", 'r', encoding='utf-8') as file:
        data_list_from_file = [line.strip() for line in file.readlines() if line.strip()]
    if not data_list_from_file:
        print("Warning: x.txt file is empty. Using dummy data.")
        data_list_from_file = ["55", "55", "AA", "55", "BB"]  # Dummy data if file is empty
except FileNotFoundError:
    print("Warning: x.txt file not found. Creating with dummy data.")
    with open("x.txt", 'w', encoding='utf-8') as file:
        file.write("55\n55\nAA\n55\nBB\n")
    data_list_from_file = ["55", "55", "AA", "55", "BB"]  # Dummy data for new file

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
HASHING_WORK_ITEMS_PER_FRAME = 4000000 # Number of nonces to test per frame/batch


class OpenCLSHA256:
    """
    A class to calculate SHA-256 hashes using OpenCL for GPU acceleration.
    """
    def __init__(self):
        print("Initializing OpenCLSHA256...")
        try:
            platforms = cl.get_platforms()
            if not platforms:
                raise RuntimeError("No OpenCL platforms found")
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
                        break 
                except cl.LogicError as e:
                    print(f"Could not create context for platform {platform.name}: {e}")
            
            if not self.ctx:
                raise RuntimeError("No suitable OpenCL device found on any platform.")

        except Exception as e:
            print(f"OpenCL initialization error: {e}")
            try:
                self.ctx = cl.create_some_context(interactive=False)
                self.queue = cl.CommandQueue(self.ctx)
                print(f"Fallback: Using default OpenCL context.")
            except Exception as fallback_e:
                print(f"FATAL: Could not create any OpenCL context: {fallback_e}")
                sys.exit(1)

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
        #define ROTR(x, n) ((x >> n) | (x << (32 - n)))
        #define Ch(x, y, z) ((x & y) ^ (~x & z))
        #define Maj(x, y, z) ((x & y) ^ (x & z) ^ (y & z))
        #define Sigma0(x) (ROTR(x, 2) ^ ROTR(x, 13) ^ ROTR(x, 22))
        #define Sigma1(x) (ROTR(x, 6) ^ ROTR(x, 11) ^ ROTR(x, 25))
        #define sigma0(x) (ROTR(x, 7) ^ ROTR(x, 18) ^ (x >> 3))
        #define sigma1(x) (ROTR(x, 17) ^ ROTR(x, 19) ^ (x >> 10))
        void bytes_to_uint(const uchar *input, uint *output, int length) {
            for (int i = 0; i < length / 4; i++) {
                output[i] = ((uint)input[i*4] << 24) | ((uint)input[i*4+1] << 16) | ((uint)input[i*4+2] << 8) | ((uint)input[i*4+3]);
            }
        }
        void uint_to_bytes(const uint *input, uchar *output, int length) {
            for (int i = 0; i < length; i++) {
                output[i*4] = (input[i] >> 24) & 0xFF; output[i*4+1] = (input[i] >> 16) & 0xFF;
                output[i*4+2] = (input[i] >> 8) & 0xFF; output[i*4+3] = input[i] & 0xFF;
            }
        }
        __kernel void calculate_sha256(
            __global const uchar* input_data, uint input_length, __global uchar* output_hashes,
            __global const char* prefix, uint prefix_length, uint nonce_offset
        ) {
            uint id = get_global_id(0); uint nonce = id + nonce_offset;
            uchar message[64];
            uint h0=0x6a09e667; uint h1=0xbb67ae85; uint h2=0x3c6ef372; uint h3=0xa54ff53a;
            uint h4=0x510e527f; uint h5=0x9b05688c; uint h6=0x1f83d9ab; uint h7=0x5be0cd19;
            for (uint i=0; i<input_length; i++) { message[i] = input_data[i]; }
            if (input_length + 4 > 55) { return; }
            message[input_length]  =(nonce >> 24)&0xFF; message[input_length+1]=(nonce >> 16)&0xFF;
            message[input_length+2]=(nonce >> 8 )&0xFF; message[input_length+3]=(nonce      )&0xFF;
            uint msg_length = input_length + 4;
            message[msg_length] = 0x80;
            for (uint i=msg_length+1; i<56; i++) { message[i] = 0; }
            ulong bit_length = (ulong)msg_length * 8;
            message[56]=(bit_length>>56)&0xFF; message[57]=(bit_length>>48)&0xFF; message[58]=(bit_length>>40)&0xFF; message[59]=(bit_length>>32)&0xFF;
            message[60]=(bit_length>>24)&0xFF; message[61]=(bit_length>>16)&0xFF; message[62]=(bit_length>>8 )&0xFF; message[63]=(bit_length    )&0xFF;
            uint w[64]; bytes_to_uint(message, w, 64);
            for (uint i=16; i<64; i++) { w[i] = sigma1(w[i-2]) + w[i-7] + sigma0(w[i-15]) + w[i-16]; }
            uint a=h0,b=h1,c=h2,d=h3,e=h4,f=h5,g=h6,h=h7;
            for (uint i=0; i<64; i++) {
                uint temp1 = h + Sigma1(e) + Ch(e,f,g) + k[i] + w[i];
                uint temp2 = Sigma0(a) + Maj(a,b,c);
                h=g; g=f; f=e; e=d+temp1; d=c; c=b; b=a; a=temp1+temp2;
            }
            h0+=a; h1+=b; h2+=c; h3+=d; h4+=e; h5+=f; h6+=g; h7+=h;
            uint final_hash_words[8];
            final_hash_words[0]=h0; final_hash_words[1]=h1; final_hash_words[2]=h2; final_hash_words[3]=h3;
            final_hash_words[4]=h4; final_hash_words[5]=h5; final_hash_words[6]=h6; final_hash_words[7]=h7;
            uchar hash_bytes[32]; uint_to_bytes(final_hash_words, hash_bytes, 8);
            char hex_hash[65]; hex_hash[64]='\\0';
            for(int i=0; i<32; i++){ uchar bt=hash_bytes[i]; char hx_cs[16]={'0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f'}; hex_hash[i*2]=hx_cs[(bt>>4)&0xF]; hex_hash[i*2+1]=hx_cs[bt&0xF]; }
            bool matches=true;
            for(uint i=0; i<prefix_length && matches; i++){ if(hex_hash[i]!=prefix[i]){matches=false;} }
            if(matches){
                output_hashes[id*37]=1;
                output_hashes[id*37+1]=(nonce>>24)&0xFF; output_hashes[id*37+2]=(nonce>>16)&0xFF;
                output_hashes[id*37+3]=(nonce>>8 )&0xFF; output_hashes[id*37+4]=(nonce    )&0xFF;
                for(int i=0; i<32; i++){ output_hashes[id*37+5+i]=hash_bytes[i]; }
            } else { output_hashes[id*37]=0; }
        }"""
        try:
            self.program = cl.Program(self.ctx, self.kernel_code).build()
            print("OpenCL kernel built successfully.")
        except cl.LogicError as e:
            print(f"OpenCL kernel compilation error: {e}")
            sys.exit(1)

    # This method calculate_hash is part of OpenCLSHA256 class
    def calculate_hash(self, data_input, prefix_str="", num_work_items=1, nonce_offset_val=0):
        if isinstance(data_input, str): data_bytes_val = data_input.encode('utf-8')
        elif isinstance(data_input, bytes): data_bytes_val = data_input
        else: raise TypeError("Input data must be a string or bytes.")

        if len(data_bytes_val) + 4 > 55:
             print(f"Warning: Input data+nonce length ({len(data_bytes_val)+4}) > 55 bytes. Kernel may not produce correct SHA256 for multi-block messages.")

        data_np_val = np.frombuffer(data_bytes_val, dtype=np.uint8)
        output_np_val = np.zeros(num_work_items * 37, dtype=np.uint8)
        mf = cl.mem_flags
        data_buf_val = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data_np_val)
        
        prefix_bytes_val = prefix_str.encode('ascii') if isinstance(prefix_str, str) else b""
        prefix_buf_val = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.frombuffer(prefix_bytes_val, dtype=np.uint8))
        output_buf_val = cl.Buffer(self.ctx, mf.WRITE_ONLY, output_np_val.nbytes)

        self.program.calculate_sha256(
            self.queue, (num_work_items,), None, data_buf_val, np.uint32(len(data_bytes_val)),
            output_buf_val, prefix_buf_val, np.uint32(len(prefix_bytes_val)), np.uint32(nonce_offset_val)
        ).wait()
        cl.enqueue_copy(self.queue, output_np_val, output_buf_val).wait()
        
        found_results = []
        for i in range(num_work_items):
            if output_np_val[i*37] == 1:
                nonce_from_gpu = int.from_bytes(output_np_val[i*37+1:i*37+5], 'big')
                hash_bytes_from_gpu = output_np_val[i*37+5:i*37+37].tobytes()
                found_results.append((nonce_from_gpu, hash_bytes_from_gpu.hex()))
        return found_results

    def verify_with_cpu(self, data_input, nonce_val):
        if isinstance(data_input, str): data_bytes_val = data_input.encode('utf-8')
        elif isinstance(data_input, bytes): data_bytes_val = data_input
        else: raise TypeError("Input must be string or bytes")
        nonce_bytes_val = nonce_val.to_bytes(4, byteorder='big')
        message_val = data_bytes_val + nonce_bytes_val
        return hashlib.sha256(message_val).hexdigest()

# --- Corrected import for HashingThread ---
from queue import Queue, Empty # Import Empty here

# --- OpenCL Hashing Thread ---
class HashingThread(QThread):
    result_ready = pyqtSignal(list, object)  # results, user_data

    def __init__(self):
        super().__init__()
        self.queue = Queue()
        self.running = True
        self.start() # QThread.start() calls run() in a new thread

    def run(self):
        while self.running:
            try:
                task = self.queue.get(timeout=0.5) # This can raise queue.Empty
                if task is None:  # Sentinel value to exit
                    self.running = False # Ensure loop terminates
                    break
                
                data_task, prefix_task, work_items_task, nonce_offset_task, user_data_task = task
                
                # Perform the calculation using the global instance and function
                results_task = calculate_sha256_with_opencl(data_task, prefix_task, work_items_task, nonce_offset_task)
                
                self.result_ready.emit(results_task, user_data_task)
                self.queue.task_done()

            except Empty: # Corrected: Catch queue.Empty specifically
                # Timeout on empty queue - just continue the loop to check self.running
                continue
            except Exception as e: # Catch other exceptions
                print(f"Error in hashing thread: {e}")
                import traceback
                traceback.print_exc()
                if 'task' in locals(): # Check if task was assigned before error
                    try:
                        self.queue.task_done() # Mark task done even on error to prevent join issues
                    except ValueError: # If task already marked done or queue issue
                        pass
        print("HashingThread finished run().")
    
    def enqueue_task(self, data_val, prefix_val="", work_items_val=1, nonce_offset_val=0, user_data_val=None):
        self.queue.put((data_val, prefix_val, work_items_val, nonce_offset_val, user_data_val))
    
    def stop(self):
        print("Stopping HashingThread...")
        self.running = False
        self.queue.put(None)  # Add sentinel to ensure thread exits run loop
        if self.isRunning():
            if not self.wait(2000): # Wait for 2 seconds
                print("HashingThread did not finish in time. Forcing termination.")
                self.terminate() # Force terminate if wait fails (use with caution)
                self.wait() # Wait again after terminate
        print("HashingThread stopped.")


# Instantiate OpenCLSHA256 once globally
sha256_cl_instance = OpenCLSHA256()

# Global function that uses the global OpenCL instance
def calculate_sha256_with_opencl(data_param, prefix_param="", work_items_param=1, nonce_offset_param=0):
    """
    Calculate SHA-256 hash of the given data using OpenCL acceleration.
    Search for a hash that starts with the given prefix.
    """
    # Calculate hash with GPU acceleration using the global instance
    results_list = sha256_cl_instance.calculate_hash(data_param, prefix_param, work_items_param, nonce_offset_param)
    
    verified_results_list = []
    for nonce_val, gpu_hash_hex_val in results_list:
        cpu_hash_val = sha256_cl_instance.verify_with_cpu(data_param, nonce_val)
        if cpu_hash_val == gpu_hash_hex_val:
            verified_results_list.append((nonce_val, gpu_hash_hex_val))
            # print(f"Match found! Data: {data_param} Nonce: {nonce_val}, Hash: {gpu_hash_hex_val}") # Verbose
        else:
            print(f"WARNING: GPU hash verification failed for Nonce {nonce_val}. GPU: {gpu_hash_hex_val}, CPU: {cpu_hash_val}")
    
    return verified_results_list
        

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


class OpenCVThread(QThread):
    frame_ready = pyqtSignal(QImage, int)
    matches_count_ready = pyqtSignal(int, int)
    status_message = pyqtSignal(str, int)
    def __init__(self, app_state_ref, camera_id):
        super().__init__()
        self.running = False; self.app_state = app_state_ref; self.camera_id = camera_id
        self.reference_frame = None; self.reference_kp = None; self.reference_des = None
        self.orb = None; self.bf_matcher = None; self._capture_next_frame_as_reference = False
        self.app_state.capture_reference_requested.connect(lambda cid: self.prepare_for_reference_capture() if cid == self.camera_id else None)
        self.app_state.reset_requested.connect(lambda cid: self.reset_reference() if cid == self.camera_id else None)
    def initialize_features(self):
        self.orb = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8)
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    def prepare_for_reference_capture(self): self._capture_next_frame_as_reference = True
    def reset_reference(self):
        self.reference_frame = None; self.reference_kp = None; self.reference_des = None
        self._capture_next_frame_as_reference = False
        self.app_state.set_camera_state(self.camera_id, AppState.STATE_WAITING_FOR_REFERENCE)
    def preprocess_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        return clahe.apply(gray)
    def run(self):
        self.running = True; self.initialize_features()
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened(): self.status_message.emit(f"Error: Cannot open camera {self.camera_id}.", self.camera_id); self.running = False; return
        while self.running:
            ret, frame = cap.read()
            if not ret: self.status_message.emit(f"Error: Can't receive frame from camera {self.camera_id}.", self.camera_id); time.sleep(0.01); continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape; bytes_per_line = ch * w
            qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.frame_ready.emit(qt_image.copy(), self.camera_id)
            num_good_matches_for_signal = 0
            if self._capture_next_frame_as_reference:
                self.reference_frame = frame.copy(); processed_ref = self.preprocess_frame(self.reference_frame)
                self.reference_kp, self.reference_des = self.orb.detectAndCompute(processed_ref, None)
                if self.reference_des is None or len(self.reference_kp) < MIN_MATCH_COUNT:
                    msg = f"Ref. Capture Failed (Cam {self.camera_id}): Not enough features ({len(self.reference_kp) if self.reference_kp is not None else 0}). Try again."
                    self.status_message.emit(msg, self.camera_id)
                    self.reference_frame = None; self.reference_kp = None; self.reference_des = None
                    self.app_state.set_camera_state(self.camera_id, AppState.STATE_WAITING_FOR_REFERENCE)
                else:
                    self.status_message.emit(f"Reference Captured (Cam {self.camera_id}): {len(self.reference_kp)} keypoints. Tracking...", self.camera_id)
                    self.app_state.set_camera_state(self.camera_id, AppState.STATE_TRACKING)
                self._capture_next_frame_as_reference = False; self.matches_count_ready.emit(0, self.camera_id)
            elif self.app_state.get_camera_state(self.camera_id) == AppState.STATE_TRACKING and self.reference_des is not None: # Ensure ref_des is not None
                processed_frame = self.preprocess_frame(frame)
                current_kp, current_des = self.orb.detectAndCompute(processed_frame, None)
                actual_good_matches_count = 0
                if current_des is not None and len(current_des) > 0 and self.reference_des is not None and len(self.reference_kp) > 0: # Ensure ref_kp also
                    try:
                        all_matches = self.bf_matcher.knnMatch(self.reference_des, current_des, k=2)
                        good_matches = []
                        for m_arr in all_matches: # Lowe's ratio test
                            if len(m_arr) == 2: 
                                m, n = m_arr
                                if m.distance < LOWE_RATIO_TEST * n.distance: good_matches.append(m)
                        actual_good_matches_count = len(good_matches)
                        if actual_good_matches_count < MIN_MATCH_COUNT:
                            self.status_message.emit(f"Low matches (Cam {self.camera_id}): {actual_good_matches_count}/{MIN_MATCH_COUNT}", self.camera_id)
                        if actual_good_matches_count >= MIN_MATCH_COUNT:
                            src_pts = np.float32([self.reference_kp[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
                            dst_pts = np.float32([current_kp[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
                            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10.0)
                            num_good_matches_for_signal = actual_good_matches_count if H is not None else actual_good_matches_count // 2
                            if H is None: self.status_message.emit(f"Homography failed (Cam {self.camera_id})", self.camera_id)
                        else: num_good_matches_for_signal = actual_good_matches_count
                    except Exception as e: self.status_message.emit(f"Match error (Cam {self.camera_id}): {str(e)}", self.camera_id); num_good_matches_for_signal = 0
                else: 
                    # self.status_message.emit(f"No/few features in current or ref frame (Cam {self.camera_id})", self.camera_id) # Can be spammy
                    num_good_matches_for_signal = 0
                self.matches_count_ready.emit(num_good_matches_for_signal, self.camera_id)
            else: self.matches_count_ready.emit(0, self.camera_id)
        cap.release(); self.status_message.emit(f"Camera {self.camera_id} released.", self.camera_id)
        print(f"OpenCVThread for camera {self.camera_id} finished run().")
    def stop(self): 
        print(f"Stopping OpenCVThread for camera {self.camera_id}...")
        self.running = False
        if self.isRunning():
             if not self.wait(2000): # Wait up to 2 seconds
                print(f"OpenCVThread {self.camera_id} did not finish in time. Forcing termination.")
                self.terminate()
                self.wait()
        print(f"OpenCVThread for camera {self.camera_id} stopped.")


class CameraTracker:
    def __init__(self, camera_id, initial_threshold=MATCH_THRESHOLD_FOR_GUESS):
        self.camera_id = camera_id; self.current_threshold = initial_threshold
        self.threshold_history = deque(maxlen=30); self.raw_match_history = deque(maxlen=MAX_CHART_POINTS)
        self.avg_match_history = deque(maxlen=MAX_CHART_POINTS); self.time_points = deque(maxlen=MAX_CHART_POINTS)
        self.current_time_step = 0; self.guess_trigger_sample_counter = 0; self.is_below_threshold = False
    def update_match_data(self, raw_match_count):
        actual_plot_count = raw_match_count if raw_match_count >= 0 else 0
        self.raw_match_history.append(actual_plot_count); self.time_points.append(self.current_time_step); self.current_time_step += 1
        self.threshold_history.append(actual_plot_count)
        current_avg = np.mean(list(self.raw_match_history)[-MOVING_AVG_WINDOW:]) if len(self.raw_match_history) > 0 and any(list(self.raw_match_history)[-MOVING_AVG_WINDOW:]) else 0.0
        self.avg_match_history.append(current_avg)
        if len(self.threshold_history) > 5: self.current_threshold = np.mean(self.threshold_history) if self.threshold_history else MATCH_THRESHOLD_FOR_GUESS
        self.is_below_threshold = actual_plot_count < self.current_threshold
        self.guess_trigger_sample_counter += 1
        return {'time_points': list(self.time_points), 'raw_history': list(self.raw_match_history), 
                'avg_history': list(self.avg_match_history), 'current_threshold': self.current_threshold,
                'is_below_threshold': self.is_below_threshold, 'should_guess': self.guess_trigger_sample_counter >= GUESS_TRIGGER_COUNT}
    def reset_guess_counter(self): self.guess_trigger_sample_counter = 0
    def clear_data(self):
        self.raw_match_history.clear(); self.avg_match_history.clear(); self.time_points.clear()
        self.threshold_history.clear(); self.current_time_step = 0
        self.current_threshold = MATCH_THRESHOLD_FOR_GUESS; self.guess_trigger_sample_counter = 0; self.is_below_threshold = False


class MainWindow(QMainWindow):
    def __init__(self, app_state_ref):
        super().__init__()
        self.app_state = app_state_ref
        self.setWindowTitle("Dual Camera Homography Tracker & SHA256 Guesser")
        self.setGeometry(100, 100, 1400, 850) # Increased height for new label

        self.camera_trackers = {CAMERA_0_ID: CameraTracker(CAMERA_0_ID), CAMERA_1_ID: CameraTracker(CAMERA_1_ID)}
        self.last_alignment_alert_time = 0
        
        self.hashing_thread = HashingThread() # Create the single hashing thread
        self.hashing_thread.result_ready.connect(self.on_hash_result)
        
        self.hashing_in_progress = False
        self.last_hash_attempt_time = 0 # Renamed for clarity
        self.hash_attempt_cooldown = 100  # ms, min time between starting new hash attempts

        central_widget = QWidget(); self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        cameras_layout = QGridLayout()
        self.video_labels = {}
        for cam_id in [CAMERA_0_ID, CAMERA_1_ID]:
            label = QLabel(f"Cam {cam_id}..."); label.setFixedSize(640, 360)
            label.setStyleSheet("border: 1px solid black; background-color: #333;"); label.setAlignment(Qt.AlignCenter)
            self.video_labels[cam_id] = label; cameras_layout.addWidget(label, cam_id // 2, cam_id % 2)
        main_layout.addLayout(cameras_layout)
        charts_layout = QGridLayout()
        self.match_charts = {}; self.raw_match_lines = {}; self.avg_match_lines = {}; self.threshold_lines = {}
        for cam_id in [CAMERA_0_ID, CAMERA_1_ID]:
            chart = pg.PlotWidget(); chart.setBackground('w'); chart.setTitle(f"Cam {cam_id} Matches", color="k", size="12pt")
            chart.setLabel('left', 'Matches', color='k'); chart.setLabel('bottom', 'Frames', color='k')
            chart.showGrid(x=True, y=True); chart.addLegend()
            tracker = self.camera_trackers[cam_id]
            threshold_line = pg.InfiniteLine(pos=tracker.current_threshold, angle=0, pen=pg.mkPen('g', width=2, style=Qt.DashLine), label=f'Thresh ({tracker.current_threshold:.2f})')
            chart.addItem(threshold_line)
            self.raw_match_lines[cam_id] = chart.plot(pen=pg.mkPen('b', width=2), name="Raw")
            self.avg_match_lines[cam_id] = chart.plot(pen=pg.mkPen('r', width=2, style=Qt.DashLine), name=f"Avg ({MOVING_AVG_WINDOW})")
            self.match_charts[cam_id] = chart; self.threshold_lines[cam_id] = threshold_line
            charts_layout.addWidget(chart, cam_id // 2, cam_id % 2)
        main_layout.addLayout(charts_layout)
        bottom_layout = QVBoxLayout(); stats_layout = QHBoxLayout()
        self.credits_label = QLabel(f"Credits: {game_state['credits']}"); self.credits_label.setFont(QFont('Arial', 14)); self.credits_label.setStyleSheet("color: green; font-weight: bold;")
        self.wins_label = QLabel(f"Successes: {game_state.get('wins',0)}"); self.wins_label.setFont(QFont('Arial', 14)); self.wins_label.setStyleSheet("color: blue;")
        self.losses_label = QLabel(f"Readies: {game_state.get('losses',0)}"); self.losses_label.setFont(QFont('Arial', 14)); self.losses_label.setStyleSheet("color: red;")
        self.hash_status_label = QLabel("Mining: Idle"); self.hash_status_label.setFont(QFont('Arial', 14)); self.hash_status_label.setStyleSheet("color: purple; font-weight: bold;")
        
        stats_layout.addWidget(self.credits_label); stats_layout.addWidget(self.wins_label); 
        stats_layout.addWidget(self.losses_label); stats_layout.addWidget(self.hash_status_label)
        bottom_layout.addLayout(stats_layout)
        instructions_label = QLabel("N: Capture/Reset Reference | Q: Quit"); instructions_label.setAlignment(Qt.AlignCenter); instructions_label.setFont(QFont('Arial', 10))
        bottom_layout.addWidget(instructions_label); main_layout.addLayout(bottom_layout)
        self.setStatusBar(QStatusBar())
        self.opencv_threads = {}
        for cam_id in [CAMERA_0_ID, CAMERA_1_ID]:
            thread = OpenCVThread(self.app_state, cam_id)
            thread.frame_ready.connect(self.update_video_frame); thread.matches_count_ready.connect(self.update_matches_chart)
            thread.status_message.connect(self.show_status_message); self.opencv_threads[cam_id] = thread
        self.app_state.state_changed.connect(self.on_state_changed_gui); self.app_state.game_state_updated.connect(self.update_game_stats)
        self.init_count = 0 # Nonce offset for hashing
        for thread in self.opencv_threads.values(): thread.start()
        for cam_id in [CAMERA_0_ID, CAMERA_1_ID]: self.on_state_changed_gui(cam_id, self.app_state.get_camera_state(cam_id))

    def update_video_frame(self, q_image, camera_id):
        if camera_id in self.video_labels: self.video_labels[camera_id].setPixmap(QPixmap.fromImage(q_image))

    def check_visual_alignment(self):
        cam0_tracker = self.camera_trackers[CAMERA_0_ID]; cam1_tracker = self.camera_trackers[CAMERA_1_ID]
        current_time_sec = time.time() # Changed to seconds for consistency
        if (current_time_sec - self.last_alignment_alert_time) < 5: return # 5 sec cooldown
        if not cam0_tracker.avg_match_history or not cam1_tracker.avg_match_history: return
        cam0_avg = cam0_tracker.avg_match_history[-1]; cam1_avg = cam1_tracker.avg_match_history[-1]
        cam0_vb = self.match_charts[CAMERA_0_ID].getViewBox(); cam1_vb = self.match_charts[CAMERA_1_ID].getViewBox()
        if not (cam0_vb and cam1_vb): return
        cam0_yr = cam0_vb.viewRange()[1]; cam1_yr = cam1_vb.viewRange()[1]
        if not (cam0_yr and cam1_yr): return
        cam0_ymin, cam0_ymax = cam0_yr; cam1_ymin, cam1_ymax = cam1_yr
        if cam0_ymax == cam0_ymin or cam1_ymax == cam1_ymin: return
        cam0_norm = (cam0_avg - cam0_ymin) / (cam0_ymax - cam0_ymin)
        cam1_norm = (cam1_avg - cam1_ymin) / (cam1_ymax - cam1_ymin)
        if abs(cam0_norm - cam1_norm) <= 0.03: 
            msg = f"ALIGNMENT DETECTED! C0_avg:{cam0_avg:.1f}({cam0_norm:.2f}), C1_avg:{cam1_avg:.1f}({cam1_norm:.2f})"
            # print(msg) # Can be spammy
            self.show_status_message(msg, 3000); self.last_alignment_alert_time = current_time_sec


    def on_hash_result(self, results_list, user_data_dict):
        camera_status_str = user_data_dict.get('camera_status', 'N/A')
        # current_value_str = user_data_dict.get('current_value', 'N/A') # Not used directly in msg
        original_nonce_offset = user_data_dict.get('nonce_offset', -1)

        self.hashing_in_progress = False # Hashing for this batch is done
        self.hash_status_label.setText("Mining: Idle")

        if results_list:
            nonce_found, hash_val_found = results_list[0]
            game_state["credits"] += WIN_CREDITS
            game_state["wins"] += 1
            success_msg_short = f"SUCCESS! Nonce {nonce_found} -> {hash_val_found[:12]}... +{WIN_CREDITS}cr"
            print(f"{success_msg_short} (Batch {original_nonce_offset}, Status: {camera_status_str})")
            self.show_status_message(success_msg_short, 4000)
        else:
            game_state["credits"] -= COST_PER_GUESS
            game_state["losses"] += 1 # Count as a "ready" state that didn't yield hash
            fail_msg_short = f"Mining attempt failed (batch {original_nonce_offset}). -{COST_PER_GUESS}cr"
            print(f"{fail_msg_short} (Status: {camera_status_str})")
            self.show_status_message(fail_msg_short, 3000)
        
        # Update nonce for the *next* attempt, regardless of this one's outcome.
        # This was previously done in process_dual_camera_guess if conditions were not met.
        # It should only be advanced after a hash attempt (successful or not) is completed.
        # The nonce_offset in user_data_dict is the one *used* for this completed task.
        self.init_count = (original_nonce_offset + HASHING_WORK_ITEMS_PER_FRAME) % 4294967294 
        self.app_state.update_game_state(game_state)


    def update_matches_chart(self, raw_match_count, camera_id):
        if camera_id not in self.camera_trackers: return
        tracker = self.camera_trackers[camera_id]; results_data = tracker.update_match_data(raw_match_count)
        self.raw_match_lines[camera_id].setData(results_data['time_points'], results_data['raw_history'])
        self.avg_match_lines[camera_id].setData(results_data['time_points'], results_data['avg_history'])
        self.threshold_lines[camera_id].setValue(results_data['current_threshold'])
        if self.threshold_lines[camera_id].label: self.threshold_lines[camera_id].label.setText(f'Thresh ({results_data["current_threshold"]:.2f})')
        
        cam0_state = self.app_state.get_camera_state(CAMERA_0_ID)
        cam1_state = self.app_state.get_camera_state(CAMERA_1_ID)
        both_tracking = (cam0_state == AppState.STATE_TRACKING and cam1_state == AppState.STATE_TRACKING)
        
        if both_tracking and len(results_data['avg_history']) > 0: self.check_visual_alignment()
        
        # Trigger guess from one camera to avoid double processing
        if both_tracking and results_data['should_guess'] and camera_id == CAMERA_0_ID: 
            self.camera_trackers[CAMERA_0_ID].reset_guess_counter() 
            self.camera_trackers[CAMERA_1_ID].reset_guess_counter() # Sync counters
            self.process_dual_camera_guess()

    def process_dual_camera_guess(self):
        current_time_ms = time.time() * 1000
        if self.hashing_in_progress or (current_time_ms - self.last_hash_attempt_time < self.hash_attempt_cooldown):
            return
        
        if game_state["credits"] <= 0: self.show_status_message("No credits! Game over.", 3000); return
        
        # data_list_from_file is used here by name 'data', ensure it's what's intended
        # If 'data' refers to the global data_list_from_file:
        if not data_list_from_file: # Check the renamed list
            self.show_status_message("Error: No data in x.txt for guessing!", 3000); return
            
        cam0_below = self.camera_trackers[CAMERA_0_ID].is_below_threshold
        cam1_below = self.camera_trackers[CAMERA_1_ID].is_below_threshold
        camera_status_str = f"C0:{'Below' if cam0_below else 'Above'},C1:{'Below' if cam1_below else 'Above'}"
        if cam0_below:
            # Win scenario
            game_state["credits"] += COST_PER_GUESS
            self.show_status_message(
                f"Win! {camera_status_str}. +{WIN_CREDITS} credits!", 2000)
        else:
            # Any other scenario is a loss
            game_state["credits"] -= COST_PER_GUESS
            self.show_status_message(
                f"Lost! {camera_status_str} " + 
                f". -{COST_PER_GUESS} credits.", 2000)
                
        # Win condition: Both cameras are below threshold AND value is 0x55
        if cam1_below:
            # Win scenario
            game_state["credits"] += COST_PER_GUESS
            self.show_status_message(
                f"Win! {camera_status_str}. +{WIN_CREDITS} credits!", 2000)
        else:
            # Any other scenario is a loss
            game_state["credits"] -= COST_PER_GUESS
            self.show_status_message(
                f"Lost! {camera_status_str} " + 
                f". -{COST_PER_GUESS} credits.", 2000)
        # Win condition for hashing: Both cameras are below threshold
        if cam0_below and cam1_below:
            self.hashing_in_progress = True
            self.hash_status_label.setText(f"Mining: Active (Nonce {self.init_count})")
            self.last_hash_attempt_time = current_time_ms
            game_state["credits"] += COST_PER_GUESS

            user_data_for_hash = {
                'camera_status': camera_status_str,
                'current_value': self.init_count, # Or other relevant data from x.txt if needed
                'nonce_offset': self.init_count # Pass the nonce offset being used
            }
            
            self.hashing_thread.enqueue_task(
                "GeorgeW",                             # This maps to data_val positionally
                prefix_val="00000000",                       # Corrected keyword
                work_items_val=HASHING_WORK_ITEMS_PER_FRAME, # Corrected keyword
                nonce_offset_val=self.init_count,      # Corrected keyword
                user_data_val=user_data_for_hash       # Corrected keyword
            )
            self.show_status_message(f"Mining... {camera_status_str} | Batch @{self.init_count}", 2000)
        else: # Camera conditions not met for hashing
            game_state["credits"] -= COST_PER_GUESS
            self.show_status_message(f"Lost. {camera_status_str} | Conditions not met. -{COST_PER_GUESS}cr.", 2000)
            # Advance nonce here if a guess was "attempted" but conditions failed,
            # so it doesn't get stuck on the same nonce if cameras never align.
            self.init_count = (self.init_count + HASHING_WORK_ITEMS_PER_FRAME) % 4294967294
            self.app_state.update_game_state(game_state)

        # This general log was inside the try block before, moved out for clarity
        # print(f"Guess Attempt Log: {camera_status_str} | Nonce for next try (if applicable): {self.init_count}, Credits: {game_state['credits']}")

    def update_game_stats(self, state_dict):
        self.credits_label.setText(f"Credits: {state_dict['credits']}")
        self.wins_label.setText(f"Successes: {state_dict.get('wins', 0)}") # Changed from "Success states"
        self.losses_label.setText(f"Readies: {state_dict.get('losses', 0)}") # Changed from "Ready states"
        # self.hash_status_label is updated in process_dual_camera_guess and on_hash_result

    def show_status_message(self, message, timeout=0, camera_id=None):
        full_message = f"Cam {camera_id}: {message}" if camera_id is not None else message
        status_bar = self.statusBar()
        if status_bar: status_bar.showMessage(full_message, timeout)
        # else: print(f"Status (no bar): {full_message}") # Avoid console spam

    def on_state_changed_gui(self, camera_id, state):
        tracker = self.camera_trackers.get(camera_id)
        if tracker:
            if state == AppState.STATE_WAITING_FOR_REFERENCE:
                self.show_status_message(f"STATE (Cam {camera_id}): Waiting for Reference. Aim & press 'N'.", 0, camera_id); tracker.clear_data()
            elif state == AppState.STATE_TRACKING:
                self.show_status_message(f"STATE (Cam {camera_id}): Tracking active. Press 'N' to reset.", 0, camera_id)

    def keyPressEvent(self, event):
        key = event.key()
        if key == KEY_TO_QUIT_QT: self.close()
        elif key == KEY_TO_CYCLE_QT:
            for cam_id in [CAMERA_0_ID, CAMERA_1_ID]:
                current_state = self.app_state.get_camera_state(cam_id)
                if current_state == AppState.STATE_WAITING_FOR_REFERENCE: self.app_state.request_capture_reference(cam_id)
                elif current_state == AppState.STATE_TRACKING: self.app_state.request_reset(cam_id)
        else: super().keyPressEvent(event)

    def closeEvent(self, event):
        print("Initiating application shutdown...")
        self.show_status_message("Closing application...", 2000)
        
        # Stop OpenCV threads first
        for cam_id, thread in self.opencv_threads.items():
            if thread.isRunning():
                thread.stop() # stop() method now includes wait and terminate logic
        
        # Stop the hashing thread
        if self.hashing_thread.isRunning():
            self.hashing_thread.stop() # stop() method now includes wait and terminate logic
        
        print("All threads requested to stop. Exiting.")
        event.accept()
        
if __name__ == '__main__':
    app = QApplication.instance()
    if not app: app = QApplication(sys.argv)
    pg.setConfigOptions(antialias=True) # Antialiasing for pyqtgraph
    main_window = MainWindow(app_state)
    main_window.show()
    if not hasattr(sys, 'flags') or not sys.flags.interactive: # Ensure app.exec_() runs
         sys.exit(app.exec_())