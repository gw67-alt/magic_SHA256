import sys
import time
import cv2
import numpy as np
from collections import deque # Not strictly used in this version but kept from original
import hashlib

# --- Configuration Constants ---
PREFIX = "0"
RAW_VALUE_FOR_GUESS = 10
STARTING_CREDITS = 10000
WIN_CREDITS = 1

MIN_MATCH_COUNT = 5
LOWE_RATIO_TEST = 0.10

# --- Performance Tuning Constants ---
ORB_N_FEATURES = 1500
PROCESSING_WIDTH = 480
PROFILE_TIMES = False # Set to True to print processing time diagnostics

# --- Game State ---
game_state = {
    "credits": STARTING_CREDITS,
    "guess_nonce": 0,
    "ready_count": 0,
    "success_count": 0
}

def calculate_sha256_with_library(data_str):
    """Calculates SHA-256 on CPU."""
    sha256_hash = hashlib.sha256()
    sha256_hash.update(data_str.encode('utf-8'))
    return sha256_hash.hexdigest()

# --- State Management Object ---
class AppState:
    STATE_WAITING_FOR_REFERENCE = 0
    STATE_TRACKING = 1

    def __init__(self):
        self._current_state = self.STATE_WAITING_FOR_REFERENCE

    @property
    def current_state(self):
        return self._current_state

    @current_state.setter
    def current_state(self, value):
        if self._current_state != value:
            old_state = self._current_state
            self._current_state = value
            print(f"STATE CHANGED: From {old_state} to {self._current_state}")
            if self._current_state == self.STATE_WAITING_FOR_REFERENCE:
                game_state["guess_nonce"] = 0
                # Note: Resetting credits here might be too aggressive if a reset to waiting
                # is triggered by something other than a deliberate game reset.
                # The current reset_reference in tracker handles credit reset.

app_state = AppState()

# --- OpenCV Feature Tracking Logic ---
class OpenCVTracker:
    def __init__(self, app_state_ref):
        self.app_state = app_state_ref
        self.orb = None
        self.bf_matcher = None
        self.reference_frame_gray_scaled = None
        self.reference_kp_scaled = None
        self.reference_des_scaled = None
        self._capture_next_frame_as_reference = False
        self.last_scale_ratio = 1.0

    def initialize_features(self):
        self.orb = cv2.ORB_create(nfeatures=ORB_N_FEATURES)
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        print(f"OpenCV features initialized (ORB nfeatures={ORB_N_FEATURES}).")

    def prepare_for_reference_capture(self):
        self._capture_next_frame_as_reference = True
        # print("Prepared to capture next frame as reference.") # Less verbose for auto-capture

    def reset_reference_and_game(self): # Renamed for clarity as it resets game too
        self.reference_frame_gray_scaled = None
        self.reference_kp_scaled = None
        self.reference_des_scaled = None
        self._capture_next_frame_as_reference = True # Prepare to capture a new one
        self.app_state.current_state = AppState.STATE_WAITING_FOR_REFERENCE
        print("Reference and game reset. Attempting new reference capture.")
        game_state["credits"] = STARTING_CREDITS
        # game_state["ready_count"] = 0 # Optional: reset these detailed stats
        # game_state["success_count"] = 0
        print(f"Game credits reset to {STARTING_CREDITS}.")


    def _get_scaled_frame_and_ratio(self, frame):
        original_height, original_width = frame.shape[:2]
        scale_ratio = 1.0

        if PROCESSING_WIDTH > 0 and original_width > PROCESSING_WIDTH:
            scale_ratio = PROCESSING_WIDTH / original_width
            processing_height = int(original_height * scale_ratio)
            scaled_frame = cv2.resize(frame, (PROCESSING_WIDTH, processing_height), interpolation=cv2.INTER_AREA)
        else:
            scaled_frame = frame.copy() 
        
        return scaled_frame, scale_ratio

    def process_frame(self, frame):
        t_start = time.perf_counter()
        
        # output_frame is no longer needed for display, but polylines was drawing on it.
        # For headless, we don't need to maintain a display copy.
        # We still need to know if homography was successful for game logic implicitly.

        scaled_frame_for_processing, current_scale_ratio = self._get_scaled_frame_and_ratio(frame)
        frame_gray_scaled = cv2.cvtColor(scaled_frame_for_processing, cv2.COLOR_BGR2GRAY)
        
        t_preprocess = time.perf_counter()
        matches_count = 0
        status_message = ""

        if self._capture_next_frame_as_reference:
            self.reference_frame_gray_scaled = frame_gray_scaled.copy()
            self.reference_kp_scaled, self.reference_des_scaled = self.orb.detectAndCompute(self.reference_frame_gray_scaled, None)
            self.last_scale_ratio = current_scale_ratio
            self._capture_next_frame_as_reference = False # Only try once per prepare call

            if self.reference_des_scaled is None or len(self.reference_kp_scaled) < MIN_MATCH_COUNT:
                status_message = f"Ref. Capture Failed: Not enough features ({len(self.reference_kp_scaled) if self.reference_kp_scaled is not None else 0}). Retrying..."
                self.reference_frame_gray_scaled = None # Clear bad reference
                self.reference_kp_scaled = None
                self.reference_des_scaled = None
                # Stay in WAITING_FOR_REFERENCE, and call prepare_for_reference_capture() again from main if needed.
                # For auto-capture, main loop will just keep calling process_frame, and this block will re-trigger.
                self._capture_next_frame_as_reference = True # Re-prime for next frame
                # self.app_state.current_state remains AppState.STATE_WAITING_FOR_REFERENCE
            else:
                status_message = f"Reference Captured ({len(self.reference_kp_scaled)} keypoints at {int(current_scale_ratio*100)}% scale). Tracking..."
                self.app_state.current_state = AppState.STATE_TRACKING
            matches_count = 0
        
        elif self.app_state.current_state == AppState.STATE_TRACKING and \
             self.reference_frame_gray_scaled is not None and self.reference_des_scaled is not None:
            
            current_kp_scaled, current_des_scaled = self.orb.detectAndCompute(frame_gray_scaled, None)
            t_detect = time.perf_counter()

            if current_des_scaled is not None and len(current_des_scaled) > 0:
                all_matches = self.bf_matcher.knnMatch(self.reference_des_scaled, current_des_scaled, k=2)
                good_matches = []
                for m_arr in all_matches:
                    if len(m_arr) == 2:
                        m, n = m_arr
                        if m.distance < LOWE_RATIO_TEST * n.distance:
                            good_matches.append(m)
                
                matches_count = len(good_matches)
                t_match = time.perf_counter()

                if matches_count >= MIN_MATCH_COUNT:
                    src_pts_scaled = np.float32([self.reference_kp_scaled[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    dst_pts_scaled = np.float32([current_kp_scaled[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    H_scaled, mask = cv2.findHomography(src_pts_scaled, dst_pts_scaled, cv2.RANSAC, 5.0)
                    t_homography = time.perf_counter()
                    
                    if H_scaled is not None:
                        # Homography successful, drawing part is removed
                        status_message = f"Tracking: {matches_count} good matches."
                    else:
                        status_message = "Homography failed. Tracking lost."
                        matches_count = -1 
                else:
                    status_message = f"Tracking: {matches_count} matches (Low: <{MIN_MATCH_COUNT})."
                    t_homography = t_match 
            else:
                status_message = "No features in current frame to match."
                matches_count = 0
                t_detect = t_preprocess 
                t_match = t_detect
                t_homography = t_match
        
        elif self.app_state.current_state == AppState.STATE_WAITING_FOR_REFERENCE:
            status_message = "STATE: Waiting for Initial Reference..." # Will keep trying to capture
            # Ensure capture is primed if we are in this state without a reference.
            if not self._capture_next_frame_as_reference:
                 self.prepare_for_reference_capture()
            matches_count = 0
            t_detect = t_preprocess
            t_match = t_detect
            t_homography = t_match
        
        t_end = time.perf_counter()

        if PROFILE_TIMES:
            profile_msg = f"Frame Proc Time: {t_end - t_start:.4f}s"
            if self.app_state.current_state == AppState.STATE_TRACKING and 't_detect' in locals():
                 profile_msg += f" | Preproc: {t_preprocess-t_start:.4f} | Detect: {t_detect-t_preprocess:.4f} | Match: {t_match-t_detect:.4f} | Homography: {t_homography-t_match:.4f}"
            print(profile_msg)
        
        # No output_frame is returned as it's not displayed
        return matches_count, status_message

# --- Game Logic (Mostly unchanged) ---
def run_guessing_game(current_match_count):
    if app_state.current_state != AppState.STATE_TRACKING:
        return True 

    # Credits check moved to main loop to control program flow (exit)
    # if game_state["credits"] <= 0: 
    #     return False 

    valid_match_count = current_match_count if current_match_count >= 0 else 0
    is_condition_met = RAW_VALUE_FOR_GUESS < valid_match_count
    
    if is_condition_met:
        game_state["ready_count"] += 1
        # game_log_parts.append("Ready!") # Less verbose for console focus
        sha_input_string = "GeorgeW" + str(game_state["guess_nonce"])
        sha = calculate_sha256_with_library(sha_input_string)
        if sha.startswith(PREFIX):
            game_state["success_count"] += 1
            game_state["credits"] += WIN_CREDITS
            print(f"Game WIN! Credits: {game_state['credits']}, Ready: {game_state['ready_count']}, Success: {game_state['success_count']}, Nonce: {game_state['guess_nonce']}, SHA: ...{sha[-6:]}")
        else:
            game_state["credits"] -= WIN_CREDITS
            # Optional: print loss due to SHA mismatch if desired for debugging
            # print(f"Game LOSS (SHA). Credits: {game_state['credits']}, Nonce: {game_state['guess_nonce']}")
    else:
        game_state["credits"] -= WIN_CREDITS
        # Optional: print loss due to condition not met
        # print(f"Game LOSS (Cond). Credits: {game_state['credits']}, Matches: {valid_match_count}")


    game_state["guess_nonce"] += 1
    if game_state["guess_nonce"] >= 10000000000: 
        game_state["guess_nonce"] = 0
    
    # Overall status print is handled by main loop's timed updates
    return True # Indicates game logic ran; credit check in main determines continuation

# --- Main Application Loop ---
def main():
    tracker = OpenCVTracker(app_state)
    tracker.initialize_features()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera.")
        return

    print(f"--- Headless OpenCV Tracker & Guesser ---")
    print(f"ORB Features: {ORB_N_FEATURES}, Processing Width: {PROCESSING_WIDTH if PROCESSING_WIDTH > 0 else 'Original'}")
    print(f"Profiling: {'Enabled' if PROFILE_TIMES else 'Disabled'}")
    print("Script will attempt to capture reference automatically.")
    print("Game will run until credits are exhausted or Ctrl+C is pressed.")
    print(f"-------------------------------------------")

    # Prepare for initial automatic reference capture
    if app_state.current_state == AppState.STATE_WAITING_FOR_REFERENCE:
        tracker.prepare_for_reference_capture()

    last_status_print_time = time.time()
    status_print_interval = 2.0 # Print general status every 2 seconds

    frame_count = 0
    max_initial_capture_attempts = 150 # Try for ~5 seconds at 30fps

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Can't receive frame from camera or end of video. Exiting.")
                break
            
            frame_count += 1
            matches_count, tracker_status_msg = tracker.process_frame(frame)
            
            current_time = time.time()
            if tracker_status_msg and (current_time - last_status_print_time > status_print_interval or \
                                      "Captured" in tracker_status_msg or "Failed" in tracker_status_msg):
                print(f"Tracker: {tracker_status_msg} (Credits: {game_state['credits']})")
                last_status_print_time = current_time

            if app_state.current_state == AppState.STATE_WAITING_FOR_REFERENCE and frame_count > max_initial_capture_attempts:
                print(f"Failed to capture a reference image automatically after {max_initial_capture_attempts} frames. Exiting.")
                break
            
            if app_state.current_state == AppState.STATE_TRACKING:
                if game_state["credits"] > 0:
                    run_guessing_game(matches_count)
                
                if game_state["credits"] <= 0:
                    print(f"Game Over: No credits left. Final Nonce: {game_state['guess_nonce']-1}. Exiting.")
                    break 
            
            # Minimal delay to prevent tight loop if camera is very fast and processing is minimal
            # This also allows some time for OS to handle Ctrl+C
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nCtrl+C detected. Exiting application.")
    finally:
        cap.release()
        print("Application Closed.")

if __name__ == '__main__':
    main()
