import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import serial
import time
import math
import random
import hashlib
# --- Global Configuration ---
ser = None
WAIT_TIMEOUT = 5  # seconds

# --- Game Constants ---
STARTING_CREDITS = 1000
COST_PER_GUESS = 10
WIN_CREDITS = 150
PREFIX = "000000000"

# --- Game State Dictionary ---
game_state = {
    "attempts": 0,
    "min_value": 0,
    "max_value": 912000000000000000,
    "max_attempts": 0,
    "credits": STARTING_CREDITS,
    "target_number": 0,
}

# ---- OpenCL acceleration ----------------------------------------------------
try:
    import pyopencl as cl
    import numpy as np
    # create one context / queue for the whole run
    _ctx   = cl.create_some_context(interactive=False)
    _queue = cl.CommandQueue(_ctx)

    _cl_kernel = r"""
    // --------- SHA-256 helpers --------------------------------------------
    #pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
    typedef unsigned int  u32;
    typedef unsigned long u64;

    inline u32 ROTR(u32 x, u32 n){ return (x>>n)|(x<<(32-n)); }
    inline u32 Ch (u32 x,u32 y,u32 z){ return (x & y) ^ (~x & z); }
    inline u32 Maj(u32 x,u32 y,u32 z){ return (x & y) ^ (x & z) ^ (y & z); }
    inline u32 Σ0(u32 x){ return ROTR(x,2) ^ ROTR(x,13) ^ ROTR(x,22); }
    inline u32 Σ1(u32 x){ return ROTR(x,6) ^ ROTR(x,11) ^ ROTR(x,25); }
    inline u32 σ0(u32 x){ return ROTR(x,7) ^ ROTR(x,18) ^ (x>>3);  }
    inline u32 σ1(u32 x){ return ROTR(x,17)^ ROTR(x,19)^ (x>>10); }

    __constant u32 K[64] = {
        0x428a2f98U,0x71374491U,0xb5c0fbcfU,0xe9b5dba5U,0x3956c25bU,0x59f111f1U,0x923f82a4U,0xab1c5ed5U,
        0xd807aa98U,0x12835b01U,0x243185beU,0x550c7dc3U,0x72be5d74U,0x80deb1feU,0x9bdc06a7U,0xc19bf174U,
        0xe49b69c1U,0xefbe4786U,0x0fc19dc6U,0x240ca1ccU,0x2de92c6fU,0x4a7484aaU,0x5cb0a9dcU,0x76f988daU,
        0x983e5152U,0xa831c66dU,0xb00327c8U,0xbf597fc7U,0xc6e00bf3U,0xd5a79147U,0x06ca6351U,0x14292967U,
        0x27b70a85U,0x2e1b2138U,0x4d2c6dfcU,0x53380d13U,0x650a7354U,0x766a0abbU,0x81c2c92eU,0x92722c85U,
        0xa2bfe8a1U,0xa81a664bU,0xc24b8b70U,0xc76c51a3U,0xd192e819U,0xd6990624U,0xf40e3585U,0x106aa070U,
        0x19a4c116U,0x1e376c08U,0x2748774cU,0x34b0bcb5U,0x391c0cb3U,0x4ed8aa4aU,0x5b9cca4fU,0x682e6ff3U,
        0x748f82eeU,0x78a5636fU,0x84c87814U,0x8cc70208U,0x90befffaU,0xa4506cebU,0xbef9a3f7U,0xc67178f2U };

    // Each work-item tests a single nonce
    __kernel void sha256_prefix(
            __global const char *base,  int base_len,
            ulong  start_nonce,
            __global const char *prefix, int prefix_len,
            __global volatile ulong *found_nonce,
            __global volatile int  *found_flag)
    {
        size_t gid = get_global_id(0);
        ulong nonce = start_nonce + gid;
        if(*found_flag) return;                     // somebody already found it

        // ---------------- build message (base || decimal(nonce)) -------------
        char msg[128];
        int  len = 0;
        for(int i=0;i<base_len;i++) msg[len++] = base[i];

        char tmp[32]; int tmp_len=0;
        ulong n = nonce;
        do{ tmp[tmp_len++] = '0' + (n%10); n/=10; }while(n);
        for(int i=tmp_len-1;i>=0;--i)  msg[len++] = tmp[i];

        // ---------------- SHA-256 padding for single-block messages ----------
        uchar chunk[64] = {0};
        for(int i=0;i<len;i++) chunk[i] = (uchar)msg[i];
        chunk[len] = 0x80;
        ulong bit_len = ((ulong)len) * 8UL;
        for(int i=0;i<8;i++) chunk[56 + (7-i)] = (uchar)(bit_len >> (i*8));

        // ---------- initialise hash state ----------------
        u32 a=0x6a09e667U,b=0xbb67ae85U,c=0x3c6ef372U,d=0xa54ff53aU;
        u32 e=0x510e527fU,f=0x9b05688cU,g=0x1f83d9abU,h=0x5be0cd19U;

        u32 w[64];
        for(int i=0;i<16;i++){
            w[i] =  (u32)chunk[i*4+0]<<24 |
                    (u32)chunk[i*4+1]<<16 |
                    (u32)chunk[i*4+2]<< 8 |
                    (u32)chunk[i*4+3];
        }
        for(int i=16;i<64;i++)
            w[i] = σ1(w[i-2]) + w[i-7] + σ0(w[i-15]) + w[i-16];

        for(int i=0;i<64;i++){
            u32 t1 = h + Σ1(e) + Ch(e,f,g) + K[i] + w[i];
            u32 t2 = Σ0(a) + Maj(a,b,c);
            h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
        }
        a+=0x6a09e667U; b+=0xbb67ae85U; c+=0x3c6ef372U; d+=0xa54ff53aU;
        e+=0x510e527fU; f+=0x9b05688cU; g+=0x1f83d9abU; h+=0x5be0cd19U;

        // ------------ first bytes are enough to test a prefix ----------------
        char digest[64];
        uint words[8]={a,b,c,d,e,f,g,h};
        for(int i=0;i<8;i++){
            for(int j=0;j<8;j++){
                uint nib = (words[i]>>(28-4*j)) & 0xF;
                digest[i*8+j] = (nib<10)? ('0'+nib):('a'+nib-10);
            }
        }

        int ok = 1;
        for(int i=0;i<prefix_len;i++){
            if(digest[i]!=prefix[i]){ ok=0; break; }
        }
        if(ok && atomic_cmpxchg(found_flag,0,1)==0){
            *found_nonce = nonce;
        }
    }
    """;
    _prg = cl.Program(_ctx, _cl_kernel).build()
    OPENCL_AVAILABLE = True
    print("OpenCL acceleration enabled")
except Exception as _e:
    print("OpenCL unavailable -> pure CPU mode. Reason:", _e)
    OPENCL_AVAILABLE = False
# -----------------------------------------------------------------------------

def sha256_search_opencl(base_str: str, nonce_start: int, nonce_range: int = 1_000_000):
    """Return (low, high) if a nonce within the range makes SHA-256 start with PREFIX, else (0,0)."""
    if not OPENCL_AVAILABLE:
        return 0, 0

    base   = base_str.encode('utf-8')
    prefix = PREFIX.encode('ascii')

    mf = cl.mem_flags
    buf_base   = cl.Buffer(_ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=base)
    buf_prefix = cl.Buffer(_ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=prefix)
    buf_found  = cl.Buffer(_ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np.int32([0]))
    buf_nonce  = cl.Buffer(_ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np.uint64([0]))

    gsize = (nonce_range,)         # one work-item per nonce
    lsize = None                   # let the driver pick

    _prg.sha256_prefix(
        _queue, gsize, lsize,
        buf_base,  np.int32(len(base)),
        np.uint64(nonce_start),
        buf_prefix, np.int32(len(prefix)),
        buf_nonce,  buf_found)

    found = np.empty(1, dtype=np.int32)
    nonce = np.empty(1, dtype=np.uint64)
    cl.enqueue_copy(_queue, found,  buf_found)
    cl.enqueue_copy(_queue, nonce,  buf_nonce)
    _queue.finish()
    import hashlib

    def sha256_hash(text):
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    print("GeorgeW"+str(int(nonce[0])))
    print(sha256_hash("GeorgeW"+str(int(nonce[0]))))
    if found[0]:
        val = int(nonce[0])
        return val-1000000000, val+1000000000
    return 0, 0

# --- Serial Communication Function ---
def send_data_to_serial(data_to_send):
    """
    Sends data to the Arduino, waits for a response, and reads a line.
    Returns the received line or None on failure/timeout.
    """
    if not ser or not ser.is_open:
        messagebox.showerror("Serial Error", "Arduino is not connected.")
        return None
    try:
        ser.reset_input_buffer()
        ser.write(data_to_send.encode('utf-8'))
        start_time = time.time()
        
        while True:
            if time.time() - start_time > WAIT_TIMEOUT:
                messagebox.showerror("Serial Error", "Timeout waiting for response from Arduino.")
                return None
            
            if ser.in_waiting > 0:
                received_line = ser.readline().decode('utf-8').strip()
                if received_line:
                    return received_line
            
            time.sleep(0.05)
            
    except serial.SerialException as e:
        messagebox.showerror("Serial Error", f"Communication failed: {e}")
        if ser: ser.close()
        check_button.config(state=tk.DISABLED)
        reset_button.config(state=tk.DISABLED)
        return None

# --- Main Game Application ---
def keyword_guessing_game():
    global ser  # Declare that we are using the global 'ser' variable

    # Initialize Tkinter window
    root = tk.Tk()
    root.title("Number Guessing Game - Arduino Edition")
    root.geometry("1700x650")
    root.resizable(False, False)
    root.configure(bg="#2E2E2E")

    # --- Style Configuration ---
    style = ttk.Style()
    style.theme_use('clam')
    style.configure('.', background='#2E2E2E', foreground='white')
    style.configure('TFrame', background='#2E2E2E')
    style.configure('TLabel', background='#2E2E2E', foreground='white', font=('Arial', 10))
    style.configure('TButton', background='#4A4A4A', foreground='white', font=('Arial', 11, 'bold'), padding=5)
    style.map('TButton', background=[('active', '#6A6A6A'), ('disabled', '#3A3A3A')], foreground=[('disabled', '#777777')])
    style.configure('TEntry', fieldbackground='#4A4A4A', foreground='white', insertcolor='white')
    style.configure('Horizontal.TScale', background='#2E2E2E', troughcolor='#4A4A4A')
    style.configure('Display.TFrame', background='#1A1A1A')
    style.configure('Display.TLabel', background='#1A1A1A', foreground="#FFEB3B", font=('Consolas', 36, 'bold'))

    # --- Game Logic Functions ---
    def validate_range_inputs():
        """Validate and update min/max values from entry fields"""
        try:
            min_val = int(min_entry.get().strip())
            max_val = int(max_entry.get().strip())
            
            if min_val >= max_val:
                messagebox.showerror("Invalid Range", "Minimum value must be less than maximum value.")
                return False
            
            if max_val - min_val < 100:
                messagebox.showwarning("Small Range", "Range is very small. Consider using a larger range for better gameplay.")
            
            game_state["min_value"] = min_val
            game_state["max_value"] = max_val
            return True
            
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid integer values for min and max.")
            return False

    def check_guess():
        if check_button['state'] == tk.DISABLED:
            return

        guess = int(round(slider.get()))
        target_number = game_state["target_number"]
        
        game_state["attempts"] += 1
        attempts_label.config(text=f"Attempts: {game_state['attempts']}/{game_state['max_attempts']}")
        # Check for a win
        A, B = sha256_search_opencl(
                "GeorgeW",
                int(game_state['min_value']),
                nonce_range=(int(game_state['max_value']) - int(game_state['min_value']))
            )
        # Check for a win
        if guess >= A and guess <= B:
            game_state["credits"] += WIN_CREDITS
            result_text = f"WIN! The number was {A} -> {B}. You win {WIN_CREDITS} credits!"
            result_label.config(text=result_text, foreground="#4CAF50")
            credits_label.config(text=f"Credits: {game_state['credits']}", foreground="#4CAF50")
            check_button.config(state=tk.DISABLED)
            slider.config(state=tk.DISABLED)
            return
        else:
            # If not a win, provide feedback and deduct credits
            game_state["credits"] -= COST_PER_GUESS
            credits_label.config(text=f"Credits: {game_state['credits']}", foreground="#FF9800")
            
     
        # Check for game over conditions
        if game_state["attempts"] >= game_state["max_attempts"]:
            result_label.config(text=f"No more attempts!.", foreground="#F44336")
            check_button.config(state=tk.DISABLED)
            slider.config(state=tk.DISABLED)
        elif game_state["credits"] <= 0:
            result_label.config(text="You have no more credits. Game Over!", foreground="#F44336")
            credits_label.config(text="Credits: 0", foreground="#F44336")
            check_button.config(state=tk.DISABLED)
            slider.config(state=tk.DISABLED)

    def setup_game():
        if game_state["credits"] <= 0 and game_state["attempts"] > 0:
            messagebox.showinfo("Game Over", "You are out of credits! Restart the application to play again.")
            return

        # Validate range inputs before starting the game
        if not validate_range_inputs():
            return

        min_val = game_state["min_value"]
        max_val = game_state["max_value"]
        
        game_state["target_number"] = random.randint(min_val, max_val)
        
        # Disable min/max entry fields during gameplay
        min_entry.config(state=tk.DISABLED)
        max_entry.config(state=tk.DISABLED)

        range_size = max_val - min_val + 1
        game_state["max_attempts"] = math.ceil(math.log2(range_size)) + 3 # More forgiving
        game_state["attempts"] = 0

        slider.config(from_=min_val, to=max_val, state=tk.NORMAL)
        slider.set((min_val + max_val) // 2)

        result_label.config(text="", foreground="#FFFFFF")
        attempts_label.config(text=f"Attempts: 0/{game_state['max_attempts']}", foreground="#CCCCCC")
        instruction_label.config(text=f"Guess a number between {min_val:,} and {max_val:,}")
        max_attempts_info_label.config(text=f"({game_state['max_attempts']} attempts max)")
        update_guess_display(slider.get())
        check_button.config(state=tk.NORMAL if ser and ser.is_open else tk.DISABLED)
        credits_label.config(text=f"Credits: {game_state['credits']}", foreground="#FFEB3B")

    def reset_game():
        """Reset game state and re-enable min/max entry fields"""
        game_state["attempts"] = 0
        game_state["target_number"] = 0
        
        # Re-enable min/max entry fields
        min_entry.config(state=tk.NORMAL)
        max_entry.config(state=tk.NORMAL)
        
        # Reset display elements
        slider.config(state=tk.DISABLED)
        check_button.config(state=tk.DISABLED)
        result_label.config(text="Set your range and click 'New Game' to start", foreground="#FFFFFF")
        attempts_label.config(text="Attempts: 0/0", foreground="#CCCCCC")
        instruction_label.config(text="Set min and max values, then start a new game")
        max_attempts_info_label.config(text="")
        current_guess_display.config(text="--")

    def update_guess_display(value):
        # Round the raw slider value to the nearest integer
        snapped_value = round(float(value))
        
        # Update the big display label with the snapped value
        current_guess_display.config(text=f"{int(snapped_value):,}")
        
        # To provide a "snapping" feel, re-set the slider's value to the snapped value.
        # The check prevents a recursive loop of the command being called.
        if slider.get() != snapped_value:
            slider.set(snapped_value)

    # --- UI Elements (Widgets) ---
    credits_label = ttk.Label(root, text=f"Credits: {game_state['credits']}", font=("Arial", 12, "bold"), foreground="#FFEB3B")
    credits_label.pack(pady=(10, 5))

    # Enhanced range frame with better layout
    range_frame = ttk.Frame(root, padding="10 5 10 5")
    range_frame.pack(pady=5, fill='x')
    
    min_label = ttk.Label(range_frame, text="Min:")
    min_label.pack(side=tk.LEFT, padx=(10, 2))
    min_entry = ttk.Entry(range_frame, width=20, font=('Arial', 10), justify='center')
    min_entry.pack(side=tk.LEFT, padx=(0, 15))
    min_entry.insert(0, str(game_state["min_value"]))  # Set default value
    
    max_label = ttk.Label(range_frame, text="Max:")
    max_label.pack(side=tk.LEFT, padx=(10, 2))
    max_entry = ttk.Entry(range_frame, width=20, font=('Arial', 10), justify='center')
    max_entry.pack(side=tk.LEFT, padx=(0, 15))
    max_entry.insert(0, str(game_state["max_value"]))  # Set default value
    
    # Button frame for better organization
    button_frame = ttk.Frame(range_frame)
    button_frame.pack(side=tk.RIGHT, padx=(10, 10))
    
    reset_button = ttk.Button(button_frame, text="Reset", command=reset_game, width=12)
    reset_button.pack(side=tk.LEFT, padx=(0, 5))
    
    new_game_button = ttk.Button(button_frame, text="New Game", command=setup_game, width=12)
    new_game_button.pack(side=tk.LEFT)

    instruction_label = ttk.Label(root, text="Set min and max values, then click 'New Game' to start", font=('Arial', 11))
    instruction_label.pack(pady=(5, 0))
    max_attempts_info_label = ttk.Label(root, text="", font=('Arial', 9), foreground="#AAAAAA")
    max_attempts_info_label.pack(pady=(0, 10))

    display_frame = ttk.Frame(root, padding="10", relief="sunken", borderwidth=2, style='Display.TFrame')
    display_frame.pack(pady=10, padx=20, fill='x')
    current_guess_display = ttk.Label(display_frame, text="--", style='Display.TLabel', anchor='center')
    current_guess_display.pack(pady=10, fill='x')

    slider = ttk.Scale(root, from_=game_state["min_value"], to=game_state["max_value"], orient=tk.HORIZONTAL, length=1400, command=update_guess_display)
    slider.pack(pady=15, padx=30)
    slider.config(state=tk.DISABLED)

    check_button = ttk.Button(root, text="Check Guess", command=check_guess, width=15)
    check_button.pack(pady=5)
    check_button.config(state=tk.DISABLED)

    attempts_label = ttk.Label(root, text="Attempts: 0/0", font=('Arial', 10), foreground="#CCCCCC")
    attempts_label.pack(pady=(5, 0))
    
    result_frame = ttk.Frame(root, height=150)
    result_frame.pack(pady=10, padx=20, fill='both', expand=True)
    result_label = tk.Label(result_frame, text="Set your range and click 'New Game' to start", font=('Arial', 10, 'bold'), anchor='nw', justify='left', wraplength=650, bg="#2E2E2E", fg="white")
    result_label.pack(pady=5, padx=5, fill='both', expand=True)
    
    # --- Establish Serial Connection ---
    port_to_use = "/dev/ttyUSB0" # Change this if your Arduino is on a different port
    try:
        ser = serial.Serial(port=port_to_use, baudrate=9600, timeout=2)
        time.sleep(2) # Wait for the connection to establish
        ser.write(b'R') # Send a ready check byte
        startup_message = ser.readline().decode('utf-8').strip()

        if "ARDUINO_READY" in startup_message:
            result_label.config(text=f"Successfully connected to Arduino on {port_to_use}.\nSet your range and click 'New Game' to start")
        else:
            messagebox.showerror("Connection Error", f"Arduino on {port_to_use} did not send ready signal. Response: {startup_message}")
            result_label.config(text="Arduino handshake failed. Set your range and click 'New Game' to start in offline mode.", foreground="#F44336")

    except serial.SerialException:
        messagebox.showerror("Connection Error", f"Could not open serial port {port_to_use}.\nMake sure the Arduino is connected and the correct port is selected.")
        result_label.config(text="Arduino not found. Set your range and click 'New Game' to start in offline mode.", foreground="#F44336")

    root.mainloop()

    if ser and ser.is_open:
        ser.close()

if __name__ == "__main__":
    keyword_guessing_game()
