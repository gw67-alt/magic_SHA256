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

def calculate_sha256_with_library(data,nonce):
    """
    Calculates the SHA-256 hash of the given data using Python's hashlib library.

    Args:
        data (bytes or str): The data to hash. If it's a string, it will be encoded to UTF-8.

    Returns:
        str: The hexadecimal representation of the SHA-256 hash.
    """
    # Create a SHA-256 hash object
    for i in range(game_state["min_value"], game_state["max_value"]): #input space
        dataX = data + str(i)
        # Update the hash object with the data
        sha256_hash = hashlib.sha256()
        sha256_hash.update(dataX.encode('utf-8'))
        # Get the hexadecimal representation of the hash digest
        hex_digest = sha256_hash.hexdigest()
        if hex_digest.startswith(PREFIX):
            print(dataX, hex_digest)
            return i-1000000000,i+1000000000 #output space detection
    return 0,0
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
    def check_guess():
        if check_button['state'] == tk.DISABLED:
            return

        guess = int(round(slider.get()))
        target_number = game_state["target_number"]
        
        game_state["attempts"] += 1
        attempts_label.config(text=f"Attempts: {game_state['attempts']}/{game_state['max_attempts']}")
        # Check for a win
        A,B = calculate_sha256_with_library("Quantum_Rig",guess)
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

        min_val = game_state["min_value"]
        max_val = game_state["max_value"]
        
        game_state["target_number"] = random.randint(min_val, max_val)
        
        min_entry.config(state=tk.NORMAL)
        max_entry.config(state=tk.NORMAL)
        min_entry.delete(0, tk.END)
        min_entry.insert(0, str(min_val))
        max_entry.delete(0, tk.END)
        max_entry.insert(0, str(max_val))
        min_entry.config(state=tk.DISABLED)
        max_entry.config(state=tk.DISABLED)

        range_size = max_val - min_val + 1
        game_state["max_attempts"] = math.ceil(math.log2(range_size)) + 3 # More forgiving
        game_state["attempts"] = 0

        slider.config(from_=min_val, to=max_val, state=tk.NORMAL)
        slider.set((min_val + max_val) // 2)

        result_label.config(text="", foreground="#FFFFFF")
        attempts_label.config(text=f"Attempts: 0/{game_state['max_attempts']}", foreground="#CCCCCC")
        instruction_label.config(text=f"Guess a number between {min_val} and {max_val}")
        max_attempts_info_label.config(text=f"({game_state['max_attempts']} attempts max)")
        update_guess_display(slider.get())
        check_button.config(state=tk.NORMAL if ser and ser.is_open else tk.DISABLED)
        credits_label.config(text=f"Credits: {game_state['credits']}", foreground="#FFEB3B")

    def update_guess_display(value):
        # Round the raw slider value to the nearest 1000
        snapped_value = round(float(value))
        
        # Update the big display label with the snapped value
        current_guess_display.config(text=f"{int(snapped_value)}")
        
        # To provide a "snapping" feel, re-set the slider's value to the snapped value.
        # The check prevents a recursive loop of the command being called.
        if slider.get() != snapped_value:
            slider.set(snapped_value)

    # --- UI Elements (Widgets) ---
    credits_label = ttk.Label(root, text=f"Credits: {game_state['credits']}", font=("Arial", 12, "bold"), foreground="#FFEB3B")
    credits_label.pack(pady=(10, 5))

    range_frame = ttk.Frame(root, padding="10 5 10 5")
    range_frame.pack(pady=5, fill='x')
    min_label = ttk.Label(range_frame, text="Min:")
    min_label.pack(side=tk.LEFT, padx=(10, 2))
    min_entry = ttk.Entry(range_frame, width=10, font=('Arial', 10), justify='center')
    min_entry.pack(side=tk.LEFT, padx=(0, 10))
    max_label = ttk.Label(range_frame, text="Max:")
    max_label.pack(side=tk.LEFT, padx=(10, 2))
    max_entry = ttk.Entry(range_frame, width=10, font=('Arial', 10), justify='center')
    max_entry.pack(side=tk.LEFT, padx=(0, 10))
    reset_button = ttk.Button(range_frame, text="New Game", command=setup_game, width=20)
    reset_button.pack(side=tk.RIGHT, padx=(10, 10))

    instruction_label = ttk.Label(root, text="Click 'New Game' to start", font=('Arial', 11))
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
    result_label = tk.Label(result_frame, text="", font=('Arial', 10, 'bold'), anchor='nw', justify='left', wraplength=650, bg="#2E2E2E", fg="white")
    result_label.pack(pady=5, padx=5, fill='both', expand=True)
    
    # --- Establish Serial Connection ---
    port_to_use = "COM5" # Change this if your Arduino is on a different port
    try:
        ser = serial.Serial(port=port_to_use, baudrate=9600, timeout=2)
        time.sleep(2) # Wait for the connection to establish
        ser.write(b'R') # Send a ready check byte
        startup_message = ser.readline().decode('utf-8').strip()

        if "ARDUINO_READY" in startup_message:
            result_label.config(text=f"Successfully connected to Arduino on {port_to_use}")
            setup_game()
        else:
            messagebox.showerror("Connection Error", f"Arduino on {port_to_use} did not send ready signal. Response: {startup_message}")
            result_label.config(text="Arduino handshake failed. Game is disabled.", foreground="#F44336")

    except serial.SerialException:
        messagebox.showerror("Connection Error", f"Could not open serial port {port_to_use}.\nMake sure the Arduino is connected and the correct port is selected.")
        result_label.config(text="Arduino not found. Game is disabled.", foreground="#F44336")

    root.mainloop()

    if ser and ser.is_open:
        ser.close()

if __name__ == "__main__":
    keyword_guessing_game()
