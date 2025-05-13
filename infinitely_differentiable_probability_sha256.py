import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import random
import math
import hashlib
import threading
import time

# Constants
STARTING_CREDITS = 100000
COST_PER_GUESS = 1 # Currently not used, but available
WIN_CREDITS = 150

# --- Helper Function: SHA-256 Calculation ---
def calculate_sha256_with_library(data):
    """
    Calculates the SHA-256 hash of the given data using Python's hashlib library.

    Args:
        data (bytes or str): The data to hash. If it's a string, it will be encoded to UTF-8.

    Returns:
        str: The hexadecimal representation of the SHA-256 hash.
    """
    try:
        sha256_hash = hashlib.sha256()
        if isinstance(data, str):
            sha256_hash.update(data.encode('utf-8'))
        elif isinstance(data, bytes):
            sha256_hash.update(data)
        else:
            raise TypeError("Input data must be a string or bytes.")
        hex_digest = sha256_hash.hexdigest()
        return hex_digest
    except TypeError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

# --- Main Game Function ---
def number_guessing_game():
    """
    Creates a Tkinter window for a number guessing game with a variable range,
    styled with a dark theme and includes a credit system and hash testing.
    Player guesses a range using two sliders.
    """
    # Initialize Tkinter window
    root = tk.Tk()
    root.title("Guess the Number Range & Hash Finder")
    root.geometry("700x600") # Adjusted height for new sliders
    root.resizable(False, False)
    root.configure(bg="#2E2E2E")

    # --- Style Configuration ---
    style = ttk.Style()
    try:
        style.theme_use('clam')
    except tk.TclError:
        print("Clam theme not available, using default.")

    style.configure('.', background='#2E2E2E', foreground='white')
    style.configure('TFrame', background='#2E2E2E')
    style.configure('TLabel', background='#2E2E2E', foreground='white', font=('Arial', 10))
    style.configure('TButton', background='#4A4A4A', foreground='white', font=('Arial', 11, 'bold'), borderwidth=0, padding=5)
    style.map('TButton',
              background=[('active', '#6A6A6A'), ('disabled', '#3A3A3A')],
              foreground=[('disabled', '#777777')])
    style.configure('TEntry', fieldbackground='#4A4A4A', foreground='white', insertcolor='white', borderwidth=1, font=('Arial',10))
    style.configure('Horizontal.TScale', background='#2E2E2E', troughcolor='#4A4A4A')
    style.map('Horizontal.TScale',
              background=[('active', '#2E2E2E')],
              troughcolor=[('active', '#5A5A5A')])
    # Display.TFrame and Display.TLabel styles are kept if needed for other things, but not for the main guess display
    style.configure('Hash.TButton', background='#5D4037', foreground='white', font=('Arial', 11, 'bold'), padding=5)
    style.map('Hash.TButton',
              background=[('active', '#8D6E63'), ('disabled', '#3E2723')],
              foreground=[('disabled', '#777777')])
    # Add.TButton style is removed as the button is removed.

    # --- Game State Variables ---
    game_state = {
        "target_number": 1,
        "attempts": 0,
        "min_value": 1,       # Overall game min
        "max_value": 100,     # Overall game max
        "max_attempts": 0,
        "credits": STARTING_CREDITS,
        "hash_testing": False,
        "hash_thread": None,
        "stop_event": threading.Event()
    }

    def setup_game():
        """ Sets up/Resets the game state and range based on entry fields. """
        stop_hash_testing()

        if game_state["credits"] <= 0 and game_state["attempts"] > 0:
            messagebox.showinfo("Game Over", "You are out of credits! Restart the application to play again.")
            return

        try:
            temp_min = int(min_entry.get())
            temp_max = int(max_entry.get())

            if temp_min >= temp_max:
                messagebox.showerror("Invalid Range", f"Overall Min ({temp_min}) must be less than Overall Max ({temp_max}). Reverting.")
                min_entry.delete(0, tk.END); min_entry.insert(0, str(game_state["min_value"]))
                max_entry.delete(0, tk.END); max_entry.insert(0, str(game_state["max_value"]))
                return

            game_state["min_value"] = temp_min
            game_state["max_value"] = temp_max
            range_size = game_state["max_value"] - game_state["min_value"] + 1
            # Max attempts for guessing a range is different, but let's keep log2 for now as an indicator
            game_state["max_attempts"] = math.ceil(math.log2(range_size)) if range_size > 1 else 1
            game_state["target_number"] = random.randint(game_state["min_value"], game_state["max_value"])
            game_state["attempts"] = 0

            # Configure and reset guess sliders
            guess_slider.config(from_=game_state["min_value"], to=game_state["max_value"], state='normal')
            guess_slider.set(game_state["min_value"])
            update_min_guess_label(game_state["min_value"])
            update_max_guess_label(game_state["max_value"])

            result_label.config(text="", foreground="#FFFFFF")
            attempts_label.config(text=f"Attempts: 0/{game_state['max_attempts']}", foreground="#CCCCCC")
            instruction_label.config(text=f"Select a guess range between {game_state['min_value']} and {game_state['max_value']}")
            max_attempts_info_label.config(text=f"({game_state['max_attempts']} optimal attempts for single number)")
            credits_label.config(text=f"Credits: {game_state['credits']}", foreground="#FFEB3B")

            print(f"New game. Overall Range: {game_state['min_value']}-{game_state['max_value']}. Target: {game_state['target_number']}")

        except ValueError as e:
            print(f"Error in setup_game: {e}")
            messagebox.showerror("Invalid Input", "Min and Max must be valid integers. Reverting.")
            min_entry.delete(0, tk.END); min_entry.insert(0, str(game_state["min_value"]))
            max_entry.delete(0, tk.END); max_entry.insert(0, str(game_state["max_value"]))

    def update_min_guess_label(value):
        """Updates the label for the min guess slider."""
        val = int(round(float(value)))
        print(val)
       
    def update_max_guess_label(value):
        """Updates the label for the max guess slider."""
        val = int(round(float(value)))
        # Ensure max_guess >= min_guess
        if guess_slider.get() > val:
            guess_slider.set(val)
            update_min_guess_label(val)

    # add_current_to_min and its button are removed.

    def toggle_hash_testing():
        """Toggle the SHA-256 hash testing state."""
        if game_state["hash_testing"]:
            stop_hash_testing()
        else:
            if guess_slider['state'] == 'disabled' and not result_label.cget("text").startswith("Correct!"):
                 messagebox.showinfo("Game Not Active", "Please start a new game before testing hashes.")
                 return
            start_hash_testing()

    def start_hash_testing():
        """Start the SHA-256 hash testing, starting from min_guess_slider's value."""
        if game_state["hash_testing"]: return

        try:
            start_num = int(round(guess_slider.get())) # Use min_guess_slider value

            game_state["stop_event"].clear()
            game_state["hash_testing"] = True
            hash_button.config(text="Stop Hash Testing")
            hash_result_label.config(text="Starting hash search...", foreground="#FFEB3B")
            prefix_str = hash_prefix_entry.get()
            if game_state["credits"] >= 0: # Use elif to avoid overwriting the 'no attempts' message if both happen
                    
                print(f"Starting hash testing from value={start_num}, prefix length={prefix_str}")

                game_state["hash_thread"] = threading.Thread(
                    target=run_hash_testing,
                    args=(start_num, prefix_str, game_state["stop_event"]),
                    daemon=True
                )
                game_state["hash_thread"].start()
        except ValueError as e:
            print(f"Error in start_hash_testing: {e}")
            hash_result_label.config(text=f"Invalid slider/prefix value: {e}", foreground="#F44336")
            update_hash_button_state(False)


    def stop_hash_testing():
        """Stop the SHA-256 hash testing."""
        if not game_state["hash_testing"]: return

        print("Stopping hash testing.")
        game_state["stop_event"].set()
        game_state["hash_thread"] = None

        root.after(0, lambda: update_hash_button_state(False))
        root.after(0, lambda: hash_result_label.config(text="Hash testing stopped.", foreground="#CCCCCC"))

        is_game_over_by_result = "Correct!" in result_label.cget("text") or "No more attempts!" in result_label.cget("text")
        if not is_game_over_by_result:
            root.after(0, lambda: guess_slider.config(state='normal'))


    def run_hash_testing(start_num, prefix_str, stop_event):
        """Run the SHA-256 hash testing in a separate thread."""
        try:
            prefix_length = int(prefix_str) if prefix_str.isdigit() and int(prefix_str) > 0 else 5
            target_hash_prefix = "0" * prefix_length
            current_num = game_state["min_value"]
            start_time = time.time()
            iterations = 0
            found = False
            last_update_time = start_time

            while not stop_event.is_set():
                hashed_value = calculate_sha256_with_library("GeorgeW" + str(current_num))
                iterations += 1

                if hashed_value.startswith(target_hash_prefix):
                    found = True
                    elapsed_time = time.time() - start_time
                    result_msg = f"FOUND! 'GeorgeW{current_num}' -> {hashed_value}\nIter: {iterations} | Time: {elapsed_time:.2f}s"
                    print(result_msg)
                    update_hash_ui(result_msg, "#4CAF50")
                    game_state["credits"] += WIN_CREDITS

                    break
                
                current_time = time.time()
                if current_time - last_update_time > 0.5 or iterations % 2000 == 0:
                    elapsed_time = current_time - start_time
                    update_hash_ui(f"Testing: {current_num} | Iter: {iterations} | Time: {elapsed_time:.2f}s", "#FFEB3B")
                    last_update_time = current_time
                current_num += 1

            if not found and not stop_event.is_set():
                game_state["credits"] -= WIN_CREDITS

                elapsed_time = time.time() - start_time
                update_hash_ui(f"Prefix '{target_hash_prefix}' not found starting from {start_num}.\nIter: {iterations} | Time: {elapsed_time:.2f}s", "#F44336")
            if not stop_event.is_set():
                root.after(0, lambda: update_hash_button_state(False))
            
        except Exception as e:
            error_msg = f"Error in hash thread: {e}"
            print(error_msg)
            update_hash_ui(error_msg, "#F44336")
            root.after(0, lambda: update_hash_button_state(False))

    def update_hash_ui(text, color):
        root.after(0, lambda: hash_result_label.config(text=text, foreground=color))

    def update_hash_button_state(testing):
        game_state["hash_testing"] = testing
        new_text = "Stop Hash Testing" if testing else "Start Hash Testing"
        hash_button.config(text=new_text)

    # --- UI Elements ---
    credits_label = ttk.Label(root, text=f"Credits: {game_state['credits']}", font=("Arial", 12, "bold"), foreground="#FFEB3B")
    credits_label.pack(pady=(10, 5))

    range_frame = ttk.Frame(root, padding="10 5 10 5")
    range_frame.pack(pady=5, fill='x')

    min_label = ttk.Label(range_frame, text="Overall Min:")
    min_label.pack(side=tk.LEFT, padx=(10, 2))
    min_entry = ttk.Entry(range_frame, width=7, font=('Arial', 10), justify='center')
    min_entry.pack(side=tk.LEFT, padx=(0, 10))
    min_entry.insert(0, str(game_state["min_value"]))

    max_label = ttk.Label(range_frame, text="Overall Max:")
    max_label.pack(side=tk.LEFT, padx=(10, 2))
    max_entry = ttk.Entry(range_frame, width=7, font=('Arial', 10), justify='center')
    max_entry.pack(side=tk.LEFT, padx=(0, 20)) # Increased padding after max_entry
    max_entry.insert(0, str(game_state["max_value"]))

    # Removed "Add Current# to Min" button
    reset_button = ttk.Button(range_frame, text="Set Overall Range / New Game", command=setup_game, width=28)
    reset_button.pack(side=tk.RIGHT, padx=(0, 10))

    instruction_label = ttk.Label(root, text="Set overall range and click 'Set Overall Range / New Game'", font=('Arial', 11))
    instruction_label.pack(pady=(5, 0))
    max_attempts_info_label = ttk.Label(root, text="", font=('Arial', 9), foreground="#AAAAAA")
    max_attempts_info_label.pack(pady=(0, 10))

    # --- Guess Range Sliders ---
    guess_range_frame = ttk.Frame(root, padding="10 5 10 10")
    guess_range_frame.pack(pady=10, padx=20, fill='x')

    min_guess_label_text = ttk.Label(guess_range_frame, text="Min Guess:", font=('Arial', 11))
    min_guess_label_text.pack(side=tk.LEFT, padx=(0,5))
    guess_slider = ttk.Scale(guess_range_frame, from_=game_state["min_value"], to=game_state["max_value"],
                                 orient=tk.HORIZONTAL, length=400, command=update_min_guess_label)
    guess_slider.pack(side=tk.LEFT, padx=5)
    guess_val_label = ttk.Label(guess_range_frame, text=str(game_state["min_value"]), font=('Arial', 11, 'bold'), width=4)
    guess_val_label.pack(side=tk.LEFT, padx=(0,15))

    guess_slider.config(state='disabled')


    attempts_label = ttk.Label(root, text="Attempts: 0/0", font=('Arial', 10), foreground="#CCCCCC")
    attempts_label.pack(pady=(5, 0))
    result_label = ttk.Label(root, text="", font=('Arial', 12, 'bold'), anchor='center', foreground="#FFFFFF", wraplength=650)
    result_label.pack(pady=10, fill='x', padx=20)

    # SHA-256 Hash Testing Section
    hash_section_separator = ttk.Separator(root, orient='horizontal')
    hash_section_separator.pack(fill='x', padx=20, pady=(5,10))
    hash_section_title = ttk.Label(root, text="SHA-256 Hash Testing ('GeorgeW' + number from Min Guess Slider)", font=('Arial', 12, 'bold'), foreground="#FFEB3B")
    hash_section_title.pack(pady=(0, 5))

    hash_prefix_frame = ttk.Frame(root)
    hash_prefix_frame.pack(pady=5)
    hash_prefix_label = ttk.Label(hash_prefix_frame, text="Leading Zero Prefix Length:")
    hash_prefix_label.pack(side=tk.LEFT, padx=(0, 5))
    hash_prefix_entry = ttk.Entry(hash_prefix_frame, width=3, justify='center')
    hash_prefix_entry.pack(side=tk.LEFT)
    hash_prefix_entry.insert(0, "5")

    hash_button = ttk.Button(root, text="Start Hash Testing", command=toggle_hash_testing, width=17, style='Hash.TButton')
    hash_button.pack(pady=5)
    hash_result_label = ttk.Label(root, text="Adjust Min Guess slider and click 'Start...' to find a hash.",
                                  font=('Arial', 10), foreground="#CCCCCC", wraplength=650, justify='center')
    hash_result_label.pack(pady=5, fill='x', padx=20)

    root.after(10, setup_game)
    root.mainloop()

if __name__ == "__main__":
    number_guessing_game()
