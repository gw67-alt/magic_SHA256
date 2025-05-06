import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import random
import math
import hashlib
import threading
import time

# Constants
STARTING_CREDITS = 100
COST_PER_GUESS = 1
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
    """
    # Initialize Tkinter window
    root = tk.Tk()
    root.title("Guess the Number & Hash Finder")
    root.geometry("700x550") # Adjusted height
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
    style.configure('TEntry', fieldbackground='#4A4A4A', foreground='white', insertcolor='white', borderwidth=1)
    style.configure('Horizontal.TScale', background='#2E2E2E', troughcolor='#4A4A4A')
    style.map('Horizontal.TScale',
              background=[('active', '#2E2E2E')],
              troughcolor=[('active', '#5A5A5A')])
    style.configure('Display.TFrame', background='#1A1A1A')
    style.configure('Display.TLabel', background='#1A1A1A', foreground="#FFEB3B", font=('Consolas', 36, 'bold')) # Yellow, large font
    style.configure('Hash.TButton', background='#5D4037', foreground='white', font=('Arial', 11, 'bold'), padding=5) # Brownish button
    style.map('Hash.TButton',
              background=[('active', '#8D6E63'), ('disabled', '#3E2723')],
              foreground=[('disabled', '#777777')])
    style.configure('Add.TButton', background='#00695C', foreground='white', font=('Arial', 10, 'bold'), padding=(3,3)) # Teal button
    style.map('Add.TButton',
               background=[('active', '#00897B'), ('disabled', '#004D40')])


    # --- Game State Variables ---
    game_state = {
        "target_number": 1,
        "attempts": 0,
        "min_value": 1,
        "max_value": 100,
        "max_attempts": 0,
        "credits": STARTING_CREDITS,
        "hash_testing": False,
        "hash_thread": None,
        "stop_event": threading.Event()
    }

    # --- Game Logic Functions ---
    def check_guess():
        """ Checks the player's guess against the target number and updates the game state. """
        target_number = game_state["target_number"]
        max_attempts = game_state["max_attempts"]

        if check_button['state'] == 'disabled':
            return

        try:
            guess = int(round(slider.get()))

            # Deduct credits for guessing (optional, uncomment if desired)
            # game_state["credits"] -= COST_PER_GUESS
            # if game_state["credits"] < 0: game_state["credits"] = 0 # Don't go below zero
            # credits_label.config(text=f"Credits: {game_state['credits']}")
            # if game_state["credits"] == 0:
            #     result_label.config(text="Out of credits! Game Over.", foreground="#F44336")
            #     check_button.config(state='disabled')
            #     slider.config(state='disabled')
            #     stop_hash_testing() # Stop hash if running out of credits
            #     return

            game_state["attempts"] += 1
            attempts_label.config(text=f"Attempts: {game_state['attempts']}/{max_attempts}")
            # Get hashes for comparison
            guess_hash = calculate_sha256_with_library("GeorgeW" + str(guess))
            
            # --- Check the guess ---
            if guess_hash.startswith(hash_prefix_entry.get()):
                game_state["credits"] += WIN_CREDITS
                result_label.config(text=f"Correct! The number was {target_number}. You win {WIN_CREDITS} credits!", foreground="#4CAF50")
                credits_label.config(text=f"Credits: {game_state['credits']}", foreground="#4CAF50") # Update color too
                check_button.config(state='disabled')
                slider.config(state='disabled')
                stop_hash_testing() # Stop hash if running and guess is correct
            elif guess < target_number:
                result_label.config(text="Too low! Try again.", foreground="#FF9800") # Orange
            else: # guess > target_number
                result_label.config(text="Too high! Try again.", foreground="#2196F3") # Blue

            # Check for game over AFTER checking the guess
            if game_state["attempts"] >= max_attempts and guess != target_number:
                result_label.config(text=f"No more attempts! The number was {target_number}.", foreground="#F44336")
                check_button.config(state='disabled')
                slider.config(state='disabled')
                stop_hash_testing() # Stop hash if running and out of attempts

        except ValueError as e:
            print(f"Error in check_guess: {e}")
            result_label.config(text=f"Invalid guess value: {e}", foreground="#F44336")

    def setup_game():
        """ Sets up/Resets the game state and range based on entry fields. """
        # Stop any ongoing hash testing before resetting
        stop_hash_testing()

        if game_state["credits"] <= 0 and game_state["attempts"] > 0: # Allow initial setup even with 0 credits
            messagebox.showinfo("Game Over", "You are out of credits! Restart the application to play again.")
            return

        try:
            temp_min = int(min_entry.get())
            temp_max = int(max_entry.get())

            if temp_min >= temp_max:
                messagebox.showerror("Invalid Range", f"Min ({temp_min}) must be less than Max ({temp_max}). Reverting to previous range.")
                min_entry.delete(0, tk.END)
                min_entry.insert(0, str(game_state["min_value"]))
                max_entry.delete(0, tk.END)
                max_entry.insert(0, str(game_state["max_value"]))
                return

            # Update game state
            game_state["min_value"] = temp_min
            game_state["max_value"] = temp_max
            range_size = game_state["max_value"] - game_state["min_value"] + 1
            game_state["max_attempts"] = math.ceil(math.log2(range_size)) if range_size > 1 else 1
            game_state["target_number"] = random.randint(game_state["min_value"], game_state["max_value"])
            game_state["attempts"] = 0

            # Reset UI elements
            slider.config(from_=game_state["min_value"], to=game_state["max_value"], state='normal')
            slider.set((game_state["min_value"] + game_state["max_value"]) // 2)
            result_label.config(text="", foreground="#FFFFFF")
            attempts_label.config(text=f"Attempts: 0/{game_state['max_attempts']}", foreground="#CCCCCC")
            instruction_label.config(text=f"Guess between {game_state['min_value']} and {game_state['max_value']}")
            max_attempts_info_label.config(text=f"({game_state['max_attempts']} optimal attempts)")
            update_guess_display(slider.get())
            check_button.config(state='normal')
            credits_label.config(text=f"Credits: {game_state['credits']}", foreground="#FFEB3B") # Reset color

            print(f" Range: {game_state['min_value']}-{game_state['max_value']}")

        except ValueError as e:
            print(f"Error in setup_game: {e}")
            messagebox.showerror("Invalid Input", "Min and Max must be valid integers. Reverting to previous range.")
            min_entry.delete(0, tk.END)
            min_entry.insert(0, str(game_state["min_value"]))
            max_entry.delete(0, tk.END)
            max_entry.insert(0, str(game_state["max_value"]))

    def update_guess_display(value):
        """Updates the central display label with the slider's current integer value."""
        try:
            display_value = int(round(float(value)))
            current_guess_display.config(text=f"{display_value}")
        except ValueError:
            current_guess_display.config(text="--") # Fallback display

    # --- Add Current Number to Min Function --- <<< NEW FUNCTION
    def add_current_to_min():
        """Adds the currently displayed number to the Min entry."""
        try:
            current_display_val_str = current_guess_display.cget("text")
            if current_display_val_str == "--":
                 messagebox.showwarning("Cannot Add", "No current number selected yet. Move the slider.")
                 return
            current_min_val = int(min_entry.get())

            current_display_val = int(current_display_val_str)-current_min_val
            
            new_min_val = current_min_val + current_display_val

            # Optional: Add a check if the new min is >= max, though setup_game handles it
            # current_max_val = int(max_entry.get())
            # if new_min_val >= current_max_val:
            #     messagebox.showwarning("Invalid Addition", f"Adding {current_display_val} would make Min ({new_min_val}) >= Max ({current_max_val}).")
            #     return

            min_entry.delete(0, tk.END)
            min_entry.insert(0, str(new_min_val))
            print(f"Added {current_display_val} to Min. New Min entry: {new_min_val}")

        except ValueError:
             messagebox.showerror("Input Error", "Could not add values. Ensure Min entry contains a valid integer.")
        except tk.TclError as e:
            # This might happen if the display label is somehow invalid
            messagebox.showerror("UI Error", f"Could not read current number display: {e}")


    # --- SHA-256 Hash Testing Functions ---
    def toggle_hash_testing():
        """Toggle the SHA-256 hash testing state."""
        print("Toggle hash testing requested.")
        if game_state["hash_testing"]:
            stop_hash_testing()
        else:
            # Check if game is active before starting hash testing
            if slider['state'] == 'disabled':
                 messagebox.showinfo("Game Not Active", "Please start a new game before testing hashes.")
                 return
            start_hash_testing()

    def start_hash_testing():
        """Start the SHA-256 hash testing with the current guess."""
        if game_state["hash_testing"]: return # Already running

        try:
            # Use the number displayed, not necessarily the target number
            start_num_from_min_entry = int(slider.get())

            # Reset stop event and update UI
            game_state["stop_event"].clear()
            game_state["hash_testing"] = True
            hash_button.config(text="Stop Hash Testing")
            hash_result_label.config(text="Starting hash search...", foreground="#FFEB3B")

            prefix_str = hash_prefix_entry.get()
            print(f"Starting hash testing from min value={start_num_from_min_entry}, prefix length={prefix_str}")

            # Start the hash testing in a separate thread
            game_state["hash_thread"] = threading.Thread(
                target=run_hash_testing,
                args=(start_num_from_min_entry, prefix_str, game_state["stop_event"]),
                daemon=True # Allow thread to exit if main program closes
            )
            game_state["hash_thread"].start()

            # Optionally disable slider/check during hash testing?
            # slider.config(state='disabled')
            # check_button.config(state='disabled')

        except ValueError as e:
            print(f"Error in start_hash_testing: {e}")
            hash_result_label.config(text=f"Invalid slider/prefix value: {e}", foreground="#F44336")
            update_hash_button_state(False) # Ensure button state is correct


    def stop_hash_testing():
        """Stop the SHA-256 hash testing."""
        if not game_state["hash_testing"]: return # Not running

        print("Stopping hash testing.")
        game_state["stop_event"].set()
        # No need to join, let the thread finish checking the event
        game_state["hash_thread"] = None # Clear the thread reference

        # Schedule UI updates on the main thread
        root.after(0, lambda: update_hash_button_state(False))
        root.after(0, lambda: hash_result_label.config(text="Hash testing stopped.", foreground="#CCCCCC"))

        # Re-enable game controls if they were disabled during testing (and game is not over)
        # Check if game is actually over by other means (attempts/credits)
        is_game_over = check_button['state'] == 'disabled' and result_label.cget("text") # Crude check if a result message is shown
        if not is_game_over:
             root.after(0, lambda: slider.config(state='normal'))
             root.after(0, lambda: check_button.config(state='normal'))


    def run_hash_testing(start_num, prefix_str, stop_event):
        """Run the SHA-256 hash testing in a separate thread."""
        try:
            prefix_length = int(prefix_str) if prefix_str.isdigit() and int(prefix_str) > 0 else 5 # Default to 5 if invalid
            target_hash_prefix = "0" * prefix_length

            current_num = start_num
            start_time = time.time()
            iterations = 0
            found = False
            last_update_time = start_time

            while not stop_event.is_set(): # Check stop event first
                hashed_value = calculate_sha256_with_library("GeorgeW" + str(current_num))
                iterations += 1

                if hashed_value.startswith(target_hash_prefix):
                    found = True
                    elapsed_time = time.time() - start_time
                    result_msg = f"FOUND! 'GeorgeW{current_num}' -> {hashed_value}\nIter: {iterations} | Time: {elapsed_time:.2f}s"
                    print(result_msg)
                    update_hash_ui(result_msg, "#4CAF50")
                    break # Exit loop on find

                # Update UI periodically (e.g., every 0.5 seconds or 1000 iterations)
                current_time = time.time()
                if current_time - last_update_time > 0.5 or iterations % 2000 == 0:
                    elapsed_time = current_time - start_time
                    update_hash_ui(f"Testing: {current_num} | Iter: {iterations} | Time: {elapsed_time:.2f}s", "#FFEB3B")
                    last_update_time = current_time
                    # time.sleep(0.001) # Small sleep to yield thread slightly if needed

                current_num += 1 # Increment number for next test

            # After loop finishes
            if not found and not stop_event.is_set(): # Finished without finding and wasn't stopped manually
                 elapsed_time = time.time() - start_time
                 update_hash_ui(f"Prefix '{target_hash_prefix}' not found starting from {start_num}.\nIter: {iterations} | Time: {elapsed_time:.2f}s", "#F44336")

            # Final state update only needs to happen if it wasn't stopped manually
            if not stop_event.is_set():
                 root.after(0, lambda: update_hash_button_state(False)) # Ensure button resets if found/exhausted

        except Exception as e:
            error_msg = f"Error in hash thread: {e}"
            print(error_msg)
            update_hash_ui(error_msg, "#F44336")
            root.after(0, lambda: update_hash_button_state(False)) # Ensure button resets on error


    def update_hash_ui(text, color):
        """ Safely update the hash result label from any thread. """
        root.after(0, lambda: hash_result_label.config(text=text, foreground=color))

    def update_hash_button_state(testing):
        """ Safely update the hash button state and internal flag from any thread. """
        game_state["hash_testing"] = testing
        new_text = "Stop Hash Testing" if testing else "Start Hash Testing"
        hash_button.config(text=new_text)


    # --- UI Elements ---

    # Credit Label
    credits_label = ttk.Label(root, text=f"Credits: {game_state['credits']}", font=("Arial", 12, "bold"), foreground="#FFEB3B")
    credits_label.pack(pady=(10, 5))

    # Frame for range selection
    range_frame = ttk.Frame(root, padding="10 5 10 5")
    range_frame.pack(pady=5, fill='x')

    min_label = ttk.Label(range_frame, text="Min:")
    min_label.pack(side=tk.LEFT, padx=(10, 2))
    min_entry = ttk.Entry(range_frame, width=7, font=('Arial', 10), justify='center')
    min_entry.pack(side=tk.LEFT, padx=(0, 10))
    min_entry.insert(0, str(game_state["min_value"]))

    max_label = ttk.Label(range_frame, text="Max:")
    max_label.pack(side=tk.LEFT, padx=(10, 2))
    max_entry = ttk.Entry(range_frame, width=7, font=('Arial', 10), justify='center')
    max_entry.pack(side=tk.LEFT, padx=(0, 10))
    max_entry.insert(0, str(game_state["max_value"]))

    # --- NEW BUTTON ---
    add_to_min_button = ttk.Button(range_frame, text="Add Current# to Min", command=add_current_to_min, width=18, style='Add.TButton')
    add_to_min_button.pack(side=tk.LEFT, padx=(10, 5)) # Place it after Max entry

    # Set Range / New Game Button (pack last on the right)
    reset_button = ttk.Button(range_frame, text="Set Range", command=setup_game, width=20)
    reset_button.pack(side=tk.RIGHT, padx=(0, 10)) # Adjusted padding

    # Instruction Labels
    instruction_label = ttk.Label(root, text="Set range and click 'Set Range / New Game'", font=('Arial', 11))
    instruction_label.pack(pady=(5, 0))
    max_attempts_info_label = ttk.Label(root, text="", font=('Arial', 9), foreground="#AAAAAA")
    max_attempts_info_label.pack(pady=(0, 10))

    # Central Display Area
    display_frame = ttk.Frame(root, padding="10", relief="sunken", borderwidth=2, style='Display.TFrame')
    display_frame.pack(pady=10, padx=20, fill='x')
    current_guess_display = ttk.Label(display_frame, text="--", style='Display.TLabel', anchor='center')
    current_guess_display.pack(pady=10, fill='x')

    # Slider
    slider = ttk.Scale(root, from_=game_state["min_value"], to=game_state["max_value"],
                       orient=tk.HORIZONTAL, length=400,
                       command=update_guess_display) # Link slider to display update
    slider.pack(pady=15, padx=30)
    slider.config(state='disabled')

    # Check Button
    check_button = ttk.Button(root, text="Check Guess", command=check_guess, width=15)
    check_button.pack(pady=5)
    check_button.config(state='disabled')

    # Attempts and Result Labels
    attempts_label = ttk.Label(root, text="Attempts: 0/0", font=('Arial', 10), foreground="#CCCCCC")
    attempts_label.pack(pady=(5, 0))
    result_label = ttk.Label(root, text="", font=('Arial', 12, 'bold'), anchor='center', foreground="#FFFFFF", wraplength=450)
    result_label.pack(pady=10, fill='x', padx=20)

    # --- SHA-256 Hash Testing Section ---
    hash_section_separator = ttk.Separator(root, orient='horizontal')
    hash_section_separator.pack(fill='x', padx=20, pady=(5,10)) # Added padding top/bottom
    hash_section_title = ttk.Label(root, text="SHA-256 Hash Testing ('GeorgeW' + number)", font=('Arial', 12, 'bold'), foreground="#FFEB3B")
    hash_section_title.pack(pady=(0, 5))

    hash_prefix_frame = ttk.Frame(root)
    hash_prefix_frame.pack(pady=5)
    hash_prefix_label = ttk.Label(hash_prefix_frame, text="Leading Zero Prefix Length:")
    hash_prefix_label.pack(side=tk.LEFT, padx=(0, 5))
    hash_prefix_entry = ttk.Entry(hash_prefix_frame, width=3, justify='center')
    hash_prefix_entry.pack(side=tk.LEFT)
    hash_prefix_entry.insert(0, "5") # Default prefix length

    hash_button = ttk.Button(root, text="Start Hash Testing", command=toggle_hash_testing, width=17, style='Hash.TButton')
    hash_button.pack(pady=5)
    hash_result_label = ttk.Label(root, text="Click 'Start...' to find a hash starting with zeros.",
                                  font=('Arial', 10), foreground="#CCCCCC", wraplength=450, justify='center')
    hash_result_label.pack(pady=5, fill='x', padx=20)

    # --- Initial Game Setup ---
    root.after(10, setup_game) # Call setup shortly after mainloop starts

    # --- Start Tkinter Event Loop ---
    root.mainloop()

# --- Run the game ---
if __name__ == "__main__":
    number_guessing_game()
