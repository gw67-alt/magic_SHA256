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
COST_PER_GUESS = 10
WIN_CREDITS = 50

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
        return hex_digest

    except TypeError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

def number_guessing_game():
    """
    Creates a Tkinter window for a number guessing game with a variable range,
    styled with a dark theme and includes a credit system.
    """
    # Initialize Tkinter window
    root = tk.Tk()
    root.title("Guess the Number")
    root.geometry("500x550")  # Increased height for hash testing section
    root.resizable(False, False)
    root.configure(bg="#2E2E2E")  # Dark background for the main window

    # --- Style Configuration ---
    style = ttk.Style()
    try:
        # Attempt to use a theme that allows more customization
        style.theme_use('clam')
    except tk.TclError:
        print("Clam theme not available, using default.")  # Fallback if 'clam' isn't available

    # Configure styles for various widgets
    style.configure('.', background='#2E2E2E', foreground='white')  # Global style
    style.configure('TFrame', background='#2E2E2E')
    style.configure('TLabel', background='#2E2E2E', foreground='white', font=('Arial', 10))
    style.configure('TButton', background='#4A4A4A', foreground='white', font=('Arial', 11, 'bold'), borderwidth=0, padding=5)  # Added padding
    style.map('TButton',
              background=[('active', '#6A6A6A'), ('disabled', '#3A3A3A')],
              foreground=[('disabled', '#777777')])
    style.configure('TEntry', fieldbackground='#4A4A4A', foreground='white', insertcolor='white', borderwidth=1)
    style.configure('Horizontal.TScale', background='#2E2E2E', troughcolor='#4A4A4A')
    style.map('Horizontal.TScale',
              background=[('active', '#2E2E2E')],
              troughcolor=[('active', '#5A5A5A')])
    # Style for the central display frame and label
    style.configure('Display.TFrame', background='#1A1A1A')
    style.configure('Display.TLabel', background='#1A1A1A', foreground="#FFEB3B", font=('Consolas', 36, 'bold'))  # Yellow, large font
    
    # Hash testing button styles
    style.configure('Hash.TButton', background='#5D4037', foreground='white', font=('Arial', 11, 'bold'), padding=5)
    style.map('Hash.TButton',
              background=[('active', '#8D6E63'), ('disabled', '#3E2723')],
              foreground=[('disabled', '#777777')])

    # --- Game State Variables ---
    # Using a dictionary to hold game state makes it easier to manage
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

        # Prevent checking if game is already over (no credits or attempts)
        if check_button['state'] == 'disabled':  # Fixed: Changed tk.DISABLED to 'disabled' string
            return

        try:
            # Get integer value from the slider
            guess = int(round(slider.get()))

            # --- Update attempts and display ---
            game_state["attempts"] += 1
            attempts_label.config(text=f"Attempts: {game_state['attempts']}/{max_attempts}")
            
            # Define target hash prefix
            try:
                prefix_length = int(hash_prefix_entry.get()) if hash_prefix_entry.get().isdigit() else 5
                target_hash_prefix = "0" * prefix_length
            except ValueError:
                target_hash_prefix = "0" * 5  # Default if invalid entry
            for i in range(guess-10000000,guess+10000000):
                # Get hashes for comparison
                guess_hash = calculate_sha256_with_library("GeorgeW" + str(i))
                
                # --- Check the guess ---
                if guess_hash.startswith(target_hash_prefix):
                    game_state["credits"] += WIN_CREDITS
                    result_label.config(text=f"Close enough! You guessed in the right range: target = {target_number} in {game_state['attempts']} tries! You win {WIN_CREDITS} credits!", foreground="#4CAF50")  # Green
                    credits_label.config(text=f"Credits: {game_state['credits']}", foreground="#4CAF50")
                    check_button.config(state='disabled')  # Fixed: Changed tk.DISABLED to 'disabled' string
                    slider.config(state='disabled')  # Fixed: Changed tk.DISABLED to 'disabled' string
                    return  # Exit function early on correct guess

            # --- Check for game over conditions (AFTER checking the guess) ---
            # Check attempts first
            if game_state["attempts"] >= max_attempts and guess != target_number:  # Added check to ensure it wasn't the winning guess
                result_label.config(text=f"No more attempts! The number was {target_number}.", foreground="#F44336")  # Red
                check_button.config(state='disabled')  # Fixed: Changed tk.DISABLED to 'disabled' string
                slider.config(state='disabled')  # Fixed: Changed tk.DISABLED to 'disabled' string
            # Then check credits (could run out on the last attempt)
            elif game_state["credits"] <= 0:  # Use elif to avoid overwriting the 'no attempts' message if both happen
                result_label.config(text="You have no more credits. Game Over!", foreground="#F44336")  # Red
                credits_label.config(text="Credits: 0", foreground="#F44336")  # Ensure credits show 0
                check_button.config(state='disabled')  # Fixed: Changed tk.DISABLED to 'disabled' string
                slider.config(state='disabled')  # Fixed: Changed tk.DISABLED to 'disabled' string

        except ValueError as e:
            # This shouldn't happen with the slider, but good practice
            print(f"Error in check_guess: {e}")  # Added debug print
            result_label.config(text=f"Error: {e}", foreground="#F44336")  # Red

    def setup_game():
        """ Sets up/Resets the game state and range based on entry fields. """
        # Check if player has credits to start a new game
        if game_state["credits"] <= 0 and game_state["attempts"] > 0:  # Check attempts > 0 to allow initial setup
            messagebox.showinfo("Game Over", "You are out of credits! Restart the application to play again.")
            return

        try:
            temp_min = int(min_entry.get())
            temp_max = int(max_entry.get())

            # Validate the range
            if temp_min >= temp_max:
                messagebox.showerror("Invalid Range", f"Min ({temp_min}) must be less than Max ({temp_max}). Reverting to previous range.")
                # Reset entry fields to current valid range
                min_entry.delete(0, tk.END)
                min_entry.insert(0, str(game_state["min_value"]))
                max_entry.delete(0, tk.END)
                max_entry.insert(0, str(game_state["max_value"]))
                return  # Stop setup if range is invalid

            # Update game state with new valid range
            game_state["min_value"] = temp_min
            game_state["max_value"] = temp_max

            # Calculate max attempts based on the new range
            range_size = game_state["max_value"] - game_state["min_value"] + 1
            if range_size <= 1:  # Handle edge case where range is 1 or less
                game_state["max_attempts"] = 1
            else:
                # Optimal number of guesses using binary search idea
                game_state["max_attempts"] = math.ceil(math.log2(range_size))

            # Configure slider and reset game variables
            slider.config(from_=game_state["min_value"], to=game_state["max_value"], state='normal')  # Fixed: Changed tk.NORMAL to 'normal' string
            slider.set((game_state["min_value"] + game_state["max_value"]) // 2)  # Center the slider
            game_state["target_number"] = random.randint(game_state["min_value"], game_state["max_value"])
            game_state["attempts"] = 0

            # Reset UI elements
            result_label.config(text="", foreground="#FFFFFF")  # Clear result
            attempts_label.config(text=f"Attempts: {game_state['attempts']}/{game_state['max_attempts']}", foreground="#CCCCCC")
            instruction_label.config(text=f"Guess between {game_state['min_value']} and {game_state['max_value']}")
            max_attempts_info_label.config(text=f"({game_state['max_attempts']} attempts)")
            update_guess_display(slider.get())  # Update display with slider's initial value
            check_button.config(state='normal')  # Fixed: Changed tk.NORMAL to 'normal' string
            # Update credits label (color might have been red/green)
            credits_label.config(text=f"Credits: {game_state['credits']}", foreground="#FFEB3B")  # Reset to default yellow

            # Debug output
            print(f"Game setup with range: {game_state['min_value']}-{game_state['max_value']}")

            # Stop any ongoing hash testing
            stop_hash_testing()

        except ValueError as e:
            print(f"Error in setup_game: {e}")  # Added debug print
            messagebox.showerror("Invalid Input", "Min and Max must be valid integers. Reverting to previous range.")
            # Reset entry fields to current valid range
            min_entry.delete(0, tk.END)
            min_entry.insert(0, str(game_state["min_value"]))
            max_entry.delete(0, tk.END)
            max_entry.insert(0, str(game_state["max_value"]))

    def update_guess_display(value):
        """Updates the central display label with the slider's current integer value."""
        try:
            # Round the float value from the slider and convert to int
            display_value = int(round(float(value)))
            current_guess_display.config(text=f"{display_value}")
            
            # If hash testing is active, stop and restart it with the new value
            if game_state["hash_testing"]:
                stop_hash_testing()
                start_hash_testing()
                
        except ValueError as e:
            print(f"Error in update_guess_display: {e}")  # Added debug print
            current_guess_display.config(text="--")  # Fallback display

    # --- SHA-256 Hash Testing Functions ---
    def toggle_hash_testing():
        """Toggle the SHA-256 hash testing state."""
        print("Hash testing button clicked")  # Added debug print
        if game_state["hash_testing"]:
            stop_hash_testing()
        else:
            start_hash_testing()

    def start_hash_testing():
        """Start the SHA-256 hash testing with the current guess."""
        check_guess()
        if game_state["hash_testing"]:
            return  # Already testing
            
        # Make sure we have a valid guess
        try:
            guess = int(round(slider.get()))
            
            # Reset stop event and update UI
            game_state["stop_event"].clear()
            game_state["hash_testing"] = True
            hash_button.config(text="Stop Hash Testing")
            hash_result_label.config(text="Testing hashes...", foreground="#FFEB3B")
            
            # Get prefix length
            prefix_str = hash_prefix_entry.get()
            print(f"Starting hash testing with guess={guess}, prefix={prefix_str}")  # Added debug print
            
            # Start the hash testing in a separate thread
            game_state["hash_thread"] = threading.Thread(
                target=run_hash_testing,
                args=(guess, prefix_str, game_state["stop_event"])
            )
            game_state["hash_thread"].daemon = True  # Thread will be terminated when main program exits
            game_state["hash_thread"].start()
            
        except ValueError as e:
            print(f"Error in start_hash_testing: {e}")  # Added debug print
            hash_result_label.config(text=f"Invalid guess value: {e}", foreground="#F44336")

    def stop_hash_testing():
        """Stop the SHA-256 hash testing."""
        if not game_state["hash_testing"]:
            return  # Not currently testing
            
        # Signal the thread to stop and wait for it
        game_state["stop_event"].set()
        if game_state["hash_thread"] and game_state["hash_thread"].is_alive():
            # We don't join() the thread to avoid blocking the UI
            pass
            
        # Update UI
        game_state["hash_testing"] = False
        hash_button.config(text="Start Hash Testing")
        hash_result_label.config(text="Hash testing stopped.", foreground="#CCCCCC")

    def run_hash_testing(start_num, prefix_str, stop_event):
        """Run the SHA-256 hash testing in a separate thread."""
        try:
            # Validate the prefix
            prefix_length = int(prefix_str) if prefix_str.isdigit() else 7
            target_hash_prefix = "0" * prefix_length
            
            # Start testing from the guess value
            current_num = start_num
            start_time = time.time()
            iterations = 0
            found = False
            
            while not stop_event.is_set() and not found:
                # Test the current number
                hashed_value = calculate_sha256_with_library("GeorgeW" + str(current_num))
                iterations += 1
                
                # Update UI periodically (every 1000 iterations)
                if iterations % 1000 == 0:
                    elapsed_time = time.time() - start_time
                    update_hash_ui(f"Testing: {current_num} | Iterations: {iterations} | Time: {elapsed_time:.2f}s", "#FFEB3B")
                
                # Check for a match
                if hashed_value.startswith(target_hash_prefix):
                    found = True
                    elapsed_time = time.time() - start_time
                    result_msg = f"FOUND! 'GeorgeW{current_num}' → {hashed_value}\nIterations: {iterations} | Time: {elapsed_time:.2f}s"
                    update_hash_ui(result_msg, "#4CAF50")
                    print(f"{result_msg}")

                    break
                    
                # Move to the next number
                current_num += 1
                
            # Final update if not found and not stopped
            if not found and not stop_event.is_set():
                elapsed_time = time.time() - start_time
                update_hash_ui(f"No match found after {iterations} iterations and {elapsed_time:.2f}s", "#F44336")
                
            # Update UI state
            root.after(0, lambda: update_hash_button_state(False))
            
        except Exception as e:
            print(f"Error in run_hash_testing: {e}")  # Added debug print
            update_hash_ui(f"Error: {str(e)}", "#F44336")
            root.after(0, lambda: update_hash_button_state(False))

    def update_hash_ui(text, color):
        """Update the hash result label from any thread."""
        # Schedule UI update on the main thread
        root.after(0, lambda: hash_result_label.config(text=text, foreground=color))

    def update_hash_button_state(testing):
        """Update the hash button state from any thread."""
        game_state["hash_testing"] = testing
        if testing:
            hash_button.config(text="Stop Hash Testing")
        else:
            hash_button.config(text="Start Hash Testing")

    # --- UI Elements ---

    # Credit Label (Top)
    credits_label = ttk.Label(root, text=f"Credits: {game_state['credits']}", font=("Arial", 12, "bold"), foreground="#FFEB3B")  # Make credits stand out
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

    # Combined Set Range and Reset Button
    reset_button = ttk.Button(range_frame, text="Set Range / New Game", command=setup_game, width=20)
    reset_button.pack(side=tk.RIGHT, padx=(10, 10))

    # Instruction Labels
    instruction_label = ttk.Label(root, text="Set range and click 'Set Range / New Game'", font=('Arial', 11))
    instruction_label.pack(pady=(5, 0))

    max_attempts_info_label = ttk.Label(root, text="", font=('Arial', 9), foreground="#AAAAAA")
    max_attempts_info_label.pack(pady=(0, 10))

    # --- Central Display Area ---
    display_frame = ttk.Frame(root, padding="10", relief="sunken", borderwidth=2, style='Display.TFrame')
    display_frame.pack(pady=10, padx=20, fill='x')

    # Label to show the guess *in real-time* as the slider moves
    current_guess_display = ttk.Label(display_frame, text="--", style='Display.TLabel', anchor='center')  # Start with placeholder
    current_guess_display.pack(pady=10, fill='x')

    # --- Slider ---
    # Link slider movement directly to the display update function
    slider = ttk.Scale(root, from_=game_state["min_value"], to=game_state["max_value"],
                       orient=tk.HORIZONTAL, length=400,
                       command=update_guess_display)
    slider.pack(pady=15, padx=30)
    slider.config(state='disabled')  # Fixed: Changed tk.DISABLED to 'disabled' string

    # --- Check Button ---
    check_button = ttk.Button(root, text="Check Guess", command=check_guess, width=15)
    check_button.pack(pady=5)
    check_button.config(state='disabled')  # Fixed: Changed tk.DISABLED to 'disabled' string

    # --- Attempts and Result Labels ---
    attempts_label = ttk.Label(root, text="Attempts: 0/0", font=('Arial', 10), foreground="#CCCCCC")
    attempts_label.pack(pady=(5, 0))

    result_label = ttk.Label(root, text="", font=('Arial', 12, 'bold'), anchor='center', foreground="#FFFFFF", wraplength=450)  # Allow wrapping for longer messages
    result_label.pack(pady=10, fill='x', padx=20)

    # --- SHA-256 Hash Testing Section ---
    hash_section_separator = ttk.Separator(root, orient='horizontal')
    hash_section_separator.pack(fill='x', padx=20, pady=10)

    hash_section_title = ttk.Label(root, text="SHA-256 Hash Testing", font=('Arial', 12, 'bold'), foreground="#FFEB3B")
    hash_section_title.pack(pady=(0, 5))

    # Frame for hash prefix input
    hash_prefix_frame = ttk.Frame(root)
    hash_prefix_frame.pack(pady=5)

    hash_prefix_label = ttk.Label(hash_prefix_frame, text="Zero Prefix Length:")
    hash_prefix_label.pack(side=tk.LEFT, padx=(0, 5))

    hash_prefix_entry = ttk.Entry(hash_prefix_frame, width=3, justify='center')
    hash_prefix_entry.pack(side=tk.LEFT)
    hash_prefix_entry.insert(0, "5")  # Changed default prefix length to 5 for easier testing

    # Hash testing button
    hash_button = ttk.Button(root, text="Start Hash Testing", command=toggle_hash_testing, width=17, style='Hash.TButton')
    hash_button.pack(pady=5)

    # Hash result label
    hash_result_label = ttk.Label(root, text="Click 'Start Hash Testing' to find a hash with the specified prefix.",
                                 font=('Arial', 10), foreground="#CCCCCC", wraplength=450, justify='center')
    hash_result_label.pack(pady=5, fill='x', padx=20)

    # --- Initial Game Setup ---
    # Call setup_game initially to set the default range (1-100) and prepare the game
    setup_game()

    # Start the Tkinter event loop
    root.mainloop()

# --- Run the game ---
if __name__ == "__main__":
    number_guessing_game()
