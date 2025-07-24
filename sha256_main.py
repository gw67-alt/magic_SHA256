import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import random
import math
import time
import datetime # Not directly used in game logic now, but kept
import psutil # Added for getting process count

# Constants
STARTING_CREDITS = 10000
COST_PER_GUESS = 10
WIN_CREDITS = 50

# --- Process Count Function ---
def get_current_process_count():
    """Returns the current number of running processes."""
    try:
        return len(psutil.pids())
    except Exception as e:
        messagebox.showerror("Error", f"Could not get process count: {e}")
        return None # Or a default fallback like 0 or -1

def number_guessing_game():
    """
    Creates a Tkinter window for a number guessing game
    styled with a dark theme and includes a credit system.
    """
    root = tk.Tk()
    root.title("Number Guessing Game") # Updated title
    root.geometry("500x500")
    root.resizable(False, False)
    root.configure(bg="#2E2E2E")

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
    style.configure('Display.TLabel', background='#1A1A1A', foreground="#FFEB3B", font=('Consolas', 36, 'bold'))

    game_state = {
        "target_number": 0, # This will be the number to guess
        "attempts": 0,
        "max_attempts": 7, # Based on log2 of range
        "credits": STARTING_CREDITS,
        "guessed_numbers": [],
        "min_range": 1,
        "max_range": 1000000000
    }

    def check_guess():
        if check_button['state'] == tk.DISABLED:
            return
        
        # Get guess from slider
        guess = int(round(guess_slider.get()))

        # Check if number was already guessed
        if guess in game_state["guessed_numbers"]:
            result_label.config(text=f"You already guessed {guess}!", foreground="#FF9800")
            return

        check_number_guess(guess)
    import hashlib

    def sha256_hash(input_string):
        """
        Generate SHA256 hash of the input string.
        
        Args:
            input_string (str): The string to hash
            
        Returns:
            str: The SHA256 hash in hexadecimal format
        """
        for i in range(10000000):
            # Convert string to bytes
            input_bytes = (input_string + str(i)).encode('utf-8')
            
            # Create SHA256 hash object
            sha256_hash_obj = hashlib.sha256()
            
            # Update the hash object with the input bytes
            sha256_hash_obj.update(input_bytes)
            
            # Get the hexadecimal representation of the hash
            hex_hash = sha256_hash_obj.hexdigest()
            if hex_hash.startswith("000000"):
                print(hex_hash)
                return i
            
        return False
    def check_number_guess(number):
        game_state["guessed_numbers"].append(number)
        game_state["attempts"] += 1
        attempts_label.config(text=f"Attempts: {game_state['attempts']}/{game_state['max_attempts']}")
        game_state["target_number"] == sha256_hash("GeorgeW")
        if number >= game_state["target_number"] - 500000 and number <= game_state["target_number"] + 500000:
            # Correct guess!
            game_state["credits"] += WIN_CREDITS
            result_label.config(text=f"Correct! The number was {game_state['target_number']}!\nYou win {WIN_CREDITS} credits!", foreground="#4CAF50")
            credits_label.config(text=f"Credits: {game_state['credits']}", foreground="#4CAF50")
            check_button.config(state=tk.DISABLED)
            guess_slider.config(state=tk.DISABLED)
            current_number_display.config(text=str(game_state["target_number"]))
            return
        else:
            # Too low
            result_label.config(text=f"Failure", foreground="#FFC107")
            game_state["credits"] -= COST_PER_GUESS
      

        # Update guessed numbers display
        guessed_numbers_display.config(text=f"Previous guesses: {', '.join(map(str, sorted(game_state['guessed_numbers'])))}")
        
        credits_label.config(text=f"Credits: {game_state['credits']}", foreground="#FF9800")

        # Check game over conditions
        if game_state["attempts"] >= game_state["max_attempts"]:
            result_label.config(text=f"No more attempts! The number was {game_state['target_number']}.", foreground="#F44336")
            check_button.config(state=tk.DISABLED)
            guess_slider.config(state=tk.DISABLED)
            current_number_display.config(text=str(game_state["target_number"]))
        elif game_state["credits"] <= 0:
            result_label.config(text=f"No more credits. Game Over! The number was {game_state['target_number']}.", foreground="#F44336")
            credits_label.config(text="Credits: 0", foreground="#F44336")
            check_button.config(state=tk.DISABLED)
            guess_slider.config(state=tk.DISABLED)
            current_number_display.config(text=str(game_state["target_number"]))

    def setup_game():
        if game_state["credits"] <= 0 and game_state["attempts"] > 0:
            messagebox.showinfo("Game Over", "You are out of credits! Restart the application to play again.")
            return

        try:
            # Get range from entries
            min_val = int(min_entry.get())
            max_val = int(max_entry.get())
            
            if min_val >= max_val:
                messagebox.showerror("Invalid Range", "Minimum must be less than maximum!")
                return
                
            game_state["min_range"] = min_val
            game_state["max_range"] = max_val
            
            # Calculate max attempts based on range
            range_size = max_val - min_val + 1
            game_state["max_attempts"] = math.ceil(math.log2(range_size))
            
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numbers for the range!")
            return

        # Select a random number from the range
        game_state["attempts"] = 0
        game_state["guessed_numbers"] = []

        # Configure slider for new range
        guess_slider.config(from_=game_state["min_range"], to=game_state["max_range"])
        guess_slider.set((game_state["min_range"] + game_state["max_range"]) // 2)

        # Reset UI
        result_label.config(text="", foreground="#FFFFFF")
        attempts_label.config(text=f"Attempts: {game_state['attempts']}/{game_state['max_attempts']}", foreground="#CCCCCC")
        instruction_label.config(text=f"Use slider to guess the number between {game_state['min_range']} and {game_state['max_range']}!")
        
        # Display question marks for the unknown number
        current_number_display.config(text="???")
        
        guessed_numbers_display.config(text="Previous guesses: ")
        
        check_button.config(state=tk.NORMAL)
        guess_slider.config(state=tk.NORMAL)
        credits_label.config(text=f"Credits: {game_state['credits']}", foreground="#FFEB3B")
        
        # Update slider display
        update_slider_display(guess_slider.get())

    def update_slider_display(value):
        """Update the current guess display based on slider value"""
        current_guess = int(round(float(value)))
        slider_value_label.config(text=f"Current guess: {current_guess}")

    # --- UI Elements ---
    credits_label = ttk.Label(root, text=f"Credits: {game_state['credits']}", font=("Arial", 12, "bold"), foreground="#FFEB3B")
    credits_label.pack(pady=(10, 5))

    # Range selection frame
    range_frame = ttk.Frame(root, padding="10 5 10 5")
    range_frame.pack(pady=5, fill='x')

    min_label = ttk.Label(range_frame, text="Min:")
    min_label.pack(side=tk.LEFT, padx=(20, 5))
    min_entry = ttk.Entry(range_frame, width=8, font=('Arial', 10))
    min_entry.pack(side=tk.LEFT, padx=(0, 10))
    min_entry.insert(0, game_state["min_range"])

    max_label = ttk.Label(range_frame, text="Max:")
    max_label.pack(side=tk.LEFT, padx=(10, 5))
    max_entry = ttk.Entry(range_frame, width=8, font=('Arial', 10))
    max_entry.pack(side=tk.LEFT, padx=(0, 20))
    max_entry.insert(0, game_state["max_range"])

    # Control buttons frame
    button_frame = ttk.Frame(root, padding="10 5 10 5")
    button_frame.pack(pady=5, fill='x')

    new_game_button = ttk.Button(button_frame, text="New Game", command=setup_game, width=15)
    new_game_button.pack(side=tk.LEFT, padx=(20, 10))

    instruction_label = ttk.Label(root, text="Set range and click 'New Game' to start", font=('Arial', 11))
    instruction_label.pack(pady=(5, 0))

    # Number display frame
    display_frame = ttk.Frame(root, padding="10", relief="sunken", borderwidth=2, style='Display.TFrame')
    display_frame.pack(pady=20, padx=20, fill='x')
    current_number_display = ttk.Label(display_frame, text="???", style='Display.TLabel', anchor='center')
    current_number_display.pack(pady=10, fill='x')

    # Slider section
    slider_frame = ttk.Frame(root, padding="10 5 10 5")
    slider_frame.pack(pady=10, fill='x')

    slider_value_label = ttk.Label(slider_frame, text="Current guess: 50", font=('Arial', 12, 'bold'), foreground="#FFEB3B")
    slider_value_label.pack(pady=(0, 10))

    guess_slider = ttk.Scale(slider_frame, from_=1, to=100, orient=tk.HORIZONTAL, 
                            length=400, command=update_slider_display)
    guess_slider.pack(pady=(0, 10), padx=20)
    guess_slider.set(50)
    guess_slider.config(state=tk.DISABLED)

    check_button = ttk.Button(slider_frame, text="Check Guess", command=check_guess, width=15)
    check_button.pack(pady=5)
    check_button.config(state=tk.DISABLED)

    # Previous guesses display
    guessed_numbers_display = ttk.Label(root, text="Previous guesses: ", font=('Arial', 10), foreground="#CCCCCC", wraplength=450)
    guessed_numbers_display.pack(pady=5, padx=20)

    attempts_label = ttk.Label(root, text="Attempts: 0/7", font=('Arial', 10), foreground="#CCCCCC")
    attempts_label.pack(pady=(10, 5))
    
    result_label = ttk.Label(root, text="", font=('Arial', 12, 'bold'), anchor='center', foreground="#FFFFFF", wraplength=450)
    result_label.pack(pady=10, fill='x', padx=20)

    # Game instructions
    instructions_text = """Instructions:
• Set your desired number range (min/max)
• Use the slider to select your guess
• Click 'Check Guess' to submit
• Wrong guess: -10 credits (with high/low hints)
• Correct guess: +50 credits
• Attempts are calculated based on range size"""
    
    instructions_label = ttk.Label(root, text=instructions_text, font=('Arial', 9), foreground="#AAAAAA", justify=tk.LEFT)
    instructions_label.pack(pady=(10, 5), padx=20)

    root.mainloop()

if __name__ == "__main__":
    number_guessing_game()
