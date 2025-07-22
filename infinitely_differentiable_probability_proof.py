

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox  # Import messagebox for error popups
import random
import math  # Import math for calculating max_attempts
# Removed time import as it's not used
import time
import datetime
import itertools

path_threshold = 9
with open("x.txt", 'r', encoding='utf-8') as file:
    # Read, lower, and split robustly
    data = file.readlines()

now = datetime.datetime.now()


# Constants
STARTING_CREDITS = 100
COST_PER_GUESS = 10
WIN_CREDITS = 50
def number_guessing_game():
    """
    Creates a Tkinter window for a number guessing game with a variable range,
    styled with a dark theme and includes a credit system.
    """
    # Initialize Tkinter window
    root = tk.Tk()
    root.title("Guess the Number")
    root.geometry("700x750")  # Adjusted size for styling
    root.resizable(False, False)
    root.configure(bg="#2E2E2E")  # Dark background for the main window

    # --- Style Configuration ---
    style = ttk.Style()
    try:
        # Attempt to use a theme that allows more customization
        style.theme_use('clam')
    except tk.TclError:
        print("Clam theme not available, using default.") # Fallback if 'clam' isn't available

    # Configure styles for various widgets
    style.configure('.', background='#2E2E2E', foreground='white')  # Global style
    style.configure('TFrame', background='#2E2E2E')
    style.configure('TLabel', background='#2E2E2E', foreground='white', font=('Arial', 10))
    style.configure('TButton', background='#4A4A4A', foreground='white', font=('Arial', 11, 'bold'), borderwidth=0, padding=5) # Added padding
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
    style.configure('Display.TLabel', background='#1A1A1A', foreground="#FFEB3B", font=('Consolas', 36, 'bold')) # Yellow, large font

    # --- Game State Variables ---
    # Using a dictionary to hold game state makes it easier to manage
    game_state = {
        "target_number": random.randint(0,99),
        "attempts": 0,
        "min_value": 1,
        "max_value": 100,
        "max_attempts": 0,
        "credits": STARTING_CREDITS,
    }

    # --- Game Logic Functions ---
    def check_guess():
        """ Checks the player's guess against the target number and updates the game state. """
        target_number = game_state["target_number"]
        max_attempts = game_state["max_attempts"]

        # Prevent checking if game is already over (no credits or attempts)
        if check_button['state'] == tk.DISABLED:
             return

        try:
            # Get integer value from the slider
            guess = int(round(slider.get()))

            # --- Update attempts and display ---
            game_state["attempts"] += 1
            attempts_label.config(text=f"Attempts: {game_state['attempts']}/{max_attempts}")

            # --- Check the guess ---
            if game_state['attempts'] < max_attempts and guess >= target_number - 25 and guess <= target_number + 25:

                game_state["credits"] += WIN_CREDITS
                result_label.config(text=f"You guessed the range correctly in {game_state['attempts']} tries! The number was {target_number}. You win {WIN_CREDITS} credits!", foreground="#4CAF50") # Green
                credits_label.config(text=f"Credits: {game_state['credits']}", foreground="#4CAF50")
                check_button.config(state=tk.DISABLED)
                slider.config(state=tk.DISABLED)
                return # Exit function early on correct guess

            elif game_state['attempts'] < max_attempts and  guess < target_number - 5:
                game_state["credits"] -= COST_PER_GUESS
                # Corrected: Use foreground instead of fg
                credits_label.config(text=f"Credits: {game_state['credits']}", foreground="#FF9800")

            elif game_state['attempts'] < max_attempts and  guess > target_number + 5:
                game_state["credits"] -= COST_PER_GUESS
                # Corrected: Use foreground instead of fg
                credits_label.config(text=f"Credits: {game_state['credits']}", foreground="#FF9800")

            # --- Check for game over conditions (AFTER checking the guess) ---
            # Check attempts first
            if game_state["attempts"] >= max_attempts and guess != target_number: # Added check to ensure it wasn't the winning guess
                result_label.config(text=f"No more attempts! The number was {target_number}.", foreground="#F44336")  # Red
                check_button.config(state=tk.DISABLED)
                slider.config(state=tk.DISABLED)
            # Then check credits (could run out on the last attempt)
            elif game_state["credits"] <= 0: # Use elif to avoid overwriting the 'no attempts' message if both happen
                result_label.config(text="You have no more credits. Game Over!", foreground="#F44336") # Red
                credits_label.config(text="Credits: 0", foreground="#F44336") # Ensure credits show 0
                check_button.config(state=tk.DISABLED)
                slider.config(state=tk.DISABLED)


        except ValueError:
            # This shouldn't happen with the slider, but good practice
            result_label.config(text="Invalid input (should not occur).", foreground="#F44336") # Red

    def setup_game():
        """ Sets up/Resets the game state and range based on entry fields. """
        # Check if player has credits to start a new game
        if game_state["credits"] <= 0 and game_state["attempts"] > 0: # Check attempts > 0 to allow initial setup
             messagebox.showinfo("Game Over", "You are out of credits! Restart the application to play again.")
             # Optionally disable range setting if out of credits
             # min_entry.config(state=tk.DISABLED)
             # max_entry.config(state=tk.DISABLED)
             # reset_button.config(state=tk.DISABLED)
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
                return # Stop setup if range is invalid

            # Update game state with new valid range
            game_state["min_value"] = temp_min
            game_state["max_value"] = temp_max

            # Calculate max attempts based on the new range
            range_size = game_state["max_value"] - game_state["min_value"] + 1
            if range_size <= 1: # Handle edge case where range is 1 or less
                game_state["max_attempts"] = 1
            else:
                 # Optimal number of guesses using binary search idea
                game_state["max_attempts"] = math.ceil(math.log2(range_size))

            # Configure slider and reset game variables
            slider.config(from_=game_state["min_value"], to=game_state["max_value"], state=tk.NORMAL)
            slider.set((game_state["min_value"] + game_state["max_value"]) // 2)  # Center the slider
            game_state["target_number"] = random.randint(game_state["min_value"], game_state["max_value"])
            game_state["attempts"] = 0

            # Reset UI elements
            result_label.config(text="", foreground="#FFFFFF") # Clear result
            attempts_label.config(text=f"Attempts: {game_state['attempts']}/{game_state['max_attempts']}", foreground="#CCCCCC")
            instruction_label.config(text=f"Guess between {game_state['min_value']} and {game_state['max_value']}")
            max_attempts_info_label.config(text=f"({game_state['max_attempts']} attempts)")
            update_guess_display(slider.get()) # Update display with slider's initial value
            check_button.config(state=tk.NORMAL) # Enable check button
            # Update credits label (color might have been red/green)
            credits_label.config(text=f"Credits: {game_state['credits']}", foreground="#FFEB3B") # Reset to default yellow


        except ValueError:
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
        except ValueError:
            current_guess_display.config(text="--") # Fallback display

    # --- UI Elements ---

    # Credit Label (Top)
    credits_label = ttk.Label(root, text=f"Credits: {game_state['credits']}", font=("Arial", 12, "bold"), foreground="#FFEB3B") # Make credits stand out
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
    current_guess_display = ttk.Label(display_frame, text="--", style='Display.TLabel', anchor='center') # Start with placeholder
    current_guess_display.pack(pady=10, fill='x')

    # --- Slider ---
    # Link slider movement directly to the display update function
    slider = ttk.Scale(root, from_=game_state["min_value"], to=game_state["max_value"],
                       orient=tk.HORIZONTAL, length=400,
                       command=update_guess_display)
    slider.pack(pady=15, padx=30)
    slider.config(state=tk.DISABLED) # Initially disabled until range is set

    # --- Check Button ---
    check_button = ttk.Button(root, text="Check Guess", command=check_guess, width=15)
    check_button.pack(pady=5)
    check_button.config(state=tk.DISABLED) # Initially disabled

    # --- Attempts and Result Labels ---
    attempts_label = ttk.Label(root, text="Attempts: 0/0", font=('Arial', 10), foreground="#CCCCCC")
    attempts_label.pack(pady=(5, 0))

    result_label = ttk.Label(root, text="", font=('Arial', 12, 'bold'), anchor='center', foreground="#FFFFFF", wraplength=450) # Allow wrapping for longer messages
    result_label.pack(pady=10, fill='x', padx=20)


    # --- Initial Game Setup ---
    # Call setup_game initially to set the default range (1-100) and prepare the game
    setup_game()

    # Start the Tkinter event loop
    root.mainloop()

# --- Run the game ---
if __name__ == "__main__":
    number_guessing_game()
