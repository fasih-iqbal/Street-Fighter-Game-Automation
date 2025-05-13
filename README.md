# GameBot Street Fighter II Turbo API

A bot development framework for automating gameplay in **Street Fighter II Turbo (U)** using Python. This project allows you to create and test your own fighting bots that interact with the game through real-time emulator integration.

---

## 🛠️ Prerequisites / Dependencies

* **Operating System**: Windows 7 or above (64-bit)
* **Python API**: Developed and tested with Python 3.6.3 (expected to work with Python 3+ with minor changes)

---

## 🚀 Getting Started

### Single or Two-Player Bot Setup

1. Navigate to the appropriate folder:

   * `single-player/` for your bot vs CPU
   * `two-players/` for bot vs bot

2. Launch the emulator:

   * Run `EmuHawk.exe`

3. Load the game:

   * Go to **File > Open ROM** (`Ctrl+O`)
   * Select `Street Fighter II Turbo (U).smc`

4. Open Tool Box:

   * Go to **Tools > Tool Box** (`Shift+T`)

5. Leave the emulator and tool box open. Open Command Prompt in the API directory and run:

   **For Python API**:

   ```bash
   python controller.py 1
   ```
   
   > 🔹 `1` is the player number. Use `1` for player 1 (left) and `2` for player 2 (right). Any other value will throw an error.

6. In the game, select your character(s) and start in **Normal Mode**. Configure controls via **Config > Controllers**.

7. Click the second icon in the top row (Gyroscope Bot) to connect the bot.

8. A successful connection will display `Connected to the game!` or `CONNECTED SUCCESSFULLY` in the terminal.

9. Once a round finishes, the bot stops. Repeat the above process for the next match.

---

### `Buttons` Class

Represents a SNES gamepad with 12 buttons (`up`, `down`, `left`, `right`, `A`, `B`, `X`, `Y`, `L`, `R`, `start`, `select`). Each button is a boolean representing its state (pressed or not).

### `Player` Class

Represents an individual player with attributes:

* `player_id`: ID of the selected character (0–11)
* `health`: Remaining health
* `x_coord`, `y_coord`: Player coordinates
* `is_jumping`, `is_crouching`: Player state
* `player_buttons`: `Buttons` object for current input
* `in_move`: Whether the player is performing a move
* `move_id`: ID of the current move

### `GameState` Class

Captures the game snapshot at a given time:

* `player1`, `player2`: `Player` objects
* `timer`: Time left in the round (max 100 seconds)
* `fight_result`: Outcome of the round (if ended)
* `has_round_started`, `is_round_over`: Round status

### `Command` Class

Represents the output from your bot—a set of buttons to press in the next game frame. This is how your bot interacts with the game.

---

## 🧠 What You Need to Do

Implement the `fight()` function in either:

* `bot.py` (for Python)

This function receives:

* `GameState` object with current game info
* A string indicating which player you're controlling (`"1"` or `"2"`)

You must return a `Command` object with the desired button inputs for the next frame.

📌 **Example** (in the bot file):
The starter code always presses the 'up' button, causing the bot to jump continuously.

---

## 🤖 Running Two Bots (Optional)

To make two of your bots fight each other:

1. Open two terminal windows.
2. Run the controller script with arguments `1` and `2` respectively:

   ```bash
   python controller.py 1
   python controller.py 2
   ```
3. In the game, select two characters and start **VS Battle Mode**.

---

## ❓ Troubleshooting

* Make sure `EmuHawk.exe` is open before running the bot.
* Use only `1` or `2` as command-line arguments.
* Ensure the ROM file is correctly loaded from the proper directory.
* Exception handling is minimal—read terminal messages carefully.

---
