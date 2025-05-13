import pandas as pd
import numpy as np
import joblib
import os
import time
from command import Command
from buttons import Buttons

class Bot:
    def __init__(self):
        # Original initialization
        self.fire_code = ["<", "!<", "v+<", "!v+!<", "v", "!v", "v+>", "!v+!>", ">+Y", "!>+!Y"]
        self.exe_code = 0
        self.start_fire = True
        self.remaining_code = []
        self.my_command = Command()
        self.buttn = Buttons()
        
        # Game state tracking for adaptive strategy
        self.last_health = 100  # Initial health value to detect damage
        self.defensive_mode = False
        self.defensive_timer = 0
        
        # Data collection for continuous learning
        self.csv_file = "gameplay_data.csv"
        self.initialize_csv()
        
        # Statistics to track ML usage and performance
        self.total_decisions = 0
        self.ml_decisions = 0
        self.rule_based_decisions = 0
        self.start_time = time.time()
        
        # Attempt to load the enhanced model first
        try:
            print("Loading enhanced ML model...")
            self.model = joblib.load('enhanced_mlp_model.joblib')
            self.scaler = joblib.load('enhanced_scaler.joblib')
            
            # Load feature columns
            with open('enhanced_feature_columns.txt', 'r') as f:
                self.feature_columns = [line.strip() for line in f.readlines()]
                
            print("ENHANCED ML MODEL LOADED SUCCESSFULLY")
            print(f"Model has {len(self.feature_columns)} input features")
            self.ml_enabled = True
            self.using_enhanced_model = True
        except Exception as e:
            print(f"Could not load enhanced ML model: {e}")
            print("Trying to load original model...")
            # Fall back to original model if enhanced model fails
            try:
                self.model = joblib.load('mlp_model.joblib')
                self.scaler = joblib.load('scaler.joblib')
                
                with open('feature_columns.txt', 'r') as f:
                    self.feature_columns = [line.strip() for line in f.readlines()]
                    
                print("Original ML MODEL LOADED SUCCESSFULLY")
                print(f"Model has {len(self.feature_columns)} input features")
                self.ml_enabled = True
                self.using_enhanced_model = False
            except Exception as e:
                print(f"ERROR: Could not load any ML model: {e}")
                print("Falling back to rule-based logic only")
                self.ml_enabled = False
                self.using_enhanced_model = False
        
        # Define button columns in correct order for ML prediction
        self.button_columns = ['up', 'down', 'left', 'right', 'A', 'B', 'X', 'Y', 'L', 'R']
    
    def initialize_csv(self):
        """Initialize CSV file for data collection and continuous learning"""
        if not os.path.exists(self.csv_file):
            import csv
            with open(self.csv_file, 'w', newline='') as file:
                writer = csv.writer(file)
                headers = [
                    'p1_character_id', 'p2_character_id',
                    'p1_health', 'p1_x', 'p1_y', 'p1_jumping', 'p1_crouching', 'p1_in_move', 'p1_move_id',
                    'p2_health', 'p2_x', 'p2_y', 'p2_jumping', 'p2_crouching', 'p2_in_move', 'p2_move_id',
                    'x_distance', 'y_distance', 'timer',
                    'up', 'down', 'left', 'right', 'A', 'B', 'X', 'Y', 'L', 'R', 'select', 'start',
                    'player_controlled', 'using_ml', 'using_enhanced_model'
                ]
                writer.writerow(headers)
                print(f"Created new CSV file {self.csv_file} for continued data collection")
    
    def record_data(self, game_state, player, buttons, using_ml=False):
        """Record game state and action data for post-analysis and model improvement"""
        try:
            import csv
            with open(self.csv_file, 'a', newline='') as file:
                writer = csv.writer(file)
                
                p1 = game_state.player1
                p2 = game_state.player2
                x_distance = p2.x_coord - p1.x_coord
                y_distance = p2.y_coord - p1.y_coord
                
                row_data = [
                    # Character IDs
                    p1.player_id, p2.player_id,
                    # Player 1 features
                    p1.health, p1.x_coord, p1.y_coord, 
                    int(p1.is_jumping), int(p1.is_crouching), 
                    int(p1.is_player_in_move), p1.move_id,
                    # Player 2 features
                    p2.health, p2.x_coord, p2.y_coord, 
                    int(p2.is_jumping), int(p2.is_crouching), 
                    int(p2.is_player_in_move), p2.move_id,
                    # Derived features
                    x_distance, y_distance, 
                    game_state.timer,
                    # Button states
                    int(buttons.up), int(buttons.down), 
                    int(buttons.left), int(buttons.right),
                    int(buttons.A), int(buttons.B), 
                    int(buttons.X), int(buttons.Y),
                    int(buttons.L), int(buttons.R),
                    int(buttons.select), int(buttons.start),
                    # Additional info
                    player,
                    int(using_ml),
                    int(self.using_enhanced_model if using_ml else 0)
                ]
                writer.writerow(row_data)
        except Exception as e:
            print(f"Warning: Could not record data: {e}")
    
    def extract_features(self, game_state):
        """Extract and prepare features from game state for ML prediction"""
        p1 = game_state.player1
        p2 = game_state.player2
        
        # Calculate distances between players
        x_distance = p2.x_coord - p1.x_coord
        y_distance = p2.y_coord - p1.y_coord
        
        # Define required numerical features
        numerical_features = ['p1_health', 'p1_x', 'p1_y', 'p1_move_id',
                             'p2_health', 'p2_x', 'p2_y', 'p2_move_id',
                             'x_distance', 'y_distance', 'timer']
        
        # Create numerical features array in the correct order
        numerical_data = np.array([
            p1.health, p1.x_coord, p1.y_coord, p1.move_id,
            p2.health, p2.x_coord, p2.y_coord, p2.move_id,
            x_distance, y_distance, game_state.timer
        ]).reshape(1, -1)  # Reshape for sklearn compatibility
        
        # Apply feature scaling to numerical features
        scaled_numerical = self.scaler.transform(numerical_data)
        
        # Create a dictionary for the final feature set
        features = {}
        
        # Add scaled numerical features
        for i, feat_name in enumerate(numerical_features):
            features[feat_name] = scaled_numerical[0, i]
        
        # Add binary features directly (no scaling needed)
        features['p1_jumping'] = int(p1.is_jumping)
        features['p1_crouching'] = int(p1.is_crouching) 
        features['p1_in_move'] = int(p1.is_player_in_move)
        features['p2_jumping'] = int(p2.is_jumping)
        features['p2_crouching'] = int(p2.is_crouching)
        features['p2_in_move'] = int(p2.is_player_in_move)
        
        # Add character one-hot encoding
        all_p1_chars = list(range(13))  # Assuming character IDs 0-12
        all_p2_chars = list(range(13))
        
        # Add character features using one-hot encoding
        for char_id in all_p1_chars:
            features[f'p1_char_{char_id}'] = 1 if p1.player_id == char_id else 0
            
        for char_id in all_p2_chars:
            features[f'p2_char_{char_id}'] = 1 if p2.player_id == char_id else 0
        
        # Convert to DataFrame with all expected columns
        features_df = pd.DataFrame([features])
        
        # Ensure all columns from training are present
        for col in self.feature_columns:
            if col not in features_df.columns:
                features_df[col] = 0
        
        # Reorder columns to match training data
        features_df = features_df[self.feature_columns]
        
        return features_df
    
    def check_health_change(self, current_health):
        """
        Monitors health changes and adjusts fighting strategy accordingly.
        Returns True if damage was taken.
        """
        health_change = self.last_health - current_health
        self.last_health = current_health
        
        # If taking significant damage, go defensive
        if health_change > 5:
            self.defensive_mode = True
            self.defensive_timer = 20  # Stay defensive for 20 frames
            return True
        
        # Update defensive timer
        if self.defensive_timer > 0:
            self.defensive_timer -= 1
            if self.defensive_timer == 0:
                self.defensive_mode = False
        
        return health_change > 0
        
    def fight(self, current_game_state, player):
        """
        Main fighting method - implements ML-based combat decisions with adaptive strategy.
        Falls back to rule-based logic if ML is unavailable or fails.
        """
        self.total_decisions += 1
        
        # Reset buttons before making a decision
        self.buttn = Buttons()
        
        # Print stats every 100 decisions
        if self.total_decisions % 100 == 0:
            elapsed_time = time.time() - self.start_time
            ml_percentage = (self.ml_decisions / self.total_decisions * 100) if self.total_decisions > 0 else 0
            print(f"\n----- BOT STATS AFTER {self.total_decisions} DECISIONS ({elapsed_time:.1f} sec) -----")
            print(f"ML Decisions: {self.ml_decisions} ({ml_percentage:.1f}%)")
            print(f"Rule-based Decisions: {self.rule_based_decisions} ({100-ml_percentage:.1f}%)")
            if self.ml_enabled:
                print(f"Using {'Enhanced' if self.using_enhanced_model else 'Original'} ML Model")
            print("----------------------------------------------------\n")
        
        if self.ml_enabled:
            try:
                # Extract features for ML prediction
                features = self.extract_features(current_game_state)
                
                # Get player positions and stats
                p1 = current_game_state.player1
                p2 = current_game_state.player2
                x_diff = p2.x_coord - p1.x_coord
                dist = abs(x_diff)
                
                # Check if player took damage and adjust strategy
                took_damage = self.check_health_change(p1.health)
                
                # Make prediction with the model
                button_preds = self.model.predict(features)[0]
                
                # Initialize button activations
                pressed = []
                
                # COMBAT STRATEGY BASED ON DISTANCE AND HEALTH
                
                # LOW HEALTH STRATEGY - prioritize defensive play
                if p1.health < 30:
                    # Keep distance and play defensively
                    if dist < 80:
                        # Back away from opponent
                        if x_diff > 0:
                            self.buttn.left = True
                            pressed.append("left")
                        else:
                            self.buttn.right = True
                            pressed.append("right")
                            
                        # Block if opponent is attacking
                        if p2.is_player_in_move and np.random.random() < 0.7:
                            self.buttn.R = True
                            pressed.append("R")
                    else:
                        # Use projectiles from a distance
                        if np.random.random() < 0.3:
                            # Execute fireball motion
                            self.buttn.down = True
                            self.buttn.right = True
                            self.buttn.Y = True
                            pressed.extend(["down", "right", "Y"])
                        else:
                            # Move cautiously to maintain distance
                            if x_diff > 0:
                                self.buttn.left = True
                                pressed.append("left")
                            else:
                                self.buttn.right = True
                                pressed.append("right")
                
                # DEFENSIVE MODE (activated after taking damage)
                elif self.defensive_mode:
                    # Higher chance to block
                    if np.random.random() < 0.6:
                        self.buttn.R = True
                        pressed.append("R")
                    
                    # Move away from opponent
                    if x_diff > 0:
                        self.buttn.left = True
                        pressed.append("left")
                    else:
                        self.buttn.right = True
                        pressed.append("right")
                    
                    # Occasional counter-attack
                    if np.random.random() < 0.2:
                        self.buttn.Y = True
                        pressed.append("Y")
                
                # CLOSE RANGE COMBAT STRATEGY
                elif dist < 60:
                    # Be more aggressive when opponent has low health
                    attack_chance = 0.7 if p2.health < 50 else 0.5
                    
                    if np.random.random() < attack_chance:
                        # Choose an attack type based on situation
                        attack_type = np.random.choice(["normal", "special", "throw"])
                        
                        if attack_type == "normal":
                            # Regular attacks
                            if np.random.random() < 0.6:
                                self.buttn.Y = True  # Punch
                                pressed.append("Y")
                            else:
                                self.buttn.B = True  # Kick
                                pressed.append("B")
                                
                            # Add directional input for different attack variants
                            if np.random.random() < 0.4:
                                if np.random.random() < 0.5:
                                    self.buttn.down = True  # Crouching attack
                                    pressed.append("down")
                                else:
                                    if x_diff > 0:
                                        self.buttn.right = True  # Forward attack
                                        pressed.append("right")
                                    else:
                                        self.buttn.left = True  # Forward attack
                                        pressed.append("left")
                        
                        elif attack_type == "special" and np.random.random() < 0.25:
                            # Special moves based on distance
                            if dist < 40:
                                # Dragon punch (Shoryuken) - anti-air and reversal
                                self.buttn.right = True
                                self.buttn.down = True
                                self.buttn.Y = True
                                pressed.extend(["right", "down", "Y"])
                            else:
                                # Fireball (Hadouken) - zoning tool
                                self.buttn.down = True
                                self.buttn.right = True
                                self.buttn.Y = True
                                pressed.extend(["down", "right", "Y"])
                        
                        elif attack_type == "throw":
                            # Throw attempt when very close
                            self.buttn.L = True
                            if x_diff > 0:
                                self.buttn.right = True
                                pressed.extend(["L", "right"])
                            else:
                                self.buttn.left = True
                                pressed.extend(["L", "left"])
                    else:
                        # Defensive options
                        if np.random.random() < 0.4:
                            self.buttn.R = True  # Block
                            pressed.append("R")
                        else:
                            # Position better for counterattack
                            if np.random.random() < 0.6:
                                if x_diff > 0:
                                    self.buttn.right = True
                                    pressed.append("right")
                                else:
                                    self.buttn.left = True
                                    pressed.append("left")
                            else:
                                # Jump to avoid sweeps
                                self.buttn.up = True
                                pressed.append("up")
                
                # MID RANGE COMBAT STRATEGY
                elif dist < 120:
                    # Mix of approach and zoning tactics
                    if np.random.random() < 0.3:
                        # Fireball from mid-range (zoning)
                        self.buttn.down = True
                        self.buttn.right = True
                        self.buttn.Y = True
                        pressed.extend(["down", "right", "Y"])
                    
                    # Approach strategies
                    else:
                        # Move toward opponent
                        if np.random.random() < 0.7:
                            if x_diff > 0:
                                self.buttn.right = True
                                pressed.append("right")
                            else:
                                self.buttn.left = True
                                pressed.append("left")
                            
                            # Jump attack sometimes (jump-in)
                            if np.random.random() < 0.3:
                                self.buttn.up = True
                                pressed.append("up")
                                
                                # Attack button for jump-in
                                if np.random.random() < 0.6:
                                    self.buttn.Y = True
                                    pressed.append("Y")
                        else:
                            # Defensive stance
                            if np.random.random() < 0.5:
                                self.buttn.down = True  # Crouch (avoid projectiles)
                                pressed.append("down")
                            else:
                                self.buttn.R = True  # Block
                                pressed.append("R")
                
                # LONG RANGE COMBAT STRATEGY
                else:
                    # Approach or zoning based on game state
                    if np.random.random() < 0.3 and dist > 150:
                        # Fireball to control space
                        self.buttn.down = True
                        self.buttn.right = True
                        self.buttn.Y = True
                        pressed.extend(["down", "right", "Y"])
                    else:
                        # Approach quickly to close distance
                        if x_diff > 0:
                            self.buttn.right = True
                            pressed.append("right")
                        else:
                            self.buttn.left = True
                            pressed.append("left")
                        
                        # Jump to close distance faster
                        if np.random.random() < 0.3:
                            self.buttn.up = True
                            pressed.append("up")
                
                # Ensure we don't have contradictory inputs
                if self.buttn.left and self.buttn.right:
                    if np.random.random() < 0.5:
                        self.buttn.left = False
                        if "left" in pressed:
                            pressed.remove("left")
                    else:
                        self.buttn.right = False
                        if "right" in pressed:
                            pressed.remove("right")
                
                if self.buttn.up and self.buttn.down:
                    if np.random.random() < 0.5:
                        self.buttn.up = False
                        if "up" in pressed:
                            pressed.remove("up")
                    else:
                        self.buttn.down = False
                        if "down" in pressed:
                            pressed.remove("down")
                
                # If no buttons were selected, ensure at least one button is pressed
                if not pressed:
                    button_to_press = np.random.choice(self.button_columns)
                    if button_to_press == 'up': self.buttn.up = True
                    elif button_to_press == 'down': self.buttn.down = True
                    elif button_to_press == 'left': self.buttn.left = True
                    elif button_to_press == 'right': self.buttn.right = True
                    elif button_to_press == 'A': self.buttn.A = True
                    elif button_to_press == 'B': self.buttn.B = True
                    elif button_to_press == 'X': self.buttn.X = True
                    elif button_to_press == 'Y': self.buttn.Y = True
                    elif button_to_press == 'L': self.buttn.L = True
                    elif button_to_press == 'R': self.buttn.R = True
                    pressed.append(button_to_press)
                
                # Update stats
                self.ml_decisions += 1
                
                # Print ML action
                model_type = "ENHANCED" if self.using_enhanced_model else "STANDARD"
                print(f"ML ACTION: {' + '.join(pressed) if pressed else 'None'}")
                
                # Set the command
                if player == "1":
                    self.my_command.player_buttons = self.buttn
                else:
                    self.my_command.player2_buttons = self.buttn
                
                # Record data for potential continued learning
                self.record_data(current_game_state, player, self.buttn, using_ml=True)
                
                return self.my_command
                
            except Exception as e:
                print(f"ERROR in ML prediction: {e}")
                print("Falling back to rule-based logic for this decision")
                # Fall back to rule-based logic if ML fails
                return self.rule_based_fight(current_game_state, player)
        else:
            return self.rule_based_fight(current_game_state, player)
    
    def rule_based_fight(self, current_game_state, player):
        """
        Original rule-based fighting logic as a fallback when ML is unavailable.
        Implements predefined sequences of moves based on player distance.
        """
        # Update stats
        self.rule_based_decisions += 1
        print("USING RULE-BASED LOGIC")
        
        # Reset buttons before making a decision
        self.buttn = Buttons()
        
        if player == "1":
            if self.exe_code != 0:
                self.run_command([], current_game_state.player1)
            diff = current_game_state.player2.x_coord - current_game_state.player1.x_coord
            if diff > 60:
                toss = np.random.randint(3)
                if toss == 0:
                    self.run_command([">", "-", "!>", "v+>", "-", "!v+!>", "v", "-", "!v", "v+<", "-", "!v+!<", "<+Y", "-", "!<+!Y"], current_game_state.player1)
                elif toss == 1:
                    self.run_command([">+^+B", ">+^+B", "!>+!^+!B"], current_game_state.player1)
                else:  # fire
                    self.run_command(["<", "-", "!<", "v+<", "-", "!v+!<", "v", "-", "!v", "v+>", "-", "!v+!>", ">+Y", "-", "!>+!Y"], current_game_state.player1)
            elif diff < -60:
                toss = np.random.randint(3)
                if toss == 0:  # spinning
                    self.run_command(["<", "-", "!<", "v+<", "-", "!v+!<", "v", "-", "!v", "v+>", "-", "!v+!>", ">+Y", "-", "!>+!Y"], current_game_state.player1)
                elif toss == 1:
                    self.run_command(["<+^+B", "<+^+B", "!<+!^+!B"], current_game_state.player1)
                else:  # fire
                    self.run_command([">", "-", "!>", "v+>", "-", "!v+!>", "v", "-", "!v", "v+<", "-", "!v+!<", "<+Y", "-", "!<+!Y"], current_game_state.player1)
            else:
                toss = np.random.randint(2)
                if toss >= 1:
                    if diff > 0:
                        self.run_command(["<", "<", "!<"], current_game_state.player1)
                    else:
                        self.run_command([">", ">", "!>"], current_game_state.player1)
                else:
                    self.run_command(["v+R", "v+R", "v+R", "!v+!R"], current_game_state.player1)
                    
            self.my_command.player_buttons = self.buttn
            
        elif player == "2":
            if self.exe_code != 0:
                self.run_command([], current_game_state.player2)
            diff = current_game_state.player1.x_coord - current_game_state.player2.x_coord
            if diff > 60:
                toss = np.random.randint(3)
                if toss == 0:
                    self.run_command([">", "-", "!>", "v+>", "-", "!v+!>", "v", "-", "!v", "v+<", "-", "!v+!<", "<+Y", "-", "!<+!Y"], current_game_state.player2)
                elif toss == 1:
                    self.run_command([">+^+B", ">+^+B", "!>+!^+!B"], current_game_state.player2)
                else:
                    self.run_command(["<", "-", "!<", "v+<", "-", "!v+!<", "v", "-", "!v", "v+>", "-", "!v+!>", ">+Y", "-", "!>+!Y"], current_game_state.player2)
            elif diff < -60:
                toss = np.random.randint(3)
                if toss == 0:
                    self.run_command(["<", "-", "!<", "v+<", "-", "!v+!<", "v", "-", "!v", "v+>", "-", "!v+!>", ">+Y", "-", "!>+!Y"], current_game_state.player2)
                elif toss == 1:
                    self.run_command(["<+^+B", "<+^+B", "!<+!^+!B"], current_game_state.player2)
                else:
                    self.run_command([">", "-", "!>", "v+>", "-", "!v+!>", "v", "-", "!v", "v+<", "-", "!v+!<", "<+Y", "-", "!<+!Y"], current_game_state.player2)
            else:
                toss = np.random.randint(2)
                if toss >= 1:
                    if diff < 0:
                        self.run_command(["<", "<", "!<"], current_game_state.player2)
                    else:
                        self.run_command([">", ">", "!>"], current_game_state.player2)
                else:
                    self.run_command(["v+R", "v+R", "v+R", "!v+!R"], current_game_state.player2)
                    
            self.my_command.player2_buttons = self.buttn
            
        # Record the state and action
        self.record_data(current_game_state, player, self.buttn, using_ml=False)
        
        return self.my_command

    def run_command(self, com, player):
        """
        Executes a sequence of button commands for rule-based fighting.
        Handles special moves and button combinations according to preset patterns.
        """
        if self.exe_code-1 == len(self.fire_code):
            self.exe_code = 0
            self.start_fire = False
            print("complete")
        elif len(self.remaining_code) == 0:
            self.fire_code = com
            self.exe_code += 1
            self.remaining_code = self.fire_code[0:]
        else:
            self.exe_code += 1
            # Down + Left combination
            if self.remaining_code[0] == "v+<":
                self.buttn.down = True
                self.buttn.left = True
                print("v+<")
            elif self.remaining_code[0] == "!v+!<":
                self.buttn.down = False
                self.buttn.left = False
                print("!v+!<")
            # Down + Right combination
            elif self.remaining_code[0] == "v+>":
                self.buttn.down = True
                self.buttn.right = True
                print("v+>")
            elif self.remaining_code[0] == "!v+!>":
                self.buttn.down = False
                self.buttn.right = False
                print("!v+!>")
            # Right + Y combination
            elif self.remaining_code[0] == ">+Y":
                self.buttn.Y = True
                self.buttn.right = True
                print(">+Y")
            elif self.remaining_code[0] == "!>+!Y":
                self.buttn.Y = False
                self.buttn.right = False
                print("!>+!Y")
            # Left + Y combination
            elif self.remaining_code[0] == "<+Y":
                self.buttn.Y = True
                self.buttn.left = True
                print("<+Y")
            elif self.remaining_code[0] == "!<+!Y":
                self.buttn.Y = False
                self.buttn.left = False
                print("!<+!Y")
            # Right + Up + L combination
            elif self.remaining_code[0] == ">+^+L":
                self.buttn.right = True
                self.buttn.up = True
                self.buttn.L = not (player.player_buttons.L)
                print(">+^+L")
            elif self.remaining_code[0] == "!>+!^+!L":
                self.buttn.right = False
                self.buttn.up = False
                self.buttn.L = False
                print("!>+!^+!L")
            # Right + Up + Y combination
            elif self.remaining_code[0] == ">+^+Y":
                self.buttn.right = True
                self.buttn.up = True
                self.buttn.Y = not (player.player_buttons.Y)
                print(">+^+Y")
            elif self.remaining_code[0] == "!>+!^+!Y":
                self.buttn.right = False
                self.buttn.up = False
                self.buttn.Y = False
                print("!>+!^+!Y")
            # Right + Up + R combination
            elif self.remaining_code[0] == ">+^+R":
                self.buttn.right = True
                self.buttn.up = True
                self.buttn.R = not (player.player_buttons.R)
                print(">+^+R")
            elif self.remaining_code[0] == "!>+!^+!R":
                self.buttn.right = False
                self.buttn.up = False
                self.buttn.R = False
                print("!>+!^+!R")
            # Right + Up + A combination
            elif self.remaining_code[0] == ">+^+A":
                self.buttn.right = True
                self.buttn.up = True
                self.buttn.A = not (player.player_buttons.A)
                print(">+^+A")
            elif self.remaining_code[0] == "!>+!^+!A":
                self.buttn.right = False
                self.buttn.up = False
                self.buttn.A = False
                print("!>+!^+!A")
            # Right + Up + B combination
            elif self.remaining_code[0] == ">+^+B":
                self.buttn.right = True
                self.buttn.up = True
                self.buttn.B = not (player.player_buttons.B)
                print(">+^+B")
            elif self.remaining_code[0] == "!>+!^+!B":
                self.buttn.right = False
                self.buttn.up = False
                self.buttn.B = False
                print("!>+!^+!B")
            # Left + Up + L combination
            elif self.remaining_code[0] == "<+^+L":
                self.buttn.left = True
                self.buttn.up = True
                self.buttn.L = not (player.player_buttons.L)
                print("<+^+L")
            elif self.remaining_code[0] == "!<+!^+!L":
                self.buttn.left = False
                self.buttn.up = False
                self.buttn.L = False
                print("!<+!^+!L")
            # Left + Up + Y combination
            elif self.remaining_code[0] == "<+^+Y":
                self.buttn.left = True
                self.buttn.up = True
                self.buttn.Y = not (player.player_buttons.Y)
                print("<+^+Y")
            elif self.remaining_code[0] == "!<+!^+!Y":
                self.buttn.left = False
                self.buttn.up = False
                self.buttn.Y = False
                print("!<+!^+!Y")
            # Left + Up + R combination
            elif self.remaining_code[0] == "<+^+R":
                self.buttn.left = True
                self.buttn.up = True
                self.buttn.R = not (player.player_buttons.R)
                print("<+^+R")
            elif self.remaining_code[0] == "!<+!^+!R":
                self.buttn.left = False
                self.buttn.up = False
                self.buttn.R = False
                print("!<+!^+!R")
            # Left + Up + A combination
            elif self.remaining_code[0] == "<+^+A":
                self.buttn.left = True
                self.buttn.up = True
                self.buttn.A = not (player.player_buttons.A)
                print("<+^+A")
            elif self.remaining_code[0] == "!<+!^+!A":
                self.buttn.left = False
                self.buttn.up = False
                self.buttn.A = False
                print("!<+!^+!A")
            # Left + Up + B combination
            elif self.remaining_code[0] == "<+^+B":
                self.buttn.left = True
                self.buttn.up = True
                self.buttn.B = not (player.player_buttons.B)
                print("<+^+B")
            elif self.remaining_code[0] == "!<+!^+!B":
                self.buttn.left = False
                self.buttn.up = False
                self.buttn.B = False
                print("!<+!^+!B")
            # Down + R combination (blocking low)
            elif self.remaining_code[0] == "v+R":
                self.buttn.down = True
                self.buttn.R = not (player.player_buttons.R)
                print("v+R")
            elif self.remaining_code[0] == "!v+!R":
                self.buttn.down = False
                self.buttn.R = False
                print("!v+!R")
            # Single button commands
            else:
                if self.remaining_code[0] == "v":
                    self.buttn.down = True
                    print("down")
                elif self.remaining_code[0] == "!v":
                    self.buttn.down = False
                    print("Not down")
                elif self.remaining_code[0] == "<":
                    print("left")
                    self.buttn.left = True
                elif self.remaining_code[0] == "!<":
                    print("Not left")
                    self.buttn.left = False
                elif self.remaining_code[0] == ">":
                    print("right")
                    self.buttn.right = True
                elif self.remaining_code[0] == "!>":
                    print("Not right")
                    self.buttn.right = False
                elif self.remaining_code[0] == "^":
                    print("up")
                    self.buttn.up = True
                elif self.remaining_code[0] == "!^":
                    print("Not up")
                    self.buttn.up = False
                elif self.remaining_code[0] == "-":
                    # This is a pause/no-op for timing
                    pass
                    
            self.remaining_code = self.remaining_code[1:]
        return