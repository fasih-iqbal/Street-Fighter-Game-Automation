"""
AI Helper module for Street Fighter II Bot

This module contains helper functions for the machine learning components
of the Street Fighter II bot. It provides functionality for data processing,
model training, and evaluation.
"""

import numpy as np
import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_model(data_file="fighter_training_data.csv", output_model="fighter_model.pkl", output_scaler="fighter_scaler.pkl"):
    """
    Train a machine learning model on the collected gameplay data
    
    Args:
        data_file: Path to the CSV file with training data
        output_model: Path to save the trained model
        output_scaler: Path to save the feature scaler
        
    Returns:
        True if training was successful, False otherwise
    """
    try:
        print(f"Training model using data from {data_file}")
        
        # Check if data file exists
        if not os.path.exists(data_file):
            print(f"Error: Data file {data_file} not found")
            return False
        
        # Load data
        print("Loading data...")
        data = pd.read_csv(data_file)
        print(f"Loaded {len(data)} samples")
        
        # Handle missing values
        # For numeric columns, fill with median
        numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            data[col] = data[col].fillna(data[col].median())
        
        # Identify button columns
        button_columns = [col for col in data.columns if col.startswith("button_")]
        feature_columns = [col for col in data.columns if col not in button_columns]
        
        print(f"Feature columns: {len(feature_columns)}")
        print(f"Button columns: {len(button_columns)}")
        
        # Split features and targets
        X = data[feature_columns]
        y = data[button_columns]
        
        # Split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest model
        print("Training Random Forest model...")
        model = RandomForestClassifier(
            n_estimators=100,  # Number of trees
            max_depth=15,      # Maximum depth of trees
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1  # Use all available processors
        )
        model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        print("Evaluating model...")
        y_pred = model.predict(X_test_scaled)
        
        # Calculate accuracy for each button
        print("\nButton-wise accuracy:")
        for i, button in enumerate(button_columns):
            acc = accuracy_score(y_test[button], y_pred[:, i])
            print(f"{button}: {acc:.4f}")
        
        # Calculate overall accuracy
        overall_acc = accuracy_score(y_test.values.astype(bool), y_pred.astype(bool))
        print(f"\nOverall accuracy: {overall_acc:.4f}")
        
        # Save model and scaler
        joblib.dump(model, output_model)
        joblib.dump(scaler, output_scaler)
        print(f"Model saved to {output_model}")
        print(f"Scaler saved to {output_scaler}")
        
        return True
        
    except Exception as e:
        print(f"Error training model: {e}")
        return False

def analyze_data(data_file="fighter_training_data.csv"):
    """
    Analyze the collected gameplay data and print statistics
    
    Args:
        data_file: Path to the CSV file with gameplay data
    """
    try:
        # Check if data file exists
        if not os.path.exists(data_file):
            print(f"Error: Data file {data_file} not found")
            return
        
        # Load data
        print(f"Analyzing data from {data_file}")
        data = pd.read_csv(data_file)
        print(f"Loaded {len(data)} samples")
        
        # Basic statistics
        print("\nData summary:")
        print(f"Number of samples: {len(data)}")
        print(f"Number of features: {len(data.columns)}")
        
        # Identify button columns
        button_columns = [col for col in data.columns if col.startswith("button_")]
        feature_columns = [col for col in data.columns if col not in button_columns]
        
        # Analyze button press distributions
        print("\nButton press distributions:")
        for button in button_columns:
            press_count = data[button].sum()
            press_percent = (press_count / len(data)) * 100
            print(f"{button}: {press_count} presses ({press_percent:.2f}%)")
        
        # Analyze health distributions
        print("\nHealth statistics:")
        print(f"Average player health: {data['my_health'].mean():.2f}")
        print(f"Average opponent health: {data['opponent_health'].mean():.2f}")
        
        # Analyze distance between players
        if 'abs_distance_x' in data.columns:
            print("\nDistance statistics:")
            print(f"Average distance between players: {data['abs_distance_x'].mean():.2f}")
            print(f"Maximum distance recorded: {data['abs_distance_x'].max():.2f}")
            print(f"Minimum distance recorded: {data['abs_distance_x'].min():.2f}")
        
        # Check for missing values
        missing_values = data.isnull().sum()
        if missing_values.sum() > 0:
            print("\nMissing values detected:")
            print(missing_values[missing_values > 0])
        else:
            print("\nNo missing values detected.")
            
    except Exception as e:
        print(f"Error analyzing data: {e}")

def predict_buttons(features, model, scaler):
    """
    Make button press predictions using the trained model
    
    Args:
        features: Dictionary of game state features
        model: Trained ML model
        scaler: Feature scaler
        
    Returns:
        Array of predicted button presses
    """
    try:
        # Convert features to array format
        feature_list = []
        for key in sorted(features.keys()):
            feature_list.append(features[key])
            
        features_array = np.array([feature_list])
        
        # Scale features
        scaled_features = scaler.transform(features_array)
        
        # Predict button presses
        button_presses = model.predict(scaled_features)[0]
        
        return button_presses
        
    except Exception as e:
        print(f"Prediction error: {e}")
        # Return default (no buttons pressed)
        return np.zeros(12)

def prepare_training_script():
    """
    Generate a standalone training script based on the collected data
    
    This function creates a Python script that can be used to train the model
    outside of the game environment.
    """
    script_content = """#!/usr/bin/env python
# Street Fighter II Bot - Model Training Script

import numpy as np
import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_model(data_file="fighter_training_data.csv"):
    # Load data
    print(f"Training model using data from {data_file}")
    data = pd.read_csv(data_file)
    print(f"Loaded {len(data)} samples")
    
    # Handle missing values
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        data[col] = data[col].fillna(data[col].median())
    
    # Identify button columns and features
    button_columns = [col for col in data.columns if col.startswith("button_")]
    feature_columns = [col for col in data.columns if col not in button_columns]
    
    # Split features and targets
    X = data[feature_columns]
    y = data[button_columns]
    
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest model
    print("Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    for i, button in enumerate(button_columns):
        acc = accuracy_score(y_test[button], y_pred[:, i])
        print(f"{button}: {acc:.4f}")
    
    # Save model and scaler
    joblib.dump(model, 'fighter_model.pkl')
    joblib.dump(scaler, 'fighter_scaler.pkl')
    print("Model and scaler saved")
    
    # Feature importance analysis
    if hasattr(model, 'feature_importances_'):
        print("\\nTop 10 most important features:")
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1]
        for i in range(min(10, len(feature_columns))):
            print(f"{feature_columns[indices[i]]}: {importance[indices[i]]:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train Street Fighter II bot model')
    parser.add_argument('--data', default='fighter_training_data.csv', 
                      help='Path to training data CSV file')
    args = parser.parse_args()
    
    train_model(args.data)
"""
    
    # Write script to file
    with open("train_model.py", "w") as f:
        f.write(script_content)
    
    print("Training script generated: train_model.py")
    print("You can run this script after collecting data to train your model.")
    print("Usage: python train_model.py --data fighter_training_data.csv")

def validate_model(model_file="fighter_model.pkl", scaler_file="fighter_scaler.pkl", data_file="fighter_training_data.csv"):
    """
    Validate the trained model on test data
    
    Args:
        model_file: Path to the trained model file
        scaler_file: Path to the feature scaler file
        data_file: Path to the data file for validation
        
    Returns:
        Accuracy score if validation was successful, None otherwise
    """
    try:
        # Check if required files exist
        if not os.path.exists(model_file):
            print(f"Error: Model file {model_file} not found")
            return None
            
        if not os.path.exists(scaler_file):
            print(f"Error: Scaler file {scaler_file} not found")
            return None
            
        if not os.path.exists(data_file):
            print(f"Error: Data file {data_file} not found")
            return None
        
        # Load model and scaler
        print(f"Loading model from {model_file}...")
        model = joblib.load(model_file)
        scaler = joblib.load(scaler_file)
        
        # Load data
        print(f"Loading data from {data_file}...")
        data = pd.read_csv(data_file)
        
        # Identify button columns
        button_columns = [col for col in data.columns if col.startswith("button_")]
        feature_columns = [col for col in data.columns if col not in button_columns]
        
        # Split data
        X = data[feature_columns]
        y = data[button_columns]
        
        # Split into train and test sets
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_test_scaled = scaler.transform(X_test)
        
        # Make predictions
        print("Making predictions on test data...")
        y_pred = model.predict(X_test_scaled)
        
        # Calculate accuracy for each button
        print("\nButton-wise accuracy:")
        accuracies = []
        for i, button in enumerate(button_columns):
            acc = accuracy_score(y_test[button], y_pred[:, i])
            accuracies.append(acc)
            print(f"{button}: {acc:.4f}")
        
        # Calculate overall accuracy
        overall_acc = np.mean(accuracies)
        print(f"\nOverall accuracy: {overall_acc:.4f}")
        
        return overall_acc
        
    except Exception as e:
        print(f"Error validating model: {e}")
        return None