# train.py
from AI_helper import analyze_data, train_model, validate_model

print("===== Analyzing collected data =====")
analyze_data()

print("\n===== Training model =====")
train_model()

print("\n===== Validating model =====")
validate_model()

print("\nTraining complete! Now set collecting_data = False in bot.py to use the model.")