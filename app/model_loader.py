import joblib
import os

# Path ke model
MODEL_PATH = os.path.join("models", "final_model.pkl")

# Load model saat aplikasi start
model = joblib.load(MODEL_PATH)
