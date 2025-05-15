# --------------------------------------------
# 📦 Model Prediction & Feedback Logger Module
# --------------------------------------------
# Loads a trained model, makes predictions from input,
# and logs feedback (with optional corrections) to the database.
# --------------------------------------------

import os
import sys
import pandas as pd
import joblib
from fastapi import HTTPException

# ✅ Add backend directory to path for direct execution
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

# ✅ Use relative imports when running this file directly
try:
    from database import SessionLocal, Feedback
except ModuleNotFoundError as e:
    print("❌ Import failed:", e)
    raise

MODEL_DIR = os.path.join(CURRENT_DIR, "../models/saved_models")

# 🔁 Load trained model from disk
def load_model(model_name: str):
    model_path = os.path.join(MODEL_DIR, model_name)
    print(f"📦 Loading model: {model_path}")
    try:
        model = joblib.load(model_path)
        return model
    except Exception:
        raise HTTPException(status_code=404, detail="Model not found")

# 🔍 Predict using model and input data
def predict(model_name: str, input_data: dict):
    print(f"🔮 Running prediction using model: {model_name}")
    model = load_model(model_name)
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    print(f"✅ Prediction result: {prediction}")
    return prediction

# 📝 Save prediction and optional correction to feedback log
def save_prediction_feedback(input_data: dict, prediction: str, correction: str = None):
    print("🗂️ Saving prediction feedback...")
    session = SessionLocal()
    entry = Feedback(
        input_data=str(input_data),
        prediction=prediction,
        user_correction=correction
    )
    session.add(entry)
    session.commit()
    session.close()
    print("✅ Feedback saved to database.")
