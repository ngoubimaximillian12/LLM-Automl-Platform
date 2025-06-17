import os
import sys
import pandas as pd
import joblib
from fastapi import HTTPException

# Ensure current directory in sys.path for imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

try:
    from database import SessionLocal, Feedback
except ModuleNotFoundError as e:
    print("❌ Import failed:", e)
    raise

MODEL_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../models/saved_models"))

def load_model(model_name: str):
    model_path = os.path.join(MODEL_DIR, model_name)
    print(f"📦 Loading model: {model_path}")
    if not os.path.isfile(model_path):
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found at {model_path}")
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

def predict(model_name: str, input_data: dict):
    print(f"🔮 Running prediction using model: {model_name}")
    model = load_model(model_name)
    input_df = pd.DataFrame([input_data])
    try:
        prediction = model.predict(input_df)[0]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")
    print(f"✅ Prediction result: {prediction}")
    return prediction

def save_prediction_feedback(input_data: dict, prediction: str, correction: str = None):
    print("🗂️ Saving prediction feedback...")
    session = SessionLocal()
    try:
        entry = Feedback(
            input_data=str(input_data),
            prediction=prediction,
            user_correction=correction
        )
        session.add(entry)
        session.commit()
        print("✅ Feedback saved to database.")
    except Exception as e:
        session.rollback()
        print(f"❌ Failed to save feedback: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save feedback: {str(e)}")
    finally:
        session.close()
