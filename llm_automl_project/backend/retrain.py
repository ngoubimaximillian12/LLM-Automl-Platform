import os
import sys
import pandas as pd
import json

# Fix imports when running this script directly
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

try:
    from model_pipeline import train_and_save_model
    from database import SessionLocal, Feedback
except ModuleNotFoundError as e:
    print("❌ Import failed:", e)
    raise

def retrain_from_feedback():
    try:
        print("🔁 Starting feedback-based retraining...")

        session = SessionLocal()
        feedback_entries = session.query(Feedback).filter(Feedback.user_correction.isnot(None)).all()

        if not feedback_entries:
            print("ℹ️ No feedback found with user corrections.")
            return "No feedback to retrain"

        records = []
        for entry in feedback_entries:
            try:
                x = json.loads(entry.input_data)
                x['target'] = entry.user_correction
                records.append(x)
            except Exception as parse_err:
                print(f"⚠️ Failed to parse feedback entry {entry.id}: {parse_err}")

        df = pd.DataFrame(records)
        file_path = os.path.join(CURRENT_DIR, "../data/feedback_retrain.csv")
        df.to_csv(file_path, index=False)

        print(f"📄 Saved retraining dataset to {file_path} with {len(df)} rows.")

        model_path = train_and_save_model(file_path)
        print(f"✅ Retrained model saved to {model_path}")
        return model_path

    except Exception as e:
        print("❌ Error during retraining:", e)
        raise
