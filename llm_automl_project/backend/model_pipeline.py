# ------------------------------------------
# 📦 ML Model Training and Saving Pipeline
# ------------------------------------------
# This script loads a dataset, trains a RandomForestClassifier,
# evaluates the model, saves it to disk, and logs metadata to the database.
# It’s designed to be run directly or imported into a FastAPI service.
# ------------------------------------------

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# ✅ Step 1: Fix Python path for direct execution
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

# ✅ Step 2: Import custom database logger
try:
    from database import save_model_metadata
except ModuleNotFoundError as e:
    print("❌ Import failed (is your working directory correct?):", e)
    raise

# ------------------------------------------
# 🔁 Main training function
# ------------------------------------------
def train_and_save_model(file_path: str) -> str:
    # Step 3: Load dataset
    print(f"📂 Loading dataset from: {file_path}")
    df = pd.read_csv(file_path)

    # Step 4: Separate features and target
    target = df.columns[-1]
    X = df.drop(columns=[target])
    y = df[target]
    print(f"📊 Dataset summary: {len(df)} rows, {len(X.columns)} features, Target: '{target}'")

    # Step 5: Split into training and testing sets
    print("🔀 Splitting data into training and testing sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Step 6: Train a Random Forest model
    print("🧠 Training RandomForestClassifier...")
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Step 7: Evaluate the model
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"📈 Model accuracy on test set: {acc:.4f}")

    # Step 8: Save the trained model to disk
    model_name = os.path.basename(file_path).split('.')[0] + "_rf_model.pkl"
    model_path = os.path.join(CURRENT_DIR, "../models/saved_models", model_name)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"✅ Trained model saved at: {model_path}")

    # Step 9: Log model metadata to the database
    print("🗃️ Logging model metadata to database...")
    save_model_metadata(name=model_name, accuracy=acc, path=model_path)

    # Step 10: Return saved model path
    print("🚀 Training pipeline complete.\n")
    return model_path
