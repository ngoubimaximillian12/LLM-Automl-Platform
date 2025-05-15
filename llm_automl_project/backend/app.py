import os
import sys
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ✅ Ensure backend imports work
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ✅ Local backend modules
from backend.model_pipeline import train_and_save_model
from backend.predict import predict, save_prediction_feedback
from backend.utils import load_dataset
from backend.eda_generator import generate_eda_report, export_eda_to_pdf
from backend.retrain import retrain_from_feedback

# ✅ Initialize FastAPI app
app = FastAPI(title="LLM AutoML Backend API")

# ✅ Allow frontend (Streamlit) to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Optional: use specific origins like ["http://localhost:8501"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Ensure 'data/' directory exists
DATA_DIR = os.path.abspath("data")
os.makedirs(DATA_DIR, exist_ok=True)


# 📁 Upload CSV dataset
@app.post("/upload-data/")
async def upload_data(file: UploadFile = File(...)):
    file_path = os.path.join(DATA_DIR, file.filename)
    try:
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        return {"status": "File uploaded", "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


# 🧠 Train model and generate EDA
@app.post("/train-model/")
def train_model(file_name: str):
    dataset_path = os.path.join(DATA_DIR, file_name)
    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=404, detail="Dataset not found")

    try:
        df = load_dataset(dataset_path)
        generate_eda_report(df)
        export_eda_to_pdf()
        model_path = train_and_save_model(dataset_path)
        return {
            "message": "Model trained and EDA generated successfully",
            "model_path": model_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


# 🔮 Predict with uploaded input
@app.post("/predict/")
def make_prediction(model_name: str, input_data: dict):
    try:
        result = predict(model_name, input_data)
        save_prediction_feedback(input_data, prediction=result)
        return {"prediction": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# 📝 Save feedback on prediction
@app.post("/predict/feedback/")
def submit_feedback(model_name: str, input_data: dict, correct_label: str):
    try:
        result = predict(model_name, input_data)
        save_prediction_feedback(input_data, prediction=result, correction=correct_label)
        return {
            "status": "Feedback saved",
            "original_prediction": result,
            "user_correction": correct_label
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feedback submission failed: {str(e)}")


# 🔁 Retrain model using feedback
@app.post("/retrain/")
def retrain_model():
    try:
        result = retrain_from_feedback()
        return {"retrain_status": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")
