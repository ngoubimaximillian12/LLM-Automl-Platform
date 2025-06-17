import os
import sys
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model_pipeline import train_and_save_model
from predict import predict, save_prediction_feedback
from utils import load_dataset
from eda_generator import generate_eda_report, export_eda_to_pdf
from retrain import retrain_from_feedback
from background_tasks import schedule_daily_monitoring

DATA_DIR = os.path.abspath("data")
os.makedirs(DATA_DIR, exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    tasks = BackgroundTasks()
    schedule_daily_monitoring(tasks)
    yield

app = FastAPI(title="LLM AutoML Backend API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload-data/")
async def upload_data(file: UploadFile = File(...)):
    file_path = os.path.join(DATA_DIR, file.filename)
    try:
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        return {"status": "File uploaded", "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

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

@app.post("/predict/")
def make_prediction(model_name: str, input_data: dict):
    try:
        result = predict(model_name, input_data)
        save_prediction_feedback(input_data, prediction=result)
        return {"prediction": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

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

@app.post("/retrain/")
def retrain_model():
    try:
        result = retrain_from_feedback()
        return {"retrain_status": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")
