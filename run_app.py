import subprocess
import threading
import time
import os

# Set working directory to project root
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def start_backend():
    print("🚀 Starting FastAPI backend at http://localhost:8000 ...")
    subprocess.run(["uvicorn", "llm_automl_project.backend.app:app", "--reload"])

def start_frontend():
    print("🎨 Launching Streamlit frontend at http://localhost:8501 ...")
    subprocess.run(["streamlit", "run", "llm_automl_project/frontend/app.py"])

if __name__ == "__main__":
    print("🔧 Booting LLM AutoML system...")

    # Run backend in separate thread
    backend_thread = threading.Thread(target=start_backend)
    backend_thread.start()

    # Delay to ensure backend starts first
    time.sleep(3)

    # Then run frontend
    start_frontend()
