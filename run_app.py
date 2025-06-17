import subprocess
import threading
import time
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def start_backend():
    print("🚀 Starting FastAPI backend at http://localhost:8000 ...")
    # Use Popen to start backend asynchronously (non-blocking)
    subprocess.Popen(["uvicorn", "llm_automl_project.backend.app:app", "--reload"])

def start_frontend():
    print("🎨 Launching Streamlit frontend at http://localhost:8501 ...")
    # This will block until frontend exits, which is fine
    subprocess.run(["streamlit", "run", "llm_automl_project/frontend/app.py"])

if __name__ == "__main__":
    print("🔧 Booting LLM AutoML system...")

    backend_thread = threading.Thread(target=start_backend)
    backend_thread.start()

    # Instead of a fixed sleep, optionally implement a health-check loop to ensure backend is ready
    time.sleep(3)

    start_frontend()
