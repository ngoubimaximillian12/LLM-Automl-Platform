import os
import requests
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("DEEPSEEK_API_KEY")

def suggest_model_or_pipeline(task_desc):
    prompt = f"Suggest the best ML model and preprocessing for: {task_desc}"
    response = requests.post(
        "https://api.deepseek.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json={
            "model": "deepseek-coder",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.5
        },
        timeout=60
    )
    return response.json()["choices"][0]["message"]["content"]
