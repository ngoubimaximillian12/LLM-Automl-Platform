import os
import requests
from dotenv import load_dotenv
load_dotenv()


DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

def deepseek_fallback(csv_text):
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "You're an expert AutoML assistant."},
            {"role": "user", "content": f"This is a CSV dataset:\n\n{csv_text}\n\nGive preprocessing suggestions and model ideas."}
        ]
    }

    response = requests.post(url, json=data, headers=headers)
    result = response.json()
    return result['choices'][0]['message']['content']
