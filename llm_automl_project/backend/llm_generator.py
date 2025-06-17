import os
import requests
from dotenv import load_dotenv

# 🔄 Load environment variables
load_dotenv()

# 🔐 Get API key from environment
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")  # Set this in your .env file


def generate_preprocessing_code(task: str) -> str:
    """Generate placeholder preprocessing code (if fallback fails or offline mode)."""
    return f"# Placeholder: code to perform '{task}'\nprint('Implement logic here')"


def deepseek_fallback(task: str) -> str:
    """Send fallback task to DeepSeek API for reasoning/code generation."""
    if not DEEPSEEK_API_KEY:
        return "❌ DeepSeek API key not set in environment."

    prompt = f"You are a Python ML engineer. Generate sklearn-compatible code to: {task}"

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "deepseek-coder",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.5
    }

    try:
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=90  # ⏱️ Increased timeout to 60 seconds
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ DeepSeek request failed: {e}"
