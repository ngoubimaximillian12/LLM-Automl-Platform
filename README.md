# 🤖 LLM-Powered AutoML Platform with Bias Auditing & Feedback Learning

A full-stack intelligent machine learning platform that:

- Automates end-to-end model building  
- Provides real-time bias and fairness auditing  
- Uses LLMs (GPT/DeepSeek) for reasoning, feedback interpretation, and code generation  
- Supports continuous learning and explainability through a conversational interface  

---

## 🔍 Motivation

Machine learning is powerful but often inaccessible due to its complexity. This project aims to **democratize ML** by enabling non-experts to build and monitor **ethical models** with minimal coding. It integrates a fine-tuned LLM, fairness auditing, and feedback-driven retraining.

---

## 🎯 Key Features

| Feature                    | Description                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
| **AutoML Engine**          | Upload → Train → Save ML model (with EDA + evaluation)                      |
| **LLM Reasoning Assistant**| Explains ML logic, bias results, generates preprocessing code                |
| **Bias Auditor**           | Audits fairness using SPD, EOD, DIR, AOD                                    |
| **Fallback LLM**           | DeepSeek or GPT-4 handles reasoning if backend fails or is limited          |
| **Auto-Retraining**        | Triggers retraining via feedback or bias/accuracy thresholds                |
| **EDA PDF + Email**        | Generates and sends PDF reports via email                                   |
| **Feedback Loop**          | Stores user correction → triggers automatic retraining                      |
| **Streamlit UI**           | Intuitive interface to train models, explore results, and interact with LLM |
| **Dockerized Setup**       | One-command launch for backend, frontend, and PostgreSQL                    |

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit  
- **Backend**: FastAPI  
- **LLMs**: Hugging Face Transformers (GPT-2), DeepSeek (fallback)  
- **Bias Tools**: `fairlearn`, `aif360`  
- **Database**: PostgreSQL  
- **Deployment**: Docker & Docker Compose  

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/ngoubimaximillian12/llm-automl-platform.git
cd llm-automl-platform
2. Configure Environment Variables
Create a .env file:

ini
Copy
Edit
DEEPSEEK_API_KEY=your_api_key
EMAIL_USER=your_email@example.com
EMAIL_PASS=your_password
3. Run the Platform
With Docker:

bash
Copy
Edit
docker-compose up --build
Or manually:

bash
Copy
Edit
# Terminal 1 - Backend
uvicorn backend.app:app --reload

# Terminal 2 - Frontend
streamlit run frontend/app.py
🧪 How to Use
➕ Upload Data: Use the Streamlit UI to upload your CSV dataset

🧠 Train Model: Click "Train Model + Generate EDA"

Preprocesses data

Trains (RandomForest by default)

Generates downloadable EDA PDF

👁️ View Bias Audit: View fairness metrics; use LLM to explain bias results

✍️ Provide Feedback: Submit corrections if predictions are incorrect

Retraining is triggered once feedback threshold is reached

📬 Send Email Report: Generate and send the EDA report to any email address

📈 Bias Metrics
SPD – Statistical Parity Difference

EOD – Equal Opportunity Difference

DIR – Disparate Impact Ratio

AOD – Average Odds Difference

🧠 LLM Roles
Task	LLM Functionality
Bias explanation	Translates metrics to natural language
Preprocessing suggestions	Generates code for missing values, encoding, etc.
Model selection logic	Suggests models based on dataset characteristics
Backend fallback	DeepSeek handles logic if internal LLM fails

📦 File Structure
bash
Copy
Edit
llm_automl_project/
├── backend/             # FastAPI backend
│   ├── app.py
│   ├── model_pipeline.py
│   ├── predict.py
│   ├── utils.py
│   ├── eda_generator.py
│   ├── database.py
│   ├── retrain.py
│   ├── llm_generator.py
│   ├── llm_bias_helper.py
│   └── background_tasks.py
├── frontend/            # Streamlit UI
│   ├── app.py
│   ├── eda_email.py
│   ├── llm_assistant.py
│   ├── llm_bias_tools.py
│   └── dashboard.py
├── run_app.py
├── docker-compose.yml
├── requirements.txt
└── .env
📚 Research Basis
This project supports the MSc dissertation:

"LLM-Powered Automated Machine Learning Platform with Integrated Bias and Fairness Auditing"

Built on:

AutoML: AutoSklearn, H2O.ai

Fairness Tools: AIF360, Fairlearn

Instruction-tuned LLMs: QLoRA, Mistral, DeepSeek

✅ Completed Functionality Checklist
✅ AutoML pipeline

✅ Bias audit & LLM explanation

✅ Code generation via LLM fallback

✅ Feedback loop & auto-retraining

✅ EDA PDF with email delivery

✅ Streamlit dashboard

✅ Dockerized setup

✅ Active learning via feedback database

🚧 Future Additions
📊 Model comparison visualizer

📚 Multi-class bias metrics

📡 Real-time alerts via Slack/Discord

🌍 Hugging Face Space deployment

📜 License
MIT © 2025 Ngoubi Maximillian Diamgha

