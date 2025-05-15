
# 🤖 LLM-Powered AutoML Platform with Bias Auditing & Feedback Learning

A full-stack intelligent machine learning platform that:
- Automates end-to-end model building
- Provides real-time bias and fairness auditing
- Uses LLMs (GPT/DeepSeek) for reasoning, feedback interpretation, and code generation
- Supports continuous learning and explanation through a conversational interface

---

## 🔍 Motivation

Machine learning is powerful, but often inaccessible due to complexity. This project aims to democratize ML by enabling non-experts to build and monitor ethical models with minimal coding. It integrates a fine-tuned LLM, fairness auditing, and automatic feedback-driven retraining.

---

## 🎯 Key Features

| Feature                            | Description                                                                 |
|-----------------------------------|-----------------------------------------------------------------------------|
| **AutoML Engine**                 | Upload → Train → Save ML model (with EDA + evaluation)                     |
| **LLM Reasoning Assistant**       | Explains ML logic, bias results, generates preprocessing code              |
| **Bias Auditor**                  | Audits fairness using metrics like SPD, EOD, Disparate Impact              |
| **Fallback LLM (DeepSeek/GPT-4)** | Handles reasoning if backend fails or needs enhancement                    |
| **Auto-Retraining**               | Triggers retraining via feedback or accuracy/bias thresholds               |
| **EDA PDF + Email**               | Generates and sends PDF reports via email                                  |
| **Feedback Loop**                 | Stores user correction → retrain pipeline automatically                    |
| **Streamlit UI**                  | Intuitive interface to train models, see results, inject code, and explore |
| **Dockerized Setup**             | Backend, frontend, and PostgreSQL launched in one command                  |

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **Backend:** FastAPI
- **LLMs:** Hugging Face Transformers (GPT-2 / DeepSeek fallback)
- **Bias Tools:** `fairlearn`, `aif360`
- **Storage:** PostgreSQL
- **Deployment:** Docker & Docker Compose

---

## 🚀 Getting Started

### 1. Clone this repo
```bash
git clone https://github.com/ngoubimaximillian12/llm-automl-platform.git
cd llm-automl-platform
2. Create .env file (optional but recommended)
ini
Copy
Edit
DEEPSEEK_API_KEY=your_api_key
EMAIL_USER=your_email@example.com
EMAIL_PASS=your_password
3. Run the entire platform (with Docker)
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
➕ Upload Data
Upload a .csv dataset using the Streamlit UI.

🧠 Train Model
Click Train Model + Generate EDA. This will:

Preprocess data

Train model (RandomForest by default)

Generate EDA charts and a downloadable PDF

👁️ View Bias Audit
See bias metrics after training. Use LLM to explain them using plain language.

✍️ Feedback & Retraining
Submit corrections if predictions are wrong. Retraining is automatic once threshold is met.

📬 Email Report
Send EDA report via email using a form.

📈 Bias Metrics Used
Statistical Parity Difference (SPD)

Equal Opportunity Difference (EOD)

Disparate Impact Ratio (DIR)

Average Odds Difference (AOD)

🧠 LLM Usage
Task	LLM Role
Bias explanation	Natural language explanation
Preprocessing suggestions	Generates code for missing values, scaling, encoding, etc.
Model selection & retraining logic	Suggests best model based on data structure
Fallback when backend fails	DeepSeek API as last resort

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
This project was developed in alignment with the MSc dissertation titled:

"LLM-Powered Automated Machine Learning Platform with Integrated Bias and Fairness Auditing"

It incorporates state-of-the-art research in:

AutoML (AutoSklearn, H2O.ai)

Fairness tools (Fairlearn, AIF360)

Instruction-tuned LLMs (QLoRA, Mistral, DeepSeek)

✅ Completed Functionality Checklist
✅ AutoML pipeline
✅ Bias audit & explanation
✅ LLM fallback + code generation
✅ Auto-retraining via feedback
✅ Email delivery of EDA
✅ Streamlit dashboard for history
✅ Docker deployment
✅ Feedback database & active learning

🧠 Future Additions
Model comparison visualizer (side-by-side performance)

Support for multi-class bias metrics

Real-time Slack/Discord alerts on retrain or drift

Hugging Face Space deployment

📜 License
MIT © 2025 Ngoubi Maximillian Diamgha

yaml
Copy
Edit
