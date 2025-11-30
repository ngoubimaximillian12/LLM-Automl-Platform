# 🤖 LLM-Powered AutoML Platform with Bias Auditing & Feedback Learning

A full-stack intelligent machine learning platform that:
- Automates end-to-end model building  
- Provides real-time bias and fairness auditing  
- Uses LLMs (GPT/DeepSeek) for reasoning, feedback interpretation, and code generation  
- Supports continuous learning and explainability through a conversational interface  

--

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
Copy code
DEEPSEEK_API_KEY=your_api_key
EMAIL_USER=your_email@example.com
EMAIL_PASS=your_password
DATABASE_URL=postgresql://postgres:password@db:5432/automl_db
HUGGINGFACE_API_KEY=your_hf_token
OPENAI_API_KEY=your_openai_key
3. Run the Platform
With Docker:

bash
Copy code
docker-compose up --build
Without Docker:

bash
Copy code
# Install dependencies
pip install -r requirements.txt

# Start PostgreSQL
sudo service postgresql start

# Run backend
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000

# Run frontend (new terminal)
cd frontend
streamlit run app.py
4. Access the Platform
Frontend: http://localhost:8501

Backend API: http://localhost:8000

API Docs: http://localhost:8000/docs

📖 User Guide
Training Your First Model
Upload Dataset: CSV files with target column

Configure Settings: Select target column, choose algorithms

Review EDA: Automated exploratory data analysis

Train Model: One-click training with progress tracking

Bias Audit: Automatic fairness assessment

Get Insights: LLM-powered explanations

Bias Auditing Metrics
Metric	Formula	Interpretation
Statistical Parity Difference (SPD)	P(Ŷ=1	A=1) - P(Ŷ=1
Equal Opportunity Difference (EOD)	TPR₁ - TPR₀	Difference in true positive rates
Disparate Impact Ratio (DIR)	P(Ŷ=1	A=1) / P(Ŷ=1
Average Odds Difference (AOD)	0.5 × [(TPR₁-TPR₀) + (FPR₁-FPR₀)]	Average of TPR and FPR differences

LLM Assistant Capabilities
python
Copy code
questions = [
    "Why is my model biased against group A?",
    "How can I improve model fairness?",
    "Generate preprocessing code for handling missing values",
    "Explain the confusion matrix results",
    "What features are most important for predictions?"
]
🏗️ Architecture
pgsql
Copy code
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │    │   FastAPI       │    │   PostgreSQL    │
│   Frontend      │◄──►│   Backend       │◄──►│   Database      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Upload   │    │   AutoML Engine │    │   Model Storage │
│   & Feedback    │    │   + Bias Audit  │    │   + Metadata    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                    ┌─────────────────┐
                    │   LLM Services  │
                    │   GPT-2/DeepSeek│
                    └─────────────────┘
🔧 Core Components
AutoML Engine
python
Copy code
class AutoMLEngine:
    def train_model(self, data, target_column, algorithms):
        """
        Automated model training pipeline:
        1. Data preprocessing and cleaning
        2. Feature engineering and selection
        3. Model training with hyperparameter tuning
        4. Cross-validation and evaluation
        5. Bias auditing and fairness assessment
        """
Bias Auditor
python
Copy code
class BiasAuditor:
    def audit_model(self, model, X_test, y_test, sensitive_attributes):
        """
        Comprehensive fairness evaluation:
        - Statistical parity assessment
        - Equal opportunity analysis
        - Disparate impact calculation
        - Average odds difference
        """
LLM Assistant
python
Copy code
class LLMAssistant:
    def explain_results(self, model_results, bias_results):
        """
        Generates human-readable explanations:
        - Model performance interpretation
        - Bias analysis and recommendations
        - Feature importance explanations
        - Actionable improvement suggestions
        """
🔧 API Reference
Model Training
http
Copy code
POST /api/train
Content-Type: multipart/form-data
{
  "file": "dataset.csv",
  "target_column": "target",
  "algorithms": ["random_forest", "xgboost"],
  "test_size": 0.2
}
Bias Auditing
http
Copy code
POST /api/audit-bias
Content-Type: application/json
{
  "model_id": "model_123",
  "sensitive_attributes": ["gender", "race"],
  "fairness_metrics": ["spd", "eod", "dir"]
}
LLM Interaction
http
Copy code
POST /api/llm/explain
Content-Type: application/json
{
  "query": "Why is my model biased?",
  "context": {
    "model_results": {...},
    "bias_results": {...}
  }
}
Feedback Submission
http
Copy code
POST /api/feedback
Content-Type: application/json
{
  "model_id": "model_123",
  "feedback_type": "bias_correction",
  "details": "Model should not discriminate based on age",
  "auto_retrain": true
}
🧪 Supported Algorithms
Classification

Random Forest

XGBoost

Logistic Regression

SVM

Neural Networks

Regression

Random Forest Regressor

XGBoost Regressor

Linear Regression

SVR

Neural Networks

Preprocessing Options

Missing Value Handling: mean, median, KNN

Scaling: Standard, MinMax, Robust

Encoding: One-hot, label, target

Feature Selection: Univariate, RFE

Outlier Detection: Isolation Forest, LOF

📊 Monitoring & Analytics
Performance Metrics

python
Copy code
metrics = {
    "accuracy": 0.85,
    "precision": 0.82,
    "recall": 0.88,
    "f1_score": 0.85,
    "auc_roc": 0.91
}
Bias Metrics

python
Copy code
bias_metrics = {
    "statistical_parity_difference": 0.05,
    "equal_opportunity_difference": 0.03,
    "disparate_impact_ratio": 0.95,
    "average_odds_difference": 0.04
}
Automated alerts for performance degradation, bias detection, data drift, and staleness.

🔄 Feedback Learning System
Prediction Corrections

Bias Reports

Feature Requests

Performance Issues

python
Copy code
class RetrainingTrigger:
    def should_retrain(self, model_id):
        conditions = [
            self.accuracy_below_threshold(model_id, 0.8),
            self.bias_above_threshold(model_id, 0.1),
            self.feedback_count_exceeded(model_id, 10),
            self.data_drift_detected(model_id)
        ]
        return any(conditions)
🛡️ Ethical AI Guidelines
Fairness Principles: individual, group, counterfactual, causal

Bias Mitigation: pre-processing, in-processing, post-processing

Transparency: explainability, audit trails, model cards

User Consent: clear communication of decisions

🧪 Testing
bash
Copy code
# Unit tests
cd backend
pytest tests/ -v
pytest tests/ --cov=automl --cov-report=html

# Integration tests
pytest tests/integration/ -v

# API tests
pytest tests/api/ -v

# Bias tests
pytest tests/bias/ -v
pytest tests/bias/test_statistical_parity.py -v
🚀 Deployment
Production Docker Setup
bash
Copy code
docker-compose -f docker-compose.prod.yml build
docker-compose -f docker-compose.prod.yml up -d
docker-compose -f docker-compose.prod.yml up -d --scale backend=3
Environment Configuration
bash
Copy code
export ENVIRONMENT=production
export DATABASE_URL=postgresql://user:pass@prod-db:5432/automl
export REDIS_URL=redis://prod-redis:6379
export SENTRY_DSN=your_sentry_dsn
Monitoring Setup
bash
Copy code
curl http://localhost:8000/metrics
curl http://localhost:8000/health
curl http://localhost:8000/api/models/status
🔒 Security
Authentication & Authorization: JWT, RBAC, rate limiting, input validation

Data Protection: AES-256 encryption, anonymization, audit logs, secure storage

📈 Performance Optimization
python
Copy code
# Async training endpoint
@app.post("/api/train-async")
async def train_model_async(background_tasks: BackgroundTasks):
    background_tasks.add_task(train_model_task, data)
    return {"status": "training_started"}

# Cached predictions
@lru_cache(maxsize=1000)
def predict_cached(model_id: str, features: tuple):
    return model.predict([features])
Database: connection pooling, query optimization, batch ops, read replicas.

🤝 Contributing
bash
Copy code
# Dev setup
git clone https://github.com/ngoubimaximillian12/llm-automl-platform.git
cd llm-automl-platform
python -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt
pre-commit install
Guidelines:

PEP8 + black formatting

90% test coverage

Conventional Commits

PRs from develop branch

📄 License
MIT License — see LICENSE.

🙏 Acknowledgments
Fairlearn Team

AIF360 (IBM)

Hugging Face

FastAPI

Streamlit

👨‍💻 Author
Ngoubi Maximillian Diangha
GitHub: @ngoubimaximillian12
Email: ngoubimaximilliandiangha@gmail.com
LinkedIn: Diangha Ngoubi

📞 Support
Docs: docs.automl-platform.com

Issues: GitHub Issues

Discussions: GitHub Discussions

Email: support@automl-platform.com

Built with ❤️ for ethical AI and democratized machine learning

vbnet
Copy code

?
