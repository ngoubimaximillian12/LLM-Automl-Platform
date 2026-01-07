# 🤖 CLAUDE.md - Development Guide & Context

> **Purpose**: This file contains all essential context, conventions, and guidelines for building the LLM-AutoML Platform. Read this first before making any code changes.

**Last Updated**: January 7, 2026
**Project Lead**: Ngoubi Maximillian Diangha
**Repository**: https://github.com/ngoubimaximillian12/LLM-Automl-Platform

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Current State](#-current-state)
3. [Implementation Phases](#-implementation-phases)
4. [Tech Stack](#-tech-stack)
5. [Project Structure](#-project-structure)
6. [Development Standards](#-development-standards)
7. [Common Commands](#-common-commands)
8. [Database Schema](#-database-schema)
9. [API Conventions](#-api-conventions)
10. [Testing Strategy](#-testing-strategy)
11. [Deployment Guide](#-deployment-guide)
12. [Important Notes](#-important-notes)

---

## 🎯 Project Overview

### Vision
Build a **next-generation AutoML platform** that democratizes AI by combining:
- Automated machine learning with multiple frameworks
- LLM-powered assistance and code generation
- Built-in fairness and bias auditing
- Model marketplace for monetization
- Enterprise features (teams, RBAC, white-label)
- Cutting-edge capabilities (Quantum ML, Neurosymbolic AI, Web3)

### Key Differentiators
1. **Most Affordable**: 10x cheaper than DataRobot ($29/mo vs $5000+/mo)
2. **Fairness-First**: Built-in bias auditing (not an afterthought)
3. **LLM Integration**: ChatGPT-style copilot for ML guidance
4. **Marketplace**: Monetize your trained models as NFTs
5. **Next-Gen Ready**: Quantum ML, Neurosymbolic AI preparation

### Target Users
- **Primary**: Data scientists, ML engineers, analysts
- **Secondary**: Business analysts, product managers (no-code)
- **Enterprise**: Teams needing collaboration, RBAC, compliance
- **Academia**: Researchers, students learning ML

### Revenue Model
- **Free Tier**: $0/mo (10 models, 1GB storage)
- **Pro Tier**: $29/mo (unlimited models, 100GB, GPU)
- **Enterprise**: $299/mo (teams, white-label, on-premise)
- **Marketplace**: 20% commission on model sales
- **Enterprise Contracts**: Custom pricing

---

## 📊 Current State

### ✅ What's Working (v1.0)

**Frontend**:
- Streamlit multi-tab UI (Upload, Fairness, Email, LLM Chat, Preview)
- Dataset upload (CSV, XLSX, JSON, Parquet)
- Real-time training progress display
- EDA visualizations with Matplotlib/Plotly
- Fairness metrics charts
- Email form for EDA report delivery

**Backend**:
- FastAPI REST API with endpoints:
  - `POST /upload-data/` - Upload datasets
  - `POST /train-model/` - Train RandomForest models
  - `POST /predict/` - Make predictions
  - `POST /predict/feedback/` - Submit feedback
  - `POST /retrain/` - Trigger retraining
- SQLAlchemy with SQLite database
- Automated EDA generation (pandas-profiling style)
- PDF report generation and email delivery (SMTP)
- Model metadata storage (accuracy, timestamps, paths)
- Feedback loop with auto-retraining
- Background monitoring tasks
- DeepSeek LLM fallback integration

**ML/AI**:
- RandomForest classifier/regressor (scikit-learn)
- Automated EDA with visualizations
- Bias/fairness auditing framework
- Active learning with user feedback
- Prediction confidence tracking

**DevOps**:
- Docker Compose setup (backend + frontend)
- GitHub Actions CI/CD workflows (4 files ready)
- SQLite database (file-based)

### 🔨 What Needs Building (v2.0+)

See [IMPLEMENTATION_ROADMAP.md](./IMPLEMENTATION_ROADMAP.md) for complete details.

**High Priority (Phase 1-3, Weeks 1-14)**:
1. Next.js 14 frontend migration
2. PostgreSQL + Redis + MinIO setup
3. Multi-user authentication (JWT + OAuth2)
4. Team collaboration with RBAC
5. AutoGluon, H2O.ai integration
6. Model versioning system

**Medium Priority (Phase 4-8, Weeks 15-44)**:
1. Computer Vision suite (YOLO, SAM, OCR)
2. NLP capabilities (BERT, transformers)
3. Time series forecasting (Prophet, LSTM)
4. Visual workflow builder (React Flow)
5. Model marketplace with Stripe
6. Kubernetes deployment

**Future (Phase 9-12, Weeks 45-68)**:
1. Neurosymbolic AI integration
2. Quantum ML preparation (PennyLane)
3. Web3 marketplace (NFTs, IPFS)
4. Mobile apps (React Native)
5. Edge deployment (Raspberry Pi)

---

## 🏗️ Implementation Phases

### **Phase 1: Foundation & Infrastructure** (Weeks 1-4) - NEXT

**Goal**: Migrate to production-ready tech stack

**Tasks**:
```bash
# 1. Frontend Setup
cd frontend
npx create-next-app@latest . --typescript --tailwind --app
npm install @shadcn/ui zustand @tanstack/react-query socket.io-client

# 2. Backend Restructure
cd backend
# Create proper folder structure:
# app/
#   ├── api/v1/          # API routes
#   ├── core/            # Config, security, dependencies
#   ├── models/          # SQLAlchemy models
#   ├── schemas/         # Pydantic schemas
#   ├── services/        # Business logic
#   └── workers/         # Celery tasks

# 3. Database Setup
createdb automl_platform
alembic init alembic
alembic revision --autogenerate -m "initial schema"
alembic upgrade head

# 4. Add Redis + MinIO
docker-compose up -d redis minio
```

**Deliverables**:
- [ ] Next.js app with basic routing
- [ ] Backend restructured with proper folders
- [ ] PostgreSQL database with Alembic migrations
- [ ] Redis + MinIO running in Docker
- [ ] Updated docker-compose.yml with all services
- [ ] Environment variables properly configured

**Success Metrics**:
- Next.js dev server runs without errors
- FastAPI connects to PostgreSQL
- Alembic migrations run successfully
- Redis cache works (test with simple key-value)
- MinIO accepts file uploads

---

### **Phase 2: Authentication & Multi-User** (Weeks 5-8)

**Goal**: Support multiple users with team collaboration

**Tasks**:
1. Implement JWT authentication
2. Add OAuth2 (Google, GitHub, Microsoft)
3. Create user registration/login flow
4. Build team management (create, invite, remove)
5. Implement RBAC (Owner, Admin, Member, Viewer)
6. Add API key generation
7. Create user profile management

**Database Tables Needed** (from DATABASE_SCHEMA.md):
- `users` - User accounts
- `teams` - Team/workspace management
- `team_members` - User-team relationships with roles
- `oauth_accounts` - OAuth provider linkage
- `api_keys` - Programmatic access

**API Endpoints**:
```python
POST   /api/v1/auth/register
POST   /api/v1/auth/login
POST   /api/v1/auth/logout
POST   /api/v1/auth/refresh
GET    /api/v1/auth/me

POST   /api/v1/teams
GET    /api/v1/teams
GET    /api/v1/teams/{team_id}
POST   /api/v1/teams/{team_id}/members
DELETE /api/v1/teams/{team_id}/members/{user_id}
```

**Success Metrics**:
- Users can register/login
- OAuth works with at least Google
- Teams can be created
- Members can be invited with proper roles
- API keys work for programmatic access

---

### **Phase 3: Core AutoML Enhancement** (Weeks 9-14)

**Goal**: Add multiple AutoML frameworks and algorithms

**Tasks**:
1. Integrate AutoGluon (priority #1)
2. Add H2O.ai integration
3. Implement FLAML (Microsoft)
4. Add 30+ algorithms (XGBoost, LightGBM, CatBoost, Neural Nets)
5. Build model versioning system
6. Create advanced EDA with more visualizations
7. Add hyperparameter tuning UI
8. Implement model comparison dashboard

**New Database Tables**:
- `model_versions` - Track model iterations
- `experiments` - Group related training runs
- `hyperparameters` - Store tuning configs

**Success Metrics**:
- AutoGluon trains models successfully
- Users can compare 3+ algorithms side-by-side
- Model versioning tracks all iterations
- Accuracy improves by 10%+ vs v1.0 RandomForest

---

## 💻 Tech Stack

### Frontend (Target: v2.0)

```typescript
// Framework
Next.js 14          // App Router, Server Components
TypeScript 5.x      // Type safety
React 18           // UI library

// Styling
Tailwind CSS 3.x   // Utility-first CSS
shadcn/ui          // Beautiful components
Radix UI           // Headless components

// State Management
Zustand            // Lightweight state
React Query        // Server state & caching
React Hook Form    // Form management

// Charts & Visualizations
Recharts           // React charts
Plotly.js          // Interactive plots
D3.js              // Custom visualizations
React Flow         // Workflow builder

// Real-time
Socket.io Client   // WebSocket connection

// Utilities
date-fns           // Date manipulation
zod                // Schema validation
clsx               // Conditional classes
```

### Backend (Target: v2.0)

```python
# Framework
fastapi==0.109.0           # REST API
uvicorn[standard]==0.27.0  # ASGI server
pydantic==2.5.0            # Data validation

# Database
sqlalchemy==2.0.25         # ORM (async)
alembic==1.13.0            # Migrations
asyncpg==0.29.0            # PostgreSQL driver
redis==5.0.1               # Cache & queue

# Task Queue
celery==5.3.4              # Distributed tasks
celery[redis]              # Redis backend

# Auth
python-jose[cryptography]  # JWT
passlib[bcrypt]            # Password hashing
python-multipart           # OAuth2

# ML/AI Libraries
autogluon==1.0.0           # AutoML (priority)
h2o==3.44.0                # H2O AutoML
flaml==2.1.1               # Microsoft AutoML
pycaret==3.2.0             # Simple AutoML
scikit-learn==1.4.0        # Traditional ML
xgboost==2.0.3             # Gradient boosting
lightgbm==4.1.0            # Fast boosting
catboost==1.2.2            # Categorical boosting

# Deep Learning
torch==2.1.2               # PyTorch
tensorflow==2.15.0         # TensorFlow
transformers==4.36.0       # Hugging Face

# LLM Integration
langchain==0.1.0           # LLM framework
openai==1.7.0              # OpenAI API
anthropic==0.8.0           # Claude API

# Data Processing
pandas==2.1.4              # DataFrames
numpy==1.26.2              # Numerical
polars==0.20.0             # Fast DataFrames
dask==2024.1.0             # Parallel computing

# Visualization
matplotlib==3.8.2          # Plotting
seaborn==0.13.0            # Statistical plots
plotly==5.18.0             # Interactive charts

# Monitoring
prometheus-client==0.19.0  # Metrics
sentry-sdk==1.39.0         # Error tracking

# Storage
boto3==1.34.0              # AWS S3 / MinIO
```

### Infrastructure

```yaml
# Databases
PostgreSQL: 15.5           # Primary database
Redis: 7.2                 # Cache + message queue
MinIO: latest              # S3-compatible storage

# Containers
Docker: 24.0+
Docker Compose: 2.23+

# Orchestration (Production)
Kubernetes: 1.28+
Helm: 3.13+

# CI/CD
GitHub Actions             # Automated testing & deployment

# Monitoring
Prometheus                 # Metrics collection
Grafana                    # Dashboards
Loki                       # Log aggregation
Jaeger                     # Distributed tracing

# Cloud (Optional)
AWS: EKS, RDS, S3, ECR
GCP: GKE, Cloud SQL, GCS
Azure: AKS, PostgreSQL, Blob Storage
```

---

## 📁 Project Structure

### Current Structure (v1.0)

```
LLM-Automl-Platform/
├── llm_automl_project/
│   ├── backend/
│   │   ├── app.py                    # FastAPI main app
│   │   ├── model_pipeline.py         # Training logic
│   │   ├── predict.py                # Prediction logic
│   │   ├── eda_generator.py          # EDA generation
│   │   ├── bias_auditor.py           # Fairness checks
│   │   ├── retrain.py                # Retraining logic
│   │   ├── database.py               # SQLAlchemy models
│   │   ├── llm_generator.py          # DeepSeek integration
│   │   └── utils.py                  # Helper functions
│   └── frontend/
│       ├── app.py                    # Streamlit main app
│       ├── data_preview_tab.py
│       ├── email_form_tab.py
│       └── multimodal_agent_tab.py
├── data/                             # User uploads
├── models/                           # Saved models
├── .github/workflows/                # CI/CD pipelines
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

### Target Structure (v2.0+)

```
LLM-Automl-Platform/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                   # FastAPI app entry
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   └── v1/
│   │   │       ├── __init__.py
│   │   │       ├── auth.py           # Auth endpoints
│   │   │       ├── users.py          # User management
│   │   │       ├── teams.py          # Team collaboration
│   │   │       ├── datasets.py       # Dataset CRUD
│   │   │       ├── models.py         # Model training/management
│   │   │       ├── predictions.py    # Inference endpoints
│   │   │       ├── workflows.py      # Workflow builder
│   │   │       └── marketplace.py    # Model marketplace
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── config.py             # Settings (Pydantic BaseSettings)
│   │   │   ├── security.py           # JWT, OAuth2, hashing
│   │   │   ├── deps.py               # FastAPI dependencies
│   │   │   └── events.py             # Startup/shutdown events
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── user.py               # User SQLAlchemy model
│   │   │   ├── team.py               # Team models
│   │   │   ├── dataset.py            # Dataset models
│   │   │   ├── model.py              # ML model metadata
│   │   │   └── marketplace.py        # Marketplace models
│   │   ├── schemas/
│   │   │   ├── __init__.py
│   │   │   ├── user.py               # User Pydantic schemas
│   │   │   ├── team.py               # Team schemas
│   │   │   ├── dataset.py            # Dataset schemas
│   │   │   ├── model.py              # Model schemas
│   │   │   └── token.py              # JWT token schemas
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── auth.py               # Authentication logic
│   │   │   ├── automl/
│   │   │   │   ├── autogluon.py      # AutoGluon wrapper
│   │   │   │   ├── h2o.py            # H2O.ai wrapper
│   │   │   │   ├── flaml.py          # FLAML wrapper
│   │   │   │   └── pycaret.py        # PyCaret wrapper
│   │   │   ├── llm/
│   │   │   │   ├── copilot.py        # AI assistant
│   │   │   │   ├── rag.py            # RAG system
│   │   │   │   └── chains.py         # LangChain pipelines
│   │   │   ├── cv/
│   │   │   │   ├── yolo.py           # Object detection
│   │   │   │   ├── sam.py            # Segmentation
│   │   │   │   └── ocr.py            # Text extraction
│   │   │   ├── nlp/
│   │   │   │   ├── classification.py
│   │   │   │   ├── sentiment.py
│   │   │   │   └── ner.py
│   │   │   └── storage.py            # MinIO/S3 wrapper
│   │   ├── workers/
│   │   │   ├── __init__.py
│   │   │   ├── celery_app.py         # Celery instance
│   │   │   ├── training_tasks.py     # Model training tasks
│   │   │   ├── eda_tasks.py          # EDA generation tasks
│   │   │   └── monitoring_tasks.py   # Health checks
│   │   ├── db/
│   │   │   ├── __init__.py
│   │   │   ├── base.py               # SQLAlchemy base
│   │   │   └── session.py            # DB session management
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── email.py              # Email utilities
│   │       ├── s3.py                 # Storage utilities
│   │       └── metrics.py            # Prometheus metrics
│   ├── alembic/                      # Database migrations
│   ├── tests/
│   │   ├── api/                      # API endpoint tests
│   │   ├── services/                 # Service layer tests
│   │   └── conftest.py               # Pytest fixtures
│   ├── requirements.txt
│   ├── Dockerfile
│   └── pyproject.toml
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   ├── layout.tsx            # Root layout
│   │   │   ├── page.tsx              # Home page
│   │   │   ├── (auth)/
│   │   │   │   ├── login/
│   │   │   │   └── register/
│   │   │   ├── dashboard/
│   │   │   ├── datasets/
│   │   │   ├── models/
│   │   │   ├── workflows/
│   │   │   └── marketplace/
│   │   ├── components/
│   │   │   ├── ui/                   # shadcn/ui components
│   │   │   ├── layout/               # Header, Sidebar, Footer
│   │   │   ├── charts/               # Chart components
│   │   │   └── forms/                # Reusable forms
│   │   ├── lib/
│   │   │   ├── api.ts                # API client
│   │   │   ├── auth.ts               # Auth helpers
│   │   │   └── utils.ts              # Utilities
│   │   ├── hooks/
│   │   │   ├── useAuth.ts
│   │   │   ├── useModels.ts
│   │   │   └── useDatasets.ts
│   │   ├── store/
│   │   │   ├── auth.ts               # Zustand auth store
│   │   │   └── ui.ts                 # UI state
│   │   └── types/
│   │       └── index.ts              # TypeScript types
│   ├── public/
│   ├── package.json
│   ├── tsconfig.json
│   ├── tailwind.config.ts
│   └── next.config.js
├── kubernetes/                       # K8s manifests
│   ├── backend-deployment.yaml
│   ├── frontend-deployment.yaml
│   ├── postgres-statefulset.yaml
│   └── ingress.yaml
├── .github/workflows/
│   ├── backend-ci.yml
│   ├── frontend-ci.yml
│   ├── deploy-production.yml
│   └── security-scan.yml
├── docs/
│   ├── API.md                        # API documentation
│   ├── ARCHITECTURE.md               # System architecture
│   └── DEPLOYMENT.md                 # Deployment guide
├── DATABASE_SCHEMA.md
├── IMPLEMENTATION_ROADMAP.md
├── CLAUDE.md                         # This file
├── README.md
├── docker-compose.yml
└── .env.example
```

---

## 🎨 Development Standards

### Code Style

**Python** (Backend):
```python
# Use Black formatter (line length: 100)
black --line-length 100 app/

# Use flake8 for linting
flake8 app/ --max-line-length 100

# Use mypy for type checking
mypy app/ --strict

# Use isort for import sorting
isort app/
```

**TypeScript** (Frontend):
```typescript
// Use Prettier
prettier --write "src/**/*.{ts,tsx}"

// Use ESLint
eslint "src/**/*.{ts,tsx}" --fix

// TypeScript strict mode enabled
```

### Naming Conventions

**Python**:
```python
# Files: snake_case
user_service.py
model_pipeline.py

# Classes: PascalCase
class UserService:
class ModelPipeline:

# Functions/methods: snake_case
def train_model():
def get_user_by_id():

# Constants: UPPER_SNAKE_CASE
MAX_UPLOAD_SIZE = 100_000_000
DEFAULT_MODEL_TYPE = "classification"

# Private: leading underscore
def _internal_helper():
class _InternalClass:
```

**TypeScript**:
```typescript
// Files: kebab-case or PascalCase for components
user-service.ts
UserProfile.tsx

// Components: PascalCase
export function DatasetUpload() {}
export const ModelCard = () => {}

// Functions: camelCase
function fetchUserData() {}
const handleSubmit = () => {}

// Constants: UPPER_SNAKE_CASE
const MAX_FILE_SIZE = 100_000_000;

// Types/Interfaces: PascalCase
interface User {}
type ModelStatus = "training" | "complete" | "failed";
```

### Git Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```bash
# Format
<type>(<scope>): <subject>

<body>

<footer>

# Types
feat:     New feature
fix:      Bug fix
docs:     Documentation only
style:    Code style (formatting, no logic change)
refactor: Code restructure (no feature/bug fix)
perf:     Performance improvement
test:     Add/update tests
chore:    Build/config changes
ci:       CI/CD changes

# Examples
feat(auth): add OAuth2 Google login
fix(models): resolve AutoGluon GPU memory leak
docs(api): update authentication endpoints
refactor(backend): migrate to async SQLAlchemy
test(datasets): add upload validation tests
chore(deps): upgrade Next.js to 14.1.0

# Include Co-Author
Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
```

### API Design Principles

1. **RESTful Routes**:
```python
# Resources are plural nouns
GET    /api/v1/users
GET    /api/v1/users/{id}
POST   /api/v1/users
PUT    /api/v1/users/{id}
DELETE /api/v1/users/{id}

# Nested resources
GET    /api/v1/teams/{team_id}/members
POST   /api/v1/teams/{team_id}/members

# Actions use verbs
POST   /api/v1/models/{id}/train
POST   /api/v1/models/{id}/deploy
```

2. **Response Format**:
```json
// Success (200, 201)
{
  "data": { ... },
  "message": "Operation successful"
}

// List response
{
  "data": [ ... ],
  "pagination": {
    "page": 1,
    "per_page": 20,
    "total": 150,
    "pages": 8
  }
}

// Error (400, 401, 403, 404, 500)
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input data",
    "details": {
      "field": "email",
      "issue": "Email already exists"
    }
  }
}
```

3. **HTTP Status Codes**:
```
200 OK              - Success
201 Created         - Resource created
204 No Content      - Success, no response body
400 Bad Request     - Invalid input
401 Unauthorized    - Not authenticated
403 Forbidden       - Authenticated but not authorized
404 Not Found       - Resource doesn't exist
422 Unprocessable   - Validation error
429 Too Many Req    - Rate limited
500 Server Error    - Internal error
503 Unavailable     - Service down
```

### Environment Variables

**Required Variables**:
```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/automl_platform
REDIS_URL=redis://localhost:6379/0

# Security
SECRET_KEY=your-secret-key-min-32-chars
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# OAuth (optional)
GOOGLE_CLIENT_ID=xxx
GOOGLE_CLIENT_SECRET=xxx
GITHUB_CLIENT_ID=xxx
GITHUB_CLIENT_SECRET=xxx

# Storage
S3_ENDPOINT=http://localhost:9000
S3_ACCESS_KEY=minioadmin
S3_SECRET_KEY=minioadmin
S3_BUCKET=automl-models

# Email
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password

# LLM APIs (optional)
OPENAI_API_KEY=sk-xxx
ANTHROPIC_API_KEY=sk-ant-xxx
DEEPSEEK_API_KEY=xxx

# Monitoring
SENTRY_DSN=https://xxx@sentry.io/xxx
PROMETHEUS_PORT=9090

# Environment
ENVIRONMENT=development  # or production, staging
DEBUG=True
```

**Usage in Code**:
```python
# backend/app/core/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str
    secret_key: str
    algorithm: str = "HS256"

    class Config:
        env_file = ".env"

settings = Settings()
```

---

## 🔧 Common Commands

### Development

**Backend**:
```bash
# Setup virtual environment
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run dev server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Run Celery worker (separate terminal)
celery -A app.workers.celery_app worker --loglevel=info

# Run migrations
alembic upgrade head

# Create new migration
alembic revision --autogenerate -m "description"

# Format code
black app/
isort app/

# Lint
flake8 app/
mypy app/

# Test
pytest tests/ -v --cov=app --cov-report=html
```

**Frontend**:
```bash
# Install dependencies
cd frontend
npm install

# Run dev server
npm run dev

# Build for production
npm run build

# Start production server
npm start

# Lint
npm run lint

# Type check
npm run type-check

# Test
npm run test
npm run test:watch
npm run test:coverage
```

### Docker

```bash
# Build and run all services
docker-compose up --build

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f backend
docker-compose logs -f frontend

# Stop services
docker-compose down

# Remove volumes (reset database)
docker-compose down -v

# Rebuild single service
docker-compose up --build backend

# Execute command in container
docker-compose exec backend bash
docker-compose exec backend alembic upgrade head
```

### Database

```bash
# Create database
createdb automl_platform

# Drop database
dropdb automl_platform

# Backup
pg_dump automl_platform > backup.sql

# Restore
psql automl_platform < backup.sql

# Connect to database
psql automl_platform

# Run SQL file
psql automl_platform -f migration.sql
```

### Git Workflow

```bash
# Create feature branch
git checkout -b feature/add-autogluon

# Make changes, stage, commit
git add .
git commit -m "feat(automl): add AutoGluon integration

- Integrate AutoGluon TabularPredictor
- Add AutoGluon config schema
- Update model service to support AutoGluon
- Add tests for AutoGluon training

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"

# Push to remote
git push origin feature/add-autogluon

# Create pull request on GitHub
# After review and approval, merge to main

# Update local main
git checkout main
git pull origin main

# Delete feature branch
git branch -d feature/add-autogluon
```

---

## 🗄️ Database Schema

See [DATABASE_SCHEMA.md](./DATABASE_SCHEMA.md) for complete details.

### Key Tables (16 total)

1. **users** - User accounts (id, email, username, password_hash, role, subscription_tier)
2. **teams** - Team/workspace management
3. **team_members** - User-team relationships with RBAC
4. **oauth_accounts** - OAuth provider linkage
5. **datasets** - Uploaded datasets metadata
6. **models** - Trained model metadata
7. **model_versions** - Model iteration tracking
8. **workflows** - Visual workflow definitions
9. **deployments** - Model deployment tracking
10. **predictions** - Prediction history
11. **api_keys** - Programmatic access
12. **subscriptions** - Billing information
13. **usage_logs** - Usage tracking for billing
14. **marketplace_listings** - Model marketplace
15. **model_reviews** - Marketplace ratings
16. **audit_logs** - Compliance trail

### Important Relationships

```sql
-- User owns multiple datasets
users (1) --> (*) datasets

-- User owns multiple models
users (1) --> (*) models

-- User belongs to multiple teams
users (*) <--> (*) teams (via team_members)

-- Dataset used to train multiple models
datasets (1) --> (*) models

-- Model has multiple versions
models (1) --> (*) model_versions

-- Model has multiple deployments
models (1) --> (*) deployments

-- Model listed on marketplace
models (1) --> (0..1) marketplace_listings
```

---

## 🔌 API Conventions

### Authentication

**Endpoints**:
```python
POST /api/v1/auth/register
POST /api/v1/auth/login
POST /api/v1/auth/logout
POST /api/v1/auth/refresh
GET  /api/v1/auth/me
POST /api/v1/auth/oauth/{provider}  # provider: google, github, microsoft
```

**Headers**:
```
Authorization: Bearer <JWT_TOKEN>
```

**Token Structure**:
```json
{
  "access_token": "eyJ...",
  "refresh_token": "eyJ...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

### Pagination

**Request**:
```
GET /api/v1/models?page=1&per_page=20&sort=created_at&order=desc
```

**Response**:
```json
{
  "data": [...],
  "pagination": {
    "page": 1,
    "per_page": 20,
    "total": 150,
    "pages": 8,
    "has_next": true,
    "has_prev": false
  }
}
```

### Filtering

```
GET /api/v1/models?status=training&algorithm=autogluon&created_after=2026-01-01
```

### Error Handling

```python
# Standard error response
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable message",
    "details": {}  # Optional
  }
}

# Error codes
VALIDATION_ERROR
AUTHENTICATION_REQUIRED
PERMISSION_DENIED
RESOURCE_NOT_FOUND
RATE_LIMIT_EXCEEDED
INTERNAL_ERROR
SERVICE_UNAVAILABLE
```

---

## 🧪 Testing Strategy

### Backend Testing

**Unit Tests** (90%+ coverage target):
```python
# tests/services/test_auth.py
import pytest
from app.services.auth import AuthService

@pytest.fixture
def auth_service():
    return AuthService()

def test_create_access_token(auth_service):
    token = auth_service.create_access_token(user_id="123")
    assert token is not None
    assert len(token) > 0

def test_verify_password(auth_service):
    password = "SecurePass123!"
    hashed = auth_service.hash_password(password)
    assert auth_service.verify_password(password, hashed)
    assert not auth_service.verify_password("WrongPass", hashed)
```

**Integration Tests**:
```python
# tests/api/test_users.py
from fastapi.testclient import TestClient

def test_register_user(client: TestClient):
    response = client.post("/api/v1/auth/register", json={
        "email": "test@example.com",
        "username": "testuser",
        "password": "SecurePass123!"
    })
    assert response.status_code == 201
    data = response.json()
    assert data["email"] == "test@example.com"
```

**ML Tests**:
```python
# tests/services/test_autogluon.py
def test_autogluon_training(sample_dataset):
    from app.services.automl.autogluon import AutoGluonService

    service = AutoGluonService()
    result = service.train(
        dataset=sample_dataset,
        target="label",
        time_limit=60
    )

    assert result.model_path is not None
    assert result.accuracy > 0.5
    assert result.status == "completed"
```

### Frontend Testing

**Unit Tests** (Components):
```typescript
// __tests__/components/ModelCard.test.tsx
import { render, screen } from '@testing-library/react';
import { ModelCard } from '@/components/ModelCard';

describe('ModelCard', () => {
  it('renders model name and accuracy', () => {
    render(
      <ModelCard
        name="My Model"
        accuracy={0.95}
        status="completed"
      />
    );

    expect(screen.getByText('My Model')).toBeInTheDocument();
    expect(screen.getByText('95.0%')).toBeInTheDocument();
  });
});
```

**Integration Tests** (API):
```typescript
// __tests__/lib/api.test.ts
import { fetchModels } from '@/lib/api';

describe('API Client', () => {
  it('fetches models successfully', async () => {
    const models = await fetchModels({ page: 1, per_page: 10 });
    expect(Array.isArray(models.data)).toBe(true);
    expect(models.pagination.page).toBe(1);
  });
});
```

**E2E Tests** (Playwright/Cypress):
```typescript
// e2e/training.spec.ts
test('user can train a model', async ({ page }) => {
  await page.goto('/datasets');
  await page.click('[data-testid="upload-dataset"]');
  await page.setInputFiles('input[type="file"]', 'test.csv');
  await page.click('[data-testid="train-model"]');

  await expect(page.locator('[data-testid="training-status"]'))
    .toContainText('Training in progress');
});
```

### Running Tests

```bash
# Backend
pytest tests/ -v --cov=app --cov-report=html
pytest tests/api/ -v  # Only API tests
pytest tests/ -k "test_auth"  # Specific test pattern

# Frontend
npm run test
npm run test:watch
npm run test:coverage
npm run test:e2e

# CI (runs automatically on PR)
.github/workflows/backend-ci.yml
.github/workflows/frontend-ci.yml
```

---

## 🚀 Deployment Guide

### Local Development

```bash
# 1. Clone repository
git clone https://github.com/ngoubimaximillian12/LLM-Automl-Platform.git
cd LLM-Automl-Platform

# 2. Setup environment variables
cp .env.example .env
# Edit .env with your settings

# 3. Start services
docker-compose up --build

# Access:
# Frontend: http://localhost:3000
# Backend API: http://localhost:8000
# API Docs: http://localhost:8000/docs
# MinIO: http://localhost:9000
```

### Staging Deployment

```bash
# Using docker-compose.staging.yml
docker-compose -f docker-compose.staging.yml up -d

# Or manual
git pull origin develop
docker build -t automl-backend:staging ./backend
docker build -t automl-frontend:staging ./frontend
docker run -d -p 8000:8000 automl-backend:staging
docker run -d -p 3000:3000 automl-frontend:staging
```

### Production Deployment (Kubernetes)

**Prerequisites**:
- Kubernetes cluster (EKS, GKE, or AKS)
- kubectl configured
- Docker images pushed to registry (ECR, GCR, or ACR)

**Deploy**:
```bash
# 1. Create namespace
kubectl create namespace automl-production

# 2. Create secrets
kubectl create secret generic db-credentials \
  --from-literal=username=postgres \
  --from-literal=password=yourpassword \
  -n automl-production

kubectl create secret generic api-keys \
  --from-literal=openai-key=sk-xxx \
  --from-literal=anthropic-key=sk-ant-xxx \
  -n automl-production

# 3. Apply manifests
kubectl apply -f kubernetes/ -n automl-production

# 4. Check deployment
kubectl get pods -n automl-production
kubectl get services -n automl-production

# 5. Get external IP
kubectl get ingress -n automl-production
```

**Using GitHub Actions** (Automated):
```bash
# Triggers on push to main branch
git push origin main

# Workflow does:
# 1. Build Docker images
# 2. Push to ECR
# 3. Deploy to Kubernetes
# 4. Run migrations
# 5. Notify Slack
```

### Environment-Specific Configs

**Development** (.env):
```bash
ENVIRONMENT=development
DEBUG=True
DATABASE_URL=postgresql://localhost:5432/automl_dev
```

**Staging** (.env.staging):
```bash
ENVIRONMENT=staging
DEBUG=True
DATABASE_URL=postgresql://staging-db:5432/automl_staging
```

**Production** (.env.production):
```bash
ENVIRONMENT=production
DEBUG=False
DATABASE_URL=postgresql://prod-db:5432/automl_production
SENTRY_DSN=https://xxx@sentry.io/xxx
```

---

## ⚠️ Important Notes

### DO's ✅

1. **Always read this file before starting work**
2. **Follow the implementation phases in order** (don't skip Phase 1)
3. **Write tests for all new features** (maintain 90%+ coverage)
4. **Use type hints** in Python, TypeScript strict mode
5. **Document all API endpoints** (docstrings, OpenAPI)
6. **Handle errors gracefully** (try-catch, proper status codes)
7. **Validate all inputs** (Pydantic, Zod)
8. **Use environment variables** (never hardcode secrets)
9. **Keep dependencies updated** (but test thoroughly)
10. **Commit often** with meaningful messages
11. **Ask questions** when unclear (don't assume)
12. **Review DATABASE_SCHEMA.md** before creating models
13. **Check IMPLEMENTATION_ROADMAP.md** for task details

### DON'Ts ❌

1. **Don't skip tests** ("I'll add them later" = never)
2. **Don't commit secrets** (.env, API keys, passwords)
3. **Don't ignore linting errors** (fix them immediately)
4. **Don't use `any` type** in TypeScript (be specific)
5. **Don't hardcode values** (use config/constants)
6. **Don't make breaking API changes** without versioning
7. **Don't deploy without testing** (even to staging)
8. **Don't mix concerns** (keep business logic out of routes)
9. **Don't optimize prematurely** (make it work, then fast)
10. **Don't duplicate code** (DRY principle)
11. **Don't ignore TypeScript errors** (fix, don't suppress)
12. **Don't modify existing v1.0 files** without backup

### Current Limitations (v1.0)

1. **Single-user only** - No authentication yet
2. **RandomForest only** - No other algorithms yet
3. **No model versioning** - Can't track iterations
4. **SQLite** - Not suitable for production scale
5. **No real-time updates** - Page refresh required
6. **No GPU support** - CPU-only training
7. **Limited file formats** - No HDF5, Avro, etc.
8. **No API authentication** - Endpoints are open
9. **No rate limiting** - Vulnerable to abuse
10. **No model monitoring** - No drift detection

### Security Considerations

1. **Never expose SECRET_KEY** in logs or errors
2. **Always hash passwords** (never store plain text)
3. **Validate file uploads** (size, type, content)
4. **Sanitize user inputs** (prevent SQL injection, XSS)
5. **Use HTTPS in production** (Let's Encrypt)
6. **Implement rate limiting** (prevent DoS)
7. **Add CORS properly** (don't use `allow_origins=["*"]` in prod)
8. **Audit sensitive operations** (login, model deletion)
9. **Encrypt data at rest** (database encryption)
10. **Regular security scans** (Snyk, CodeQL, Trivy)

### Performance Tips

1. **Use async/await** for I/O operations
2. **Cache frequently accessed data** (Redis)
3. **Paginate large results** (don't return 10k records)
4. **Index database columns** (foreign keys, search fields)
5. **Lazy load images/models** on frontend
6. **Use CDN** for static assets
7. **Compress responses** (gzip)
8. **Optimize database queries** (avoid N+1)
9. **Use connection pooling** (SQLAlchemy pool)
10. **Monitor slow endpoints** (Prometheus, APM)

---

## 📚 Additional Resources

### Documentation
- [IMPLEMENTATION_ROADMAP.md](./IMPLEMENTATION_ROADMAP.md) - Complete 12-phase plan
- [DATABASE_SCHEMA.md](./DATABASE_SCHEMA.md) - Full database design
- [README.md](./README.md) - Project overview
- [SETUP_COMPLETE.md](./SETUP_COMPLETE.md) - What's been created
- [COMPLETED_TASKS.md](./COMPLETED_TASKS.md) - Accomplishments summary

### External Docs
- [FastAPI](https://fastapi.tiangolo.com/)
- [Next.js](https://nextjs.org/docs)
- [AutoGluon](https://auto.gluon.ai/stable/tutorials/)
- [SQLAlchemy 2.0](https://docs.sqlalchemy.org/en/20/)
- [Pydantic](https://docs.pydantic.dev/)
- [shadcn/ui](https://ui.shadcn.com/)
- [Tailwind CSS](https://tailwindcss.com/docs)

### Community
- GitHub Discussions: https://github.com/ngoubimaximillian12/LLM-Automl-Platform/discussions
- Project Repository: https://github.com/ngoubimaximillian12/LLM-Automl-Platform

---

## 🆘 Troubleshooting

### Common Issues

**Issue**: Docker containers won't start
```bash
# Solution
docker-compose down -v  # Remove volumes
docker system prune -a  # Clean up Docker
docker-compose up --build
```

**Issue**: Database migration fails
```bash
# Solution
alembic downgrade -1  # Rollback one migration
# Fix migration file
alembic upgrade head
```

**Issue**: Frontend can't connect to backend
```bash
# Check backend is running
curl http://localhost:8000/docs

# Check CORS settings in backend/app/main.py
# Check API_URL in frontend/.env
```

**Issue**: Import errors in Python
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Check virtual environment is activated
which python  # Should show venv path
```

**Issue**: TypeScript errors after npm install
```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install
```

---

## 📞 Contact

**Project Lead**: Ngoubi Maximillian Diangha
**Email**: ngoubimaximilliandiangha@gmail.com
**GitHub**: [@ngoubimaximillian12](https://github.com/ngoubimaximillian12)
**Repository**: https://github.com/ngoubimaximillian12/LLM-Automl-Platform

---

## 📝 Changelog

### [Unreleased]
- Phase 1: Foundation & Infrastructure (Next)
- Phase 2: Authentication & Multi-User
- Phase 3: Core AutoML Enhancement

### [1.0.0] - 2026-01-07
- Initial working version with Streamlit + FastAPI
- RandomForest AutoML
- EDA generation with PDF/Email export
- Bias auditing and fairness metrics
- LLM fallback (DeepSeek)
- Feedback loop and auto-retraining
- Docker deployment
- CI/CD pipelines created

---

**Last Updated**: January 7, 2026
**Next Review**: When starting Phase 1 implementation

---

## ⚡ Quick Start Checklist

Before starting development, ensure:

- [ ] Read this entire CLAUDE.md file
- [ ] Read IMPLEMENTATION_ROADMAP.md (at least Phase 1)
- [ ] Read DATABASE_SCHEMA.md (understand tables)
- [ ] Environment variables configured (.env)
- [ ] Docker + Docker Compose installed
- [ ] Python 3.11+ installed
- [ ] Node.js 20+ installed
- [ ] PostgreSQL client installed (psql)
- [ ] Git configured with commit signing
- [ ] IDE configured (VS Code recommended)
- [ ] Extensions installed (Prettier, ESLint, Black, mypy)
- [ ] Understand current v1.0 codebase
- [ ] Know which phase you're working on
- [ ] Tests passing before making changes
- [ ] Ready to write tests for new code

**Now you're ready to build! 🚀**

Good luck, and remember: Make it work, make it right, make it fast - in that order!
