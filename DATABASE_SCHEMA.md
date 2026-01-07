# 🗄️ DATABASE SCHEMA DOCUMENTATION

## Overview

Complete PostgreSQL database schema for the LLM-AutoML Platform supporting multi-user authentication, team collaboration, model marketplace, and billing.

---

## 📊 Entity Relationship Diagram

```
Users ──< OAuth_Accounts
  │
  ├──< Teams (as owner)
  ├──< Team_Members
  ├──< Datasets
  ├──< Models
  ├──< Workflows
  ├──< Deployments
  ├──< Predictions
  ├──< API_Keys
  ├──< Subscriptions
  ├──< Usage_Logs
  └──< Notifications

Teams ──< Team_Members
  ├──< Datasets (team datasets)
  ├──< Models (team models)
  └──< Workflows (team workflows)

Models ──< Predictions
  ├──< Deployments
  ├──< Marketplace_Listings
  └──< Model_Reviews
```

---

## 📋 TABLE DEFINITIONS

### **1. users**

Core user authentication and profile information.

```sql
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(100) UNIQUE NOT NULL,
    full_name VARCHAR(255),
    password_hash VARCHAR(255) NOT NULL,
    avatar_url TEXT,
    bio TEXT,

    -- Role & Subscription
    role VARCHAR(50) DEFAULT 'user' CHECK (role IN ('user', 'admin', 'enterprise')),
    subscription_tier VARCHAR(50) DEFAULT 'free' CHECK (subscription_tier IN ('free', 'pro', 'enterprise')),

    -- Status
    is_active BOOLEAN DEFAULT true,
    is_verified BOOLEAN DEFAULT false,
    email_verified_at TIMESTAMP,
    last_login TIMESTAMP,

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Indexes
    INDEX idx_users_email (email),
    INDEX idx_users_username (username)
);
```

**Columns**:
- `id`: Unique user identifier (UUID for security)
- `email`: User email (unique, required for login)
- `username`: Public username (unique)
- `password_hash`: Bcrypt hashed password
- `avatar_url`: Profile picture URL (S3/MinIO)
- `role`: user, admin, enterprise
- `subscription_tier`: free, pro, enterprise
- `is_verified`: Email verification status

---

### **2. oauth_accounts**

OAuth provider connections (Google, GitHub, Microsoft).

```sql
CREATE TABLE oauth_accounts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    provider VARCHAR(50) NOT NULL CHECK (provider IN ('google', 'github', 'microsoft', 'apple')),
    provider_account_id VARCHAR(255) NOT NULL,
    access_token TEXT,
    refresh_token TEXT,
    expires_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(provider, provider_account_id),
    INDEX idx_oauth_user_id (user_id)
);
```

---

### **3. teams**

Organizations/workspaces for collaboration.

```sql
CREATE TABLE teams (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    slug VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    avatar_url TEXT,
    owner_id UUID REFERENCES users(id) ON DELETE SET NULL,
    subscription_tier VARCHAR(50) DEFAULT 'free',

    -- Settings
    settings JSONB DEFAULT '{}',

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_teams_owner (owner_id),
    INDEX idx_teams_slug (slug)
);
```

---

### **4. team_members**

Team membership with roles.

```sql
CREATE TABLE team_members (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    team_id UUID REFERENCES teams(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    role VARCHAR(50) DEFAULT 'member' CHECK (role IN ('owner', 'admin', 'member', 'viewer')),
    joined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(team_id, user_id),
    INDEX idx_team_members_team (team_id),
    INDEX idx_team_members_user (user_id)
);
```

**Permissions Matrix**:
| Action | Owner | Admin | Member | Viewer |
|--------|-------|-------|--------|--------|
| View resources | ✅ | ✅ | ✅ | ✅ |
| Create model | ✅ | ✅ | ✅ | ❌ |
| Delete model | ✅ | ✅ | Own | ❌ |
| Invite members | ✅ | ✅ | ❌ | ❌ |
| Remove members | ✅ | ✅ | ❌ | ❌ |
| Change settings | ✅ | ❌ | ❌ | ❌ |
| Billing | ✅ | ❌ | ❌ | ❌ |

---

### **5. datasets**

Uploaded datasets with metadata.

```sql
CREATE TABLE datasets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    team_id UUID REFERENCES teams(id) ON DELETE SET NULL,

    -- Basic info
    name VARCHAR(255) NOT NULL,
    description TEXT,

    -- File info
    file_path TEXT NOT NULL,
    file_size BIGINT,
    file_type VARCHAR(50),

    -- Data info
    num_rows INTEGER,
    num_columns INTEGER,
    columns JSONB,
    statistics JSONB,

    -- Visibility
    is_public BOOLEAN DEFAULT false,
    visibility VARCHAR(20) DEFAULT 'private' CHECK (visibility IN ('private', 'team', 'public')),

    -- Tags & Search
    tags TEXT[],

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_datasets_user (user_id),
    INDEX idx_datasets_team (team_id),
    INDEX idx_datasets_public (is_public),
    INDEX idx_datasets_tags USING GIN(tags)
);
```

**JSONB Schemas**:

`columns`:
```json
[
  {
    "name": "age",
    "type": "int64",
    "null_count": 5,
    "unique_count": 42,
    "min": 18,
    "max": 90
  }
]
```

`statistics`:
```json
{
  "numeric_columns": 5,
  "categorical_columns": 3,
  "missing_percentage": 2.5,
  "duplicates": 10
}
```

---

### **6. models**

Trained machine learning models.

```sql
CREATE TABLE models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    team_id UUID REFERENCES teams(id) ON DELETE SET NULL,
    dataset_id UUID REFERENCES datasets(id) ON DELETE SET NULL,

    -- Basic info
    name VARCHAR(255) NOT NULL,
    description TEXT,
    model_type VARCHAR(100) CHECK (model_type IN (
        'classification', 'regression', 'clustering',
        'time_series', 'nlp', 'computer_vision',
        'recommendation', 'anomaly_detection', 'reinforcement_learning'
    )),
    algorithm VARCHAR(100),

    -- Status
    status VARCHAR(50) DEFAULT 'pending' CHECK (status IN (
        'pending', 'training', 'completed', 'failed', 'deploying', 'deployed'
    )),

    -- Training config
    config JSONB,
    hyperparameters JSONB,

    -- Metrics
    accuracy FLOAT,
    precision FLOAT,
    recall FLOAT,
    f1_score FLOAT,
    metrics JSONB,

    -- Bias & Fairness
    bias_metrics JSONB,
    fairness_score FLOAT,

    -- Files
    model_path TEXT,
    model_size BIGINT,

    -- Training info
    training_time INTEGER,
    training_started_at TIMESTAMP,
    training_completed_at TIMESTAMP,
    training_logs TEXT,

    -- Versioning
    version INTEGER DEFAULT 1,
    parent_model_id UUID REFERENCES models(id) ON DELETE SET NULL,

    -- Visibility
    is_public BOOLEAN DEFAULT false,
    visibility VARCHAR(20) DEFAULT 'private',
    is_featured BOOLEAN DEFAULT false,
    downloads INTEGER DEFAULT 0,

    tags TEXT[],
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_models_user (user_id),
    INDEX idx_models_team (team_id),
    INDEX idx_models_status (status),
    INDEX idx_models_type (model_type),
    INDEX idx_models_public (is_public),
    INDEX idx_models_tags USING GIN(tags)
);
```

---

### **7. workflows**

Visual workflow pipelines (React Flow).

```sql
CREATE TABLE workflows (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    team_id UUID REFERENCES teams(id) ON DELETE SET NULL,

    name VARCHAR(255) NOT NULL,
    description TEXT,

    -- React Flow JSON
    workflow_json JSONB NOT NULL,

    -- Status
    status VARCHAR(50) DEFAULT 'draft' CHECK (status IN ('draft', 'active', 'archived')),

    -- Execution
    last_run_at TIMESTAMP,
    run_count INTEGER DEFAULT 0,

    -- Visibility
    is_public BOOLEAN DEFAULT false,

    tags TEXT[],
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_workflows_user (user_id),
    INDEX idx_workflows_team (team_id),
    INDEX idx_workflows_status (status)
);
```

**workflow_json** example:
```json
{
  "nodes": [
    {
      "id": "1",
      "type": "data_source",
      "data": { "dataset_id": "uuid" },
      "position": { "x": 0, "y": 0 }
    },
    {
      "id": "2",
      "type": "preprocessing",
      "data": { "operations": ["scale", "impute"] },
      "position": { "x": 200, "y": 0 }
    }
  ],
  "edges": [
    { "id": "e1-2", "source": "1", "target": "2" }
  ]
}
```

---

### **8. deployments**

Model deployment instances.

```sql
CREATE TABLE deployments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    model_id UUID REFERENCES models(id) ON DELETE CASCADE,

    name VARCHAR(255) NOT NULL,
    endpoint_url TEXT,
    deployment_type VARCHAR(50) CHECK (deployment_type IN (
        'api', 'docker', 'kubernetes', 'edge', 'mobile',
        'browser', 'sagemaker', 'vertex_ai', 'azure_ml'
    )),

    status VARCHAR(50) DEFAULT 'pending' CHECK (status IN (
        'pending', 'deploying', 'active', 'stopped', 'failed'
    )),

    config JSONB,

    -- Usage metrics
    total_requests INTEGER DEFAULT 0,
    total_errors INTEGER DEFAULT 0,
    avg_latency_ms FLOAT,
    last_request_at TIMESTAMP,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_deployments_user (user_id),
    INDEX idx_deployments_model (model_id),
    INDEX idx_deployments_status (status)
);
```

---

### **9. predictions**

Prediction logs for monitoring and feedback.

```sql
CREATE TABLE predictions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    model_id UUID REFERENCES models(id) ON DELETE CASCADE,
    deployment_id UUID REFERENCES deployments(id) ON DELETE SET NULL,

    input_data JSONB NOT NULL,
    prediction JSONB NOT NULL,
    confidence FLOAT,
    latency_ms INTEGER,

    -- Feedback loop
    feedback JSONB,
    is_correct BOOLEAN,
    user_correction JSONB,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_predictions_model (model_id),
    INDEX idx_predictions_deployment (deployment_id),
    INDEX idx_predictions_created (created_at DESC)
);
```

---

### **10. api_keys**

API key management for programmatic access.

```sql
CREATE TABLE api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,

    name VARCHAR(255) NOT NULL,
    key_hash VARCHAR(255) NOT NULL UNIQUE,
    prefix VARCHAR(20) NOT NULL,

    scopes TEXT[],

    last_used_at TIMESTAMP,
    expires_at TIMESTAMP,
    is_active BOOLEAN DEFAULT true,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_api_keys_user (user_id),
    INDEX idx_api_keys_prefix (prefix)
);
```

**Scopes**:
- `models:read`
- `models:write`
- `datasets:read`
- `datasets:write`
- `predictions:create`
- `admin:*`

---

### **11. usage_logs**

Track resource usage for billing.

```sql
CREATE TABLE usage_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    team_id UUID REFERENCES teams(id) ON DELETE SET NULL,

    resource_type VARCHAR(50) CHECK (resource_type IN (
        'training', 'prediction', 'storage', 'gpu_time', 'api_call'
    )),
    resource_id UUID,

    quantity FLOAT,
    unit VARCHAR(20),
    cost DECIMAL(10, 4),

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_usage_user (user_id),
    INDEX idx_usage_team (team_id),
    INDEX idx_usage_created (created_at DESC)
);
```

---

### **12. subscriptions**

Stripe subscription tracking.

```sql
CREATE TABLE subscriptions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    team_id UUID REFERENCES teams(id) ON DELETE SET NULL,

    stripe_customer_id VARCHAR(255),
    stripe_subscription_id VARCHAR(255) UNIQUE,

    plan VARCHAR(50) CHECK (plan IN ('free', 'pro', 'enterprise')),
    status VARCHAR(50) CHECK (status IN (
        'active', 'trialing', 'past_due', 'canceled', 'incomplete'
    )),

    current_period_start TIMESTAMP,
    current_period_end TIMESTAMP,
    cancel_at TIMESTAMP,
    canceled_at TIMESTAMP,

    trial_start TIMESTAMP,
    trial_end TIMESTAMP,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_subscriptions_user (user_id),
    INDEX idx_subscriptions_stripe (stripe_subscription_id)
);
```

---

### **13. marketplace_listings**

Model marketplace.

```sql
CREATE TABLE marketplace_listings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id UUID REFERENCES models(id) ON DELETE CASCADE,
    seller_id UUID REFERENCES users(id) ON DELETE CASCADE,

    title VARCHAR(255) NOT NULL,
    description TEXT,
    price DECIMAL(10, 2),
    license VARCHAR(50) CHECK (license IN ('MIT', 'Apache-2.0', 'GPL-3.0', 'Commercial', 'Custom')),

    category VARCHAR(100),
    demo_url TEXT,
    documentation_url TEXT,

    downloads INTEGER DEFAULT 0,
    revenue DECIMAL(12, 2) DEFAULT 0,

    rating FLOAT,
    num_ratings INTEGER DEFAULT 0,

    is_active BOOLEAN DEFAULT true,
    is_featured BOOLEAN DEFAULT false,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_marketplace_seller (seller_id),
    INDEX idx_marketplace_category (category),
    INDEX idx_marketplace_active (is_active),
    INDEX idx_marketplace_rating (rating DESC)
);
```

---

### **14. model_reviews**

User reviews for marketplace models.

```sql
CREATE TABLE model_reviews (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id UUID REFERENCES models(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,

    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
    title VARCHAR(255),
    comment TEXT,

    helpful_count INTEGER DEFAULT 0,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(model_id, user_id),
    INDEX idx_reviews_model (model_id),
    INDEX idx_reviews_rating (rating DESC)
);
```

---

### **15. notifications**

In-app notifications.

```sql
CREATE TABLE notifications (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,

    type VARCHAR(50) CHECK (type IN (
        'training_complete', 'training_failed', 'deployment_ready',
        'payment_success', 'payment_failed', 'new_review', 'new_message'
    )),

    title VARCHAR(255) NOT NULL,
    message TEXT,
    data JSONB,

    is_read BOOLEAN DEFAULT false,
    read_at TIMESTAMP,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_notifications_user (user_id),
    INDEX idx_notifications_unread (user_id, is_read, created_at DESC)
);
```

---

### **16. audit_logs**

Security audit trail.

```sql
CREATE TABLE audit_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,

    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(50),
    resource_id UUID,

    ip_address INET,
    user_agent TEXT,

    metadata JSONB,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_audit_user (user_id),
    INDEX idx_audit_created (created_at DESC),
    INDEX idx_audit_action (action)
);
```

**Common Actions**:
- `user.login`
- `user.logout`
- `model.create`
- `model.delete`
- `dataset.upload`
- `deployment.create`
- `api_key.create`
- `subscription.update`

---

## 🔧 INDEXES & PERFORMANCE

### **Additional Indexes**
```sql
-- Full-text search on models
CREATE INDEX idx_models_search ON models USING GIN(
    to_tsvector('english', name || ' ' || COALESCE(description, ''))
);

-- Full-text search on datasets
CREATE INDEX idx_datasets_search ON datasets USING GIN(
    to_tsvector('english', name || ' ' || COALESCE(description, ''))
);

-- Composite indexes for common queries
CREATE INDEX idx_models_user_status ON models(user_id, status);
CREATE INDEX idx_predictions_model_created ON predictions(model_id, created_at DESC);
```

### **Partitioning** (for large tables)
```sql
-- Partition predictions by month
CREATE TABLE predictions_2026_01 PARTITION OF predictions
    FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');

CREATE TABLE predictions_2026_02 PARTITION OF predictions
    FOR VALUES FROM ('2026-02-01') TO ('2026-03-01');
```

---

## 🔒 ROW-LEVEL SECURITY

```sql
-- Enable RLS on models table
ALTER TABLE models ENABLE ROW LEVEL SECURITY;

-- Policy: Users can see their own models + public models + team models
CREATE POLICY model_access_policy ON models
    USING (
        user_id = current_user_id() OR
        is_public = true OR
        team_id IN (SELECT team_id FROM team_members WHERE user_id = current_user_id())
    );
```

---

## 📊 DATABASE VIEWS

### **Active Subscriptions View**
```sql
CREATE VIEW active_subscriptions AS
SELECT
    u.id AS user_id,
    u.email,
    s.plan,
    s.status,
    s.current_period_end
FROM users u
JOIN subscriptions s ON u.id = s.user_id
WHERE s.status = 'active';
```

### **Model Leaderboard View**
```sql
CREATE VIEW model_leaderboard AS
SELECT
    m.id,
    m.name,
    u.username AS author,
    m.accuracy,
    m.downloads,
    ml.rating,
    ml.num_ratings
FROM models m
JOIN users u ON m.user_id = u.id
LEFT JOIN marketplace_listings ml ON m.id = ml.model_id
WHERE m.is_public = true
ORDER BY m.accuracy DESC, m.downloads DESC
LIMIT 100;
```

---

## 🔄 TRIGGERS

### **Update Timestamps**
```sql
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Repeat for all tables with updated_at column
```

### **Increment Downloads**
```sql
CREATE OR REPLACE FUNCTION increment_model_downloads()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE models
    SET downloads = downloads + 1
    WHERE id = NEW.model_id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER model_download_trigger
    AFTER INSERT ON marketplace_listings
    FOR EACH ROW
    EXECUTE FUNCTION increment_model_downloads();
```

---

## 💾 BACKUP STRATEGY

### **Daily Backups**
```bash
# Backup command
pg_dump -U postgres -d automl_platform > backup_$(date +%Y%m%d).sql

# Automated with cron
0 2 * * * /usr/local/bin/backup_db.sh
```

### **Point-in-Time Recovery**
```sql
-- Enable WAL archiving
archive_mode = on
archive_command = 'cp %p /var/lib/postgresql/wal_archive/%f'
```

---

## 📈 MONITORING QUERIES

### **Active Users (Last 24h)**
```sql
SELECT COUNT(DISTINCT user_id)
FROM audit_logs
WHERE action = 'user.login'
  AND created_at > NOW() - INTERVAL '24 hours';
```

### **Training Jobs Status**
```sql
SELECT
    status,
    COUNT(*) as count,
    AVG(training_time) as avg_time_seconds
FROM models
WHERE created_at > NOW() - INTERVAL '7 days'
GROUP BY status;
```

### **Top Users by Models Trained**
```sql
SELECT
    u.username,
    COUNT(m.id) as models_count,
    AVG(m.accuracy) as avg_accuracy
FROM users u
JOIN models m ON u.id = m.user_id
WHERE m.status = 'completed'
GROUP BY u.id, u.username
ORDER BY models_count DESC
LIMIT 10;
```

---

**Last Updated**: January 7, 2026
**Version**: 1.0.0
