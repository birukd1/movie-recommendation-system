# Deployment Guide

This guide covers different deployment options for the Movie Recommendation System.

## 🚀 Local Development

### Prerequisites
- Python 3.8+
- pip package manager

### Setup
```bash
# Clone the repository
git clone https://github.com/birukd1/movie-recommendation-system.git
cd movie-recommendation-system

# Install dependencies
pip install -r requirements.txt

# Run the demo
python demo.py

# Start the API server
python api_server.py
```

The API will be available at `http://localhost:8000`

## 🐳 Docker Deployment

### Build and Run
```bash
# Build the Docker image
docker build -f docker/Dockerfile -t movie-rec-system .

# Run the container
docker run -p 8000:8000 movie-rec-system
```

### Using Docker Compose
```bash
# Start all services
docker-compose -f docker/docker-compose.yml up -d

# View logs
docker-compose -f docker/docker-compose.yml logs -f

# Stop services
docker-compose -f docker/docker-compose.yml down
```

## ☁️ Cloud Deployment

### AWS Deployment

#### Option 1: AWS ECS (Elastic Container Service)
1. Push Docker image to ECR
2. Create ECS task definition
3. Deploy to ECS cluster
4. Configure load balancer

#### Option 2: AWS Lambda + API Gateway
1. Package application for Lambda
2. Create Lambda function
3. Set up API Gateway
4. Configure triggers

### Google Cloud Platform

#### Cloud Run Deployment
```bash
# Build and deploy to Cloud Run
gcloud run deploy movie-rec-system \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### Heroku Deployment

#### Create Heroku App
```bash
# Install Heroku CLI and login
heroku login

# Create app
heroku create movie-rec-system-app

# Deploy
git push heroku main
```

#### Procfile
```
web: python api_server.py
```

## 🔧 Production Configuration

### Environment Variables
```bash
# API Configuration
export API_HOST=0.0.0.0
export API_PORT=8000
export API_WORKERS=4

# Model Configuration
export MODEL_PATH=/app/models/recommendation_model.pkl
export N_RECOMMENDATIONS=10

# Database (if using external DB)
export DATABASE_URL=postgresql://user:pass@host:port/db

# Redis (if using caching)
export REDIS_URL=redis://host:port/0
```

### Performance Optimization

#### 1. Model Caching
```python
# Use Redis for caching recommendations
import redis
r = redis.Redis(host='localhost', port=6379, db=0)

# Cache user recommendations
def get_cached_recommendations(user_id, method):
    cache_key = f"rec:{user_id}:{method}"
    cached = r.get(cache_key)
    if cached:
        return json.loads(cached)
    return None
```

#### 2. Database Integration
```python
# Use PostgreSQL for storing ratings
import psycopg2
import pandas as pd

def load_ratings_from_db():
    conn = psycopg2.connect(DATABASE_URL)
    query = "SELECT user_id, movie_id, rating FROM ratings"
    return pd.read_sql(query, conn)
```

#### 3. Batch Processing
```python
# Pre-compute recommendations for active users
def batch_compute_recommendations():
    active_users = get_active_users()
    for user_id in active_users:
        recommendations = rec_system.get_recommendations(user_id)
        cache_recommendations(user_id, recommendations)
```

## 📊 Monitoring and Logging

### Health Checks
```python
# Add comprehensive health checks
@app.get("/health/detailed")
async def detailed_health():
    return {
        "status": "healthy",
        "model_loaded": rec_system is not None,
        "memory_usage": get_memory_usage(),
        "response_time": measure_response_time(),
        "cache_status": check_cache_connection()
    }
```

### Logging Configuration
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
```

### Metrics Collection
```python
# Use Prometheus for metrics
from prometheus_client import Counter, Histogram, generate_latest

REQUEST_COUNT = Counter('requests_total', 'Total requests')
REQUEST_LATENCY = Histogram('request_duration_seconds', 'Request latency')

@app.middleware("http")
async def add_metrics(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    REQUEST_COUNT.inc()
    REQUEST_LATENCY.observe(time.time() - start_time)
    return response
```

## 🔒 Security Considerations

### API Security
```python
# Add API key authentication
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

@app.get("/recommendations")
async def get_recommendations(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    if not validate_api_key(credentials.credentials):
        raise HTTPException(status_code=401, detail="Invalid API key")
    # ... rest of the function
```

### Rate Limiting
```python
# Implement rate limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.get("/recommendations")
@limiter.limit("10/minute")
async def get_recommendations(request: Request):
    # ... function implementation
```

## 📈 Scaling Strategies

### Horizontal Scaling
- Use load balancers (nginx, HAProxy)
- Deploy multiple API instances
- Implement session affinity if needed

### Vertical Scaling
- Increase CPU and memory resources
- Optimize model size and complexity
- Use GPU acceleration for large models

### Database Scaling
- Read replicas for recommendation queries
- Sharding for large user bases
- Caching layer (Redis/Memcached)

## 🔄 CI/CD Pipeline

### GitHub Actions Example
```yaml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: python -m pytest test_system.py

  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to production
        run: |
          # Add deployment commands here
          echo "Deploying to production..."
```

## 📋 Maintenance

### Model Updates
- Schedule regular model retraining
- A/B test new model versions
- Monitor model performance metrics

### Data Pipeline
- Implement ETL processes for new data
- Data quality checks and validation
- Backup and recovery procedures

### System Updates
- Regular security updates
- Dependency management
- Performance monitoring and optimization