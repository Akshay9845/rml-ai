# RML-AI Deployment Guide

This guide covers deploying the RML-AI system in various environments.

## 🚀 Quick Start

### Local Development

```bash
# Clone and setup
git clone https://github.com/your-username/rml-ai.git
cd rml-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run interactive CLI
python -m rml_ai.cli

# Or start API server
python -m rml_ai.server
```

## 🐳 Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Set environment variables
ENV PYTHONPATH=/app/src
ENV RML_DEVICE=cpu
ENV RML_DISABLE_WEB_SEARCH=1
ENV RML_DISABLE_WORLD_KNOWLEDGE=1

# Expose port
EXPOSE 8000

# Run the server
CMD ["python", "-m", "rml_ai.server"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  rml-ai:
    build: .
    ports:
      - "8000:8000"
    environment:
      - RML_DEVICE=cpu
      - RML_DATASET_PATH=/app/data/rml_data.jsonl
      - RML_MAX_ENTRIES=1000
    volumes:
      - ./data:/app/data
    restart: unless-stopped
```

## ☁️ Cloud Deployment

### AWS Lambda

```python
# lambda_function.py
import json
from rml_ai import RMLSystem, RMLConfig

def lambda_handler(event, context):
    try:
        # Initialize RML system
        config = RMLConfig()
        rml = RMLSystem(config)
        
        # Get query from event
        query = event.get('query', '')
        if not query:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'No query provided'})
            }
        
        # Process query
        response = rml.query(query)
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'answer': response.answer,
                'sources': response.sources,
                'response_ms': response.response_ms
            })
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
```

### Google Cloud Run

```yaml
# cloudbuild.yaml
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/rml-ai', '.']
  
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/rml-ai']
  
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    args:
      - gcloud
      - run
      - deploy
      - rml-ai
      - --image
      - gcr.io/$PROJECT_ID/rml-ai
      - --region
      - us-central1
      - --platform
      - managed
      - --allow-unauthenticated
```

## 🔧 Production Configuration

### Environment Variables

```bash
# Production environment variables
export RML_DEVICE="cpu"                    # Force CPU for stability
export RML_ENCODER_BATCH_SIZE=16           # Larger batches for production
export RML_ENCODER_MAX_LEN=256             # Longer sequences
export RML_MAX_ENTRIES=10000               # More entries
export RML_DISABLE_WEB_SEARCH=1            # Disable external search
export RML_DISABLE_WORLD_KNOWLEDGE=1       # Disable world knowledge
export RML_ENCODER_MODEL="intfloat/e5-base-v2"
export RML_DECODER_MODEL="/path/to/fine-tuned/model"
```

### Performance Tuning

```python
# config.py
class ProductionConfig(RMLConfig):
    def __init__(self):
        super().__init__()
        self.encoder_batch_size = 32      # Larger batches
        self.encoder_max_length = 512     # Longer sequences
        self.max_entries = 50000          # More entries
        self.device = "cpu"               # Stable CPU
        self.enable_caching = True        # Enable response caching
        self.cache_ttl = 3600            # Cache TTL in seconds
```

## 📊 Monitoring & Logging

### Health Checks

```python
@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    try:
        # Check RML system
        rml_status = rml_system is not None
        
        # Check memory
        memory_stats = rml_system.memory.get_stats() if rml_system else {}
        
        # Check models
        encoder_status = rml_system.encoder.model is not None if rml_system else False
        decoder_status = rml_system.decoder.model is not None if rml_system else False
        
        return {
            "status": "healthy" if all([rml_status, encoder_status, decoder_status]) else "unhealthy",
            "timestamp": time.time(),
            "rml_system": rml_status,
            "encoder": encoder_status,
            "decoder": decoder_status,
            "memory": memory_stats
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }
```

### Logging Configuration

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rml-ai.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
```

## 🔒 Security Considerations

### API Security

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API token"""
    token = credentials.credentials
    if not is_valid_token(token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    return token

@app.post("/chat")
async def chat(request: ChatRequest, token: str = Depends(verify_token)):
    # Process chat request
    pass
```

### Rate Limiting

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/chat")
@limiter.limit("10/minute")
async def chat(request: ChatRequest):
    # Process chat request
    pass
```

## 📈 Scaling Considerations

### Horizontal Scaling

- Use load balancers for multiple instances
- Implement shared memory stores (Redis, etc.)
- Use distributed file systems for model storage

### Vertical Scaling

- Increase CPU/memory resources
- Use GPU instances for encoding
- Optimize batch sizes for your hardware

## 🚨 Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch sizes or use CPU
2. **Model Loading Failures**: Check model paths and permissions
3. **Slow Responses**: Optimize batch sizes and device selection
4. **API Timeouts**: Increase timeout values and optimize queries

### Debug Mode

```bash
export RML_DEBUG=1
export RML_LOG_LEVEL=DEBUG
python -m rml_ai.server
```

## 📚 Additional Resources

- [Performance Tuning Guide](performance.md)
- [Model Fine-tuning Guide](fine-tuning.md)
- [API Reference](api.md)
- [Contributing Guide](contributing.md) 