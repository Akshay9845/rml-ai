# RML-AI Project Structure

This document outlines the clean, professional structure of the RML-AI repository.

## 📁 Directory Structure

```
rml-ai/
├── README.md                 # Professional README with badges and documentation
├── LICENSE                   # MIT License
├── setup.py                  # Package setup and configuration
├── requirements.txt          # Clean dependencies list
├── .env.example             # Environment configuration template
├── .gitignore               # Comprehensive gitignore
├── PROJECT_STRUCTURE.md     # This file
│
├── src/                     # Source code package
│   └── rml_ai/             # Main package
│       ├── __init__.py      # Package initialization and exports
│       ├── core.py          # Core RML system classes
│       ├── memory.py        # Memory store and search
│       ├── config.py        # Configuration management
│       ├── server.py        # FastAPI server
│       └── cli.py           # Command line interface
│
├── examples/                 # Usage examples
│   └── basic_usage.py       # Basic RML system usage
│
├── docs/                     # Documentation
│   └── deployment.md        # Deployment guide
│
├── scripts/                  # Utility scripts
│   └── setup_data.py        # Data directory setup
│
├── tests/                    # Test suite
│   └── test_basic.py        # Basic unit tests
│
├── data/                     # Data directory (created by setup)
│   ├── rml_data.jsonl       # Sample dataset
│   └── README.md            # Data directory documentation
│
├── models/                   # Model storage (not in git)
├── outputs/                  # Output files (not in git)
└── venv/                     # Virtual environment (not in git)
```

## 🚀 Key Features

### Clean Architecture
- **Modular Design**: Separated concerns with clear interfaces
- **Professional Structure**: Follows Python packaging best practices
- **Type Hints**: Full type annotations for better code quality
- **Error Handling**: Comprehensive error handling and fallbacks

### Core Components
- **RMLSystem**: Main orchestrator class
- **RMLEncoder**: E5-based semantic encoder
- **RMLDecoder**: Phi-based text generator
- **MemoryStore**: Vector-based memory with semantic search
- **RMLConfig**: Environment-based configuration

### Multiple Interfaces
- **CLI**: Interactive command-line interface
- **API**: FastAPI-based HTTP server
- **Library**: Direct Python import and usage

## 🔧 Configuration

### Environment Variables
```bash
# Model Configuration
RML_ENCODER_MODEL=intfloat/e5-base-v2
RML_DECODER_MODEL=microsoft/phi-1_5

# Device Configuration
RML_DEVICE=auto              # auto, cpu, mps, cuda

# Dataset Configuration
RML_DATASET_PATH=data/rml_data.jsonl
RML_API_ENTRIES=1000

# Performance Configuration
RML_ENCODER_BATCH_SIZE=8
RML_ENCODER_MAX_LEN=192

# Feature Flags
RML_DISABLE_WEB_SEARCH=1
RML_DISABLE_WORLD_KNOWLEDGE=1
```

## 📦 Installation & Usage

### Quick Start
```bash
# Clone repository
git clone https://github.com/your-username/rml-ai.git
cd rml-ai

# Setup environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup data
python scripts/setup_data.py

# Run interactive CLI
python -m rml_ai.cli

# Or start API server
python -m rml_ai.server
```

### As a Library
```python
from rml_ai import RMLSystem, RMLConfig

config = RMLConfig()
rml = RMLSystem(config)
response = rml.query("What is machine learning?")
print(response.answer)
```

## 🧪 Testing

### Run Tests
```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python tests/test_basic.py

# Run with coverage
python -m pytest tests/ --cov=rml_ai
```

## 🚀 Deployment

### Docker
```bash
# Build image
docker build -t rml-ai .

# Run container
docker run -p 8000:8000 rml-ai
```

### Cloud Deployment
- **AWS Lambda**: Serverless deployment
- **Google Cloud Run**: Container-based deployment
- **Kubernetes**: Scalable container orchestration

## 📚 Documentation

- **README.md**: Comprehensive project overview
- **docs/deployment.md**: Detailed deployment guide
- **examples/**: Usage examples and tutorials
- **API Docs**: Auto-generated FastAPI documentation at `/docs`

## 🔒 Security & Best Practices

- **Environment Variables**: Secure configuration management
- **Input Validation**: Pydantic models for request validation
- **Error Handling**: Graceful error handling without information leakage
- **Rate Limiting**: Built-in rate limiting support
- **Authentication**: Token-based authentication support

## 🎯 What Was Cleaned Up

### Removed (Unnecessary Files)
- ❌ Test scripts and validation files
- ❌ Demo runners and pipeline files
- ❌ Old training scripts and logs
- ❌ Cache directories and temporary files
- ❌ Backup and old version files
- ❌ Unused utility scripts

### Kept (Essential Components)
- ✅ Core RML system implementation
- ✅ Tested and working pipeline
- ✅ Professional documentation
- ✅ Clean package structure
- ✅ Essential examples and tests
- ✅ Deployment configurations

## 🌟 Ready for Production

This clean structure is now ready for:
- **GitHub**: Professional repository hosting
- **Hugging Face**: Model and space deployment
- **PyPI**: Python package distribution
- **Enterprise**: Production deployment
- **Research**: Academic and research use

## 📞 Support

- **Issues**: GitHub Issues for bug reports
- **Discussions**: GitHub Discussions for questions
- **Documentation**: Comprehensive docs and examples
- **Examples**: Working code examples for all use cases

---

**RML-AI**: A clean, professional, production-ready AI system! 🚀 