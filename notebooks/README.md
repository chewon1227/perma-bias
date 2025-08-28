# Career-Pathway Installation Guide

## Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for local LLM inference)

## Installation Methods

### Method 1: Using pip (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/Career-Pathway.git
cd Career-Pathway

# Install the package and dependencies
pip install -e .

# The NLTK data will be automatically downloaded during installation
```

### Method 2: Using requirements.txt

```bash
# Clone the repository
git clone https://github.com/yourusername/Career-Pathway.git
cd Career-Pathway

# Install dependencies
pip install -r requirements.txt

# Download NLTK data manually
python download_nltk_data.py

# Install the package in development mode
pip install -e .
```

### Method 3: For Development

```bash
# Clone the repository
git clone https://github.com/yourusername/Career-Pathway.git
cd Career-Pathway

# Install with development dependencies
pip install -e ".[dev]"

# Or install GPU version if you have CUDA
pip install -e ".[gpu]"
```

## Verify Installation

```python
# Test basic functionality
from careerpathway.utils import load_api_config
from careerpathway.data import load_reddit_data

# Test NLTK
from nltk.tokenize import word_tokenize
from nrclex import NRCLex

print("Installation successful!")
```

## Environment Setup

1. Create API configuration file:
```bash
cp configs/api.yaml.example configs/api.yaml
# Edit configs/api.yaml with your actual API keys
```

2. Set environment variables (optional):
```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GOOGLE_API_KEY="your-google-key"
```

## Troubleshooting

### NLTK Data Issues
If NLTK data download fails during installation:
```bash
python download_nltk_data.py
```

### GPU Issues
For CUDA-related issues:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### SSL Certificate Issues
If you encounter SSL issues with NLTK downloads:
```bash
python -c "import ssl; ssl._create_default_https_context = ssl._create_unverified_context"
python download_nltk_data.py
```