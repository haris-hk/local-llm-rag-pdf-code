# Installation Instructions for Pehchaan AI

## Prerequisites
- Python 3.8 or higher
- pip package manager

## Installation Steps

### 1. Install CPU-only version (for basic usage):
```bash
pip install -r requirements.txt
```

### 2. For GPU/CUDA support (optional, recommended for faster processing):
First install the CPU version, then upgrade torch:
```bash
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Verify installation:
```bash
python test_requirements.py
```

## Package Versions
The requirements.txt includes minimum version specifications to ensure compatibility:

- **pandas>=1.5.0**: Data manipulation and analysis
- **numpy>=1.21.0**: Numerical computing
- **scikit-learn>=1.2.0**: Machine learning library
- **joblib>=1.2.0**: Lightweight pipelining
- **transformers>=4.21.0**: NLP models (Hugging Face)
- **torch>=1.12.0**: Deep learning framework
- **pyspellchecker>=0.7.0**: Spell checking functionality
- **python-Levenshtein>=0.20.0**: Fast string distance calculations

## Troubleshooting

### Common Issues:

1. **Import errors**: Make sure you're using Python 3.8+
2. **CUDA not available**: Install GPU version of PyTorch if you have a CUDA-capable GPU
3. **Memory issues**: Reduce batch size or use CPU if running out of GPU memory

### Check CUDA availability:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
```
