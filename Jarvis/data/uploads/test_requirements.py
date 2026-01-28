"""
Test script to verify all required packages can be imported successfully
"""

def test_imports():
    """Test importing all required packages"""
    
    print("Testing package imports...")
    
    try:
        import pandas as pd
        print("✓ pandas imported successfully")
    except ImportError as e:
        print(f"✗ pandas import failed: {e}")
    
    try:
        import numpy as np
        print("✓ numpy imported successfully")
    except ImportError as e:
        print(f"✗ numpy import failed: {e}")
    
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler, OrdinalEncoder
        from sklearn.impute import KNNImputer
        print("✓ scikit-learn components imported successfully")
    except ImportError as e:
        print(f"✗ scikit-learn import failed: {e}")
    
    try:
        import joblib
        print("✓ joblib imported successfully")
    except ImportError as e:
        print(f"✗ joblib import failed: {e}")
    
    try:
        from transformers import pipeline
        print("✓ transformers imported successfully")
    except ImportError as e:
        print(f"✗ transformers import failed: {e}")
    
    try:
        import torch
        print(f"✓ torch imported successfully (CUDA available: {torch.cuda.is_available()})")
    except ImportError as e:
        print(f"✗ torch import failed: {e}")
    
    try:
        from spellchecker import SpellChecker
        print("✓ pyspellchecker imported successfully")
    except ImportError as e:
        print(f"✗ pyspellchecker import failed: {e}")
    
    try:
        from Levenshtein import distance as edit_distance
        print("✓ python-Levenshtein imported successfully")
    except ImportError as e:
        print(f"✗ python-Levenshtein import failed: {e}")
    
    try:
        import json
        print("✓ json (built-in) imported successfully")
    except ImportError as e:
        print(f"✗ json import failed: {e}")
    
    print("\nImport test completed!")

if __name__ == "__main__":
    test_imports()
