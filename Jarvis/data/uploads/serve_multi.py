from fastapi import FastAPI, Request, HTTPException
import joblib
import numpy as np
import pandas as pd
import json
import os
from typing import Dict, Any, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Pehchaan AI Multi-Skill Prediction API", version="2.0.0")

# Global variables for model components
models = {}
target_encoders = {}
label_encoders = {}
scaler = None
feature_info = {}

def load_model_components():
    """Load all model components for multi-output prediction"""
    global models, target_encoders, label_encoders, scaler, feature_info
    
    try:
        logger.info("Loading multi-output model components...")
        
        # Load all components
        models = joblib.load('multi_output_models.pkl')
        target_encoders = joblib.load('target_encoders.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        scaler = joblib.load('scaler.pkl')
        feature_info = joblib.load('feature_info.pkl')
        
        logger.info(f"Multi-output models loaded successfully:")
        logger.info(f"  - {len(models)} skill models")
        logger.info(f"  - {len(feature_info['feature_columns'])} features")
        logger.info(f"  - {len(target_encoders)} target encoders")
        
    except Exception as e:
        logger.error(f"Error loading model components: {e}")
        raise

# Load model components on startup
load_model_components()

@app.get('/')
def root():
    return {
        'message': 'Pehchaan AI Multi-Skill Prediction API is running',
        'version': '2.0.0',
        'models_loaded': len(models),
        'skills_available': list(feature_info.get('target_names', {}).values()),
        'features_count': len(feature_info.get('feature_columns', []))
    }

@app.get('/health')
def health_check():
    """Health check endpoint"""
    return {
        'status': 'healthy',
        'models_loaded': len(models),
        'components_loaded': {
            'models': len(models) > 0,
            'target_encoders': len(target_encoders) > 0,
            'label_encoders': len(label_encoders) > 0,
            'scaler': scaler is not None,
            'feature_info': len(feature_info) > 0
        }
    }

@app.get('/skills')
def get_available_skills():
    """Get list of available skills for prediction"""
    if not feature_info:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    skills = []
    for target, name in feature_info.get('target_names', {}).items():
        if target in models:
            # Get possible recommendation types
            encoder = target_encoders.get(target)
            recommendations = list(encoder.classes_) if encoder else []
            
            skills.append({
                'skill_name': name,
                'target_column': target,
                'possible_recommendations': recommendations,
                'model_available': True
            })
    
    return {
        'total_skills': len(skills),
        'skills': skills
    }

@app.get('/model-info')
def model_info():
    """Get detailed model information"""
    if not models:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    model_details = {}
    for target, model in models.items():
        skill_name = feature_info.get('target_names', {}).get(target, target)
        encoder = target_encoders.get(target)
        
        model_details[skill_name] = {
            'target_column': target,
            'model_type': str(type(model).__name__),
            'classes': list(encoder.classes_) if encoder else [],
            'n_classes': len(encoder.classes_) if encoder else 0
        }
    
    return {
        'total_models': len(models),
        'feature_count': len(feature_info.get('feature_columns', [])),
        'skills': model_details
    }

def preprocess_input(data: Dict[str, Any]) -> np.ndarray:
    """Preprocess input data for prediction"""
    try:
        feature_columns = feature_info['feature_columns']
        
        # Create DataFrame from input
        df = pd.DataFrame([data])
        
        # Handle missing columns - fill with appropriate defaults
        for col in feature_columns:
            if col not in df.columns:
                # Determine appropriate default based on column type
                if col in label_encoders:
                    df[col] = 'unknown'  # String default for categorical
                else:
                    df[col] = 0  # Numeric default
        
        # Select only the required columns in the correct order
        df = df[feature_columns]
        
        # Encode categorical features
        for col, le in label_encoders.items():
            if col in df.columns:
                try:
                    # Handle unknown categories
                    df[col] = df[col].astype(str).fillna('unknown')
                    transformed_values = []
                    for val in df[col]:
                        if val in le.classes_:
                            transformed_values.append(le.transform([val])[0])
                        else:
                            # Use 'unknown' if it exists, otherwise use first class
                            if 'unknown' in le.classes_:
                                transformed_values.append(le.transform(['unknown'])[0])
                            else:
                                transformed_values.append(0)
                    df[col] = transformed_values
                except Exception as e:
                    logger.warning(f"Error encoding column {col}: {e}")
                    df[col] = 0
        
        # Convert all columns to numeric
        for col in df.columns:
            if col not in label_encoders:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    pass
        
        # Fill any remaining missing values
        df = df.fillna(0)
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # Scale features
        X_scaled = scaler.transform(df)
        
        return X_scaled
        
    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        raise HTTPException(status_code=400, detail=f"Preprocessing error: {str(e)}")

@app.post('/predict')
async def predict_all_skills(request: Request):
    """Make predictions for all available skills"""
    if not models:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    try:
        data = await request.json()
        logger.info(f"Received prediction request with {len(data)} features")
        
        # Preprocess input
        X_scaled = preprocess_input(data)
        
        # Make predictions for all skills
        predictions = {}
        
        for target, model in models.items():
            try:
                # Get skill name
                skill_name = feature_info.get('target_names', {}).get(target, target)
                
                # Make prediction
                pred = model.predict(X_scaled)[0]
                pred_proba = model.predict_proba(X_scaled)[0] if hasattr(model, 'predict_proba') else None
                
                # Decode prediction
                encoder = target_encoders.get(target)
                if encoder:
                    pred_label = encoder.inverse_transform([pred])[0]
                    confidence = float(max(pred_proba)) if pred_proba is not None else None
                    
                    predictions[skill_name] = {
                        'recommendation': pred_label,
                        'confidence': confidence,
                        'raw_prediction': int(pred)
                    }
                else:
                    predictions[skill_name] = {
                        'recommendation': str(pred),
                        'confidence': None,
                        'raw_prediction': int(pred)
                    }
                    
            except Exception as e:
                logger.warning(f"Error predicting for {target}: {e}")
                skill_name = feature_info.get('target_names', {}).get(target, target)
                predictions[skill_name] = {
                    'recommendation': 'Error',
                    'confidence': None,
                    'error': str(e)
                }
        
        result = {
            'predictions': predictions,
            'total_skills': len(predictions),
            'input_features_count': len(data),
            'successful_predictions': len([p for p in predictions.values() if 'error' not in p])
        }
        
        logger.info(f"Predictions made for {len(predictions)} skills")
        return result
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post('/predict/{skill_name}')
async def predict_single_skill(skill_name: str, request: Request):
    """Make prediction for a specific skill"""
    if not models:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    # Find the target column for this skill
    target_column = None
    for target, name in feature_info.get('target_names', {}).items():
        if name.lower() == skill_name.lower() or target == skill_name:
            target_column = target
            break
    
    if not target_column or target_column not in models:
        available_skills = list(feature_info.get('target_names', {}).values())
        raise HTTPException(status_code=404, detail=f"Skill '{skill_name}' not found. Available skills: {available_skills}")
    
    try:
        data = await request.json()
        
        # Preprocess input
        X_scaled = preprocess_input(data)
        
        # Make prediction for specific skill
        model = models[target_column]
        pred = model.predict(X_scaled)[0]
        pred_proba = model.predict_proba(X_scaled)[0] if hasattr(model, 'predict_proba') else None
        
        # Decode prediction
        encoder = target_encoders.get(target_column)
        pred_label = encoder.inverse_transform([pred])[0] if encoder else str(pred)
        confidence = float(max(pred_proba)) if pred_proba is not None else None
        
        return {
            'skill': skill_name,
            'recommendation': pred_label,
            'confidence': confidence,
            'raw_prediction': int(pred)
        }
        
    except Exception as e:
        logger.error(f"Prediction error for {skill_name}: {e}")
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.get('/sample-input')
def get_sample_input():
    """Get a sample input for testing"""
    sample = {
        "gender": "male",
        "number_of_siblings": 2,
        "educational_history_academic_performance": "Good",
        "educational_history_learning_difficulty": "mild",
        "psychological_analysis_verbal_iq_score": 85,
        "psychological_analysis_non_verbal_iq_score": 75,
        "psychological_analysis_autism_id": True,
        "psychological_analysis_communication": "mild",
        "occupational_therapy_adl_score": 15,
        "occupational_therapy_cognitive_score": 20
    }
    return {
        "sample_input": sample,
        "usage": "POST this data to /predict for all skills or /predict/{skill_name} for specific skill",
        "available_endpoints": [
            "POST /predict - Get recommendations for all skills",
            "POST /predict/Arts%20And%20Crafts - Get recommendation for specific skill",
            "GET /skills - List all available skills"
        ]
    }
