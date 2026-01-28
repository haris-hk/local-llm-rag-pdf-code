"""
Analysis of Pehchaan AI Model Predictions
=========================================

This script analyzes what the current model is predicting and identifies limitations.
"""

import pandas as pd
import joblib
import requests

def analyze_current_model():
    print("🔍 CURRENT MODEL ANALYSIS")
    print("=" * 50)
    
    # Load the original data to understand the target
    df = pd.read_csv('dataset.csv')
    target_col = 'skill_recommendations_arts_and_crafts_recommendation'
    
    print(f"📊 The model predicts: {target_col}")
    print(f"📈 Original values distribution:")
    value_counts = df[target_col].value_counts()
    for value, count in value_counts.items():
        print(f"   {value}: {count} students")
    
    # Load the target encoder to understand the mapping
    target_encoder = joblib.load('target_encoder.pkl')
    
    # Try to reverse engineer the mapping
    print(f"\n🔢 Model output classes: {target_encoder.classes_}")
    
    # The issue: target was converted to string then encoded
    # Let's see what the string values would be
    original_values = df[target_col].dropna().unique()
    print(f"\n❌ PROBLEM IDENTIFIED:")
    print(f"   Original categorical values: {sorted(original_values)}")
    print(f"   But model returns: {target_encoder.classes_}")
    print(f"   The .astype(str) in training converted everything to string representations!")
    
    return target_col, original_values

def test_current_predictions():
    print(f"\n🧪 TESTING CURRENT PREDICTIONS")
    print("=" * 50)
    
    BASE_URL = "http://localhost:8080"
    headers = {"Content-Type": "application/json"}
    
    # Test with different student profiles
    test_cases = [
        {
            "name": "High-performing student",
            "data": {
                "gender": "female",
                "educational_history_academic_performance": "Good",
                "psychological_analysis_verbal_iq_score": 95,
                "psychological_analysis_autism_id": False
            }
        },
        {
            "name": "Student with autism",
            "data": {
                "gender": "male",
                "educational_history_academic_performance": "Adequate",
                "psychological_analysis_autism_id": True,
                "psychological_analysis_communication": "moderate"
            }
        }
    ]
    
    for test_case in test_cases:
        print(f"\n👤 {test_case['name']}:")
        try:
            response = requests.post(f"{BASE_URL}/predict", 
                                   json=test_case['data'], 
                                   headers=headers)
            if response.status_code == 200:
                result = response.json()
                prediction = result['prediction']
                confidence = result.get('confidence', 'N/A')
                
                # Try to interpret the prediction
                interpretation = interpret_prediction(prediction)
                print(f"   Raw prediction: {prediction}")
                print(f"   Confidence: {confidence:.3f}" if confidence != 'N/A' else f"   Confidence: {confidence}")
                print(f"   Likely meaning: {interpretation}")
            else:
                print(f"   Error: {response.text}")
        except Exception as e:
            print(f"   Error: {e}")

def interpret_prediction(prediction):
    """Try to interpret the numeric prediction"""
    # Based on the original data distribution:
    # 'Not Recommended': 31, 'Recommended': 4, 'Highly Recommended': 3, 'One-Term': 2
    # LabelEncoder typically sorts alphabetically, so likely:
    interpretations = {
        '0': "Highly Recommended (alphabetically first)",
        '1': "Not Recommended (most common class)",  
        '2': "One-Term",
        '3': "Recommended"
    }
    return interpretations.get(str(prediction), f"Unknown mapping for {prediction}")

def suggest_improvements():
    print(f"\n💡 SUGGESTED IMPROVEMENTS")
    print("=" * 50)
    
    print("1. 🎯 MULTI-OUTPUT MODEL:")
    print("   Currently predicting: Arts & Crafts recommendation only")
    print("   Should predict: ALL skill recommendations")
    
    # Count all skill recommendation columns
    df = pd.read_csv('dataset.csv')
    skill_cols = [col for col in df.columns if 'skill_recommendations' in col and 'recommendation' in col]
    
    print(f"   Available skills: {len(skill_cols)} different skills")
    for i, col in enumerate(skill_cols[:5], 1):
        skill_name = col.replace('skill_recommendations_', '').replace('_recommendation', '')
        print(f"   {i}. {skill_name.replace('_', ' ').title()}")
    if len(skill_cols) > 5:
        print(f"   ... and {len(skill_cols) - 5} more skills")
    
    print(f"\n2. 🔧 FIX LABEL ENCODING:")
    print("   Current: Converts categories to strings then numbers")
    print("   Should: Keep original categorical labels")
    
    print(f"\n3. 🏗️ MODEL ARCHITECTURE OPTIONS:")
    print("   Option A: Multi-output model (one model, multiple targets)")
    print("   Option B: Separate models per skill")
    print("   Option C: Hierarchical model (recommendation type → skill type)")

if __name__ == "__main__":
    print("🎯 PEHCHAAN AI MODEL ANALYSIS")
    print("=" * 60)
    
    analyze_current_model()
    test_current_predictions()
    suggest_improvements()
    
    print(f"\n✅ SUMMARY:")
    print("The current model only predicts Arts & Crafts recommendations,")
    print("but the system has data for 14+ different skills!")
    print("Consider building a comprehensive multi-skill prediction system.")
