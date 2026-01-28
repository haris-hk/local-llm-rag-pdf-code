import requests
import json

def demo_predictions():
    """Demonstrate the API with different types of input"""
    
    print("🎯 Pehchaan AI Model - Prediction Demo")
    print("=" * 50)
    
    BASE_URL = "http://localhost:8080"
    headers = {"Content-Type": "application/json"}
    
    # Test cases with different profiles
    test_cases = [
        {
            "name": "Student with mild learning difficulty",
            "data": {
                "gender": "male",
                "number_of_siblings": 1,
                "educational_history_academic_performance": "Good",
                "educational_history_learning_difficulty": "mild",
                "psychological_analysis_verbal_iq_score": 90,
                "psychological_analysis_non_verbal_iq_score": 85,
                "psychological_analysis_autism_id": False,
                "psychological_analysis_communication": "mild",
                "occupational_therapy_adl_score": 20,
                "occupational_therapy_cognitive_score": 25
            }
        },
        {
            "name": "Student with autism spectrum disorder",
            "data": {
                "gender": "female",
                "number_of_siblings": 2,
                "educational_history_academic_performance": "Adequate",
                "educational_history_learning_difficulty": "moderate",
                "psychological_analysis_verbal_iq_score": 75,
                "psychological_analysis_non_verbal_iq_score": 80,
                "psychological_analysis_autism_id": True,
                "psychological_analysis_communication": "moderate",
                "occupational_therapy_adl_score": 15,
                "occupational_therapy_cognitive_score": 18
            }
        },
        {
            "name": "Student with minimal data",
            "data": {
                "gender": "male",
                "educational_history_academic_performance": "Weak in academics"
            }
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. Testing: {test_case['name']}")
        print("-" * 40)
        
        response = requests.post(f"{BASE_URL}/predict", 
                               json=test_case['data'], 
                               headers=headers)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Prediction: {result['prediction']}")
            print(f"   Confidence: {result.get('confidence', 'N/A'):.3f}" if result.get('confidence') else "   Confidence: N/A")
            print(f"   Features used: {result.get('input_features_count', 'N/A')}")
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
    
    # Get model information
    print(f"\n📊 Model Information")
    print("-" * 40)
    response = requests.get(f"{BASE_URL}/model-info")
    if response.status_code == 200:
        info = response.json()
        print(f"Model Type: {info['model_type']}")
        print(f"Total Features: {info['feature_count']}")
        print(f"Target Classes: {info['target_classes']}")
        print(f"Categorical Features: {len(info['categorical_features'])}")

if __name__ == "__main__":
    demo_predictions()
