import requests
import json

def test_multi_output_api():
    """Test the multi-output API comprehensively"""
    
    print("🎯 Testing Pehchaan AI Multi-Output API")
    print("=" * 50)
    
    BASE_URL = "http://localhost:8080"
    headers = {"Content-Type": "application/json"}
    
    # Sample input data
    sample_data = {
        "gender": "male",
        "number_of_siblings": 1,
        "educational_history_academic_performance": "Good",
        "educational_history_learning_difficulty": "mild",
        "psychological_analysis_verbal_iq_score": 90,
        "psychological_analysis_non_verbal_iq_score": 85,
        "psychological_analysis_autism_id": False,
        "psychological_analysis_communication": "mild"
    }
    
    try:
        # Test 1: Root endpoint
        print("\n1. Testing root endpoint...")
        response = requests.get(f"{BASE_URL}/", timeout=10)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"API Version: {data.get('version')}")
            print(f"Models loaded: {data.get('models_loaded')}")
            print(f"Skills available: {len(data.get('skills_available', []))}")
            print(f"Skills: {data.get('skills_available', [])[:5]}...")  # Show first 5
        
        # Test 2: Get available skills
        print("\n2. Testing available skills...")
        response = requests.get(f"{BASE_URL}/skills", timeout=10)
        if response.status_code == 200:
            skills_data = response.json()
            print(f"Total skills: {skills_data.get('total_skills')}")
            for skill in skills_data.get('skills', [])[:3]:  # Show first 3
                print(f"   - {skill['skill_name']}: {skill['possible_recommendations']}")
        
        # Test 3: Multi-skill prediction
        print("\n3. Testing multi-skill prediction...")
        response = requests.post(f"{BASE_URL}/predict", 
                               json=sample_data, 
                               headers=headers,
                               timeout=30)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Successful predictions: {result.get('successful_predictions')}/{result.get('total_skills')}")
            print(f"Sample predictions:")
            for skill, pred in list(result.get('predictions', {}).items())[:5]:
                if 'error' not in pred:
                    print(f"   {skill}: {pred['recommendation']} (confidence: {pred.get('confidence', 'N/A'):.3f})" if pred.get('confidence') else f"   {skill}: {pred['recommendation']}")
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
        
        # Test 4: Single skill prediction
        print("\n4. Testing single skill prediction...")
        response = requests.post(f"{BASE_URL}/predict/Arts%20And%20Crafts", 
                               json=sample_data, 
                               headers=headers,
                               timeout=10)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Arts & Crafts: {result['recommendation']} (confidence: {result.get('confidence', 'N/A'):.3f})" if result.get('confidence') else f"✅ Arts & Crafts: {result['recommendation']}")
        else:
            print(f"❌ Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Connection error: {e}")
        print("Make sure the API server is running on port 8080")

if __name__ == "__main__":
    test_multi_output_api()
