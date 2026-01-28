import requests
import json

# API base URL
BASE_URL = "http://localhost:8080"

def test_api():
    print("🧪 Testing Pehchaan AI Model API...")
    
    # Test 1: Root endpoint
    print("\n1. Testing root endpoint...")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    
    # Test 2: Health check
    print("\n2. Testing health check...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    
    # Test 3: Model info
    print("\n3. Testing model info...")
    response = requests.get(f"{BASE_URL}/model-info")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Model type: {data.get('model_type')}")
        print(f"Features: {data.get('feature_count')}")
        print(f"Target classes: {data.get('target_classes')}")
    
    # Test 4: Get sample input
    print("\n4. Getting sample input...")
    response = requests.get(f"{BASE_URL}/sample-input")
    sample_data = response.json()["sample_input"]
    print(f"Sample input: {sample_data}")
    
    # Test 5: Make prediction with sample data
    print("\n5. Testing prediction with sample data...")
    headers = {"Content-Type": "application/json"}
    response = requests.post(f"{BASE_URL}/predict", 
                           json=sample_data, 
                           headers=headers)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Prediction: {result.get('prediction')}")
        print(f"Confidence: {result.get('confidence')}")
        print(f"Input features count: {result.get('input_features_count')}")
    else:
        print(f"Error: {response.text}")
    
    # Test 6: Test with minimal data
    print("\n6. Testing prediction with minimal data...")
    minimal_data = {
        "gender": "female",
        "psychological_analysis_autism_id": False
    }
    response = requests.post(f"{BASE_URL}/predict", 
                           json=minimal_data, 
                           headers=headers)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Prediction: {result.get('prediction')}")
        print(f"Confidence: {result.get('confidence')}")
    else:
        print(f"Error: {response.text}")
    
    print("\n✅ API testing completed!")

if __name__ == "__main__":
    test_api()
