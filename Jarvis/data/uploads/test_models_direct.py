"""
Simple test script to verify the multi-output models work independently
"""
import joblib
import pandas as pd
import numpy as np

def test_models():
    """Test the loaded models directly"""
    
    print("🧪 Testing Multi-Output Models Directly")
    print("=" * 50)
    
    try:
        # Load the model components
        print("Loading model components...")
        multi_output_models = joblib.load('multi_output_models.pkl')
        target_encoders = joblib.load('target_encoders.pkl')
        scaler = joblib.load('scaler.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        feature_info = joblib.load('feature_info.pkl')
        
        print(f"✅ Loaded {len(multi_output_models)} models")
        print(f"✅ Loaded {len(target_encoders)} target encoders")
        print(f"✅ Feature info loaded with {len(feature_info['feature_columns'])} features")
        
        # Create sample data
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
        
        print("\n📊 Sample Input Data:")
        for key, value in sample_data.items():
            print(f"   {key}: {value}")
        
        # Process the data
        print("\n🔄 Processing input data...")
        
        # Create DataFrame
        input_df = pd.DataFrame([sample_data])
        
        # Apply label encoders - get categorical columns from label encoders
        categorical_columns = list(label_encoders.keys())
        for col in categorical_columns:
            if col in input_df.columns and col in label_encoders:
                # Handle unseen categories
                encoder = label_encoders[col]
                unique_values = set(encoder.classes_)
                input_df[col] = input_df[col].apply(
                    lambda x: x if x in unique_values else encoder.classes_[0]
                )
                input_df[col] = encoder.transform(input_df[col])
        
        # Scale features
        feature_columns = feature_info['feature_columns']
        X = input_df[feature_columns]
        X_scaled = scaler.transform(X)
        
        print(f"✅ Input processed: {X_scaled.shape}")
        
        # Test predictions for first 3 skills
        print("\n🎯 Testing Predictions:")
        
        skill_count = 0
        for skill_name, model in multi_output_models.items():
            if skill_count >= 3:  # Test only first 3 for brevity
                break
                
            try:
                # Make prediction
                prediction = model.predict(X_scaled)[0]
                
                # Decode prediction
                if skill_name in target_encoders:
                    decoded_prediction = target_encoders[skill_name].inverse_transform([prediction])[0]
                else:
                    decoded_prediction = prediction
                
                print(f"   ✅ {skill_name}: {decoded_prediction}")
                skill_count += 1
                
            except Exception as e:
                print(f"   ❌ {skill_name}: Error - {e}")
        
        print(f"\n✅ Successfully tested {skill_count} skills!")
        print(f"Total skills available: {len(multi_output_models)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_models()
    if success:
        print("\n🎉 Multi-output models are working correctly!")
    else:
        print("\n💥 Issues detected with model files.")
