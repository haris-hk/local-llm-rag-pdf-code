"""
Check what's in the feature_info.pkl file
"""
import joblib

def inspect_feature_info():
    try:
        feature_info = joblib.load('feature_info.pkl')
        print("Feature info keys:", list(feature_info.keys()))
        for key, value in feature_info.items():
            print(f"{key}: {type(value)} - {len(value) if hasattr(value, '__len__') else value}")
            if hasattr(value, '__len__') and len(value) < 10:
                print(f"  Content: {value}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_feature_info()
