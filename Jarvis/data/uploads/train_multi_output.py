import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings
import os
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    xgb_available = True
except ImportError:
    print("⚠️ XGBoost not installed. Install with: pip install xgboost")
    xgb_available = False

print("🚀 Starting Multi-Output AI model training...")

# Load the dataset
try:
    data_path = "dataset_copy.csv" if os.path.exists("dataset_copy.csv") else "dataset.csv"
except Exception:
    data_path = "dataset.csv"

df = pd.read_csv(data_path)
print(f"📊 Dataset loaded from {data_path}: {df.shape[0]} rows, {df.shape[1]} columns")

# Define all skill recommendation targets
skill_targets = [
    'skill_recommendations_arts_and_crafts_recommendation',
    'skill_recommendations_carpentry_recommendation', 
    'skill_recommendations_culinary_recommendation',
    'skill_recommendations_ict_recommendation',
    'skill_recommendations_block_printing_recommendation',
    'skill_recommendations_screen_printing_recommendation',
    'skill_recommendations_machine_embroidery_recommendation',
    'skill_recommendations_music_recommendation',
    'skill_recommendations_pottery_painting_recommendation',
    'skill_recommendations_polishing_recommendation',
    'skill_recommendations_zari_work_recommendation',
    'skill_recommendations_beauty_salon_recommendation',
    'skill_recommendations_tailoring_recommendation',
    'skill_recommendations_gardening_recommendation'
]

# Filter to only include targets that exist in the dataset
available_targets = [target for target in skill_targets if target in df.columns]
print(f"🎯 Found {len(available_targets)} skill recommendation targets:")
for i, target in enumerate(available_targets, 1):
    skill_name = target.replace('skill_recommendations_', '').replace('_recommendation', '')
    print(f"   {i}. {skill_name.replace('_', ' ').title()}")

if not available_targets:
    print("❌ No skill recommendation targets found in dataset")
    exit(1)

# Prepare features (exclude all target columns and metadata)
exclude_columns = [
    'name', 'father_name', 'id', 'assessment_date', 'date_of_birth'
] + [col for col in df.columns if 'skill_recommendations' in col]

feature_columns = [col for col in df.columns if col not in exclude_columns]
print(f"📈 Number of features: {len(feature_columns)}")

# Create feature matrix
X = df[feature_columns].copy()

# Create multi-target matrix
y_dict = {}
target_encoders = {}
valid_rows = []

print(f"\n🔍 Analyzing target distributions:")
for target in available_targets:
    # Remove rows where this target is null
    non_null_mask = df[target].notna()
    
    if non_null_mask.sum() == 0:
        print(f"   ❌ {target}: No valid data")
        continue
        
    # Encode the target
    le = LabelEncoder()
    valid_values = df.loc[non_null_mask, target].astype(str)
    y_encoded = le.fit_transform(valid_values)
    
    # Store the encoded target
    y_dict[target] = pd.Series(index=df.index, dtype='float64')
    y_dict[target].loc[non_null_mask] = y_encoded
    target_encoders[target] = le
    
    # Track which rows have valid data for this target
    valid_rows.append(non_null_mask)
    
    # Show distribution
    value_counts = df[target].value_counts()
    skill_name = target.replace('skill_recommendations_', '').replace('_recommendation', '')
    print(f"   ✅ {skill_name.title()}: {len(value_counts)} classes, {non_null_mask.sum()} samples")

# Find rows that have at least one valid target
if valid_rows:
    has_any_target = pd.concat(valid_rows, axis=1).any(axis=1)
    print(f"\n📋 Rows with at least one valid target: {has_any_target.sum()}")
else:
    print("❌ No valid targets found")
    exit(1)

# Filter to rows with at least one valid target
X_filtered = X.loc[has_any_target].copy()

# Create target matrix with NaN for missing values
y_matrix = pd.DataFrame(index=X_filtered.index)
for target in y_dict:
    y_matrix[target] = y_dict[target].loc[X_filtered.index]

print(f"🎯 Final dataset: {X_filtered.shape[0]} rows, {X_filtered.shape[1]} features, {y_matrix.shape[1]} targets")

# Handle missing values in features
print(f"\n🔧 Preprocessing features...")
X_processed = X_filtered.fillna('unknown')

# Encode categorical variables
label_encoders = {}
for column in X_processed.columns:
    if X_processed[column].dtype == 'object':
        le = LabelEncoder()
        X_processed[column] = le.fit_transform(X_processed[column].astype(str))
        label_encoders[column] = le

print(f"   Encoded {len(label_encoders)} categorical features")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_processed)

# For multi-output learning, we need to handle NaN values in targets
# Strategy: Use -1 for missing targets, then ignore them during training
y_matrix_filled = y_matrix.fillna(-1)

print(f"\n🤖 Training Multi-Output Model...")

# Create base models
base_models = []

# Random Forest (handles missing targets well)
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
base_models.append(('rf', rf_model))

# Logistic Regression with multi-output wrapper
lr_model = MultiOutputClassifier(
    LogisticRegression(random_state=42, max_iter=1000),
    n_jobs=-1
)
base_models.append(('lr', lr_model))

# XGBoost if available
if xgb_available:
    xgb_model = MultiOutputClassifier(
        XGBClassifier(random_state=42, eval_metric='logloss'),
        n_jobs=-1
    )
    base_models.append(('xgb', xgb_model))

# Create ensemble
if len(base_models) > 1:
    ensemble_model = VotingClassifier(
        estimators=base_models,
        voting='soft'  # Use soft voting for probabilities
    )
    print(f"   Created ensemble with {len(base_models)} base models")
else:
    ensemble_model = base_models[0][1]
    print(f"   Using single model: {base_models[0][0]}")

# Custom training to handle missing targets
print(f"   Training on targets with valid data...")

# Train individual models per target
individual_models = {}
individual_scalers = {}

for target in available_targets:
    if target not in y_matrix.columns:
        continue
        
    print(f"   🎯 Training model for {target.replace('skill_recommendations_', '').replace('_recommendation', '')}...")
    
    # Get rows with valid data for this target
    valid_mask = y_matrix[target].notna()
    if valid_mask.sum() < 10:  # Need at least 10 samples
        print(f"      ⚠️ Skipping {target}: insufficient data ({valid_mask.sum()} samples)")
        continue
    
    X_target = X_scaled[valid_mask]
    y_target = y_matrix[target][valid_mask].astype(int)
    
    # Check if we have multiple classes
    if len(np.unique(y_target)) < 2:
        print(f"      ⚠️ Skipping {target}: only one class present")
        continue
    
    # Split data
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_target, y_target, 
            test_size=0.2, 
            random_state=42,
            stratify=y_target if len(np.unique(y_target)) > 1 else None
        )
    except:
        # If stratify fails, use random split
        X_train, X_test, y_train, y_test = train_test_split(
            X_target, y_target, 
            test_size=0.2, 
            random_state=42
        )
    
    # Train model for this target
    target_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    target_model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = target_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"      ✅ Accuracy: {accuracy:.3f}")
    
    # Store the model
    individual_models[target] = target_model

print(f"\n💾 Saving models and encoders...")

# Save individual models
joblib.dump(individual_models, "multi_output_models.pkl")
joblib.dump(target_encoders, "target_encoders.pkl")  # Note: plural
joblib.dump(label_encoders, "label_encoders.pkl")
joblib.dump(scaler, "scaler.pkl")

# Save feature information
feature_info = {
    'feature_columns': feature_columns,
    'available_targets': available_targets,
    'target_names': {target: target.replace('skill_recommendations_', '').replace('_recommendation', '').replace('_', ' ').title() 
                    for target in available_targets}
}
joblib.dump(feature_info, "feature_info.pkl")

print(f"\n✅ Multi-Output Model Training Complete!")
print(f"📁 Saved files:")
print(f"  - multi_output_models.pkl ({len(individual_models)} individual models)")
print(f"  - target_encoders.pkl (target label encoders)")
print(f"  - label_encoders.pkl (feature encoders)")
print(f"  - scaler.pkl (feature scaler)")
print(f"  - feature_info.pkl (model metadata)")

print(f"\n📊 Model Summary:")
print(f"   🎯 Skills covered: {len(individual_models)}")
print(f"   📈 Features used: {len(feature_columns)}")
print(f"   📋 Training samples: {X_filtered.shape[0]}")

print(f"\n🎉 Ready for multi-skill prediction deployment!")
