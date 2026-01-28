import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, hamming_loss
import joblib
import warnings
import os
from preprocess import wrangle
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    xgb_available = True
except ImportError:
    print("⚠️ XGBoost not installed. Install with: pip install xgboost")
    xgb_available = False

print("🚀 Starting AI model training...")

# Run preprocessing pipeline using preprocess.py
print("🔧 Running preprocessing pipeline...")
df = wrangle()
print(f"📊 Dataset preprocessed: {df.shape[0]} rows, {df.shape[1]} columns")

# ============================================================================
# MULTI-LABEL SKILL RECOMMENDATION TARGET SELECTION
# ============================================================================
# Automatically detect all skill recommendation columns (not level columns)
skill_recommendation_columns = [col for col in df.columns if col.endswith('_recommendation')]

if len(skill_recommendation_columns) == 0:
    print("❌ No skill recommendation columns found in dataset")
    print("Available columns:", df.columns.tolist())
    exit(1)

print(f"🎯 Found {len(skill_recommendation_columns)} skill recommendation targets:")
for i, col in enumerate(skill_recommendation_columns, 1):
    print(f"   {i}. {col}")


# ============================================================================
# MULTI-LABEL TARGET MATRIX PREPARATION
# ============================================================================
# Remove rows with any null values in skill recommendation columns
df_clean = df.dropna(subset=skill_recommendation_columns)
print(f"📋 After removing rows with null targets: {df_clean.shape[0]} rows")

if df_clean.empty:
    print("❌ No valid data remaining after cleaning")
    exit(1)

# Prepare features and target matrix
# Features: exclude identifier columns and ALL skill recommendation columns
exclude_columns = [
    'name', 'father_name', 'id', 'assessment_date', 'date_of_birth'
] + skill_recommendation_columns

feature_columns = [col for col in df_clean.columns if col not in exclude_columns]
X = df_clean[feature_columns].copy()
Y_raw = df_clean[skill_recommendation_columns].copy()

print(f"🎯 Multi-label setup:")
print(f"   - Number of features: {len(feature_columns)}")
print(f"   - Number of skills to predict: {len(skill_recommendation_columns)}")
print(f"   - Target matrix shape: {Y_raw.shape}")

# Print class distribution for each skill
print(f"\n📊 Class distribution per skill:")
for skill_col in skill_recommendation_columns:
    dist = Y_raw[skill_col].value_counts().to_dict()
    print(f"   {skill_col}: {dist}")


# ============================================================================
# FEATURE ENCODING
# ============================================================================
# Handle missing values: fill numeric columns with 0 and categorical with 'unknown'
for col in X.columns:
    if X[col].dtype in ['int64', 'float64', 'Int64']:
        X[col] = X[col].fillna(0)
    else:
        X[col] = X[col].fillna('unknown')

# Encode categorical features
label_encoders_features = {}
for column in X.columns:
    if X[column].dtype == 'object':
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column].astype(str))
        label_encoders_features[column] = le

# ============================================================================
# MULTI-LABEL TARGET ENCODING (One encoder per skill)
# ============================================================================
# Create a separate label encoder for each skill recommendation column
label_encoders_skills = {}
Y_encoded = pd.DataFrame()

for skill_col in skill_recommendation_columns:
    le = LabelEncoder()
    Y_encoded[skill_col] = le.fit_transform(Y_raw[skill_col].astype(str))
    label_encoders_skills[skill_col] = le
    print(f"📝 Encoded {skill_col}: classes {le.classes_}")

print(f"\n✅ Encoded target matrix shape: {Y_encoded.shape}")

# ============================================================================
# TRAIN-TEST SPLIT
# ============================================================================
# Adjust test size for small datasets
total_samples = len(Y_encoded)
if total_samples < 10:
    test_size = 0.3  # Use 30% for very small datasets
elif total_samples < 50:
    test_size = 0.2  # Use 20% for small datasets
else:
    test_size = 0.2  # Standard 20%

print(f"📊 Dataset size: {total_samples} samples")
print(f"📊 Test size: {test_size:.0%}")

# Train-test split (using first skill for stratification reference, not critical for multi-label)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y_encoded, test_size=test_size, random_state=42
)

print(f"📊 Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
print(f"📊 Test set: {X_test.shape[0]} samples")


# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# MULTI-OUTPUT ENSEMBLE MODEL CREATION
# ============================================================================
# For multi-label classification, wrap each base classifier with MultiOutputClassifier
# This treats each skill as an independent classification problem

print("🤖 Creating multi-output ensemble models...")
lr_model = LogisticRegression(
    random_state=42,
    class_weight='balanced',
    max_iter=1000,
    solver='liblinear'
)

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    max_depth=None,
    min_samples_leaf=2,
    class_weight='balanced',
    n_jobs=-1
)

models_for_ensemble = [
    ('logistic', lr_model),
    ('random_forest', rf_model)
]

if xgb_available:
    xgb_model = XGBClassifier(
        random_state=42,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        verbosity=0
    )
    models_for_ensemble.append(('xgboost', xgb_model))
    
# Wrap each base model with MultiOutputClassifier
base_ensemble = VotingClassifier(
    estimators=models_for_ensemble,
    voting='soft',
    n_jobs=-1
)

ensemble_model = MultiOutputClassifier(base_ensemble)

print("🚀 Training multi-output ensemble model...")
ensemble_model.fit(X_train_scaled, Y_train)
print("✅ Training complete!")


# ============================================================================
# MULTI-LABEL EVALUATION METRICS
# ============================================================================
# Make predictions
Y_pred = ensemble_model.predict(X_test_scaled)

Y_true_np = Y_test.values
Y_pred_np = Y_pred

# Manual Hamming Loss
h_loss = np.mean(Y_true_np != Y_pred_np)

print(f"\n{'='*70}")
print(f"📊 MULTI-OUTPUT CLASSIFICATION RESULTS")
print(f"{'='*70}")
print(f"📊 Hamming Loss (manual): {h_loss:.4f}")

# ============================================================================
# PER-SKILL CLASSIFICATION REPORTS
# ============================================================================
print(f"\n{'='*70}")
print(f"📊 PER-SKILL CLASSIFICATION REPORTS")
print(f"{'='*70}")

for idx, skill_col in enumerate(skill_recommendation_columns):
    # Get the label encoder for this skill
    skill_encoder = label_encoders_skills[skill_col]
    
    # Get predictions and targets for this skill
    y_test_skill = Y_test.iloc[:, idx]
    y_pred_skill = Y_pred[:, idx]
    
    # Get class labels for reporting
    classes_in_test = np.unique(np.concatenate([y_test_skill, y_pred_skill]))
    class_names = [skill_encoder.classes_[i] for i in classes_in_test]
    
    print(f"\n🎯 {skill_col}:")
    print(f"   Classes: {class_names}")
    try:
        report = classification_report(
            y_test_skill,
            y_pred_skill,
            labels=classes_in_test,
            target_names=class_names,
            zero_division=0
        )
        print(report)
    except Exception as e:
        print(f"   Could not generate report: {e}")
        print(f"   Test: {y_test_skill.values}")
        print(f"   Pred: {y_pred_skill}")



# ============================================================================
# SAVE MULTI-LABEL MODEL AND ENCODERS
# ============================================================================
# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Save the trained multi-output ensemble model
joblib.dump(ensemble_model, "models/model.pkl")

# Save the feature scaler
joblib.dump(scaler, "models/scaler.pkl")

# Save feature label encoders (for encoding input features)
joblib.dump(label_encoders_features, "models/label_encoders.pkl")

# Create a simple mapping dict for decoding predictions
# Maps: 0 -> 'not recommended', 1 -> 'recommended', 2 -> 'highly recommended', 3 -> 'one-term'
prediction_mapping = {
    0: 'not recommended',
    1: 'recommended',
    2: 'highly recommended',
    3: 'one-term'
}
joblib.dump(prediction_mapping, "models/prediction_mapping.pkl")

# Save metadata about skills
skill_metadata = {
    'skill_columns': skill_recommendation_columns,
    'n_skills': len(skill_recommendation_columns),
    'feature_columns': feature_columns
}
joblib.dump(skill_metadata, "models/skill_metadata.pkl")

print(f"\n{'='*70}")
print(f"💾 MODEL AND ENCODERS SAVED TO models/")
print(f"{'='*70}")
print(f"  ✓ models/model.pkl (multi-output ensemble)")
print(f"  ✓ models/scaler.pkl (feature scaler)")
print(f"  ✓ models/label_encoders.pkl (feature encoders)")
print(f"  ✓ models/prediction_mapping.pkl (0,1,2,3 → labels)")
print(f"  ✓ models/skill_metadata.pkl (skill configuration)")


# Handle small dataset warning
if len(X_train) < 50:
    print(f"\n⚠️  Warning: Small dataset detected ({len(X_train)} training samples)")
    print(f"   Model performance may be limited. Consider collecting more data.")

print(f"\n{'='*70}")
print(f"🎉 MULTI-LABEL MODEL TRAINING COMPLETE")
print(f"{'='*70}")
print(f"📊 Hamming Loss: {h_loss:.4f}")
print(f"📊 Skills trained: {len(skill_recommendation_columns)}")
print(f"{'='*70}\n")
