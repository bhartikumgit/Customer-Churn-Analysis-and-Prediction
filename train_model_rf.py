import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
import joblib

print("\n" + "="*70)
print("🌲 TRAINING RANDOM FOREST MODEL (NOT GRADIENT BOOSTING)")
print("="*70)

# Load data
print("\n📂 Loading data...")
df = pd.read_excel("churn analysis.xlsx")
print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")

# Find churn column
target_col = None
for col in df.columns:
    if 'churn' in col.lower() and 'reason' not in col.lower():
        target_col = col
        break

if not target_col:
    print("❌ ERROR: No churn column found!")
    exit()

print(f"✅ Target column: '{target_col}'")

# Prepare features and target
cols_to_drop = [target_col, 'Customer_ID', 'Churn Reason']
X = df.drop(columns=cols_to_drop, errors='ignore')
y = df[target_col]

print(f"✅ Features: {X.shape[1]}")
print(f"   Feature names: {list(X.columns)[:5]}...")

# Encode target
if y.dtype == 'object':
    y = (y == 'Yes').astype(int)
    print(f"✅ Encoded target: Yes → 1, No → 0")

print(f"✅ Churn rate: {y.mean():.2%}")

# Encode categorical features
print("\n🔄 Encoding categorical features...")
encoders = {}
for col in X.columns:
    if X[col].dtype == 'object':
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
        print(f"   ✓ Encoded: {col}")

print(f"✅ Encoded {len(encoders)} categorical features")

# Split data
print("\n✂️ Splitting data (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✅ Train: {len(X_train)} rows")
print(f"✅ Test: {len(X_test)} rows")

# Train Random Forest
print("\n🌲 Training Random Forest Classifier...")
print("   Parameters:")
print("   - n_estimators: 200")
print("   - max_depth: 15")
print("   - min_samples_split: 10")
print("   - min_samples_leaf: 4")

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42,
    n_jobs=-1,
    verbose=0
)

rf_model.fit(X_train, y_train)
print("✅ Training complete!")

# Evaluate
print("\n📊 Evaluating model...")
y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\n" + "="*70)
print("🎯 RANDOM FOREST RESULTS")
print("="*70)
print(f"✅ Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
print(f"✅ F1 Score:  {f1:.4f}")
print(f"✅ AUC-ROC:   {auc:.4f}")
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

# Feature importance
print("\n🔑 Top 10 Most Important Features:")
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

for i, row in feature_importance.head(10).iterrows():
    print(f"   {i+1}. {row['feature']:30} {row['importance']:.4f}")

# Save model
print("\n💾 Saving model and encoders...")
joblib.dump(rf_model, 'rf_model.pkl')
joblib.dump(encoders, 'label_encoders.pkl')

print("✅ Saved: rf_model.pkl")
print("✅ Saved: label_encoders.pkl")

# Verify model type
loaded_model = joblib.load('rf_model.pkl')
print(f"\n✅ Model type verification: {type(loaded_model).__name__}")

if 'RandomForest' in type(loaded_model).__name__:
    print("✅ SUCCESS! Model is Random Forest")
else:
    print("❌ WARNING! Model is NOT Random Forest")

print("\n" + "="*70)
print("🎉 TRAINING COMPLETE!")
print("="*70 + "\n")