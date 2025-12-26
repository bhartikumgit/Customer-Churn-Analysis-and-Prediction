import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os
import glob

print("="*70)
print("🌲 TRAINING RANDOM FOREST MODEL")
print("="*70)

# Find CSV file
print("\n📂 Looking for CSV file...")
csv_files = glob.glob('*.csv')
csv_files = [f for f in csv_files if 'temp' not in f.lower()]

if not csv_files:
    print("❌ No CSV files found!")
    exit(1)

csv_file = csv_files[0]
print(f"✅ Using: {csv_file}")

# Load data
print(f"\n📂 Loading data...")
df = pd.read_csv(csv_file)
print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")

# Find target column
target_col = None
for col in df.columns:
    if 'churn' in col.lower() and 'label' in col.lower():
        target_col = col
        break
    elif col.lower() == 'churn':
        target_col = col
        break

if not target_col:
    print("❌ No churn column found!")
    exit(1)

print(f"✅ Target: {target_col}")

# Exclude columns
exclude_cols = [target_col, 'Customer_ID', 'customer_id', 'CustomerID', 'ID', 
                'Churn_Reason', 'Churn_Category', 'Customer_Status']

feature_cols = [col for col in df.columns 
                if col not in exclude_cols 
                and not col.lower().startswith('unnamed')]

print(f"✅ Using {len(feature_cols)} features")

# Prepare data
X = df[feature_cols].copy()
y = df[target_col].copy()

# Encode target
if y.dtype == 'object':
    y = y.map({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0})
else:
    y = (y == 1).astype(int)

print(f"\n✅ Target encoded: {df[target_col].unique()} → 0/1")
print(f"   Churn rate: {(y.sum() / len(y) * 100):.1f}%")

# Encode categorical features
print(f"\n🔄 Encoding features...")
encoders = {}
for col in X.columns:
    if X[col].dtype == 'object':
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
        print(f"   ✓ {col}")

print(f"✅ Encoded {len(encoders)} features")

# Fill missing values
X = X.fillna(X.median())

# Split data
print(f"\n✂️ Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✅ Train: {len(X_train)}, Test: {len(X_test)}")

# Train model
print(f"\n🌲 Training Random Forest...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
print(f"✅ Training complete!")

# Evaluate
print(f"\n📊 Evaluating...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {accuracy * 100:.2f}%")

print(f"\n📈 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

# Feature importance
print(f"\n⭐ Top 10 Important Features:")
importances = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for idx, row in importances.head(10).iterrows():
    print(f"   {row['feature']}: {row['importance']:.4f}")

# Save model
print(f"\n💾 Saving model...")
with open('rf_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print(f"✅ Saved: rf_model.pkl")

# Save encoders
print(f"💾 Saving encoders...")
with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)
print(f"✅ Saved: label_encoders.pkl")

# Save feature names
print(f"💾 Saving feature names...")
with open('feature_names.txt', 'w') as f:
    for feat in feature_cols:
        f.write(f"{feat}\n")
print(f"✅ Saved: feature_names.txt")

# Summary
print("\n" + "="*70)
print("✅ TRAINING COMPLETE!")
print("="*70)
print(f"Model: RandomForestClassifier")
print(f"Features: {len(feature_cols)}")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Files saved:")
print(f"  - rf_model.pkl")
print(f"  - label_encoders.pkl")
print(f"  - feature_names.txt")
print("="*70)
print("\n✅ Ready to use! Run: python app.py")
print("="*70)