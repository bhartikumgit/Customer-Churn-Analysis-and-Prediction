import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import os
import glob

print("="*70)
print("🌲 TRAINING RANDOM FOREST CHURN PREDICTION MODEL")
print("="*70)

# --- 1. Find CSV File ---
print("\n📂 Looking for CSV file...")
csv_files = glob.glob('*.csv')

if not csv_files:
    print("❌ No CSV files found in current directory!")
    print(f"   Current directory: {os.getcwd()}")
    print("\n💡 Please place your CSV file in this folder:")
    print(f"   {os.getcwd()}")
    exit(1)

print(f"✅ Found {len(csv_files)} CSV file(s):")
for i, f in enumerate(csv_files, 1):
    size_mb = os.path.getsize(f) / (1024 * 1024)
    print(f"   {i}. {f} ({size_mb:.2f} MB)")

# Use the first CSV file (or you can add logic to choose)
csv_file = csv_files[0]
if len(csv_files) > 1:
    print(f"\n⚠️  Multiple CSV files found. Using: {csv_file}")
    print(f"   (If this is wrong, delete other CSV files or rename the correct one)")

# --- 2. Load Data ---
print(f"\n📂 Loading data from: {csv_file}")
try:
    df = pd.read_csv(csv_file)
    print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
except Exception as e:
    print(f"❌ Error loading CSV: {str(e)}")
    exit(1)

print(f"\n📋 Columns in dataset:")
for col in df.columns:
    print(f"   • {col}")

# --- 3. Identify Target Column ---
target_col = None
possible_target_names = ['Churn_Label', 'Churn Label', 'churn_label', 'Churn', 'churn', 
                         'Customer_Status', 'customer_status', 'Status']

for col in df.columns:
    col_lower = col.lower()
    if 'churn' in col_lower and 'label' in col_lower:
        target_col = col
        break
    elif col_lower == 'churn':
        target_col = col
        break
    elif 'customer' in col_lower and 'status' in col_lower:
        target_col = col
        break

if not target_col:
    print("\n❌ Error: No churn/target column found!")
    print("   Looking for columns like: Churn_Label, Churn, Customer_Status")
    print("\n📋 Available columns are:")
    for col in df.columns:
        print(f"   • {col}")
    print("\n💡 Please specify which column indicates churn (Yes/No or 1/0)")
    exit(1)

print(f"\n✅ Target column: '{target_col}'")

# --- 4. Check Target Distribution ---
print(f"\n📊 Target distribution:")
target_counts = df[target_col].value_counts()
print(target_counts)

# Handle different target encodings
if df[target_col].dtype == 'object':
    # String values
    churn_values = ['Yes', 'yes', 'YES', 'Y', 'Churned', 'churned']
    churn_count = sum(df[target_col].isin(churn_values))
else:
    # Numeric values
    churn_count = (df[target_col] == 1).sum()

churn_rate = (churn_count / len(df)) * 100
print(f"\n✅ Churn rate: {churn_rate:.2f}%")

if churn_rate == 0:
    print("\n❌ ERROR: No churned customers found!")
    print("   Cannot train model - need both churned and non-churned customers.")
    print(f"\n   Current target values: {df[target_col].unique()}")
    print("\n💡 Check if the target column is correct.")
    exit(1)

if churn_rate == 100:
    print("\n❌ ERROR: All customers are churned!")
    print("   Cannot train model - need both churned and non-churned customers.")
    exit(1)

# --- 5. Define Features ---
# Exclude non-feature columns
exclude_cols = [target_col, 'Customer_ID', 'customer_id', 'ID', 'id', 
                'Churn_Reason', 'churn_reason', 'Churn_Category', 'churn_category']

available_features = [col for col in df.columns 
                     if col not in exclude_cols 
                     and not col.lower().startswith('unnamed')]

print(f"\n✅ Using {len(available_features)} features:")
for feat in available_features[:15]:
    print(f"   ✓ {feat}")
if len(available_features) > 15:
    print(f"   ... and {len(available_features) - 15} more")

# --- 6. Prepare Data ---
X = df[available_features].copy()
y = df[target_col].copy()

# Encode target
if y.dtype == 'object':
    # Map string values to 0/1
    churn_map = {
        'Yes': 1, 'yes': 1, 'YES': 1, 'Y': 1,
        'No': 0, 'no': 0, 'NO': 0, 'N': 0,
        'Churned': 1, 'churned': 1,
        'Stayed': 0, 'stayed': 0,
        'Active': 0, 'active': 0
    }
    y = y.map(churn_map)
    
    # Handle any unmapped values
    if y.isna().any():
        print(f"\n⚠️  Warning: Some target values couldn't be mapped:")
        print(f"   Unique values: {df[target_col].unique()}")
        y = y.fillna(0)
else:
    # Already numeric, ensure 0/1
    y = (y == 1).astype(int)

print(f"\n✅ Encoded target: {df[target_col].unique()} → 0/1")

# --- 7. Encode Categorical Features ---
print(f"\n🔄 Encoding categorical features...")
encoders = {}
categorical_cols = X.select_dtypes(include=['object']).columns

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le
    print(f"   ✓ Encoded: {col}")

print(f"✅ Encoded {len(encoders)} categorical features")

# Handle missing values
X = X.fillna(X.median())

# --- 8. Split Data ---
print(f"\n✂️ Splitting data (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✅ Train: {len(X_train)} rows")
print(f"✅ Test: {len(X_test)} rows")

# --- 9. Train Random Forest ---
print(f"\n🌲 Training Random Forest Classifier...")
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
print(f"✅ Training complete!")

# --- 10. Evaluate ---
print(f"\n📊 Evaluating model...")
y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {accuracy * 100:.2f}%")

print(f"\n📈 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

print(f"\n🎯 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"   [[TN={cm[0,0]} FP={cm[0,1]}]")
print(f"    [FN={cm[1,0]} TP={cm[1,1]}]]")

# --- 11. Feature Importance ---
print(f"\n⭐ Top 10 Most Important Features:")
feature_importance = pd.DataFrame({
    'feature': available_features,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

for idx, row in feature_importance.head(10).iterrows():
    print(f"   {row['feature']}: {row['importance']:.4f}")

# --- 12. Save Model ---
print(f"\n💾 Saving model and encoders...")

model_data = {
    'model': rf_model,
    'encoders': encoders,
    'feature_names': available_features,
    'target_name': target_col
}

joblib.dump(model_data, 'rf_model.pkl')
print(f"✅ Saved to: rf_model.pkl")

# --- 13. Summary ---
print("\n" + "="*70)
print("✅ MODEL TRAINING COMPLETE!")
print("="*70)
print(f"CSV File: {csv_file}")
print(f"Model Type: Random Forest Classifier")
print(f"Features: {len(available_features)}")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Model File: rf_model.pkl")
print("="*70)
print("\n✅ You can now use this model in your Flask app!")
print("   Run: python app.py")
print("="*70)