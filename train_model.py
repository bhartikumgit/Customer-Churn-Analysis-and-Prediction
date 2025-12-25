
import pandas as pd
import numpy as np
import joblib
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, roc_curve, accuracy_score,
                             precision_recall_curve, f1_score)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("CUSTOMER CHURN MODEL TRAINING PIPELINE")
print("="*70)

# 1. LOAD DATA
print("\n[1/8] Loading Dataset...")

try:
    df = pd.read_excel('churn analysis.xlsx')
    print(f"✓ Successfully loaded: churn analysis.xlsx")
except Exception as e:
    print(f"Error loading file: {e}")
    exit(1)

print(f"Dataset shape: {df.shape}")
print(f"\nFirst few columns: {df.columns.tolist()[:10]}")

# Clean column names
df.columns = [c.strip() for c in df.columns]

# 2. PREPROCESSING
print("\n[2/8] Data Preprocessing...")

# Find target column
target_col = None
for col in df.columns:
    if 'churn' in col.lower() and 'category' not in col.lower() and 'reason' not in col.lower():
        target_col = col
        break

if target_col is None:
    if 'Customer_Status' in df.columns:
        df['Churn'] = df['Customer_Status'].apply(lambda x: 1 if str(x).lower() == 'churned' else 0)
        target_col = 'Churn'
    else:
        print("Error: Churn target column not found!")
        exit(1)

print(f"Target column: {target_col}")
print(f"\nTarget distribution:\n{df[target_col].value_counts()}")

# Find Customer ID column
customer_id_col = None
for col in df.columns:
    if 'customer' in col.lower() and 'id' in col.lower():
        customer_id_col = col
        break

# Remove unnecessary columns
cols_to_drop = ['Churn_Category', 'Churn_Reason', 'Customer_Status']
if customer_id_col:
    cols_to_drop.append(customer_id_col)
    
df_model = df.drop([c for c in cols_to_drop if c in df.columns], axis=1)

# Separate features and target
y = df_model[target_col].copy()
X = df_model.drop(target_col, axis=1)

# Convert target to binary
if y.dtype == 'object':
    y = y.map({'Yes': 1, 'No': 0, 'Churned': 1, 'Stayed': 0})

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Handle missing values
print(f"\nHandling missing values...")
for col in X.columns:
    if X[col].dtype in ['int64', 'float64']:
        X[col].fillna(X[col].median(), inplace=True)
    else:
        X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 'Unknown', inplace=True)

# 3. FEATURE ENGINEERING
print("\n[3/8] Feature Engineering...")

label_encoders = {}
categorical_cols = X.select_dtypes(include=['object']).columns

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

print(f"✓ Encoded {len(categorical_cols)} categorical columns")

# 4. CHECK IMBALANCE
print("\n[4/8] Checking Class Imbalance...")

class_counts = y.value_counts()
imbalance_ratio = class_counts.max() / class_counts.min()
print(f"Class distribution:\n{class_counts}")
print(f"Imbalance ratio: {imbalance_ratio:.2f}:1")

use_smote = imbalance_ratio > 1.5
if use_smote:
    print("⚠ Dataset is imbalanced. SMOTE will be applied.")
else:
    print("✓ Dataset is balanced.")

# 5. SPLIT DATA
print("\n[5/8] Splitting Data...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# 6. APPLY SMOTE
if use_smote:
    print("\n[6/8] Applying SMOTE...")
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    print(f"Balanced training set: {X_train_balanced.shape}")
else:
    print("\n[6/8] Skipping SMOTE...")
    X_train_balanced, y_train_balanced = X_train, y_train

# 7. TRAIN MODELS
print("\n[7/8] Training Models...")

models = {
    'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
}

results = {}

for name, model in models.items():
    print(f"\n--- Training {name} ---")
    model.fit(X_train_balanced, y_train_balanced)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
    
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'f1_score': f1,
        'auc': auc
    }
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    if auc:
        print(f"AUC-ROC: {auc:.4f}")

# 8. SELECT BEST MODEL
print("\n[8/8] Selecting Best Model...")

best_model_name = max(results, key=lambda x: results[x]['f1_score'])
print(f"\n✓ Best model: {best_model_name} (F1={results[best_model_name]['f1_score']:.4f})")

best_model = results[best_model_name]['model']

# Final evaluation
y_pred_final = best_model.predict(X_test)
y_proba_final = best_model.predict_proba(X_test)[:, 1]

print("\n" + "="*70)
print("FINAL MODEL PERFORMANCE")
print("="*70)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_final, target_names=['No Churn', 'Churn']))

cm = confusion_matrix(y_test, y_pred_final)
print("\nConfusion Matrix:")
print(cm)

print(f"\nAccuracy: {accuracy_score(y_test, y_pred_final):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred_final):.4f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_proba_final):.4f}")

# Feature importance
if hasattr(best_model, 'feature_importances_'):
    print("\n" + "="*70)
    print("TOP 15 FEATURES")
    print("="*70)
    
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(feature_importance.head(15).to_string(index=False))

# 9. SAVE MODEL
print("\n" + "="*70)
print("SAVING MODEL")
print("="*70)

joblib.dump(best_model, 'rf_model.pkl')
print("✓ Model saved: rf_model.pkl")

joblib.dump(label_encoders, 'label_encoders.pkl')
print("✓ Encoders saved: label_encoders.pkl")

with open('feature_names.txt', 'w') as f:
    f.write('\n'.join(X.columns.tolist()))
print("✓ Feature names saved: feature_names.txt")

# 10. VISUALIZATIONS
print("\n" + "="*70)
print("GENERATING VISUALIZATIONS")
print("="*70)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Churn', 'Churn'],
            yticklabels=['No Churn', 'Churn'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: confusion_matrix.png")

plt.figure(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_test, y_proba_final)
plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, y_proba_final):.3f}')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: roc_curve.png")

if hasattr(best_model, 'feature_importances_'):
    plt.figure(figsize=(10, 8))
    top_features = feature_importance.head(15)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance')
    plt.title('Top 15 Feature Importances')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: feature_importance.png")

print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)