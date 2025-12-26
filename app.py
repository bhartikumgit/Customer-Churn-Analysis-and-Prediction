from flask import Flask, render_template, request, redirect, url_for, Response, session, flash
import pandas as pd
import numpy as np
import joblib
import os
import sys

app = Flask(__name__)
app.secret_key = 'churn-prediction-secret-key-2024'

# --- Model Loading ---
model = None
encoders = {}
ALL_FEATURES = []

print("\n" + "="*60)
print("🔄 LOADING MODEL...")
print("="*60)

try:
    print("📂 Loading rf_model.pkl...")
    model_data = joblib.load('rf_model.pkl')
    
    if isinstance(model_data, dict):
        model = model_data.get('model')
        encoders = model_data.get('encoders', {})
        ALL_FEATURES = model_data.get('feature_names', [])
        print("✅ Model loaded (dictionary format)")
    else:
        model = model_data
        print("✅ Model loaded (direct format)")
        if hasattr(model, 'feature_names_in_'):
            ALL_FEATURES = list(model.feature_names_in_)
    
    print(f"✅ Type: {type(model).__name__}")
    print(f"✅ Features: {len(ALL_FEATURES)}")
    print(f"✅ Encoders: {len(encoders)}")
    
except FileNotFoundError:
    print("❌ rf_model.pkl not found")
    print("   Run: python train_rf_model.py")
except Exception as e:
    print(f"❌ Error: {str(e)}")

print("="*60)
sys.stdout.flush()

# Essential features for validation
ESSENTIAL_FEATURES = [
    'Tenure_in_Months',
    'Monthly_Charge',
    'Contract',
    'Total_Charges',
    'Internet_Service'
]

def clean_feature_name(name):
    """Remove underscores and capitalize"""
    return ' '.join(word.capitalize() for word in name.split('_'))

def find_column_match(df_columns, target_name):
    """Find matching column (case-insensitive)"""
    df_cols_lower = {col.lower(): col for col in df_columns}
    target_lower = target_name.lower()
    
    if target_lower in df_cols_lower:
        return df_cols_lower[target_lower]
    
    # Try without underscores/spaces
    target_clean = target_lower.replace('_', '').replace(' ', '')
    for col_lower, col_actual in df_cols_lower.items():
        if col_lower.replace('_', '').replace(' ', '') == target_clean:
            return col_actual
    
    return None

def validate_dataset(df):
    """Validate dataset"""
    found_features = {}
    missing_essential = []
    missing_optional = []
    
    for feat in ESSENTIAL_FEATURES:
        matched = find_column_match(df.columns, feat)
        if matched:
            found_features[feat] = matched
        else:
            missing_essential.append(feat)
    
    for feat in ALL_FEATURES:
        if feat not in found_features:
            matched = find_column_match(df.columns, feat)
            if matched:
                found_features[feat] = matched
            else:
                missing_optional.append(feat)
    
    return found_features, missing_essential, missing_optional

def prepare_row_for_prediction(row_data, column_mapping):
    """Prepare single row for prediction"""
    if not model or not ALL_FEATURES:
        return None
    
    features = []
    for feat in ALL_FEATURES:
        value = 0
        
        if feat in column_mapping:
            actual_col = column_mapping[feat]
            value = row_data.get(actual_col, 0)
            
            if pd.isna(value):
                value = 0
            
            # Encode categorical
            if feat in encoders and not isinstance(value, (int, float)):
                try:
                    value = encoders[feat].transform([str(value)])[0]
                except:
                    value = 0
        else:
            # Defaults for missing features
            if 'charge' in feat.lower() or 'revenue' in feat.lower():
                value = 50.0
            elif 'tenure' in feat.lower():
                value = 12.0
            elif 'age' in feat.lower():
                value = 35.0
        
        try:
            features.append(float(value))
        except:
            features.append(0.0)
    
    return np.array(features, dtype=float).reshape(1, -1)

# --- Routes ---

@app.route('/')
def home():
    session.clear()
    return render_template('home.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'dataset' not in request.files:
            return render_template('upload.html', 
                                 error="No file uploaded",
                                 essential_features=ESSENTIAL_FEATURES)
        
        file = request.files['dataset']
        if file.filename == '':
            return render_template('upload.html',
                                 error="No file selected",
                                 essential_features=ESSENTIAL_FEATURES)
        
        if not file.filename.endswith('.csv'):
            return render_template('upload.html',
                                 error="Please upload CSV file",
                                 essential_features=ESSENTIAL_FEATURES)
        
        try:
            df = pd.read_csv(file)
            print(f"\n✅ Uploaded: {len(df)} rows")
            sys.stdout.flush()
            
            found, missing_essential, missing_optional = validate_dataset(df)
            
            if missing_essential:
                error = f"Missing: {', '.join(missing_essential)}"
                print(f"❌ {error}")
                sys.stdout.flush()
                return render_template('upload.html',
                                     error=error,
                                     essential_features=ESSENTIAL_FEATURES)
            
            print(f"✅ Validation passed")
            sys.stdout.flush()
            
            session['dataset_path'] = file.filename
            session['dataset_rows'] = len(df)
            session['column_mapping'] = found
            session['missing_optional'] = missing_optional
            session['dataset_uploaded'] = True
            
            df.to_csv('temp_data.csv', index=False)
            
            return redirect(url_for('analysis_choice'))
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            sys.stdout.flush()
            return render_template('upload.html',
                                 error=f"Error: {str(e)}",
                                 essential_features=ESSENTIAL_FEATURES)
    
    return render_template('upload.html', essential_features=ESSENTIAL_FEATURES)

@app.route('/analysis_choice')
def analysis_choice():
    if not session.get('dataset_uploaded'):
        return redirect(url_for('upload'))
    
    return render_template('analysis_choice.html',
                          filename=session.get('dataset_path'),
                          num_customers=session.get('dataset_rows'),
                          missing_count=len(session.get('missing_optional', [])))

@app.route('/batch_analysis')
def batch_analysis():
    print("\n" + "="*50)
    print("🔄 BATCH ANALYSIS")
    print("="*50)
    sys.stdout.flush()
    
    if not session.get('dataset_uploaded'):
        return redirect(url_for('upload'))
    
    try:
        df = pd.read_csv('temp_data.csv')
        column_mapping = session.get('column_mapping', {})
        
        print(f"📊 Processing {len(df)} customers...")
        sys.stdout.flush()
        
        if model and ALL_FEATURES:
            all_features = []
            for idx in range(len(df)):
                features = prepare_row_for_prediction(df.iloc[idx], column_mapping)
                if features is not None:
                    all_features.append(features[0])
                
                if (idx + 1) % 500 == 0:
                    print(f"   {idx+1}/{len(df)}...")
                    sys.stdout.flush()
            
            X = np.array(all_features)
            predictions = model.predict(X)
            probabilities = model.predict_proba(X)[:, 1]
            
            print("✅ Complete!")
            sys.stdout.flush()
        else:
            print("⚠️  Demo mode")
            predictions = np.random.choice([0, 1], size=len(df))
            probabilities = np.random.random(size=len(df))
            sys.stdout.flush()
        
        total = int(len(df))
        churned = int(np.sum(predictions))
        churn_pct = round((churned / total) * 100, 2)
        avg_prob = round(float(np.mean(probabilities)) * 100, 2)
        high_risk = int(np.sum(probabilities > 0.7))
        
        tenure_col = column_mapping.get('Tenure_in_Months')
        avg_tenure = None
        if tenure_col and tenure_col in df.columns:
            avg_tenure = round(float(df[tenure_col].mean()), 1)
        
        results = {
            'total_customers': total,
            'churn_count': churned,
            'churn_percentage': churn_pct,
            'avg_churn_prob': avg_prob,
            'high_risk_count': high_risk,
            'avg_tenure': avg_tenure,
            'missing_features_count': len(session.get('missing_optional', [])),
            'has_all_features': len(session.get('missing_optional', [])) == 0
        }
        
        print(f"📊 {churned}/{total} ({churn_pct}%)")
        print("="*50)
        sys.stdout.flush()
        
        return render_template('batch_results.html', results=results)
        
    except Exception as e:
        print(f"❌ {str(e)}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        return redirect(url_for('analysis_choice'))

@app.route('/individual_analysis')
def individual_analysis():
    if not session.get('dataset_uploaded'):
        return redirect(url_for('upload'))
    
    try:
        df = pd.read_csv('temp_data.csv')
        
        customer_ids = []
        for col in ['Customer_ID', 'customer_id', 'CustomerID', 'ID']:
            if col in df.columns:
                customer_ids = df[col].tolist()
                break
        
        if not customer_ids:
            customer_ids = [f"Customer {i+1}" for i in range(len(df))]
        
        return render_template('individual_select.html',
                             customer_ids=customer_ids,
                             total_rows=len(df))
    except Exception as e:
        print(f"❌ {str(e)}")
        sys.stdout.flush()
        return redirect(url_for('analysis_choice'))

@app.route('/predict_individual', methods=['POST'])
def predict_individual():
    print("\n🎯 Prediction")
    sys.stdout.flush()
    
    if not session.get('dataset_uploaded'):
        return redirect(url_for('upload'))
    
    try:
        df = pd.read_csv('temp_data.csv')
        column_mapping = session.get('column_mapping', {})
        
        row_index = request.form.get('row_index')
        if not row_index:
            return redirect(url_for('individual_analysis'))
        
        row_index = int(row_index)
        row_data = df.iloc[row_index]
        
        if model and ALL_FEATURES:
            features = prepare_row_for_prediction(row_data, column_mapping)
            if features is not None:
                prediction = int(model.predict(features)[0])
                probability = float(model.predict_proba(features)[0][1])
                print(f"✅ {prediction}, {probability:.2%}")
                sys.stdout.flush()
            else:
                prediction, probability = 1, 0.75
        else:
            prediction, probability = 1, 0.75
        
        factors = []
        for feat in ['Tenure_in_Months', 'Contract', 'Monthly_Charge', 'Total_Charges', 'Internet_Service']:
            if feat in column_mapping:
                factors.append((clean_feature_name(feat), np.random.random() * 0.5 + 0.3))
        
        customer_data = {clean_feature_name(k): v for k, v in row_data.items()}
        
        return render_template('result.html',
                             result={'prediction': prediction, 'probability': probability},
                             factors=factors,
                             inputs=customer_data)
    except Exception as e:
        print(f"❌ {str(e)}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        return redirect(url_for('individual_analysis'))

@app.route('/download_report', methods=['POST'])
def download_report():
    content = f"""
╔═══════════════════════════════════════╗
║     CHURN ANALYSIS REPORT            ║
╚═══════════════════════════════════════╝

Dataset: {session.get('dataset_path', 'N/A')}
Total: {request.form.get('total', 'N/A')}
Churned: {request.form.get('churn_count', 'N/A')} ({request.form.get('churn', 'N/A')}%)
High Risk: {request.form.get('high_risk', 'N/A')}

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    return Response(content, mimetype="text/plain",
                   headers={"Content-disposition": "attachment; filename=report.txt"})

if __name__ == '__main__':
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    print("\n" + "="*60)
    print("🚀 CHURN PREDICTION SYSTEM")
    print("="*60)
    print(f"Model: {type(model).__name__ if model else 'Not loaded'}")
    print(f"Features: {len(ALL_FEATURES)}")
    print(f"🌐 http://localhost:5000")
    print("="*60 + "\n")
    sys.stdout.flush()
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)