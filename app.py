from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import gzip
import numpy as np
from werkzeug.utils import secure_filename
import os
import urllib.request

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Global variables for model
model = None
label_encoders = None
feature_names = None

def download_file(url, local_path):
    """Download file if it doesn't exist"""
    if not os.path.exists(local_path):
        print(f"Downloading {url}...")
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        urllib.request.urlretrieve(url, local_path)
        print(f"Downloaded to {local_path}")
    return local_path

def load_model():
    """Load model and encoders (download if needed)"""
    global model, label_encoders, feature_names
    
    # Use /tmp directory (Vercel's writable directory)
    model_dir = '/tmp/model_files'
    os.makedirs(model_dir, exist_ok=True)
    
    # GitHub raw URLs (you'll update these after uploading)
    BASE_URL = "https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/"
    
    try:
        # Check if files exist locally first
        model_path = 'rf_model.pkl.gz'
        encoders_path = 'label_encoders.pkl'
        features_path = 'feature_names.txt'
        
        # Load compressed model
        print("Loading compressed model...")
        with gzip.open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Load label encoders
        print("Loading label encoders...")
        with open(encoders_path, 'rb') as f:
            label_encoders = pickle.load(f)
        
        # Load feature names
        print("Loading feature names...")
        with open(features_path, 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
        
        print(f"✓ Model loaded successfully with {len(feature_names)} features")
        return True
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

@app.route('/')
def home():
    # Load model on first request
    if model is None:
        if not load_model():
            return "Error loading model. Please check server logs.", 500
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global model, label_encoders, feature_names
    
    # Ensure model is loaded
    if model is None:
        if not load_model():
            return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        if 'file' in request.files:
            # Batch prediction from CSV
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            if file and file.filename.endswith('.csv'):
                # Create temp directory
                os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Read CSV
                df = pd.read_csv(filepath)
                
                # Process data
                df_processed = preprocess_data(df.copy())
                
                # Make predictions
                predictions = model.predict(df_processed)
                probabilities = model.predict_proba(df_processed)
                
                # Add predictions to dataframe
                df['Churn_Prediction'] = predictions
                df['Churn_Probability'] = probabilities[:, 1]
                
                # Clean up
                os.remove(filepath)
                
                return jsonify({
                    'success': True,
                    'predictions': predictions.tolist(),
                    'probabilities': probabilities[:, 1].tolist(),
                    'total': len(predictions),
                    'churn_count': int(predictions.sum())
                })
        else:
            # Single prediction from form
            data = request.json
            
            # Create dataframe from input
            df = pd.DataFrame([data])
            
            # Process data
            df_processed = preprocess_data(df)
            
            # Make prediction
            prediction = model.predict(df_processed)[0]
            probability = model.predict_proba(df_processed)[0]
            
            return jsonify({
                'success': True,
                'prediction': int(prediction),
                'probability': float(probability[1]),
                'message': 'Customer will churn' if prediction == 1 else 'Customer will not churn'
            })
            
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

def preprocess_data(df):
    """Preprocess the input data"""
    df = df.copy()
    
    categorical_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 
                          'MultipleLines', 'InternetService', 'OnlineSecurity',
                          'OnlineBackup', 'DeviceProtection', 'TechSupport',
                          'StreamingTV', 'StreamingMovies', 'Contract',
                          'PaperlessBilling', 'PaymentMethod']
    
    for col in categorical_columns:
        if col in df.columns and col in label_encoders:
            le = label_encoders[col]
            df[col] = df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
            df[col] = le.transform(df[col])
    
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(0, inplace=True)
    
    for feature in feature_names:
        if feature not in df.columns:
            df[feature] = 0
    
    df = df[feature_names]
    
    return df

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy', 
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)