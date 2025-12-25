from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import gzip
import numpy as np
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the compressed model and encoders
print("Loading compressed model...")
with gzip.open('rf_model.pkl.gz', 'rb') as f:
    model = pickle.load(f)

print("Loading label encoders...")
with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

# Load feature names
with open('feature_names.txt', 'r') as f:
    feature_names = [line.strip() for line in f.readlines()]

print(f"Model loaded successfully with {len(feature_names)} features")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' in request.files:
            # Batch prediction from CSV
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            if file and file.filename.endswith('.csv'):
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
                
                # Save results
                result_filename = f'predicted_{filename}'
                result_filepath = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
                df.to_csv(result_filepath, index=False)
                
                # Clean up
                os.remove(filepath)
                
                return jsonify({
                    'success': True,
                    'predictions': predictions.tolist(),
                    'probabilities': probabilities[:, 1].tolist(),
                    'result_file': result_filename
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
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Encode categorical variables
    categorical_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 
                          'MultipleLines', 'InternetService', 'OnlineSecurity',
                          'OnlineBackup', 'DeviceProtection', 'TechSupport',
                          'StreamingTV', 'StreamingMovies', 'Contract',
                          'PaperlessBilling', 'PaymentMethod']
    
    for col in categorical_columns:
        if col in df.columns and col in label_encoders:
            # Handle unseen labels
            le = label_encoders[col]
            df[col] = df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
            df[col] = le.transform(df[col])
    
    # Convert TotalCharges to numeric
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(0, inplace=True)
    
    # Ensure all required features are present
    for feature in feature_names:
        if feature not in df.columns:
            df[feature] = 0
    
    # Select only the features used in training
    df = df[feature_names]
    
    return df

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)