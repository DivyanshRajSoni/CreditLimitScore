from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import pickle
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

app = Flask(__name__, static_folder='..', static_url_path='')
CORS(app)

# Load the trained model and scaler
model = None
scaler = None

def load_model():
    global model, scaler
    try:
        # Get the base directory
        base_dir = os.path.dirname(os.path.dirname(__file__))
        model_path = os.path.join(base_dir, 'model.pkl')
        scaler_path = os.path.join(base_dir, 'scaler.pkl')
        
        print(f"Looking for model at: {model_path}")
        print(f"Looking for scaler at: {scaler_path}")
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            print("Model and scaler loaded successfully!")
            return True
        else:
            print("Model or scaler files not found!")
            return False
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

# Try to load model on startup
try:
    load_model()
except Exception as e:
    print(f"Initial model load failed: {e}")

@app.route('/')
def index():
    try:
        return send_from_directory('..', 'creditcardpred.html')
    except Exception as e:
        return jsonify({'error': f'HTML file not found: {str(e)}'}), 404

@app.route('/api/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 200
        
    try:
        # Load model if not already loaded
        if model is None or scaler is None:
            if not load_model():
                return jsonify({'error': 'Model not available. Please ensure model.pkl and scaler.pkl are included in the deployment.'}), 500
        
        # Get data from request
        data = request.json
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Create DataFrame with the input
        input_df = pd.DataFrame({
            'SEX': [float(data.get('sex', 1))],
            'EDUCATION': [float(data.get('education', 1))],
            'MARRIAGE': [float(data.get('marriage', 1))],
            'AGE': [float(data.get('age', 30))],
            'PAY_0': [float(data.get('pay_0', 0))],
            'PAY_2': [float(data.get('pay_2', 0))],
            'PAY_3': [float(data.get('pay_3', 0))],
            'PAY_4': [float(data.get('pay_4', 0))],
            'PAY_5': [float(data.get('pay_5', 0))],
            'PAY_6': [float(data.get('pay_6', 0))],
            'BILL_AMT3': [float(data.get('bill_amt3', 0))],
            'BILL_AMT4': [float(data.get('bill_amt4', 0))],
            'BILL_AMT5': [float(data.get('bill_amt5', 0))],
            'BILL_AMT6': [float(data.get('bill_amt6', 0))]
        })
        
        # Scale the input
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        # Ensure positive value
        prediction = max(0, prediction)
        
        return jsonify({'prediction': float(prediction)})
    
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 400

@app.route('/api/health')
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None
    })

# This is for Vercel
def handler(event, context):
    return app(event, context)
