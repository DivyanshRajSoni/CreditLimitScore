from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import pickle
import os

app = Flask(__name__)
CORS(app)

# Load the trained model and scaler
model = None
scaler = None

def load_model():
    global model, scaler
    if os.path.exists('model.pkl') and os.path.exists('scaler.pkl'):
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return True
    return False

@app.route('/')
def index():
    return send_file('creditcardpred.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Load model if not already loaded
        if model is None or scaler is None:
            if not load_model():
                return jsonify({'error': 'Model not trained yet. Please run the notebook first.'}), 500
        
        # Get data from request
        data = request.json
        
        # Create DataFrame with the input
        input_df = pd.DataFrame({
            'SEX': [float(data['sex'])],
            'EDUCATION': [float(data['education'])],
            'MARRIAGE': [float(data['marriage'])],
            'AGE': [float(data['age'])],
            'PAY_0': [float(data['pay_0'])],
            'PAY_2': [float(data['pay_2'])],
            'PAY_3': [float(data['pay_3'])],
            'PAY_4': [float(data['pay_4'])],
            'PAY_5': [float(data['pay_5'])],
            'PAY_6': [float(data['pay_6'])],
            'BILL_AMT3': [float(data['bill_amt3'])],
            'BILL_AMT4': [float(data['bill_amt4'])],
            'BILL_AMT5': [float(data['bill_amt5'])],
            'BILL_AMT6': [float(data['bill_amt6'])]
        })
        
        # Scale the input
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        # Ensure positive value
        prediction = max(0, prediction)
        
        return jsonify({'prediction': float(prediction)})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
