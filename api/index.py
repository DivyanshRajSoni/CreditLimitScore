from http.server import BaseHTTPRequestHandler
import json
import pandas as pd
import pickle
import os
import sys

# Load model and scaler
model = None
scaler = None

def load_model():
    global model, scaler
    try:
        base_dir = os.path.dirname(os.path.dirname(__file__))
        model_path = os.path.join(base_dir, 'model.pkl')
        scaler_path = os.path.join(base_dir, 'scaler.pkl')
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            return True
        return False
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

# Load model on import
load_model()

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_POST(self):
        try:
            # Read request body
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            # Check if model is loaded
            if model is None or scaler is None:
                if not load_model():
                    self.send_response(500)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    response = json.dumps({'error': 'Model not loaded'})
                    self.wfile.write(response.encode())
                    return
            
            # Create DataFrame
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
            
            # Scale and predict
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]
            prediction = max(0, prediction)
            
            # Send response
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            response = json.dumps({'prediction': float(prediction)})
            self.wfile.write(response.encode())
            
        except Exception as e:
            self.send_response(400)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            response = json.dumps({'error': str(e)})
            self.wfile.write(response.encode())
    
    def do_GET(self):
        # Health check
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        response = json.dumps({
            'status': 'ok',
            'model_loaded': model is not None,
            'scaler_loaded': scaler is not None
        })
        self.wfile.write(response.encode())
