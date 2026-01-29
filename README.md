# Credit Card Limit Prediction System

This project connects a machine learning model (trained in Jupyter notebook) with a web interface for real-time credit limit predictions.

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
Open and run all cells in `creditcardhistory.ipynb` to:
- Load and process the credit card data
- Train the Linear Regression model
- Save the model and scaler files (model.pkl, scaler.pkl)

### 3. Start the Flask Server
```bash
python app.py
```

The server will start on http://localhost:5000

### 4. Use the Web Interface
Open your browser and go to: http://localhost:5000

Fill in the form with customer information and click "Predict Now" to get the estimated credit limit.

## Files

- `creditcardhistory.ipynb` - Jupyter notebook with model training
- `creditcardhist.csv` - Training data
- `app.py` - Flask backend API
- `creditcardpred.html` - Web interface
- `model.pkl` - Trained model (generated after running notebook)
- `scaler.pkl` - Data scaler (generated after running notebook)

## How It Works

1. The notebook trains a Linear Regression model on historical credit card data
2. The model and scaler are saved as pickle files
3. Flask API loads the trained model
4. HTML frontend sends customer data to the Flask API
5. API returns the predicted credit limit
6. Frontend displays the result
