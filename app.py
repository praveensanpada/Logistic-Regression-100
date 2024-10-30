from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

app = Flask(__name__)

# Load the trained model if it exists
try:
    with open('logistic_regression_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    global model
    try:
        # Get the file from the request
        file = request.files['file']
        df = pd.read_csv(file)
        
        # Check if the expected columns are present
        if 'CGPA' not in df.columns or 'IQ' not in df.columns or 'LPA' not in df.columns:
            return jsonify({'error': 'Invalid CSV format. Ensure the columns are CGPA, IQ, and LPA.'}), 400
        
        # Prepare the data
        X = df[['CGPA', 'IQ']].values  # CGPA and IQ as features
        y = df['LPA'].values  # LPA as the binary target (0 or 1)
        
        # Check if the target contains only binary values (0 and 1)
        if not set(y).issubset({0, 1}):
            return jsonify({'error': 'LPA column must contain only binary values (0 or 1).'}), 400

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
        
        # Train the logistic regression model
        model = LogisticRegression()
        model.fit(X_train, y_train)
        
        # Save the model
        with open('logistic_regression_model.pkl', 'wb') as f:
            pickle.dump(model, f)

        # Test the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        return jsonify({'message': 'Model trained successfully!', 'accuracy': accuracy, 'report': report})
    
    except Exception as e:
        # Log the actual error for debugging purposes
        return jsonify({'error': f'Failed to train the model: {str(e)}'}), 500

@app.route('/predict', methods=['POST'])
def predict():
    global model
    if not model:
        return jsonify({'error': 'Model not trained yet. Please upload a dataset and train the model first.'})

    try:
        # Get the input values from the request
        cgpa = float(request.form['cgpa'])
        iq = float(request.form['iq'])
        
        # Predict the binary LPA (0 or 1) based on the input CGPA and IQ
        prediction = model.predict(np.array([[cgpa, iq]]))[0]
        predicted_class = 'Above 20 LPA' if prediction == 1 else 'Below 20 LPA'
        
        return jsonify({'cgpa': cgpa, 'iq': iq, 'predicted_lpa': predicted_class})
    
    except Exception as e:
        return jsonify({'error': f'Failed to predict the LPA: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
