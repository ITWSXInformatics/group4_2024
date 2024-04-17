from flask import Flask, render_template, request, jsonify
import pandas as pd
import xgboost as xgb
import numpy as np

app = Flask(__name__)

# Load the trained XGBoost model
model = xgb.Booster(model_file='xgb_model.bin')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/diagnose', methods=['POST'])
def diagnose():
    # Get the uploaded CSV file
    file = request.files['file']

    # Read the CSV file into a DataFrame
    test_fault_x = pd.read_csv(file)

    # Make predictions using the trained model
    y_pred = model.predict(xgb.DMatrix(test_fault_x))
    yprob = np.argmax(y_pred, axis=1)
    predictions = [round(value) for value in yprob]

    # Prepare the response
    response = {
        'predictions': predictions
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
