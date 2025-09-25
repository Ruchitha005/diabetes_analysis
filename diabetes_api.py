from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

# ===========================
# Load Model and Scaler
# ===========================
model = joblib.load("best_diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

# Create Flask App
app = Flask(__name__)

# ===========================
# Home Route
# ===========================
@app.route('/')
def home():
    return jsonify({
        "message": "Welcome to the Diabetes Prediction API!",
        "status": "API is running successfully"
    })

# ===========================
# Prediction Route
# ===========================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Get JSON Data
        data = request.get_json()

        # Required Features
        expected_features = [
            "Age", "Gender", "Polyuria", "Polydipsia", "sudden weight loss",
            "weakness", "Polyphagia", "Genital thrush", "visual blurring",
            "Itching", "Irritability", "delayed healing", "partial paresis",
            "muscle stiffness", "Alopecia", "Obesity"
        ]

        # 2. Validate input
        for feature in expected_features:
            if feature not in data:
                return jsonify({"error": f"Missing feature: {feature}"}), 400

        # 3. Convert to DataFrame
        input_data = pd.DataFrame([data], columns=expected_features)

        # 4. Scale only the Age column
        input_data['Age'] = scaler.transform(input_data[['Age']])

        # 5. Make prediction
        prediction = model.predict(input_data)[0]

        # 6. Convert prediction to readable label
        prediction_label = "Positive" if prediction == 1 else "Negative"

        return jsonify({
            "prediction": prediction_label,
            "message": "Diabetes risk prediction completed successfully"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ===========================
# Run the Flask App
# ===========================
if __name__ == '__main__':
    app.run(debug=True)
