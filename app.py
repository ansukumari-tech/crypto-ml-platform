from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("models/model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    previous_price = data["previous_price"]

    prediction = model.predict([[previous_price]])
    return jsonify({"predicted_price": float(prediction[0])})

if __name__ == "__main__":
    app.run(debug=True)