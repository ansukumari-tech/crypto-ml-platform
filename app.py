from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)
model = joblib.load("models/model.pkl")

@app.route("/")
def home():
    return "Crypto ML Platform is Running 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    previous_price = data["previous_price"]

    prediction = model.predict([[previous_price]])
    return jsonify({"predicted_price": float(prediction[0])})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))