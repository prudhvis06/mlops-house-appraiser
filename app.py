import os
import pickle
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)
MODEL_PATH = os.path.join("models", "model.pkl")

def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return None

model = load_model()

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "model_loaded": model is not None})

@app.route("/predict", methods=["POST"])
def predict():
    global model
    if model is None:
        model = load_model()

    if model is None:
        return jsonify({"error": "Model artifact missing! Train model first."}), 500

    try:
        data = request.get_json(force=True)
        features = np.array(data["features"]).reshape(1, -1)
        pred = model.predict(features)
        return jsonify({
            "status": "success",
            "predicted_house_value_usd": round(float(pred[0]), 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400
