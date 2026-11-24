from flask import Flask, request, jsonify, send_from_directory
import tensorflow as tf
import numpy as np
import json
from PIL import Image
from io import BytesIO

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model(
    "plant_disease_model.keras"
)

# Load class names
with open("class_names.json", "r") as f:
    class_names = json.load(f)
    disease_info = {
    "Apple___Cedar_apple_rust": {
        "description": "Cedar Apple Rust is a fungal disease affecting apple trees.",

        "symptoms": "Yellow or orange spots appear on leaves. Leaves may develop brown lesions.",

        "treatment": "Remove severely infected leaves and seek local agricultural advice about suitable fungicides.",

        "prevention": "Maintain good air circulation and manage nearby cedar or juniper hosts where practical."
    }
}


@app.route("/")
def home():
    return send_from_directory(".", "index.html")


@app.route("/style.css")
def css():
    return send_from_directory(".", "style.css")


@app.route("/script.js")
def js():
    return send_from_directory(".", "script.js")


@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({"error": "Please upload an image"}), 400

    try:
        file = request.files["image"]

        img = Image.open(BytesIO(file.read())).convert("RGB")
        img = img.resize((160, 160))

        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array, verbose=0)

        index = int(np.argmax(prediction[0]))

        disease = class_names[index]
        confidence = float(prediction[0][index]) * 100

        info = disease_info.get(disease, {
            "description": "Information about this disease is not available yet.",
            "symptoms": "Please consult an agricultural expert.",
            "treatment": "Seek advice from a plant protection specialist.",
            "prevention": "Maintain proper plant hygiene."
        })

        return jsonify({
            "disease": disease,
            "confidence": round(confidence, 2),
            "description": info["description"],
            "symptoms": info["symptoms"],
            "treatment": info["treatment"],
            "prevention": info["prevention"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)