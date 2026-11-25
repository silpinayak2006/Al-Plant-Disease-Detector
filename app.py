import os

# Limit TensorFlow resource usage
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from flask import Flask, request, jsonify, send_from_directory
import tensorflow as tf
import numpy as np
import json
from PIL import Image, UnidentifiedImageError
from io import BytesIO

app = Flask(__name__)

# Configuration
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

MODEL_PATH = "plant_disease_model.keras"
CLASS_NAMES_PATH = "class_names.json"

# TensorFlow thread configuration
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# Load trained model
try:
    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )
    app.logger.info("Model loaded successfully")

except Exception:
    app.logger.exception("Model loading failed")
    raise

# Load class names
with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

# Disease information
disease_info = {
    "Apple___Cedar_apple_rust": {
        "description": "Cedar Apple Rust is a fungal disease affecting apple trees.",
        "symptoms": "Yellow or orange spots appear on leaves.",
        "treatment": "Remove severely infected leaves and seek agricultural advice.",
        "prevention": "Maintain good air circulation."
    }
}


# Image preprocessing
def preprocess_image(file):
    try:
        image = Image.open(
            BytesIO(file.read())
        ).convert("RGB")

        image = image.resize((160, 160))

        image_array = np.asarray(
            image,
            dtype=np.float32
        )

        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        return image_array

    except (UnidentifiedImageError, OSError, ValueError) as e:
        raise ValueError(
            "Invalid or corrupted image file."
        ) from e


# Disease prediction
def predict_disease(image_array):
    try:
        predictions = model(
            image_array,
            training=False
        ).numpy()

        if predictions.ndim != 2:
            raise ValueError("Unexpected model output.")

        if predictions.shape[0] != 1:
            raise ValueError("Invalid prediction batch.")

        if predictions.shape[1] != len(class_names):
            raise ValueError(
                "Model output does not match class names."
            )

        if not np.all(np.isfinite(predictions)):
            raise ValueError("Invalid prediction values.")

        index = int(np.argmax(predictions[0]))

        disease = class_names[index]

        confidence = float(
            predictions[0][index]
        ) * 100

        return {
            "disease": disease,
            "confidence": round(confidence, 2)
        }

    except Exception as e:
        app.logger.exception("Prediction failed")
        raise RuntimeError(
            "Unable to process prediction."
        ) from e


# Home page
@app.route("/")
def home():
    return send_from_directory(".", "index.html")


# CSS
@app.route("/style.css")
def css():
    return send_from_directory(".", "style.css")


# JavaScript
@app.route("/script.js")
def js():
    return send_from_directory(".", "script.js")


# Prediction API
@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({
            "success": False,
            "error": "Please upload an image."
        }), 400

    file = request.files["image"]

    if not file.filename:
        return jsonify({
            "success": False,
            "error": "No image selected."
        }), 400

    try:
        # Process image
        image_array = preprocess_image(file)

        # Predict disease
        result = predict_disease(image_array)

        # Get disease information
        info = disease_info.get(
            result["disease"],
            {
                "description": "Information about this disease is not available yet.",
                "symptoms": "Please consult an agricultural expert.",
                "treatment": "Seek advice from a plant protection specialist.",
                "prevention": "Maintain proper plant hygiene."
            }
        )

        return jsonify({
            "success": True,
            **result,
            **info
        }), 200

    except ValueError as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

    except RuntimeError:
        return jsonify({
            "success": False,
            "error": "Prediction failed. Please try again."
        }), 500

    except Exception:
        app.logger.exception("Unexpected server error")

        return jsonify({
            "success": False,
            "error": "An unexpected server error occurred."
        }), 500


# File size error handler
@app.errorhandler(413)
def file_too_large(error):
    return jsonify({
        "success": False,
        "error": "Image is too large. Maximum size is 5 MB."
    }), 413


if __name__ == "__main__":
    app.run(debug=False)
