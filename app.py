import os

# TensorFlow resource optimization
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

app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

MODEL_PATH = "plant_disease_model.keras"
CLASS_NAMES_PATH = "class_names.json"

# TensorFlow configuration
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# Load AI model
try:
    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )
    app.logger.info("AI model loaded successfully")

except Exception:
    app.logger.exception("Model loading failed")
    raise

# Load class names
with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)


# ------------------------------------------------
# DISEASE INFORMATION DATABASE
# ------------------------------------------------

DISEASE_GUIDANCE = {

    "Apple_scab": (
        "Apple scab is a fungal disease affecting apple leaves and fruits.",
        "Olive-green or dark spots appear on leaves. Severe infection can cause leaf drop.",
        "Remove infected fallen leaves and follow locally recommended fungicide guidance.",
        "Prune trees for better airflow and remove fallen infected leaves."
    ),

    "Apple_Black_rot": (
        "Black rot is a fungal disease affecting apple trees.",
        "Brown leaf spots, dead branches and dark fruit rot may occur.",
        "Remove infected plant parts and manage the disease using locally recommended fungicides.",
        "Maintain orchard sanitation and remove mummified fruits."
    ),

    "Apple_Cedar_apple_rust": (
        "Cedar apple rust is a fungal disease affecting apple trees.",
        "Yellow or orange spots appear on leaves, sometimes with orange structures underneath.",
        "Remove severely infected leaves and follow agricultural disease-management advice.",
        "Maintain airflow and manage nearby alternate host plants where practical."
    ),

    "Cherry_Powdery_mildew": (
        "Powdery mildew is a fungal disease affecting cherry plants.",
        "White powder-like growth appears on leaves and shoots.",
        "Remove badly infected shoots and use locally recommended disease controls.",
        "Improve airflow and avoid excessive nitrogen fertilizer."
    ),

    "Corn_Cercospora_leaf_spot Gray_leaf_spot": (
        "Gray leaf spot is a fungal disease affecting maize.",
        "Long rectangular gray or tan lesions develop between leaf veins.",
        "Use locally recommended disease-management practices and remove crop residues where appropriate.",
        "Grow resistant varieties and rotate crops."
    ),

    "Corn_Common_rust": (
        "Common rust is a fungal disease of maize.",
        "Small reddish-brown raised pustules appear on both leaf surfaces.",
        "Use resistant varieties and seek local agricultural advice about fungicide use.",
        "Plant resistant hybrids and monitor crops regularly."
    ),

    "Corn_Northern_Leaf_Blight": (
        "Northern leaf blight is a fungal disease affecting maize.",
        "Long cigar-shaped gray-green lesions develop on leaves.",
        "Remove infected crop residues and use locally recommended disease controls.",
        "Use resistant varieties and practise crop rotation."
    ),

    "Grape_Black_rot": (
        "Black rot is a fungal disease affecting grapevines.",
        "Brown circular leaf spots and shrivelled black fruits may appear.",
        "Remove infected berries and plant debris and follow local disease-control advice.",
        "Maintain good airflow and remove infected plant material."
    ),

    "Grape_Esca_(Black_Measles)": (
        "Esca, also called black measles, is a complex grapevine trunk disease.",
        "Leaves may show yellow or red patterns with green veins. Fruits may develop dark spots.",
        "Remove severely affected wood and consult a grape-production specialist.",
        "Protect pruning wounds and use healthy planting material."
    ),

    "Grape_Leaf_blight_(Isariopsis_Leaf_Spot)": (
        "Grape leaf blight is a fungal leaf disease.",
        "Brown or reddish-brown spots develop on leaves and may cause premature leaf fall.",
        "Remove infected leaves and seek local recommendations for disease management.",
        "Improve air circulation and maintain vineyard sanitation."
    ),

    "Orange_Haunglongbing_(Citrus_greening)": (
        "Citrus greening, or Huanglongbing, is a serious bacterial disease spread by psyllid insects.",
        "Leaves develop irregular yellow patterns. Fruits may become small, misshapen and poorly coloured.",
        "There is no reliable cure for infected trees. Consult agricultural authorities about tree removal and vector management.",
        "Use certified planting material and monitor and manage citrus psyllids."
    ),

    "Peach_Bacterial_spot": (
        "Bacterial spot is a disease affecting peach leaves and fruits.",
        "Small dark spots develop on leaves and may lead to holes and premature leaf fall.",
        "Remove severely infected material and seek local agricultural advice.",
        "Use suitable resistant varieties and avoid working with wet plants."
    ),

    "Pepper,_bell_Bacterial_spot": (
        "Bacterial spot is a bacterial disease affecting bell pepper plants.",
        "Small water-soaked spots develop on leaves and may become dark brown.",
        "Remove infected plants or leaves where appropriate. Use disease-free seeds and locally recommended controls.",
        "Avoid overhead watering, rotate crops and maintain field sanitation."
    ),

    "Potato_Early_blight": (
        "Early blight is a fungal disease affecting potato plants.",
        "Brown circular leaf spots with concentric rings appear, often on older leaves.",
        "Remove infected leaves and follow locally recommended fungicide guidance.",
        "Rotate crops and avoid leaving infected plant debris in the field."
    ),

    "Potato_Late_blight": (
        "Late blight is a serious disease caused by a water mould.",
        "Water-soaked dark lesions develop on leaves and may spread rapidly in humid conditions.",
        "Remove infected plant material and seek immediate agricultural advice about disease control.",
        "Use healthy seed potatoes, resistant varieties and avoid prolonged leaf wetness."
    ),

    "Squash_Powdery_mildew": (
        "Powdery mildew is a fungal disease affecting squash plants.",
        "White powdery patches appear on leaves and may spread across the plant.",
        "Remove severely affected leaves and use locally recommended disease controls.",
        "Provide sufficient spacing and airflow and avoid excessive nitrogen."
    ),

    "Strawberry_Leaf_scorch": (
        "Leaf scorch is a fungal disease affecting strawberry plants.",
        "Small dark purple spots develop on leaves, with reddish-brown or scorched areas.",
        "Remove infected leaves and seek local agricultural disease-management advice.",
        "Maintain field sanitation and avoid excessive leaf wetness."
    ),

    "Tomato_Bacterial_spot": (
        "Bacterial spot is a bacterial disease affecting tomato plants.",
        "Small dark spots appear on leaves, stems and fruits.",
        "Remove infected plant material and use disease-free seeds and locally recommended controls.",
        "Rotate crops, avoid overhead irrigation and maintain sanitation."
    ),

    "Tomato_Early_blight": (
        "Early blight is a fungal disease affecting tomato plants.",
        "Brown spots with concentric rings appear on older leaves.",
        "Remove infected leaves and follow locally recommended fungicide guidance.",
        "Use crop rotation, mulch and proper plant spacing."
    ),

    "Tomato_Late_blight": (
        "Late blight is a destructive tomato disease caused by a water mould.",
        "Large water-soaked dark patches appear on leaves and stems. White growth may develop underneath.",
        "Remove infected plants or plant parts and seek immediate agricultural advice.",
        "Use healthy seedlings, avoid overhead watering and monitor crops regularly."
    ),

    "Tomato_Leaf_Mold": (
        "Tomato leaf mold is a fungal disease that commonly develops in humid conditions.",
        "Pale yellow patches appear on upper leaf surfaces, with olive-green mold underneath.",
        "Remove infected leaves and improve ventilation. Follow local disease-control advice.",
        "Reduce humidity around plants and provide good airflow."
    ),

    "Tomato_Septoria_leaf_spot": (
        "Septoria leaf spot is a fungal disease affecting tomato leaves.",
        "Small circular spots with gray centres and dark borders develop on leaves.",
        "Remove infected lower leaves and follow locally recommended disease controls.",
        "Use mulch, rotate crops and avoid wetting foliage."
    ),

    "Tomato_Spider_mites Two-spotted_spider_mite": (
        "Two-spotted spider mites are tiny pests that feed on tomato leaves.",
        "Leaves develop small yellow dots, bronzing and sometimes fine webbing.",
        "Inspect leaf undersides and use suitable locally approved mite-control methods.",
        "Monitor plants regularly and conserve beneficial predatory insects."
    ),

    "Tomato_Target_Spot": (
        "Target spot is a fungal disease affecting tomato plants.",
        "Brown circular leaf spots develop, sometimes with concentric rings.",
        "Remove infected leaves and follow locally recommended disease-management practices.",
        "Maintain airflow, rotate crops and avoid prolonged leaf wetness."
    ),

    "Tomato_Tomato_Yellow_Leaf_Curl_Virus": (
        "Tomato yellow leaf curl is a viral disease transmitted mainly by whiteflies.",
        "Leaves curl upward, become yellow and plants may show stunted growth.",
        "There is no cure for infected plants. Remove infected plants and manage whitefly populations.",
        "Use resistant varieties and monitor whiteflies."
    ),

    "Tomato_Tomato_mosaic_virus": (
        "Tomato mosaic virus is a viral disease affecting tomato plants.",
        "Leaves may show light and dark green mosaic patterns and become distorted.",
        "There is no cure. Remove infected plants and disinfect tools to reduce spread.",
        "Use virus-free seeds and maintain good hygiene."
    )
}


# ------------------------------------------------
# HEALTHY PLANT INFORMATION
# ------------------------------------------------

HEALTHY_PLANTS = {
    "Apple": "apple",
    "Blueberry": "blueberry",
    "Cherry": "cherry",
    "Corn": "corn",
    "Grape": "grape",
    "Peach": "peach",
    "Pepper,_bell": "bell pepper",
    "Potato": "potato",
    "Raspberry": "raspberry",
    "Soybean": "soybean",
    "Strawberry": "strawberry",
    "Tomato": "tomato"
}


def get_disease_info(label):

    if label == "Background_without_leaves":
        return {
            "description": "No plant leaf was detected. Please upload a clear plant leaf image.",
            "symptoms": "Not applicable.",
            "treatment": "Upload a suitable plant leaf image.",
            "prevention": "Use clear images containing plant leaves."
        }

    parts = label.split("___", 1)

    if len(parts) != 2:
        return {
            "description": "Information is not available for this class.",
            "symptoms": "Please consult an agricultural expert.",
            "treatment": "Seek advice from a plant protection specialist.",
            "prevention": "Maintain proper plant hygiene."
        }

    crop, condition = parts

    if condition == "healthy":
        crop_name = HEALTHY_PLANTS.get(crop, crop)

        return {
            "description": f"The {crop_name} plant appears healthy according to the AI model.",
            "symptoms": "No disease symptoms were identified by the model.",
            "treatment": "No disease treatment is indicated. Continue normal plant care.",
            "prevention": "Provide proper watering, sunlight, nutrition and regular monitoring."
        }

    guidance = DISEASE_GUIDANCE.get(condition)

    if guidance:
        return {
            "description": guidance[0],
            "symptoms": guidance[1],
            "treatment": guidance[2],
            "prevention": guidance[3]
        }

    return {
        "description": "Information about this condition is not available yet.",
        "symptoms": "Please consult an agricultural expert.",
        "treatment": "Seek advice from a plant protection specialist.",
        "prevention": "Maintain proper plant hygiene."
    }


# ------------------------------------------------
# IMAGE PREPROCESSING
# ------------------------------------------------

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
        raise ValueError("Invalid or corrupted image file.") from e


# ------------------------------------------------
# AI PREDICTION
# ------------------------------------------------

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
            raise ValueError("Model output does not match class names.")

        if not np.all(np.isfinite(predictions)):
            raise ValueError("Invalid prediction values.")

        index = int(np.argmax(predictions[0]))

        disease = class_names[index]

        confidence = float(predictions[0][index]) * 100

        return {
            "disease": disease,
            "confidence": round(confidence, 2)
        }

    except Exception as e:
        app.logger.exception("Prediction failed")
        raise RuntimeError("Unable to process prediction.") from e


# ------------------------------------------------
# WEBSITE ROUTES
# ------------------------------------------------

@app.route("/")
def home():
    return send_from_directory(".", "index.html")


@app.route("/style.css")
def css():
    return send_from_directory(".", "style.css")


@app.route("/script.js")
def js():
    return send_from_directory(".", "script.js")


# ------------------------------------------------
# PREDICTION API
# ------------------------------------------------

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

        image_array = preprocess_image(file)

        result = predict_disease(image_array)

        info = get_disease_info(result["disease"])

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


# ------------------------------------------------
# FILE SIZE ERROR
# ------------------------------------------------

@app.errorhandler(413)
def file_too_large(error):

    return jsonify({
        "success": False,
        "error": "Image is too large. Maximum size is 5 MB."
    }), 413


# ------------------------------------------------
# RUN APPLICATION
# ------------------------------------------------

if __name__ == "__main__":
    app.run(debug=False)
