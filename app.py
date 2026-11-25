import os

# Limit TensorFlow resource usage on Render
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

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

with open(CLASS_NAMES_PATH, "r") as file:
    class_names = json.load(file)


# Disease information for all detected classes
DISEASE_GUIDANCE = {

    "Apple___Apple_scab": {
        "information": "Apple scab is a fungal disease that affects apple leaves and fruits.",
        "symptoms": "Olive-green or dark spots appear on leaves and fruits. Infected leaves may fall.",
        "treatment": "Remove infected leaves and fallen plant material. Consult an agricultural expert about suitable fungicides.",
        "prevention": "Maintain orchard hygiene, prune trees for good air circulation, and use resistant varieties."
    },

    "Apple___Black_rot": {
        "information": "Black rot is a fungal disease that affects apple leaves, branches, and fruits.",
        "symptoms": "Dark leaf spots, dead branches, and black, rotting fruit may appear.",
        "treatment": "Remove infected fruits and branches. Use suitable registered fungicides on expert advice.",
        "prevention": "Remove fallen fruits and dead branches and maintain proper orchard sanitation."
    },

    "Apple___Cedar_apple_rust": {
        "information": "Cedar apple rust is a fungal disease that affects apple leaves and fruits.",
        "symptoms": "Yellow-orange spots appear on leaves. Orange structures may develop on the underside.",
        "treatment": "Remove nearby infected plant material where practical. Consult an expert about suitable fungicide treatment.",
        "prevention": "Choose resistant apple varieties and manage nearby alternate host plants where appropriate."
    },

    "Apple___Apple_healthy": {
        "information": "The apple leaf appears healthy, with no disease detected by the model.",
        "symptoms": "No obvious disease symptoms are detected.",
        "treatment": "No disease treatment is indicated by this prediction.",
        "prevention": "Continue proper watering, balanced nutrition, pruning, and regular plant inspection."
    },

    "Blueberry___healthy": {
        "information": "The blueberry leaf appears healthy, with no disease detected by the model.",
        "symptoms": "No obvious disease symptoms are detected.",
        "treatment": "No disease treatment is indicated by this prediction.",
        "prevention": "Maintain suitable soil acidity, proper watering, and regular plant inspection."
    },

    "Cherry_(including_sour)___Powdery_mildew": {
        "information": "Powdery mildew is a fungal disease that affects cherry leaves and shoots.",
        "symptoms": "White powdery patches appear on leaves and young shoots. Leaves may become distorted.",
        "treatment": "Remove badly infected plant parts and consult an expert about suitable fungicides.",
        "prevention": "Provide good air circulation, avoid excessive nitrogen, and maintain proper spacing."
    },

    "Cherry_(including_sour)___healthy": {
        "information": "The cherry leaf appears healthy, with no disease detected by the model.",
        "symptoms": "No obvious disease symptoms are detected.",
        "treatment": "No disease treatment is indicated by this prediction.",
        "prevention": "Maintain proper watering, pruning, and regular monitoring."
    },

    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "information": "Gray leaf spot is a fungal disease affecting maize leaves.",
        "symptoms": "Long, rectangular gray or brown lesions develop between leaf veins.",
        "treatment": "Remove infected crop residue where practical. Consult an agricultural expert about suitable fungicides.",
        "prevention": "Use resistant varieties, crop rotation, and proper field sanitation."
    },

    "Corn_(maize)___Common_rust_": {
        "information": "Common rust is a fungal disease affecting maize plants.",
        "symptoms": "Small reddish-brown raised pustules appear on both surfaces of leaves.",
        "treatment": "Use resistant varieties and seek expert advice if the infection is severe.",
        "prevention": "Grow resistant maize varieties and monitor crops regularly."
    },

    "Corn_(maize)___Northern_Leaf_Blight": {
        "information": "Northern leaf blight is a fungal disease affecting maize leaves.",
        "symptoms": "Long, cigar-shaped gray-green or brown lesions appear on leaves.",
        "treatment": "Remove infected crop residue and consult an expert about appropriate disease management.",
        "prevention": "Use resistant varieties, crop rotation, and proper field sanitation."
    },

    "Corn_(maize)___healthy": {
        "information": "The maize leaf appears healthy, with no disease detected by the model.",
        "symptoms": "No obvious disease symptoms are detected.",
        "treatment": "No disease treatment is indicated by this prediction.",
        "prevention": "Maintain balanced fertilization, proper irrigation, and regular crop monitoring."
    },

    "Grape___Black_rot": {
        "information": "Black rot is a fungal disease that affects grape leaves and berries.",
        "symptoms": "Brown circular leaf spots and dark, shriveled berries may develop.",
        "treatment": "Remove infected berries and leaves. Consult an expert about suitable fungicides.",
        "prevention": "Maintain good vineyard sanitation, prune for airflow, and remove infected fruit."
    },

    "Grape___Esca_(Black_Measles)": {
        "information": "Esca, also called black measles, is a complex grapevine disease.",
        "symptoms": "Leaves may develop striped discoloration, while berries can show dark spots.",
        "treatment": "Remove severely affected plant material where appropriate and seek professional vineyard advice.",
        "prevention": "Use healthy planting material and protect pruning wounds using recommended practices."
    },

    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "information": "Grape leaf blight is a fungal disease that damages grape leaves.",
        "symptoms": "Brown or dark spots develop on leaves, sometimes leading to drying and leaf fall.",
        "treatment": "Remove infected fallen leaves and consult an expert about suitable fungicide options.",
        "prevention": "Maintain vineyard hygiene, proper spacing, and good air circulation."
    },

    "Grape___healthy": {
        "information": "The grape leaf appears healthy, with no disease detected by the model.",
        "symptoms": "No obvious disease symptoms are detected.",
        "treatment": "No disease treatment is indicated by this prediction.",
        "prevention": "Maintain proper pruning, irrigation, and regular vineyard inspection."
    },

    "Orange___Haunglongbing_(Citrus_greening)": {
        "information": "Citrus greening, also called Huanglongbing (HLB), is a serious bacterial disease of citrus plants.",
        "symptoms": "Leaves may show uneven yellowing, blotchy mottling, and fruits may become small or misshapen.",
        "treatment": "There is no reliable cure for infected trees. Contact an agricultural expert for management advice.",
        "prevention": "Use certified disease-free planting material and control the Asian citrus psyllid vector."
    },

    "Peach___Bacterial_spot": {
        "information": "Bacterial spot is a disease that affects peach leaves and fruits.",
        "symptoms": "Small dark, water-soaked spots appear on leaves and fruits. Leaves may develop holes.",
        "treatment": "Remove infected plant material where practical and consult an expert about approved treatments.",
        "prevention": "Use healthy planting material, avoid overhead watering, and maintain orchard hygiene."
    },

    "Peach___healthy": {
        "information": "The peach leaf appears healthy, with no disease detected by the model.",
        "symptoms": "No obvious disease symptoms are detected.",
        "treatment": "No disease treatment is indicated by this prediction.",
        "prevention": "Maintain proper pruning, watering, and regular orchard inspection."
    },

    "Pepper,_bell___Bacterial_spot": {
        "information": "Bacterial spot is a bacterial disease affecting pepper leaves and fruits.",
        "symptoms": "Small, dark, water-soaked spots appear on leaves. Spots may become brown, and infected leaves may fall.",
        "treatment": "Remove infected plant parts where practical. Use disease-free seeds and consult an expert about registered treatments.",
        "prevention": "Use healthy seeds, avoid overhead watering, maintain proper spacing, and remove infected plant debris."
    },

    "Pepper,_bell___healthy": {
        "information": "The bell pepper leaf appears healthy, with no disease detected by the model.",
        "symptoms": "No obvious disease symptoms are detected.",
        "treatment": "No disease treatment is indicated by this prediction.",
        "prevention": "Maintain balanced watering, proper nutrition, and regular crop monitoring."
    },

    "Potato___Early_blight": {
        "information": "Early blight is a fungal disease affecting potato leaves and sometimes tubers.",
        "symptoms": "Brown spots with circular rings develop on older leaves, often creating a target-like appearance.",
        "treatment": "Remove infected plant debris and consult an expert about suitable fungicides.",
        "prevention": "Use healthy seed potatoes, rotate crops, and avoid prolonged leaf wetness."
    },

    "Potato___Late_blight": {
        "information": "Late blight is a serious disease of potato caused by a fungus-like organism.",
        "symptoms": "Water-soaked dark patches develop on leaves. White growth may appear underneath in humid conditions.",
        "treatment": "Seek prompt agricultural advice. Remove infected plants where recommended and use approved treatments.",
        "prevention": "Use healthy seed potatoes, resistant varieties, proper spacing, and regular field monitoring."
    },

    "Potato___healthy": {
        "information": "The potato leaf appears healthy, with no disease detected by the model.",
        "symptoms": "No obvious disease symptoms are detected.",
        "treatment": "No disease treatment is indicated by this prediction.",
        "prevention": "Use healthy seed potatoes and maintain proper irrigation and crop rotation."
    },

    "Raspberry___healthy": {
        "information": "The raspberry leaf appears healthy, with no disease detected by the model.",
        "symptoms": "No obvious disease symptoms are detected.",
        "treatment": "No disease treatment is indicated by this prediction.",
        "prevention": "Maintain proper pruning, watering, and regular plant inspection."
    },

    "Soybean___healthy": {
        "information": "The soybean leaf appears healthy, with no disease detected by the model.",
        "symptoms": "No obvious disease symptoms are detected.",
        "treatment": "No disease treatment is indicated by this prediction.",
        "prevention": "Use quality seeds, maintain balanced nutrition, and monitor the crop regularly."
    },

    "Squash___Powdery_mildew": {
        "information": "Powdery mildew is a fungal disease commonly affecting squash leaves.",
        "symptoms": "White powdery patches develop on leaf surfaces. Leaves may turn yellow and dry.",
        "treatment": "Remove badly infected leaves and consult an expert about suitable fungicide options.",
        "prevention": "Provide good air circulation, avoid overcrowding, and water near the soil."
    },

    "Strawberry___Leaf_scorch": {
        "information": "Leaf scorch is a fungal disease affecting strawberry leaves.",
        "symptoms": "Small purple or reddish spots appear on leaves. The centers may become brown or gray.",
        "treatment": "Remove infected leaves and consult an expert about appropriate disease management.",
        "prevention": "Use healthy planting material, maintain good spacing, and remove infected plant debris."
    },

    "Strawberry___healthy": {
        "information": "The strawberry leaf appears healthy, with no disease detected by the model.",
        "symptoms": "No obvious disease symptoms are detected.",
        "treatment": "No disease treatment is indicated by this prediction.",
        "prevention": "Maintain proper watering, spacing, and regular plant inspection."
    },

    "Tomato___Bacterial_spot": {
        "information": "Bacterial spot is a disease affecting tomato leaves, stems, and fruits.",
        "symptoms": "Small dark spots appear on leaves and fruits. Leaves may yellow and fall.",
        "treatment": "Remove infected plant material and consult an expert about approved treatments.",
        "prevention": "Use disease-free seeds, avoid overhead watering, and maintain crop hygiene."
    },

    "Tomato___Early_blight": {
        "information": "Early blight is a fungal disease affecting tomato leaves, stems, and fruits.",
        "symptoms": "Brown spots with concentric rings appear on older leaves. Yellowing may develop around them.",
        "treatment": "Remove infected leaves and consult an expert about suitable fungicides.",
        "prevention": "Rotate crops, use healthy seedlings, and avoid wetting leaves during irrigation."
    },

    "Tomato___Late_blight": {
        "information": "Late blight is a serious disease affecting tomato plants.",
        "symptoms": "Water-soaked dark lesions develop on leaves and stems. White fungal-like growth may appear in humid weather.",
        "treatment": "Seek prompt agricultural advice, remove infected material where recommended, and use approved treatments.",
        "prevention": "Use healthy seedlings, avoid overhead irrigation, and monitor plants regularly."
    },

    "Tomato___Leaf_Mold": {
        "information": "Tomato leaf mold is a fungal disease that mainly affects tomato leaves.",
        "symptoms": "Pale yellow patches appear on upper leaf surfaces, with olive-green or gray growth underneath.",
        "treatment": "Remove affected leaves and consult an expert about suitable fungicide options.",
        "prevention": "Improve ventilation, avoid overcrowding, and reduce excessive humidity."
    },

    "Tomato___Septoria_leaf_spot": {
        "information": "Septoria leaf spot is a fungal disease affecting tomato leaves.",
        "symptoms": "Small circular spots with gray centers and dark edges appear on older leaves.",
        "treatment": "Remove infected leaves and consult an agricultural expert about suitable fungicides.",
        "prevention": "Rotate crops, remove plant debris, and avoid overhead watering."
    },

    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "information": "Two-spotted spider mites are tiny pests that feed on tomato leaves.",
        "symptoms": "Leaves develop tiny pale spots, yellowing, and sometimes fine webbing.",
        "treatment": "Inspect the undersides of leaves. Use suitable mite-control methods recommended by an agricultural expert.",
        "prevention": "Monitor plants regularly, reduce dust, and protect beneficial insects."
    },

    "Tomato___Target_Spot": {
        "information": "Target spot is a fungal disease affecting tomato leaves, stems, and fruits.",
        "symptoms": "Brown circular spots with concentric rings develop on leaves and may cause leaf drop.",
        "treatment": "Remove infected plant material and consult an expert about suitable fungicides.",
        "prevention": "Maintain good air circulation, rotate crops, and avoid prolonged leaf wetness."
    },

    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "information": "Tomato yellow leaf curl is a viral disease commonly spread by whiteflies.",
        "symptoms": "Leaves curl upward, turn yellow, and become smaller. Plant growth may be stunted.",
        "treatment": "There is no direct cure for viral infection. Remove severely infected plants where recommended and manage whiteflies.",
        "prevention": "Use healthy seedlings, control whiteflies, and remove infected plants and weeds."
    },

    "Tomato___Tomato_mosaic_virus": {
        "information": "Tomato mosaic virus is a viral disease affecting tomato plants.",
        "symptoms": "Leaves may show light and dark green mosaic patterns, distortion, and reduced growth.",
        "treatment": "There is no direct cure. Remove infected plants where recommended and disinfect tools.",
        "prevention": "Use virus-free seeds, wash hands before handling plants, and disinfect gardening tools."
    },

    "Tomato___healthy": {
        "information": "The tomato leaf appears healthy, with no disease detected by the model.",
        "symptoms": "No obvious disease symptoms are detected.",
        "treatment": "No disease treatment is indicated by this prediction.",
        "prevention": "Maintain proper watering, balanced nutrition, and regular crop monitoring."
    }
}


def get_disease_info(label):

    if label == "Background_without_leaves":
        return {
            "information": "The uploaded image may not contain a recognizable plant leaf.",
            "symptoms": "No plant disease symptoms can be identified from this image.",
            "treatment": "Upload a clear image of a plant leaf.",
            "prevention": "Take a well-lit photograph with the leaf clearly visible."
        }

    if label in DISEASE_GUIDANCE:
        return DISEASE_GUIDANCE[label]

    return {
        "information": "Detailed information for this class is not available yet.",
        "symptoms": "Please consult an agricultural expert.",
        "treatment": "Seek advice from a plant protection specialist.",
        "prevention": "Maintain proper plant hygiene."
    }


@app.route("/")
def home():
    return send_from_directory(".", "index.html")


@app.route("/style.css")
def css():
    return send_from_directory(".", "style.css")


@app.route("/script.js")
def javascript():
    return send_from_directory(".", "script.js")


@app.route("/predict", methods=["POST"])
def predict():

    try:
        if "file" not in request.files:
            return jsonify({
                "success": False,
                "error": "Please upload a leaf image."
            }), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({
                "success": False,
                "error": "No image selected."
            }), 400

        image = Image.open(BytesIO(file.read())).convert("RGB")
        image = image.resize((160, 160))

        image_array = np.asarray(image, dtype=np.float32)
        image_array = np.expand_dims(image_array, axis=0)

        predictions = model(image_array, training=False).numpy()

        if predictions.ndim != 2 or predictions.shape[0] != 1:
            raise ValueError("Unexpected model output.")

        predicted_index = int(np.argmax(predictions[0]))

        if isinstance(class_names, dict):
            disease = class_names.get(
                str(predicted_index),
                class_names.get(predicted_index)
            )
        else:
            disease = class_names[predicted_index]

        if not disease:
            raise ValueError("Disease class not found.")

        confidence = float(predictions[0][predicted_index]) * 100

        info = get_disease_info(disease)

        return jsonify({
            "success": True,
            "disease": disease,
            "confidence": round(confidence, 2),
            "information": info["information"],
            "symptoms": info["symptoms"],
            "treatment": info["treatment"],
            "prevention": info["prevention"]
        })

    except UnidentifiedImageError:
        return jsonify({
            "success": False,
            "error": "Please upload a valid image."
        }), 400

    except Exception as e:
        print("Prediction error:", str(e))
        return jsonify({
            "success": False,
            "error": "Unable to process the image. Please try again."
        }), 500


@app.errorhandler(413)
def file_too_large(error):
    return jsonify({
        "success": False,
        "error": "Image is too large. Please upload an image under 5 MB."
    }), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
