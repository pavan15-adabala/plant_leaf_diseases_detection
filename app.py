import json
import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from flask import Flask, request, jsonify
import io

# ======================================================
# CONFIGURATION
# ======================================================
MODEL_PATH = "model/plant_disease_model.h5"
LABELS_PATH = "class_labels.json"
KB_PATH = "knowledge_base.json"
IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 60

# ======================================================
# LOAD MODEL & LABELS (ONCE)
# ======================================================
app = Flask(__name__)

model = tf.keras.models.load_model(MODEL_PATH)

with open(LABELS_PATH, "r") as f:
    class_indices = json.load(f)

labels = {v: k for k, v in class_indices.items()}

with open(KB_PATH, "r") as f:
    knowledge_base = json.load(f)

# ======================================================
# HELPER FUNCTIONS
# ======================================================
def get_risk_level(confidence):
    if confidence >= 85:
        return "High Risk"
    elif confidence >= 65:
        return "Medium Risk"
    else:
        return "Low Risk"

def preprocess_image(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise ValueError("Invalid image file")

    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def get_recommendations(crop, disease, risk_level):
    crop_data = knowledge_base.get(crop)
    if crop_data:
        disease_data = crop_data.get(disease)
        if disease_data:
            remedies = disease_data.get(risk_level)
            if remedies:
                return remedies, "crop-specific"

    if risk_level == "High Risk":
        return [
            "Immediate action required",
            "Remove infected plant parts",
            "Consult an agricultural expert"
        ], "generic"
    elif risk_level == "Medium Risk":
        return [
            "Monitor plant condition regularly",
            "Apply preventive treatment",
            "Ensure proper irrigation and nutrition"
        ], "generic"
    else:
        return [
            "No immediate danger detected",
            "Maintain good crop hygiene",
            "Continue regular monitoring"
        ], "generic"

# ======================================================
# FLASK API ENDPOINT
# ======================================================
@app.route("/predict", methods=["POST"])
def predict_disease():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files["image"]

    if image_file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    image_bytes = image_file.read()

    if len(image_bytes) == 0:
        return jsonify({"error": "Empty image file"}), 400

    if not image_file.mimetype.startswith("image/"):
        return jsonify({"error": "Uploaded file is not an image"}), 400

    try:
        processed = preprocess_image(image_bytes)
    except ValueError:
        return jsonify({"error": "Invalid image. Upload JPG/PNG only."}), 400

    preds = model.predict(processed)

    idx = int(np.argmax(preds))
    confidence = float(np.max(preds) * 100)

    # ---------- UNKNOWN / OOD ----------
    if confidence < CONFIDENCE_THRESHOLD:
        return jsonify({
            "crop": "Unknown",
            "disease": "Uncertain Prediction",
            "confidence": round(confidence, 2),
            "risk_level": "Unknown",
            "goal": "Input Rejected",
            "recommendations": [
                "Image does not match trained crops",
                "Upload a clear Tomato, Potato, or Pepper leaf image"
            ]
        })

    raw_label = labels[idx]

    # ---------- HEALTHY ----------
    if "healthy" in raw_label.lower():
        crop = raw_label.split("_")[0]
        return jsonify({
            "crop": crop,
            "disease": "Healthy Leaf",
            "confidence": round(confidence, 2),
            "risk_level": "No Risk",
            "goal": "No Action Required",
            "recommendations": [
                "No disease detected",
                "Maintain regular watering and nutrition"
            ]
        })

    # ---------- NORMALIZATION ----------
    crop = raw_label.split("_")[0]
    disease = raw_label.replace(crop + "_", "").replace("_", " ").title()

    risk_level = get_risk_level(confidence)

    # ---------- GOAL ----------
    if risk_level == "High Risk":
        goal = "Immediate Treatment Required"
    elif risk_level == "Medium Risk":
        goal = "Preventive Treatment Required"
    else:
        goal = "Monitoring Only"

    remedies, knowledge_source = get_recommendations(crop, disease, risk_level)

    return jsonify({
        "crop": crop,
        "disease": disease,
        "confidence": round(confidence, 2),
        "risk_level": risk_level,
        "goal": goal,
        "knowledge_source": knowledge_source,
        "recommendations": remedies
    })

# ======================================================
# RUN SERVER (IMPORTANT CHANGE HERE)
# ======================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)