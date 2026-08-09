"""
OryzaScan — Rice Leaf Pathology Scanner
Flask backend: loads the trained CNN and serves image classification
over a small JSON API for the frontend in templates/index.html.
"""

import glob
import io
import os
from datetime import datetime

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image

# ---------------------------------------------------------------------------
# Config & Path Handling for Vercel Serverless Environments
# ---------------------------------------------------------------------------

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(APP_DIR, "model")
TEMPLATE_DIR = os.path.join(APP_DIR, "templates")
STATIC_DIR = os.path.join(APP_DIR, "static")

IMG_HEIGHT = 300
IMG_WIDTH = 300

CLASS_NAMES = ["Bacterialblight", "Brownspot", "Leafsmut"]
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB

# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

BASELINE_HISTORY = {
    "label": "Baseline (no augmentation / dropout)",
    "epochs": 15,
    "accuracy": [0.4511, 0.5833, 0.7105, 0.8177, 0.8292, 0.8836, 0.8867, 0.9041,
                 0.9195, 0.9347, 0.9352, 0.9441, 0.9318, 0.9449, 0.9576],
    "loss": [1.2402, 0.8834, 0.6900, 0.5243, 0.4559, 0.3219, 0.2745, 0.2387,
             0.1742, 0.1319, 0.1192, 0.0914, 0.1137, 0.0921, 0.0864],
    "val_accuracy": [0.5342, 0.6677, 0.7799, 0.8088, 0.8226, 0.8194, 0.8825, 0.8942,
                     0.9177, 0.9017, 0.9124, 0.9006, 0.9199, 0.9231, 0.9519],
    "val_loss": [0.9873, 0.7258, 0.5791, 0.4876, 0.4472, 0.3946, 0.2958, 0.2543,
                 0.1881, 0.2114, 0.1807, 0.1979, 0.1886, 0.1654, 0.2142],
    "params": 1709799,
}

AUGMENTED_HISTORY = {
    "label": "Augmented + dropout + early stopping",
    "epochs": 26,
    "accuracy": [0.5275, 0.7773, 0.8195, 0.8403, 0.8556, 0.8631, 0.8723, 0.8819,
                 0.8918, 0.9032, 0.8988, 0.9040, 0.9153, 0.9246, 0.9276, 0.9347,
                 0.9263, 0.9156, 0.9390, 0.9318, 0.9548, 0.9636, 0.9558, 0.9495,
                 0.9503, 0.9566],
    "loss": [3.2018, 0.5693, 0.4580, 0.3820, 0.3425, 0.3322, 0.3169, 0.2732,
             0.2687, 0.2460, 0.2426, 0.2494, 0.2045, 0.1800, 0.1673, 0.1547,
             0.1752, 0.2403, 0.1477, 0.1769, 0.1140, 0.0926, 0.0961, 0.1284,
             0.1432, 0.1210],
    "val_accuracy": [0.8066, 0.8376, 0.8248, 0.8558, 0.8248, 0.8761, 0.9006, 0.9028,
                     0.9071, 0.8942, 0.9092, 0.8996, 0.9348, 0.9316, 0.9231, 0.9573,
                     0.9338, 0.9423, 0.9274, 0.9135, 0.9722, 0.9679, 0.9701, 0.9519,
                     0.9744, 0.9669],
    "val_loss": [0.5941, 0.4529, 0.4084, 0.3505, 0.4102, 0.3095, 0.2537, 0.2533,
                 0.2482, 0.2386, 0.1973, 0.2201, 0.1653, 0.1545, 0.1795, 0.1409,
                 0.1505, 0.1409, 0.1520, 0.2051, 0.0755, 0.0756, 0.0839, 0.1307,
                 0.0764, 0.1052],
    "params": 43675075,
    "best_epoch_by_val_loss": 21,
}

DISEASE_INFO = {
    "Bacterialblight": {
        "display_name": "Bacterial Leaf Blight",
        "type": "Bacterial",
        "pathogen": "Xanthomonas oryzae pv. oryzae",
        "symptoms": "Long, wavy, water-soaked streaks starting at the leaf tip and margins...",
        "conditions": "Favoured by warm, humid weather (roughly 25–34°C)...",
        "notes": "Managed mainly through resistant varieties, disease-free seed, and balanced nitrogen use.",
        "solutions": [
            "Remove and burn infected plant debris immediately to reduce inoculum load.",
            "Apply copper-based bactericides (e.g., Copper Oxychloride 50 WP at 3 g/L)...",
        ],
        "color": "rust",
    },
    "Brownspot": {
        "display_name": "Brown Spot",
        "type": "Fungal",
        "pathogen": "Cochliobolus miyabeanus",
        "symptoms": "Small circular-to-oval lesions on the leaf with a brown margin...",
        "conditions": "Linked to nutrient-poor soil plus high humidity.",
        "notes": "Balanced soil fertility, certified seed, and resistant varieties are the standard countermeasures.",
        "solutions": [
            "Correct soil nutrient deficiencies first...",
        ],
        "color": "coffee",
    },
    "Leafsmut": {
        "display_name": "Leaf Smut",
        "type": "Fungal",
        "pathogen": "Entyloma oryzae",
        "symptoms": "Small, slightly raised, angular black specks scattered across the leaf surface...",
        "conditions": "Widespread wherever rice is grown, but usually appears late in the season.",
        "notes": "Generally a minor disease with limited yield impact.",
        "solutions": [
            "Monitor infection level...",
        ],
        "color": "soot",
    },
}

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

_model = None
_model_error = None
_model_path = None


def _find_model_file():
    candidates = sorted(
        glob.glob(os.path.join(MODEL_DIR, "*.keras"))
        + glob.glob(os.path.join(MODEL_DIR, "*.h5"))
    )
    return candidates[0] if candidates else None


def load_model_once():
    global _model, _model_error, _model_path

    if _model is not None:
        return

    path = _find_model_file()
    if path is None:
        _model_error = (
            "No model file found in /model. Export your trained model as "
            "a .h5 or .keras file and place it in the model/ folder."
        )
        return

    try:
        import tensorflow as tf

        _model = tf.keras.models.load_model(path, compile=False)
        _model_path = path
    except Exception as exc:
        _model_error = f"Found {os.path.basename(path)} but couldn't load it: {exc}"


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(file_bytes):
    img = Image.open(io.BytesIO(file_bytes))
    img = img.convert("RGB")
    img = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    return arr


# ---------------------------------------------------------------------------
# Flask app instantiation with explicit template/static folder paths
# ---------------------------------------------------------------------------

app = Flask(
    __name__,
    template_folder=TEMPLATE_DIR,
    static_folder=STATIC_DIR
)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH


@app.errorhandler(413)
def too_large(_e):
    return jsonify(error="That image is over the 10 MB limit."), 413


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def status():
    load_model_once()
    return jsonify(
        model_loaded=_model is not None,
        model_file=os.path.basename(_model_path) if _model_path else None,
        error=_model_error,
        classes=CLASS_NAMES,
        input_size=[IMG_WIDTH, IMG_HEIGHT],
    )


@app.route("/api/model-info")
def model_info():
    return jsonify(
        classes=CLASS_NAMES,
        disease_info=DISEASE_INFO,
        baseline=BASELINE_HISTORY,
        augmented=AUGMENTED_HISTORY,
        headline_accuracy=max(AUGMENTED_HISTORY["val_accuracy"]),
    )


@app.route("/api/predict", methods=["POST"])
def predict():
    load_model_once()

    if _model is None:
        return jsonify(error=_model_error or "Model not loaded."), 503

    if "image" not in request.files:
        return jsonify(error="No image was sent with the request."), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify(error="No file selected."), 400

    if not allowed_file(file.filename):
        return jsonify(error="Please upload a JPG, PNG, WEBP, or BMP image."), 400

    try:
        raw = file.read()
        batch = preprocess_image(raw)
    except Exception:
        return jsonify(error="Couldn't read that file as an image."), 400

    try:
        preds = _model.predict(batch, verbose=0)[0]
    except Exception as exc:
        return jsonify(error=f"The model failed on this image: {exc}"), 500

    preds = [float(p) for p in preds]
    top_idx = int(np.argmax(preds))

    probabilities = {CLASS_NAMES[i]: preds[i] for i in range(len(CLASS_NAMES))}

    return jsonify(
        predicted_class=CLASS_NAMES[top_idx],
        display_name=DISEASE_INFO[CLASS_NAMES[top_idx]]["display_name"],
        confidence=preds[top_idx],
        probabilities=probabilities,
        case_id=datetime.now().strftime("OS-%Y%m%d-%H%M%S"),
    )


# Expose application handler for Vercel WSGI environment
app_handler = app

if __name__ == "__main__":
    app.run(debug=True, port=5000)
