import os
import hashlib
import logging
import tempfile
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from dotenv import load_dotenv

load_dotenv()

# -------------------
# Config
# -------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "newpneumonia.keras")
MODEL_URL = os.getenv("MODEL_URL")  

THRESHOLD = 0.5  # sigmoid threshold

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PneumoniaAPI")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# -------------------
# Helpers
# -------------------
def md5sum(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

# -------------------
# Download model if missing
# -------------------
logger.info(f"Checking model path: {MODEL_PATH}")
if not os.path.isfile(MODEL_PATH):
    if not MODEL_URL:
        logger.error("MODEL_URL not set and model file is missing!")
        raise FileNotFoundError("No model found and MODEL_URL not provided.")

    logger.info(f"Model not found locally. Downloading from {MODEL_URL}...")
    try:
        response = requests.get(MODEL_URL, stream=True, timeout=60)
        response.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        logger.info(f"✅ Model downloaded successfully to {MODEL_PATH}")
    except Exception as e:
        logger.exception(f"Failed to download model: {e}")
        raise

# -------------------
# Load model
# -------------------
if os.path.isfile(MODEL_PATH):
    logger.info("🔄 Loading model...")
    try:
        model = load_model(MODEL_PATH, compile=False)
        MODEL_MD5 = md5sum(MODEL_PATH)
        logger.info(f"✅ Model loaded. MD5 checksum: {MODEL_MD5}")
    except Exception as e:
        logger.exception(f"Failed to load model: {e}")
        raise
else:
    logger.error(f"Model file still missing: {MODEL_PATH}")
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

# Detect expected input size (ignore batch dimension)
if isinstance(model.input_shape, (list, tuple)) and isinstance(model.input_shape[0], (list, tuple)):
    in_h, in_w = model.input_shape[0][1:3]
else:
    in_h, in_w = model.input_shape[1:3]

logger.info(f"✅ Model input size: ({in_h}, {in_w})")

# Optimize predict call
@tf.function
def model_predict(batch):
    return model(batch, training=False)

# -------------------
# Preprocessing
# -------------------
def preprocess_image(path):
    img = image.load_img(path, target_size=(in_h, in_w))
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# -------------------
# Routes
# -------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Pneumonia Detection API is running!",
        "tf_version": tf.__version__,
        "input_shape": model.input_shape,
        "output_shape": model.output_shape,
        "model_md5": MODEL_MD5
    })

@app.route("/healthz", methods=["GET"])
def healthz():
    return jsonify({"status": "ok", "model_loaded": True})

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        file_path = tmp.name
        file.save(file_path)

    logger.info(f"Saved uploaded file to: {file_path}")

    try:
        img_array = preprocess_image(file_path)
        logger.info(f"✅ Preprocessed shape: {img_array.shape}")

        prediction = model_predict(img_array).numpy()
        logger.info(f"✅ Raw prediction: {prediction}")

        if prediction.shape[1] == 1:  # sigmoid model
            diagnosis = "Pneumonia" if prediction[0][0] > THRESHOLD else "Normal"
            response = {
                "diagnosis": diagnosis,
                "raw": prediction[0].tolist(),
                "threshold": THRESHOLD,
                "pneumonia_score": float(prediction[0][0]),
                "normal_score": float(1 - prediction[0][0]),
                "input_size_used": [in_h, in_w],
            }
        else:  # softmax model
            classes = ["Normal", "Pneumonia"]
            idx = int(np.argmax(prediction))
            diagnosis = classes[idx]
            response = {
                "diagnosis": diagnosis,
                "raw": prediction[0].tolist(),
                "confidence": float(np.max(prediction)),
                "input_size_used": [in_h, in_w],
            }

        return jsonify(response)

    except Exception as e:
        logger.exception("❌ Prediction failed")
        return jsonify({"error": str(e)}), 500

    finally:
        try:
            os.remove(file_path)
            logger.info(f"Deleted temp file: {file_path}")
        except OSError:
            logger.warning(f"Failed to delete temp file: {file_path}")

# -------------------
# Run (for local dev)
# -------------------
if __name__ == "__main__":
    PORT = int(os.getenv("PORT", 8000))
    logger.info(f"Starting app on port {PORT}...")
    app.run(host="0.0.0.0", port=PORT, debug=True)
