import os
import logging
import tempfile
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from dotenv import load_dotenv
from PIL import Image  # using PIL instead of keras.preprocessing.image

load_dotenv()

# -------------------
# Config
# -------------------
MODEL_URL = os.getenv("MODEL_URL")  # must be set
LOCAL_MODEL_PATH = "newpneumonia.keras"
THRESHOLD = 0.5  # sigmoid threshold

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PneumoniaAPI")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# -------------------
# Load Model (local first, fallback to download)
# -------------------
model = None
if os.path.exists(LOCAL_MODEL_PATH):
    logger.info(f"📂 Found local model: {LOCAL_MODEL_PATH}, loading...")
    model = load_model(LOCAL_MODEL_PATH, compile=False)
    logger.info("✅ Local model loaded successfully.")
else:
    if not MODEL_URL:
        logger.error("MODEL_URL not set and no local model found!")
        raise ValueError("MODEL_URL environment variable is required if no local model is present.")
    
    logger.info(f"🌐 Local model not found, downloading from {MODEL_URL} ...")
    try:
        response = requests.get(MODEL_URL, stream=True, timeout=60)
        response.raise_for_status()

        # Save to local file for reuse
        with open(LOCAL_MODEL_PATH, "wb") as f:
            f.write(response.content)

        model = load_model(LOCAL_MODEL_PATH, compile=False)
        logger.info(f"✅ Model downloaded and saved as {LOCAL_MODEL_PATH}")
    except Exception as e:
        logger.exception(f"❌ Failed to download/load model: {e}")
        raise

# Detect expected input size
if isinstance(model.input_shape, (list, tuple)) and isinstance(model.input_shape[0], (list, tuple)):
    in_h, in_w = model.input_shape[0][1:3]
else:
    in_h, in_w = model.input_shape[1:3]

logger.info(f"✅ Model input size: ({in_h}, {in_w})")

# -------------------
# Predict function (no tf.function → eager mode)
# -------------------
def model_predict(batch):
    return model(batch, training=False)

# -------------------
# Warmup the model once (so first real request is fast)
# -------------------
dummy = np.zeros((1, in_h, in_w, 3), dtype=np.float32)
_ = model_predict(dummy)
logger.info("🔥 Model warmed up and ready")

# -------------------
# Preprocessing (using Pillow directly)
# -------------------
def preprocess_image(path):
    img = Image.open(path).convert("RGB")
    img = img.resize((in_h, in_w))
    img_array = np.array(img) / 255.0
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
        "model_in_use": LOCAL_MODEL_PATH if os.path.exists(LOCAL_MODEL_PATH) else "downloaded"
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
