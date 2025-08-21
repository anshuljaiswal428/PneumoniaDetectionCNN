import os, hashlib, logging, tempfile
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# -------------------
# Config
# -------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "pneumonia_detector_model_new.h5")
THRESHOLD = 0.5  # sigmoid threshold

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PneumoniaAPI")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000", "http://127.0.0.1:3000", "https://pneumoniadetectionbyanshul.vercel.app/"]}})

# -------------------
# Helpers
# -------------------
def md5sum(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

# Load model once
logger.info("🔄 Loading model...")
model = load_model(MODEL_PATH, compile=False)
MODEL_MD5 = md5sum(MODEL_PATH)

# Detect expected input size (ignore batch dimension)
if isinstance(model.input_shape, (list, tuple)) and isinstance(model.input_shape[0], (list, tuple)):
    in_h, in_w = model.input_shape[0][1:3]
else:
    in_h, in_w = model.input_shape[1:3]
logger.info(f"✅ Model loaded. Expected input size: ({in_h}, {in_w})")

# Optimize predict call
@tf.function
def model_predict(batch):
    return model(batch, training=False)

# -------------------
# Preprocessing (SAME as your script)
# -------------------
def preprocess_image(path):
    """Load image, resize with keras.load_img, convert to array, normalize"""
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

    # Use temp file to avoid storage bloat
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        file_path = tmp.name
        file.save(file_path)

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
        # Clean up temp file
        try:
            os.remove(file_path)
        except OSError:
            pass

# -------------------
# Run
# -------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
