import os
import io
import cv2
import time
import uuid
import base64
import numpy as np
import tensorflow as tf
import keras
from keras import layers, ops
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from supabase import create_client, Client
from PIL import Image
from werkzeug.utils import secure_filename

# ==========================================
# 0. CONFIGURATION & SETUP
# ==========================================
load_dotenv()

app = Flask(__name__)
CORS(app)

# Supabase Setup (Optional - Server will run even if keys are missing)
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key) if url and key else None

# Paths to Models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAMMO_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'vit_mammogram_model.keras')
ULTRA_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'ultrasound_unet_model.h5')

# Global Model Variables
mammogram_model = None
ultrasound_model = None

# ==========================================
# 1. ViT MODEL ARCHITECTURE (For Mammograms)
# ==========================================
# These classes are required to load the SavedModel correctly.

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=keras.activations.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        input_shape = ops.shape(images)
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]
        channels = input_shape[3]
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        patches = keras.ops.image.extract_patches(images, size=self.patch_size)
        patches = ops.reshape(
            patches,
            (batch_size, num_patches_h * num_patches_w, self.patch_size * self.patch_size * channels),
        )
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)

    def call(self, patch):
        positions = ops.expand_dims(ops.arange(start=0, stop=self.num_patches, step=1), axis=0)
        projected_patches = self.projection(patch)
        return projected_patches + self.position_embedding(positions)

    def get_config(self):
        config = super().get_config()
        config.update({"num_patches": self.num_patches})
        return config

def create_vit_classifier():
    inputs = keras.Input(shape=(224, 224, 1))
    patches = Patches(patch_size=16)(inputs)
    num_patches = (224 // 16) ** 2
    encoded_patches = PatchEncoder(num_patches, projection_dim=64)(patches)
    for _ in range(8):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(num_heads=4, key_dim=64, dropout=0.1)(x1, x1)
        x2 = layers.Add()([attention_output, encoded_patches])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, hidden_units=[128, 64], dropout_rate=0.1)
        encoded_patches = layers.Add()([x3, x2])
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    features = mlp(representation, hidden_units=[2048, 1024], dropout_rate=0.5)
    logits = layers.Dense(2)(features)
    model = keras.Model(inputs=inputs, outputs=logits)
    return model

# ==========================================
# 2. MODEL LOADING LOGIC
# ==========================================

def load_models():
    global mammogram_model, ultrasound_model
    
    # --- Load Mammogram Model ---
    # We use a try-except block so the server starts even if this model is missing
    try:
        if os.path.exists(MAMMO_MODEL_PATH):
            print(f"🔹 Loading Mammogram ViT from {MAMMO_MODEL_PATH}...")
            mammogram_model = create_vit_classifier()
            mammogram_model.load_weights(MAMMO_MODEL_PATH)
            print("✅ Mammogram Model loaded.")
        else:
            print(f"⚠️ Warning: Mammogram model NOT found at {MAMMO_MODEL_PATH}. (Skipping)")
    except Exception as e:
        print(f"❌ Error loading Mammogram model: {e}")

    # --- Load Ultrasound Model ---
    try:
        if os.path.exists(ULTRA_MODEL_PATH):
            print(f"🔹 Loading Ultrasound U-Net from {ULTRA_MODEL_PATH}...")
            # compile=False is critical for avoiding custom loss function errors during inference
            ultrasound_model = tf.keras.models.load_model(ULTRA_MODEL_PATH, compile=False)
            print("✅ Ultrasound Model loaded.")
        else:
            print(f"⚠️ Warning: Ultrasound model NOT found at {ULTRA_MODEL_PATH}. (Skipping)")
    except Exception as e:
        print(f"❌ Error loading Ultrasound model: {e}")

# Trigger loading on startup
load_models()

# ==========================================
# 3. HELPER FUNCTIONS (Preprocessing)
# ==========================================

def preprocess_mammogram(image_bytes):
    """
    Converts raw image bytes -> Grayscale -> Resized (224x224) -> Normalized (-1 to 1)
    """
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert('L') # Grayscale
    img = img.resize((224, 224))
    
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=-1) # Add channel dim
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dim
    
    img_array = img_array.astype("float32")
    img_array = (img_array - 0.5) / 0.5 # Normalize to [-1, 1]
    
    return img_array

def preprocess_ultrasound(image_bytes):
    """
    Converts raw bytes -> RGB -> Resized (128x128) -> Normalized (0 to 1)
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # Keep RGB
    
    if img is None:
        raise ValueError("Could not decode image")
        
    img_resized = cv2.resize(img, (128, 128))
    img_norm = img_resized / 255.0 # Normalize to [0, 1]
    img_input = np.expand_dims(img_norm, axis=0) # Add batch dim
    
    return img_input, img_resized

# ==========================================
# 4. API ENDPOINTS
# ==========================================

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "Breast Cancer AI Backend is Running",
        "models_status": {
            "mammogram": "Active" if mammogram_model else "Inactive",
            "ultrasound": "Active" if ultrasound_model else "Inactive"
        }
    })

# --- MAMMOGRAM PREDICTION ---
@app.route('/predict', methods=['POST'])
def predict_mammogram():
    if not mammogram_model:
        return jsonify({"error": "Mammogram model is not active on the server."}), 503

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    try:
        file_bytes = file.read()
        
        # 1. Preprocess
        processed_img = preprocess_mammogram(file_bytes)
        
        # 2. Predict
        logits = mammogram_model.predict(processed_img)
        probabilities = keras.ops.softmax(logits).numpy()[0]
        
        prob_benign = float(probabilities[0])
        prob_malignant = float(probabilities[1])
        label = "Malignant" if prob_malignant > prob_benign else "Benign"
        confidence = prob_malignant if label == "Malignant" else prob_benign
        
        # 3. Save to Supabase (If configured)
        image_url = ""
        if supabase:
            # Add your Supabase saving logic here if needed
            pass 

        return jsonify({
            "prediction": label,
            "confidence": confidence,
            "raw_output": probabilities.tolist()
        })

    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({"error": str(e)}), 500

# --- ULTRASOUND PREDICTION ---
@app.route('/ultrasound', methods=['POST'])
def predict_ultrasound():
    if not ultrasound_model:
        return jsonify({"error": "Ultrasound model is not active on the server."}), 503

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    try:
        file_bytes = file.read()
        
        # 1. Preprocess
        input_tensor, original_img = preprocess_ultrasound(file_bytes)
        
        # 2. Predict (Segmentation Map)
        pred_mask = ultrasound_model.predict(input_tensor)
        
        # 3. Post-Process Mask
        # Threshold at 0.5 (Pixels > 0.5 are tumor)
        mask = (pred_mask > 0.5).astype(np.uint8) * 255
        mask_2d = mask[0, :, :, 0] # Remove extra dims to get 128x128 image
        
        # 4. Check Diagnosis
        has_tumor = np.sum(mask_2d) > 0 # If any white pixels exist, tumor is found
        confidence = float(np.max(pred_mask)) # Max probability in the map
        
        # 5. Convert Mask to Base64 (For frontend display)
        _, buffer = cv2.imencode('.png', mask_2d)
        mask_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            "type": "ultrasound",
            "diagnosis": "Potential Abnormality Detected" if has_tumor else "No Abnormality Detected",
            "tumor_detected": bool(has_tumor),
            "confidence": confidence,
            "mask_image": f"data:image/png;base64,{mask_base64}"
        })

    except Exception as e:
        print(f"Ultrasound Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # Threaded=False can help with some TensorFlow memory issues on Windows
    app.run(host="0.0.0.0", port=port, debug=True, threaded=False)