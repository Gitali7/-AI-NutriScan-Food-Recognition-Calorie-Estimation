import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image
import io
import os
import json

class FoodModel:
    def __init__(self):
        self.custom_model_path = 'model/food_model.h5'
        self.classes_path = 'model/classes.json'
        
        # Always load the base ImageNet model for fallback/general foods
        print("Loading Base MobileNetV2 (ImageNet)...")
        self.base_model = MobileNetV2(weights='imagenet', include_top=True)
        
        # Try to load Custom Model
        self.custom_model = None
        self.classes = {}
        
        if os.path.exists(self.custom_model_path) and os.path.exists(self.classes_path):
            print(f"Loading Custom Fine-Tuned Model from {self.custom_model_path}...")
            try:
                self.custom_model = load_model(self.custom_model_path)
                with open(self.classes_path, 'r') as f:
                    self.classes = json.load(f)
                # Convert keys to int just in case JSON loaded them as strings
                self.classes = {int(k): v for k, v in self.classes.items()}
                print("Custom Model Loaded Successfully.")
            except Exception as e:
                print(f"Failed to load custom model: {e}.")
        else:
            print("Custom model not found. System will rely on ImageNet only.")

    def preprocess_image(self, img_bytes):
        img = Image.open(io.BytesIO(img_bytes))
        img = img.resize((224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        
        # MobileNetV2 preprocess is standard (scale to -1..1)
        x = preprocess_input(x)
        return x

    def predict(self, img_bytes):
        try:
            processed_image = self.preprocess_image(img_bytes)
            
            # 1. Try Custom Model First (if available)
            custom_result = None
            if self.custom_model:
                preds = self.custom_model.predict(processed_image)
                top_idx = np.argmax(preds[0])
                confidence = float(preds[0][top_idx])
                class_name = self.classes.get(top_idx, "Unknown")
                
                custom_result = {
                    "label": class_name.replace("_", " ").title(),
                    "confidence": confidence,
                    "raw_label": class_name
                }
            
            # 2. Decision Logic: Fallback if low confidence or no custom model
            # Threshold for custom model (e.g., 60%)
            CONFIDENCE_THRESHOLD = 0.6
            
            if custom_result and custom_result['confidence'] >= CONFIDENCE_THRESHOLD:
                return {
                    "label": custom_result['label'],
                    "confidence": custom_result['confidence'],
                    "model_type": "Custom Fine-Tuned"
                }
            
            # 3. Fallback to ImageNet
            print("Low confidence on custom model (or no model). Falling back to ImageNet...")
            preds_imagenet = self.base_model.predict(processed_image)
            decoded_preds = decode_predictions(preds_imagenet, top=3)[0]
            top_pred = decoded_preds[0]
            
            return {
                "label": top_pred[1].replace("_", " ").title(),
                "confidence": float(top_pred[2]),
                "model_type": "ImageNet Base (Fallback)"
            }

        except Exception as e:
            print(f"Error during prediction: {e}")
            return None
