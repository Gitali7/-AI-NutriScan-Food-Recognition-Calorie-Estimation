import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
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
        self.use_custom = False
        
        if os.path.exists(self.custom_model_path) and os.path.exists(self.classes_path):
            print(f"Loading Custom Fine-Tuned Model from {self.custom_model_path}...")
            try:
                self.model = load_model(self.custom_model_path)
                with open(self.classes_path, 'r') as f:
                    self.classes = json.load(f)
                # Convert keys to int just in case JSON loaded them as strings
                self.classes = {int(k): v for k, v in self.classes.items()}
                self.use_custom = True
                print("Custom Model Loaded Successfully.")
            except Exception as e:
                print(f"Failed to load custom model: {e}. Falling back to ImageNet.")
                self._load_imagenet()
        else:
            print("Custom model not found. Loading Base MobileNetV2 (ImageNet)...")
            self._load_imagenet()

    def _load_imagenet(self):
        from tensorflow.keras.applications import MobileNetV2
        self.model = MobileNetV2(weights='imagenet', include_top=True)
        self.use_custom = False

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
            preds = self.model.predict(processed_image)
            
            if self.use_custom:
                # Custom Model Logic
                # preds is [ [prob_0, prob_1, ...] ]
                top_idx = np.argmax(preds[0])
                confidence = float(preds[0][top_idx])
                class_name = self.classes.get(top_idx, "Unknown")
                
                # Format name
                formatted_name = class_name.replace("_", " ").title()
                
                return {
                    "label": formatted_name,
                    "confidence": confidence,
                    "model_type": "Custom Fine-Tuned"
                }
            else:
                # ImageNet Logic
                decoded_preds = decode_predictions(preds, top=3)[0]
                top_pred = decoded_preds[0]
                class_name = top_pred[1]
                confidence = float(top_pred[2])
                formatted_name = class_name.replace("_", " ").title()
                
                return {
                    "label": formatted_name,
                    "confidence": confidence,
                    "model_type": "ImageNet Base"
                }

        except Exception as e:
            print(f"Error during prediction: {e}")
            return None
