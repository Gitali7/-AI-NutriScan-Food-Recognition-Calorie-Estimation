from flask import Flask, render_template, request, jsonify
from utils.model_handler import FoodModel
from utils.calorie_data import get_calorie_info
import os

app = Flask(__name__)

# Initialize model (Load once on startup)
print("Initializing Food Model...")
food_model = FoodModel()
print("Model Initialized.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        # Read file bytes
        img_bytes = file.read()
        
        # Get prediction
        result = food_model.predict(img_bytes)
        
        if result:
            label = result['label']
            confidence = result['confidence']
            
            # Get calories
            calories = get_calorie_info(label)
            
            response = {
                'label': label,
                'confidence': f"{confidence*100:.1f}%",
                'calories': calories if calories else "Not in database",
                'message': "Success"
            }
            return jsonify(response)
        else:
            return jsonify({'error': 'Prediction failed'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
