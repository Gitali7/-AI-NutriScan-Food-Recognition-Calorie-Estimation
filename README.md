# 🍽️ AI NutriScan: Food Recognition & Calorie Estimation

A fast, lightweight AI system that identifies food items from images and estimates their calorie content using Transfer Learning with MobileNetV2.

## 🚀 Key Features
- **Instant Recognition**: Identifies 80+ Indian dishes + fast foods (Pizza, Burger, Ice Cream).
- **Calorie Estimation**: Display calories per serving from a verified database.
- **Privacy First**: No external APIs. Everything runs locally on your machine.
- **Premium UI**: Modern, responsive, and dark-themed interface.
- **Custom AI**: Fine-tuned on a custom dataset for high accuracy on regional foods.

## 🛠️ Tech Stack
- **Frontend**: HTML5, CSS3 (Modern Glassmorphism)
- **Backend**: Flask (Python)
- **AI Model**: MobileNetV2 (TensorFlow/Keras) - Transfer Learning
- **Dataset**: Indian Food 101 + Synthetic Data (Pizza/Burger/Ice Cream)

## 📂 Project Structure
```
food_calorie_app/
├── app.py                 # Main Flask Application
├── train_model.py         # AI Training Script (Transfer Learning)
├── requirements.txt       # Dependencies
├── static/
│   └── style.css          # Premium Styling
├── templates/
│   └── index.html         # Frontend Interface
├── utils/
│   ├── calorie_data.py    # Calorie Database
│   └── model_handler.py   # AI Inference Logic
├── model/
│   ├── food_model.h5      # Fine-Tuned Model Weights
│   └── classes.json       # Class Labels
└── data/                  # Training Dataset
```

## ⚙️ How to Run

### 1. Setup Environment
Open your terminal/command prompt in the project folder:
```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install Dependencies
pip install -r requirements.txt
```

### 2. (Optional) Retrain Model
If you want to train the model yourself (already trained currently):
```bash
python train_model.py
```

### 3. Run the App
```bash
python app.py
```
Go to your browser and open: **http://127.0.0.1:5000**

## 📸 Usage
1. Click **"Choose an Image"** or Drag & Drop.
2. Wait a second for the AI to analyze.
3. View the **detected food name**, **calories**, and **confidence score**.

## 📊 Performance
- **Model**: MobileNetV2 (Alpha 1.0)
- **Inference Time**: < 200ms
- **Accuracy**: ~85% on verified test set (5 Epochs)

---
*Created for Final Year CSV Project.*
