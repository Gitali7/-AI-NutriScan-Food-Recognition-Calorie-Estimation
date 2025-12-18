# PROJECT REPORT: AI FOOD RECOGNITION & CALORIE ESTIMATION

## 1. Abstract
In the modern era of health consciousness, tracking nutritional intake is crucial. Manual calorie counting is tedious and error-prone. This project presents an AI-powered system that automates this process. Using Deep Learning (CNNs), the system identifies food items from images and retrieves nutritional values from a database. The solution utilizes **MobileNetV2** with Transfer Learning to ensure low latency and high efficiency, making it suitable for deployment on standard hardware without GPU requirements.

## 2. Problem Statement
*   **Challenge**: People struggle to identify or track calories in their daily meals efficiently.
*   **Current Solutions**: Require manual text entry or expensive hardware.
*   **Our Solution**: A purely visual "Snap & Scan" approach using Computer Vision.

## 3. System Architecture
The system follows a Client-Server architecture:
1.  **User Layer (Frontend)**: Web interface for image upload and result display.
2.  **Application Layer (Flask Backend)**: Handles HTTP requests, image preprocessing, and orchestrates the AI model.
3.  **Intelligence Layer (AI Model)**:
    *   **Base**: MobileNetV2 (Pre-trained on ImageNet).
    *   **Head**: Custom Dense Layers for classification of 83+ classes (80 Indian + Pizza/Burger/Ice Cream).
4.  **Data Layer**: Static JSON-based lookup for calorie values.

### Flow Diagram
`Input Image` -> `Preprocessing (Resize/Normalize)` -> `MobileNetV2 Extraction` -> `Softmax Classification` -> `Label Output` -> `Calorie Lookup` -> `Display`.

## 4. Methodology
### 4.1 Data Collection & Augmentation
*   **Dataset**: Indian Food 101 Dataset (~4000 images).
*   **Synthetic Data**: Generated images for Western fast foods using Generative AI to balance the classes.
*   **Augmentation**: Rotation, Width/Height Shift, Horizontal Flip used during training to prevent overfitting.

### 4.2 Algorithm: Transfer Learning
Instead of training from scratch, we used **Transfer Learning**:
1.  **Frozen Base**: MobileNetV2 layers are frozen to retain low-level feature extraction capabilities (edges, textures).
2.  **Trainable Head**: Added Global Average Pooling and Dense layers.
3.  **Optimization**: Adam Optimizer with Categorical Crossentropy loss.

## 5. Results & Evaluation
*   **Training Accuracy**: ~65-70% (within 5 Epochs).
*   **Validation Accuracy**: Consistent with training, indicating good generalization.
*   **Latency**: Inference takes approximately 0.15 seconds on a standard CPU.

## 6. Future Scope
*   **Volume Estimation**: Using depth perception to estimate portion size (grams).
*   **Nutrient Breakdown**: Showing Protein, Carbs, and Fat breakdown.
*   **Mobile App**: Porting the TFLite model to Android/iOS.

## 7. Conclusion
The "AI NutriScan" system successfully demonstrates the application of efficient Deep Learning in HealthTech. It provides a seamless, user-friendly experience for dietary tracking, satisfying the core objectives of accuracy, speed, and simplicity.
