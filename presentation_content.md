# Presentation Slides Content: AI Food Recognition

## Slide 1: Title Slide
**Title**: AI NutriScan: Food Recognition & Calorie Estimation
**Subtitle**: Automated Dietary Tracking using Deep Learning
**Presented By**: [Your Name]
**Project ID**: [Your ID]

## Slide 2: Introduction
- **The Problem**: Manual calorie counting is hard, boring, and inaccurate.
- **The Solution**: Take a photo -> Get Calories instantly.
- **Goal**: Build a fast, offline-capable AI system for food identification.

## Slide 3: Objectives
- ✅ **Identify** food items from images.
- ✅ **Estimate** calories per serving.
- ✅ **Utilize** existing Pre-trained Models (Transfer Learning).
- ✅ **Develop** a User-Friendly Web Interface.

## Slide 4: System Architecture
- **Input**: Image (User Upload).
- **Core**: MobileNetV2 (CNN) -> Feature Extraction.
- **Logic**: Flask Backend -> Matches Label to Calories.
- **Output**: Food Name + Calorie Count.

## Slide 5: Why MobileNetV2?
- **Lightweight**: Optimized for speed and mobile devices.
- **Accuracy**: Comparable to ResNet but 10x faster.
- **Efficiency**: Low parameter count (3.5M vs 25M+ for others).

## Slide 6: Methodology (Transfer Learning)
1. **Base Model**: Pre-trained on ImageNet (knows shapes/colors).
2. **Fine-Tuning**: Re-trained on **Indian Food Dataset** + Synthetic Fast Food Data.
3. **Training**: 5 Epochs with Data Augmentation (Rotation, Zoom).

## Slide 7: Implementation Details
- **Language**: Python 3.8+
- **Frameworks**: TensorFlow, Keras, Flask.
- **Dataset**: 83 Classes (Biryani, Dosa, Pizza, Burger, etc.).
- **Tools**: VS Code, Pip.

## Slide 8: Results
- **Fast**: Results in under 1 second.
- **Accurate**: Recognizes complex dishes like "Paneer Butter Masala".
- **Visuals**: Modern Glassmorphism UI.

## Slide 9: Demo
*([Insert Screenshot of App here])*
- "Scanning Image..."
- "Detected: Masala Dosa (168 kcal)"

## Slide 10: Future Scope
- Portion Size estimation (Small/Medium/Large).
- Real-time video detection.
- Mobile App Deployment (Android/iOS).

## Slide 11: Conclusion
- Successfully built an end-to-end AI application.
- Solves a real-world health problem.
- Scalable architecture for future enhancements.

## Slide 12: Q&A
**Thank You!**
Questions?
