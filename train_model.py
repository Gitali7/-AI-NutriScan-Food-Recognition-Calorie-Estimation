import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import json
import os

# Configuration
DATA_DIR = 'data/Indian Food Images/Indian Food Images'
MODEL_DIR = 'model'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5

def train():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    print(f"Loading data from {DATA_DIR}...")
    
    # Data Generator (with validation split)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    print(f"Classes found: {list(train_generator.class_indices.keys())}")
    
    # Save Class Indices
    class_indices = train_generator.class_indices
    # Invert to {0: "burger", ...} for lookup
    idx_to_class = {v: k for k, v in class_indices.items()}
    with open(os.path.join(MODEL_DIR, 'classes.json'), 'w') as f:
        json.dump(idx_to_class, f)
    print("Class mappings saved.")

    # Base Model
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Freeze base model
    base_model.trainable = False

    # Add Custom Head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(len(class_indices), activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print("Starting training...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator
    )

    print("Training complete. Saving model...")
    model.save(os.path.join(MODEL_DIR, 'food_model.h5'))
    print("Model saved successfully.")

if __name__ == "__main__":
    train()
