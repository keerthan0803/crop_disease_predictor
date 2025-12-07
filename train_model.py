import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
import pickle
import json
from pathlib import Path

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration
DATASET_PATH = r"plantvillage dataset\color"
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
VALIDATION_SPLIT = 0.2

def get_disease_info(class_name):
    """Extract plant name and disease status from class name"""
    parts = class_name.split('___')
    plant = parts[0].replace('_', ' ').replace(',', '')
    
    if len(parts) > 1:
        disease = parts[1].replace('_', ' ')
        if 'healthy' in disease.lower():
            disease = 'Healthy'
    else:
        disease = 'Unknown'
    
    return plant, disease

def prepare_data():
    """Prepare and load dataset"""
    print("Loading dataset...")
    
    images = []
    labels = []
    class_names = []
    
    dataset_dir = Path(DATASET_PATH)
    
    for idx, class_dir in enumerate(sorted(dataset_dir.iterdir())):
        if class_dir.is_dir():
            class_names.append(class_dir.name)
            class_path = str(class_dir)
            
            # Load images from this class
            img_count = 0
            for img_file in class_dir.iterdir():
                if img_file.is_file() and img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    try:
                        img = keras.preprocessing.image.load_img(
                            img_file, 
                            target_size=(IMAGE_SIZE, IMAGE_SIZE)
                        )
                        img_array = keras.preprocessing.image.img_to_array(img) / 255.0
                        images.append(img_array)
                        labels.append(idx)
                        img_count += 1
                    except Exception as e:
                        print(f"Error loading {img_file}: {e}")
            
            print(f"Loaded {img_count} images from {class_dir.name}")
    
    # Convert to numpy arrays
    X = np.array(images)
    y = np.array(labels)
    
    print(f"\nTotal images loaded: {len(X)}")
    print(f"Total classes: {len(class_names)}")
    
    return X, y, class_names

def build_model(num_classes):
    """Build the CNN model"""
    print("\nBuilding model...")
    
    # Use EfficientNetB0 as base model for better performance
    base_model = keras.applications.EfficientNetB0(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model, base_model

def train_model():
    """Main training function"""
    # Prepare data
    X, y, class_names = prepare_data()
    
    # Split data
    split_idx = int(len(X) * (1 - VALIDATION_SPLIT))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Build model
    model, base_model = build_model(len(class_names))
    print(model.summary())
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.2
    )
    
    # Train phase 1: frozen base model
    print("\nPhase 1: Training with frozen base model...")
    history1 = model.fit(
        train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        validation_data=(X_val, y_val),
        epochs=20,
        verbose=1
    )
    
    # Unfreeze last layers of base model for fine-tuning
    print("\nPhase 2: Fine-tuning with unfrozen base model...")
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history2 = model.fit(
        train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        validation_data=(X_val, y_val),
        epochs=30,
        verbose=1
    )
    
    # Save model
    print("\nSaving model...")
    model.save('crop_disease_model.h5')
    
    # Save class names
    with open('class_names.pkl', 'wb') as f:
        pickle.dump(class_names, f)
    
    # Save disease info
    disease_info = {}
    for class_name in class_names:
        plant, disease = get_disease_info(class_name)
        disease_info[class_name] = {'plant': plant, 'disease': disease}
    
    with open('disease_info.json', 'w') as f:
        json.dump(disease_info, f, indent=2)
    
    print("\nModel training completed!")
    print(f"Model saved as 'crop_disease_model.h5'")
    print(f"Class names saved as 'class_names.pkl'")
    print(f"Disease info saved as 'disease_info.json'")
    
    # Print final accuracy
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nFinal Validation Accuracy: {val_acc*100:.2f}%")

if __name__ == "__main__":
    train_model()
