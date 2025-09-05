
import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import json

# Use ABSOLUTE paths to be sure
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
train_data_dir = os.path.join(BASE_DIR, "data", "plants", "train")
validation_data_dir = os.path.join(BASE_DIR, "data", "plants", "validation")
model_path = os.path.join(BASE_DIR, "models", "plant_model.h5")

print(f"Training data directory: {train_data_dir}")
print(f"Validation data directory: {validation_data_dir}")

# Parameters
img_size = (128, 128)
batch_size = 16  
epochs = 25  # More epochs for better learning

#  DATA AUGMENTATION 
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,       # Rotate images ±30 degrees
    width_shift_range=0.2,   # Shift width ±20%
    height_shift_range=0.2,  # Shift height ±20%
    shear_range=0.2,         # Shear transformations
    zoom_range=0.2,          # Random zoom ±20%
    horizontal_flip=True,    # Random horizontal flipping
    fill_mode='nearest'      # Fill in missing pixels
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Flow from the pre-split directories
print("Loading training data...")
train_gen = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical"
)

print("Loading validation data...")
val_gen = val_datagen.flow_from_directory(
    validation_data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical"
)

# Check what classes were found
print(f"Found classes: {train_gen.class_indices}")

#  MODEL ARCHITECTURE 
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(128, (3,3), activation='relu'),  # Added more layers
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),  
    Dense(128, activation='relu'),
    Dense(len(train_gen.class_indices), activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.0001),  # Lower learning rate
              loss="categorical_crossentropy", 
              metrics=["accuracy"])

# Train
print("Training model...")
history = model.fit(train_gen, 
                   validation_data=val_gen, 
                   epochs=epochs,
                   verbose=1)

#    Save model
model.save(model_path)
print(f"Plant model saved to {model_path}")

# Save class label mapping
labels_path = os.path.join(BASE_DIR, "models", "plant_labels.json")
with open(labels_path, "w") as f:
    json.dump(train_gen.class_indices, f)
print(f"Class indices saved to {labels_path}")
print(f"Class mapping: {train_gen.class_indices}")