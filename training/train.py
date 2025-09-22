# train.py

import os
import cv2
import numpy as np
import pickle
#    I just added these   : import padndas as pd 
#from sklearn.tree import DecisionTreeClassifier 


# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
faces_dir = os.path.join(BASE_DIR, "modules", "datasets", "faces")
models_dir = os.path.join(BASE_DIR, "models")

# Ensure models directory exists
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

#   Haar cascade for face detection
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

# Recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Variables
current_id = 0
label_ids = {}
x_train = []
y_labels = []

# Loop through faces dataset
for root, dirs, files in os.walk(faces_dir):
    for file in files:
        if file.lower().endswith(("png", "jpg", "jpeg")):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "_").lower()

            if label not in label_ids:
                label_ids[label] = current_id
                current_id += 1

            id_ = label_ids[label]

            # Read image
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f" Could not read image: {path}")
                continue

            # Detect face(s)
            faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
            for (x, y, w, h) in faces:
                roi = img[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)
                print(f"[INFO] Processed {path}")

# Save labels
with open(os.path.join(models_dir, "labels.pkl"), "wb") as f:
    pickle.dump(label_ids, f)

# Train model
if len(x_train) > 0 and len(y_labels) > 0:
    recognizer.train(x_train, np.array(y_labels))
    recognizer.save(os.path.join(models_dir, "face_trainer.yml"))
    print("\n Training complete! Model saved to 'models/face_trainer.yml'")
    print(" Labels saved to 'models/labels.pkl'")
else:
    print("\n No training data found. Please add face images to 'modules/datasets/faces/'")
