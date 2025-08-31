import cv2
import numpy as np
from tensorflow.keras.models import load_model
import json
import os

class PlantRecognition:
    def __init__(self):
        # Load the trained plant classification model
        self.model = load_model(os.path.join("models", "plant_model.h5"))
        
        # Load the mapping between class numbers and plant names
        with open(os.path.join("models", "plant_labels.json"), "r") as f:
            class_dict = json.load(f)
        
        # Convert {0: 'Cucumber', 1: 'Potato', 2: 'Tomato'} to names
        self.class_names = {v: k for k, v in class_dict.items()}

        # Minimum confidence required to show a detection (65%)
        self.threshold = 65

    def predict(self, frame):
        # This will track if we found any plants
        plants_found = []
        
        # Convert to HSV color space for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define color range for plant-like colors (green, yellow, red tones)
        lower_color = np.array([25, 40, 40])
        upper_color = np.array([100, 255, 255])
        
        # Create a mask where plant-colored pixels are white
        mask = cv2.inRange(hsv, lower_color, upper_color)
        
        # Clean up the mask to remove noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Fill small holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # Remove small dots
        
        # Find continuous regions in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process each detected region
        for contour in contours:
            # Skip very small areas (noise)
            if cv2.contourArea(contour) > 1000:
                # Get the bounding box coordinates
                x, y, w, h = cv2.boundingRect(contour)
                
                # Extract this region from the original frame
                plant_roi = frame[y:y+h, x:x+w]
                
                if plant_roi.size > 0:
                    # Prepare image for the model
                    img = cv2.resize(plant_roi, (128, 128))
                    img_array = np.expand_dims(img, axis=0) / 255.0

                    # Ask the model what plant this is
                    prediction = self.model.predict(img_array, verbose=0)
                    predicted_class = int(np.argmax(prediction, axis=1)[0])
                    confidence = float(np.max(prediction) * 100)

                    # Only keep confident detections
                    if confidence >= self.threshold:
                        plants_found.append({
                            'box': (x, y, w, h),
                            'class': predicted_class,
                            'confidence': confidence
                        })
        
        # Draw results on the frame
        for plant in plants_found:
            x, y, w, h = plant['box']
            label = f"{self.class_names[plant['class']]} ({plant['confidence']:.1f}%)"
            
            # Draw green box around the plant
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            # Put label above the box
            cv2.putText(frame, label, (x, y - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Show message if no plants found
        if not plants_found:
            cv2.putText(frame, "No plants detected", (50, 50),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        return frame