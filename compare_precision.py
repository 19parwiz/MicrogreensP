#!/usr/bin/env python3
"""
Compare loose vs precise bounding boxes side by side
"""

import cv2
import numpy as np
from production_detection import ProductionMicrogreensRecognition
from precise_detection import PreciseMicrogreensRecognition
from src.camera import Camera

def compare_detection_precision():
    """Compare loose vs precise detection side by side"""
    
    # Setup camera
    cam = Camera(index=0)
    if not cam.open():
        print("Error: Could not open camera")
        return
    
    # Create both recognizers
    loose_recognizer = ProductionMicrogreensRecognition(
        conf=0.25,
        subdivision_enabled=True,
        min_detection_size=0.4,
        max_detections=8
    )
    
    precise_recognizer = PreciseMicrogreensRecognition(
        conf=0.25,
        subdivision_enabled=True,
        min_detection_size=0.4,
        max_detections=8
    )
    
    print("Comparison: Loose vs Precise Detection")
    print("Left: Loose boxes | Right: Precise boxes (tight around microgreens)")
    print("Press 'q' to quit")
    
    while True:
        frame = cam.read()
        if frame is None:
            print("No frame captured")
            break
        
        # Run both detections on copies of the frame
        loose_result = loose_recognizer.predict(frame.copy())
        precise_result = precise_recognizer.predict(frame.copy())
        
        # Add labels
        cv2.putText(loose_result, "LOOSE BOXES", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(precise_result, "PRECISE BOXES", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Resize both to same height for comparison
        height = 500
        h1, w1 = loose_result.shape[:2]
        h2, w2 = precise_result.shape[:2]
        
        scale1 = height / h1
        scale2 = height / h2
        
        new_w1 = int(w1 * scale1)
        new_w2 = int(w2 * scale2)
        
        loose_resized = cv2.resize(loose_result, (new_w1, height))
        precise_resized = cv2.resize(precise_result, (new_w2, height))
        
        # Combine side by side
        combined = np.hstack([loose_resized, precise_resized])
        
        cv2.imshow("Detection Comparison: Loose vs Precise", combined)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    compare_detection_precision()
