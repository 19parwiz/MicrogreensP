#!/usr/bin/env python3
"""
Compare original vs enhanced detection side by side
"""

import cv2
import glob
import numpy as np
from src.plant_recognition_module import PlantRecognition
from quick_fix_detection import MultiMicrogreensRecognition

def compare_detections():
    """Compare original vs enhanced detection"""
    
    # Get test images
    test_images = glob.glob("test_images/*.jpg") + glob.glob("test_images/*.jpeg") + glob.glob("test_images/*.JPG")
    
    if not test_images:
        print("No test images found")
        return
    
    # Create both recognizers
    original_recognizer = PlantRecognition(conf=0.25, safe_mode=True)
    enhanced_recognizer = MultiMicrogreensRecognition(
        conf=0.20,
        subdivision_enabled=True,
        min_detection_size=0.3
    )
    
    print(f"Comparing detection methods on {len(test_images)} images")
    print("Left: Original Detection | Right: Enhanced Multi-Detection")
    print("Press any key to continue, 'q' to quit")
    
    for img_path in test_images[:10]:  # Test first 10 images
        print(f"\nTesting: {img_path}")
        
        frame = cv2.imread(img_path)
        if frame is None:
            continue
        
        # Run both detections
        original_result = original_recognizer.predict(frame.copy())
        enhanced_result = enhanced_recognizer.predict(frame.copy())
        
        # Add labels
        cv2.putText(original_result, "Original Detection", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(enhanced_result, "Enhanced Multi-Detection", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Resize both to same height for comparison
        height = 600
        h1, w1 = original_result.shape[:2]
        h2, w2 = enhanced_result.shape[:2]
        
        scale1 = height / h1
        scale2 = height / h2
        
        new_w1 = int(w1 * scale1)
        new_w2 = int(w2 * scale2)
        
        original_resized = cv2.resize(original_result, (new_w1, height))
        enhanced_resized = cv2.resize(enhanced_result, (new_w2, height))
        
        # Combine side by side
        combined = np.hstack([original_resized, enhanced_resized])
        
        cv2.imshow("Detection Comparison", combined)
        
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    compare_detections()
