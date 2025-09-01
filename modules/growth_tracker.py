import cv2
import numpy as np
import json
import os
from datetime import datetime

class GrowthTracker:
    def __init__(self, config=None):
        self.config = config or {}
        self.growth_data = {}
        self.data_file = self.config.get('data_file', 'outputs/growth_data.json')
        self.is_active = False
        self.reference_object_size = 2.5  # cm (coin diameter)
        self.pixels_per_cm = None
        self.load_data()
        
    def start(self):
        """Activate growth tracking"""
        self.is_active = True
        print("Growth tracking started - Place a 2.5cm coin for calibration")
        
    def stop(self):
        """Deactivate growth tracking"""
        self.is_active = False
        self.save_data()
        print("Growth tracking stopped")
        
    def process_frame(self, frame, plant_id="default"):
        """Main processing method - called by main application"""
        if not self.is_active:
            return frame, None
            
        try:
            # Add calibration instructions to frame
            cv2.putText(frame, "Press 'c' to calibrate with coin, 's' to measure", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            return frame, None
        except Exception as e:
            print(f"Growth tracking error: {e}")
            return frame, None



    def automatic_calibration(self, frame):
        """Auto-detect calibration object (coin) and set pixels_per_cm"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        
        # Detect circles (coins)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100,
                                 param1=100, param2=30, minRadius=20, maxRadius=100)
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                radius = circle[2]
                self.pixels_per_cm = (radius * 2) / self.reference_object_size
                
                # Draw detected circle
                cv2.circle(frame, (circle[0], circle[1]), radius, (0, 255, 255), 3)
                cv2.putText(frame, f"Calibrated: {self.pixels_per_cm:.1f} px/cm", 
                           (circle[0]-50, circle[1]-radius-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                return True
        return False



    def measure_plant_dimensions(self, frame, plant_id="plant_1"):
        """Measure plant dimensions in cm/mm and return results"""
        if self.pixels_per_cm is None:
            return frame, None, "Not calibrated"
            
        # Segment plant from background
        mask = self.segment_plant(frame)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return frame, None, "No plant detected"

        # Find largest plant contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Convert pixels to real measurements
        width_cm = w / self.pixels_per_cm
        height_cm = h / self.pixels_per_cm
        area_cm2 = cv2.contourArea(largest_contour) / (self.pixels_per_cm ** 2)
        
        # Store measurement
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
        measurement = {
            'height_cm': round(height_cm, 2),
            'width_cm': round(width_cm, 2),
            'area_cm2': round(area_cm2, 2),
            'pixels': int(cv2.contourArea(largest_contour))
        }
        
        if plant_id not in self.growth_data:
            self.growth_data[plant_id] = {}
        self.growth_data[plant_id][current_time] = measurement
        self.save_data()
        

        # Create visualization
        result_frame = frame.copy()
        cv2.drawContours(result_frame, [largest_contour], -1, (0, 255, 0), 3)
        cv2.rectangle(result_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Display measurements
        info_text = [
            f"H: {height_cm:.1f}cm, W: {width_cm:.1f}cm",
            f"Area: {area_cm2:.1f}cm²",
            f"Plant: {plant_id}"
        ]
        
        for i, text in enumerate(info_text):
            cv2.putText(result_frame, text, (20, 40 + i * 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        
        return result_frame, measurement, current_time
    


    def segment_plant(self, frame):
        """Advanced plant segmentation with multiple color ranges"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Multiple color ranges for different plants
        lower_green1 = np.array([25, 40, 40])
        upper_green1 = np.array([100, 255, 255])
        
        lower_green2 = np.array([15, 30, 30])
        upper_green2 = np.array([80, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_green1, upper_green1)
        mask2 = cv2.inRange(hsv, lower_green2, upper_green2)
        mask = cv2.bitwise_or(mask1, mask2)
        
             # Clean up the mask here
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    


    def save_data(self):
        """Save growth data to JSON file"""
        try:
            os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
            with open(self.data_file, 'w') as f:
                json.dump(self.growth_data, f, indent=4)
        except Exception as e:
            print(f"Error saving data: {e}")

    def load_data(self):
        """Load growth data from JSON file"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    self.growth_data = json.load(f)
        except Exception as e:
            print(f"Error loading data: {e}")

    def get_growth_report(self, plant_id="plant_1"):
        """Generate growth analysis report"""
        if plant_id not in self.growth_data or not self.growth_data[plant_id]:
            return "No data available"
        
        data = self.growth_data[plant_id]
        dates = sorted(data.keys())
        
        if len(dates) < 2:
            return "Not enough data for analysis"
        
        
        
# Calculate growth metrics
        first = data[dates[0]]
        last = data[dates[-1]]
        
        height_growth = last['height_cm'] - first['height_cm']
        time_diff = (datetime.strptime(dates[-1], "%Y-%m-%d %H:%M") - 
                    datetime.strptime(dates[0], "%Y-%m-%d %H:%M"))
        growth_days = time_diff.days + time_diff.seconds / 86400
        
        daily_growth = height_growth / growth_days if growth_days > 0 else 0
        
        return {
            'plant_id': plant_id,
            'period': f"{dates[0]} to {dates[-1]}",
            'initial_height': first['height_cm'],
            'current_height': last['height_cm'],
            'total_growth_cm': round(height_growth, 2),
            'daily_growth_cm': round(daily_growth, 2),
            'growth_days': round(growth_days, 1),
            'measurements_count': len(data)
        }