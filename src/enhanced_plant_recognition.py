"""
Enhanced Plant Recognition Module with Multi-Microgreens Detection
This replaces the need for multiple detection files
"""

import cv2
import numpy as np
from ultralytics import YOLO

class EnhancedPlantRecognition:
    """
    Enhanced plant recognition that can detect multiple microgreens with precise bounding boxes
    Includes smart filtering to compensate for limited training data
    """
    
    def __init__(self, weights_path=None, conf=0.25, iou=0.45, device=None, 
                 detection_mode="smart", max_detections=6, min_detection_size=0.4):
        """
        detection_mode: "original", "multi", "precise", or "smart"
        - original: Single detection (like before)
        - multi: Multiple detections with subdivision
        - precise: Tight bounding boxes around microgreens
        - smart: Advanced filtering for limited training data
        """
        if weights_path is None:
            weights_path = "D:/AgroTech/runs/detect/train9/weights/best.pt"
        
        self.weights_path = str(weights_path)
        print(f"[EnhancedPlantRecognition] Loading YOLO weights: {self.weights_path}")
        self.model = YOLO(self.weights_path)
        self.conf = conf
        self.iou = iou
        self.device = device
        self.detection_mode = detection_mode
        self.max_detections = max_detections
        self.min_detection_size = min_detection_size
        
        # Initialize class names
        self.class_names = {}
        try:
            names = self.model.names
            self.class_names = self._normalize_names(names)
            print(f"[EnhancedPlantRecognition] Class names: {self.class_names}")
            print(f"[EnhancedPlantRecognition] Detection mode: {detection_mode}")
        except Exception:
            self.class_names = {0: "Arugula", 1: "Basil", 2: "Beetroot", 3: "Mangold", 4: "Tarragon"}
    
    @staticmethod
    def _normalize_names(names_obj):
        """Return a dict[int,str] from ultralytics names"""
        try:
            if isinstance(names_obj, dict):
                return {int(k): str(v) for k, v in names_obj.items()}
            if isinstance(names_obj, (list, tuple)):
                return {i: str(v) for i, v in enumerate(names_obj)}
        except Exception:
            pass
        return {0: "Arugula", 1: "Basil", 2: "Beetroot", 3: "Mangold", 4: "Tarragon"}
    
    def _is_likely_microgreen(self, roi):
        """Advanced microgreen validation with strict filtering"""
        if roi.size == 0:
            return False, 0.0
        
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Multi-range green color detection
        lower_green1 = np.array([25, 40, 40])
        upper_green1 = np.array([85, 255, 255])
        lower_green2 = np.array([35, 30, 30])
        upper_green2 = np.array([75, 255, 200])
        
        mask1 = cv2.inRange(hsv, lower_green1, upper_green1)
        mask2 = cv2.inRange(hsv, lower_green2, upper_green2)
        green_mask = cv2.bitwise_or(mask1, mask2)
        
        total_pixels = roi.shape[0] * roi.shape[1]
        green_pixels = cv2.countNonZero(green_mask)
        green_ratio = green_pixels / total_pixels
        
        # Texture analysis for organic surfaces
        texture_variance = np.var(gray)
        
        # Edge density for leaf-like structures
        edges = cv2.Canny(gray, 50, 150)
        edge_density = cv2.countNonZero(edges) / total_pixels
        
        # Color variation analysis
        color_variation = np.mean(np.std(roi, axis=(0, 1)))
        
        # Brightness and saturation checks
        brightness = np.mean(gray)
        saturation = np.mean(hsv[:, :, 1])
        
        # Strict scoring system
        score = 0.0
        
        # Primary green content requirement
        if green_ratio > 0.20:
            score += 0.5
        elif green_ratio > 0.12:
            score += 0.3
        elif green_ratio > 0.08:
            score += 0.1
        else:
            return False, 0.0  # Immediate rejection if not green enough
        
        # Organic texture requirement
        if 300 < texture_variance < 1800:
            score += 0.2
        elif texture_variance < 150:
            score -= 0.3  # Too smooth (artificial surface)
        
        # Natural edge patterns
        if 0.08 < edge_density < 0.25:
            score += 0.15
        
        # Natural color variation
        if 12 < color_variation < 45:
            score += 0.15
        elif color_variation < 8:
            score -= 0.25  # Too uniform (artificial)
        
        # Appropriate brightness
        if 40 < brightness < 180:
            score += 0.1
        elif brightness < 25 or brightness > 200:
            score -= 0.2
        
        # Sufficient saturation for living plants
        if saturation > 50:
            score += 0.1
        elif saturation < 30:
            score -= 0.15
        
        # Strong penalties for obvious false positives
        if green_ratio < 0.06:
            score -= 0.5
        if color_variation < 6:
            score -= 0.4
        if texture_variance < 100:
            score -= 0.3
        
        # Require higher threshold for acceptance
        return score > 0.65, score
    
    def _validate_frame_has_microgreens(self, frame):
        """Check if frame contains any microgreen-like content before processing"""
        h, w = frame.shape[:2]
        
        # Sample multiple regions across the frame
        sample_size = 100
        regions_to_check = [
            frame[h//4:3*h//4, w//4:3*w//4],  # Center region
            frame[0:h//2, 0:w//2],            # Top-left
            frame[0:h//2, w//2:w],            # Top-right
            frame[h//2:h, 0:w//2],            # Bottom-left
            frame[h//2:h, w//2:w]             # Bottom-right
        ]
        
        microgreen_evidence = 0
        total_regions = len(regions_to_check)
        
        for region in regions_to_check:
            if region.size == 0:
                continue
                
            # Resize region for faster processing
            if region.shape[0] > sample_size or region.shape[1] > sample_size:
                scale = sample_size / max(region.shape[0], region.shape[1])
                new_h = int(region.shape[0] * scale)
                new_w = int(region.shape[1] * scale)
                region = cv2.resize(region, (new_w, new_h))
            
            is_likely, score = self._is_likely_microgreen(region)
            if is_likely or score > 0.4:
                microgreen_evidence += 1
        
        # Require at least one region to show microgreen evidence
        return microgreen_evidence > 0
    
    def _create_precise_mask(self, roi):
        """Create precise mask for microgreen pixels"""
        if roi.size == 0:
            return None
        
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Multiple green ranges
        lower_green1 = np.array([35, 40, 40])
        upper_green1 = np.array([85, 255, 255])
        mask1 = cv2.inRange(hsv, lower_green1, upper_green1)
        
        lower_green2 = np.array([25, 30, 30])
        upper_green2 = np.array([95, 255, 200])
        mask2 = cv2.inRange(hsv, lower_green2, upper_green2)
        
        combined_mask = cv2.bitwise_or(mask1, mask2)
        
        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        
        return combined_mask
    
    def _get_microgreen_detections(self, frame, region_bbox, original_class, original_conf):
        """Get microgreen detections in a region based on detection mode"""
        x1, y1, x2, y2 = region_bbox
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return []
        
        # Smart mode filtering
        if self.detection_mode == "smart":
            is_likely, likelihood_score = self._is_likely_microgreen(roi)
            if not is_likely:
                return []
        
        # Create mask based on mode
        if self.detection_mode in ["precise", "smart"]:
            mask = self._create_precise_mask(roi)
        else:
            # Simple green mask for multi mode
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            lower_green = np.array([35, 40, 40])
            upper_green = np.array([85, 255, 255])
            mask = cv2.inRange(hsv, lower_green, upper_green)
        
        if mask is None:
            return []
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        roi_area = roi.shape[0] * roi.shape[1]
        min_area = roi_area * 0.02
        max_area = roi_area * 0.8
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue
            
            rx, ry, rw, rh = cv2.boundingRect(contour)
            
            # Filter by aspect ratio
            aspect_ratio = max(rw, rh) / max(min(rw, rh), 1)
            if aspect_ratio > 4.0:
                continue
            
            if min(rw, rh) < 10:
                continue
            
            # Add padding for better visualization
            padding = 5 if self.detection_mode == "precise" else 8
            rx = max(0, rx - padding)
            ry = max(0, ry - padding)
            rw = min(roi.shape[1] - rx, rw + 2 * padding)
            rh = min(roi.shape[0] - ry, rh + 2 * padding)
            
            # Convert to frame coordinates
            abs_x1 = max(0, min(x1 + rx, frame.shape[1] - 1))
            abs_y1 = max(0, min(y1 + ry, frame.shape[0] - 1))
            abs_x2 = max(abs_x1 + 1, min(x1 + rx + rw, frame.shape[1]))
            abs_y2 = max(abs_y1 + 1, min(y1 + ry + rh, frame.shape[0]))
            
            conf_multiplier = 0.85 if self.detection_mode == "smart" else 0.8
            final_conf = min(original_conf * conf_multiplier, 0.95)
            
            detections.append({
                "bbox": [abs_x1, abs_y1, abs_x2, abs_y2],
                "label": original_class,
                "conf": final_conf
            })
        
        # Limit detections per region
        max_per_region = 3 if self.detection_mode == "smart" else 4
        return detections[:max_per_region]
    
    def _subdivide_detection(self, frame, detection):
        """Subdivide large detection into smaller regions"""
        x1, y1, x2, y2 = detection["bbox"]
        width = x2 - x1
        height = y2 - y1
        
        frame_area = frame.shape[0] * frame.shape[1]
        detection_area = width * height
        area_ratio = detection_area / frame_area
        
        if area_ratio > 0.8:
            grid_size = 3  # 3x3 grid
        elif area_ratio > 0.6:
            grid_size = 2  # 2x2 grid
        else:
            return self._get_microgreen_detections(frame, detection["bbox"], 
                                                 detection["label"], detection["conf"])
        
        all_detections = []
        cell_width = width // grid_size
        cell_height = height // grid_size
        
        for i in range(grid_size):
            for j in range(grid_size):
                cell_x1 = x1 + j * cell_width
                cell_y1 = y1 + i * cell_height
                cell_x2 = min(x2, cell_x1 + cell_width)
                cell_y2 = min(y2, cell_y1 + cell_height)
                
                cell_detections = self._get_microgreen_detections(
                    frame, [cell_x1, cell_y1, cell_x2, cell_y2], 
                    detection["label"], detection["conf"]
                )
                all_detections.extend(cell_detections)
        
        return all_detections if all_detections else [detection]
    
    def predict(self, frame):
        """Main prediction method with different detection modes"""
        # Pre-validation: check if frame likely contains microgreens
        if self.detection_mode in ["smart", "precise"]:
            if not self._validate_frame_has_microgreens(frame):
                cv2.putText(frame, "No microgreens detected", (30, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                return frame
        
        # Run YOLO detection
        results = self.model.predict(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False
        )
        
        if not results or len(results[0].boxes) == 0:
            cv2.putText(frame, "No microgreens detected", (30, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            return frame
        
        r = results[0]
        
        # Original mode - just draw the YOLO boxes as-is
        if self.detection_mode == "original":
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = self.class_names.get(cls, f"class{cls}")
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}",
                           (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                           0.7, (0, 255, 0), 2)
            return frame
        
        # Enhanced modes - process detections
        h_frame, w_frame = frame.shape[:2]
        frame_area = h_frame * w_frame
        all_detections = []
        
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = self.class_names.get(cls, f"class{cls}")
            
            detection = {
                "bbox": [x1, y1, x2, y2],
                "label": label,
                "conf": conf
            }
            
            detection_area = (x2 - x1) * (y2 - y1)
            area_ratio = detection_area / frame_area
            
            if area_ratio > self.min_detection_size:
                enhanced_detections = self._subdivide_detection(frame, detection)
                all_detections.extend(enhanced_detections)
            else:
                enhanced_detections = self._get_microgreen_detections(
                    frame, detection["bbox"], detection["label"], detection["conf"]
                )
                all_detections.extend(enhanced_detections if enhanced_detections else [detection])
        
        # Remove overlapping detections
        all_detections = self._remove_overlaps(all_detections)
        
        # Sort by confidence and limit
        all_detections.sort(key=lambda x: x["conf"], reverse=True)
        all_detections = all_detections[:self.max_detections]
        
        # Final validation: ensure we have quality detections
        if not all_detections:
            cv2.putText(frame, "No microgreens detected", (30, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            return frame
        
        # Draw detections
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), 
                 (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0)]
        
        for i, det in enumerate(all_detections):
            x1, y1, x2, y2 = det["bbox"]
            label = det["label"]
            conf = det["conf"]
            
            color = colors[i % len(colors)]
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Label with background
            label_text = f"{label} {conf:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width + 10, y1), color, -1)
            cv2.putText(frame, label_text, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add info text
        mode_text = f"Mode: {self.detection_mode} | Detections: {len(all_detections)}"
        cv2.putText(frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def _remove_overlaps(self, detections):
        """Remove overlapping detections using NMS"""
        if len(detections) <= 1:
            return detections
        
        boxes = []
        scores = []
        
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            boxes.append([x1, y1, x2, y2])
            scores.append(det["conf"])
        
        boxes = np.array(boxes, dtype=np.float32)
        scores = np.array(scores, dtype=np.float32)
        
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 0.3, 0.4)
        
        if len(indices) > 0:
            indices = indices.flatten()
            return [detections[i] for i in indices]
        else:
            return detections
