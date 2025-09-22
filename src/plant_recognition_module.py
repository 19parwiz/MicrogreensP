# AgroTech/modules/plant_recognition_module.py
"""
YOLO11-based PlantRecognition module.
Keeps same API: class PlantRecognition with predict(frame) -> frame (BGR numpy)
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # AgroTech/modules -> parent = AgroTech
MODELS_DIR = ROOT / "models"                # relative: AgroTech/models

class PlantRecognition:
    def __init__(self, weights_path=None, conf=0.5, iou=0.45, device=None):
        """
        weights_path: path to .pt file (default: AgroTech/models/plant_yolo.pt)
        conf: confidence threshold (0..1)
        iou: nms IoU threshold
        device: 'cuda' or 'cpu' or None (auto)
        """
        from pathlib import Path
       # point to your trained weights
        weights_path = Path("D:/AgroTech/models/yolo11_microgreens/weights/best.pt")



        self.weights_path = str(weights_path)
        print("[PlantRecognition] Loading YOLO weights:", self.weights_path)
        self.model = YOLO(self.weights_path)
        self.conf = conf
        self.iou = iou
        self.device = device  # used by model.predict(... device=...)

        try:
            # model.names is mapping index->name
            self.class_names = self.model.names
        except Exception:
            self.class_names = {0: "class0"}

    def predict(self, frame):
        """
        frame: OpenCV BGR image (numpy)
        returns: frame with bounding boxes and labels drawn
        """
        # run prediction (ultralytics accepts numpy BGR)
        results = self.model.predict(source=frame, conf=self.conf, iou=self.iou, device=self.device, verbose=False)

        # results is list-like; single image -> first item
        if not results:
            cv2.putText(frame, "No plants detected", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            return frame

        r = results[0]
        # r.boxes has xyxy, cls, conf
        try:
            xyxy = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy().astype(int)
            scores = r.boxes.conf.cpu().numpy()
        except Exception:
            # CPU-only fallback
            xyxy = r.boxes.xyxy.numpy()
            classes = r.boxes.cls.numpy().astype(int)
            scores = r.boxes.conf.numpy()

        found = False
        for box, cls, score in zip(xyxy, classes, scores):
            x1, y1, x2, y2 = map(int, box)
            label = self.class_names.get(int(cls), str(int(cls)))
            conf_pct = score * 100.0
            # Draw box
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            # Draw filled background for text for readability
            text = f"{label} {conf_pct:.0f}%"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            # ensure y1-th-6 not negative
            ty1 = max(y1 - th - 6, 0)
            cv2.rectangle(frame, (x1, ty1), (x1 + tw, y1), color, -1)
            cv2.putText(frame, text, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            found = True

        if not found:
            cv2.putText(frame, "No plants detected", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        return frame
