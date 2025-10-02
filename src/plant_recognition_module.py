"""
YOLO-based PlantRecognition module.
Keeps same API: class PlantRecognition with predict(frame) -> frame (BGR numpy)
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path


class PlantRecognition:
    def __init__(self, weights_path=None, conf=0.25, iou=0.45, device=None, shrink_ratio=0.6, display_conf=0.35, show_no_detection_text=True, min_area_ratio=0.005, min_side_px=20, persist_frames=2, persist_iou=0.5, safe_mode=False):
        """
        weights_path: path to .pt file (use the working model path)
        conf: confidence threshold (0..1)
        iou: nms IoU threshold
        device: 'cuda' or 'cpu' or None (auto)
        """
        # DEFAULT MODEL PATH (you can change this if needed)
        if weights_path is None:
            weights_path = "D:/AgroTech/runs/detect/train9/weights/best.pt"

        self.weights_path = str(weights_path)
        print("[PlantRecognition] Loading YOLO weights:", self.weights_path)
        self.model = YOLO(self.weights_path)
        self.conf = conf
        self.iou = iou
        self.device = device
        # shrink_ratio in (0,1], 1.0 means full YOLO box, 0.6 means 60% sized box centered
        self.shrink_ratio = float(max(0.05, min(1.0, shrink_ratio)))
        # display_conf: minimum confidence required to draw a box/label
        self.display_conf = float(max(0.0, min(1.0, display_conf)))
        self.show_no_detection_text = bool(show_no_detection_text)
        # filters to suppress tiny/background detections
        self.min_area_ratio = float(max(0.0, min(1.0, min_area_ratio)))
        self.min_side_px = int(max(0, min_side_px))
        # temporal persistence settings
        self.persist_frames = int(max(1, persist_frames))
        self.persist_iou = float(max(0.0, min(1.0, persist_iou)))
        # simple track store: list of dicts {bbox:[x1,y1,x2,y2], hits:int, last_seen:int, label:str, conf:float}
        self._tracks = []
        # safe_mode disables strict filters/persistence to ensure visibility
        self.safe_mode = bool(safe_mode)
        # Initialize class names safely after model loads
        self.class_names = {}
        try:
            names = self.model.names
            self.class_names = self._normalize_names(names)
            print("[PlantRecognition] Class names:", self.class_names)
        except Exception:
            self.class_names = {0: "class0"}
    @staticmethod
    def _iou(box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter = iw * ih
        if inter == 0:
            return 0.0
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = max(1, area_a + area_b - inter)
        return inter / union

        

    @staticmethod
    def _normalize_names(names_obj):
        """Return a dict[int,str] from ultralytics names which may be list or dict."""
        try:
            if isinstance(names_obj, dict):
                return {int(k): str(v) for k, v in names_obj.items()}
            if isinstance(names_obj, (list, tuple)):
                return {i: str(v) for i, v in enumerate(names_obj)}
        except Exception:
            pass
        return {0: "class0"}

    def predict(self, frame):
        """
        frame: OpenCV BGR image (numpy)
        returns: frame with bounding boxes and labels drawn
        """

        # Run YOLO detection
        results = self.model.predict(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False
        )

        #  MODIFIED: Only draw boxes if there are detections
        if not results or len(results[0].boxes) == 0:
            print("[PlantRecognition] No detections in frame")
            if self.show_no_detection_text:
                cv2.putText(frame, "No microgreens", (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            return frame

        r = results[0]
        # Defensive: ensure class_names exists; try to pull from result if empty
        if not getattr(self, 'class_names', None) or len(self.class_names) == 0:
            try:
                self.class_names = self._normalize_names(getattr(r, 'names', {}))
            except Exception:
                self.class_names = {0: "class0"}

        # If safe_mode, draw detections directly using model confidence only
        if self.safe_mode:
            h_frame, w_frame = frame.shape[:2]
            frame_area = float(max(1, h_frame * w_frame))
            drawn = 0
            for box in r.boxes:
                ox1, oy1, ox2, oy2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = self.class_names.get(cls, f"class{cls}")
                print(f"[PlantRecognition][safe] det conf={conf:.3f} cls={cls} label={label}")
                # In safe mode, still apply display confidence and size sanity checks
                if conf < self.display_conf:
                    continue
                bw = max(0, ox2 - ox1)
                bh = max(0, oy2 - oy1)
                if bw == 0 or bh == 0:
                    continue
                if (bw * bh) / frame_area < self.min_area_ratio:
                    continue
                if min(bw, bh) < self.min_side_px:
                    continue
                x1, y1, x2, y2 = ox1, oy1, ox2, oy2
                if self.shrink_ratio < 1.0:
                    w = x2 - x1
                    h = y2 - y1
                    if w > 0 and h > 0:
                        cx = x1 + w / 2.0
                        cy = y1 + h / 2.0
                        new_w = max(2, int(w * self.shrink_ratio))
                        new_h = max(2, int(h * self.shrink_ratio))
                        nx1 = int(cx - new_w / 2.0)
                        ny1 = int(cy - new_h / 2.0)
                        nx2 = nx1 + new_w
                        ny2 = ny1 + new_h
                        nx1 = max(0, min(nx1, w_frame - 1))
                        ny1 = max(0, min(ny1, h_frame - 1))
                        nx2 = max(0, min(nx2, w_frame - 1))
                        ny2 = max(0, min(ny2, h_frame - 1))
                        x1, y1, x2, y2 = nx1, ny1, nx2, ny2
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}",
                            (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)
                drawn += 1
            if drawn == 0 and self.show_no_detection_text:
                cv2.putText(frame, "No microgreens", (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            return frame

        #  MODIFIED: Prepare detections and apply display and size filters
        h_frame, w_frame = frame.shape[:2]
        frame_area = float(max(1, h_frame * w_frame))
        current_detections = []
        print(f"[PlantRecognition] Detections before filtering: {len(r.boxes)}")
        for box in r.boxes:
            # Extract box coordinates
            ox1, oy1, ox2, oy2 = map(int, box.xyxy[0])   # original top-left, bottom-right
            conf = float(box.conf[0])                # confidence
            cls = int(box.cls[0])                    # class index
            label = self.class_names.get(cls, f"class{cls}")
            print(f"[PlantRecognition] det conf={conf:.3f} cls={cls} label={label}")

            # Skip low-confidence detections for display
            if conf < self.display_conf:
                continue

            # Filter by physical size (area and side length)
            box_w = max(0, ox2 - ox1)
            box_h = max(0, oy2 - oy1)
            if box_w == 0 or box_h == 0:
                continue
            box_area_ratio = (box_w * box_h) / frame_area
            if box_area_ratio < self.min_area_ratio:
                continue
            if min(box_w, box_h) < self.min_side_px:
                continue

            # Store ORIGINAL bbox for tracking; shrinking will be applied at draw time
            current_detections.append({
                "bbox": [ox1, oy1, ox2, oy2],
                "label": label,
                "conf": conf,
            })

        # Update temporal tracks with current detections
        for det in current_detections:
            best_idx = -1
            best_iou = 0.0
            for idx, tr in enumerate(self._tracks):
                iou_val = self._iou(det["bbox"], tr["bbox"])
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_idx = idx
            if best_idx >= 0 and best_iou >= self.persist_iou:
                tr = self._tracks[best_idx]
                tr["bbox"] = det["bbox"]
                tr["label"] = det["label"]
                tr["conf"] = det["conf"]
                tr["hits"] = min(tr.get("hits", 0) + 1, 1000000)
                tr["last_seen"] = 0
            else:
                self._tracks.append({
                    "bbox": det["bbox"],
                    "label": det["label"],
                    "conf": det["conf"],
                    "hits": 1,
                    "last_seen": 0,
                })

        # Age tracks and drop stale ones
        to_keep = []
        for tr in self._tracks:
            tr["last_seen"] = tr.get("last_seen", 0) + 1
            if tr["last_seen"] <= 2:
                to_keep.append(tr)
        self._tracks = to_keep

        # Draw only stable tracks from this frame
        drawn = 0
        for tr in self._tracks:
            if tr.get("hits", 0) >= self.persist_frames and tr.get("last_seen", 0) == 0:
                x1, y1, x2, y2 = tr["bbox"]
                # Apply shrinking at draw time based on current frame dimensions
                if self.shrink_ratio < 1.0:
                    w = x2 - x1
                    h = y2 - y1
                    if w > 0 and h > 0:
                        cx = x1 + w / 2.0
                        cy = y1 + h / 2.0
                        new_w = max(2, int(w * self.shrink_ratio))
                        new_h = max(2, int(h * self.shrink_ratio))
                        nx1 = int(cx - new_w / 2.0)
                        ny1 = int(cy - new_h / 2.0)
                        nx2 = nx1 + new_w
                        ny2 = ny1 + new_h
                        nx1 = max(0, min(nx1, w_frame - 1))
                        ny1 = max(0, min(ny1, h_frame - 1))
                        nx2 = max(0, min(nx2, w_frame - 1))
                        ny2 = max(0, min(ny2, h_frame - 1))
                        x1, y1, x2, y2 = nx1, ny1, nx2, ny2
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{tr['label']} {tr['conf']:.2f}",
                            (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)
                drawn += 1

        # Fallback: if no stable tracks, draw current frame's filtered detections directly
        if drawn == 0:
            print("[PlantRecognition] No stable tracks; drawing current detections as fallback")
            for det in current_detections:
                x1, y1, x2, y2 = det["bbox"]
                label = det["label"]
                conf = det["conf"]
                if self.shrink_ratio < 1.0:
                    w = x2 - x1
                    h = y2 - y1
                    if w > 0 and h > 0:
                        cx = x1 + w / 2.0
                        cy = y1 + h / 2.0
                        new_w = max(2, int(w * self.shrink_ratio))
                        new_h = max(2, int(h * self.shrink_ratio))
                        nx1 = int(cx - new_w / 2.0)
                        ny1 = int(cy - new_h / 2.0)
                        nx2 = nx1 + new_w
                        ny2 = ny1 + new_h
                        nx1 = max(0, min(nx1, w_frame - 1))
                        ny1 = max(0, min(ny1, h_frame - 1))
                        nx2 = max(0, min(nx2, w_frame - 1))
                        ny2 = max(0, min(ny2, h_frame - 1))
                        x1, y1, x2, y2 = nx1, ny1, nx2, ny2
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
                cv2.putText(frame, f"{label} {conf:.2f}",
                            (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 200, 255), 2)
                drawn += 1

        if drawn == 0 and self.show_no_detection_text:
            print("[PlantRecognition] No stable tracks to draw (after filtering/persistence)")
            cv2.putText(frame, "No microgreens", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        return frame
