# AgroTech/training/test_inference.py
# Quick test to run model on a single image and save output
from ultralytics import YOLO
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[1]
weights = ROOT / "models" / "plant_yolo.pt"
img = ROOT / "data" / "microgreens" / "val" / "Basil" / "0001.jpg"  # change to a real image path

model = YOLO(str(weights))
res = model.predict(source=str(img), conf=0.5)
# res[0].plot() returns PIL image (for older versions), or you can use res[0].plot() methods
out = res[0].plot()  # this returns an image with boxes (numpy)
if out is not None:
    cv2.imwrite(str(ROOT / "outputs" / "detections" / "test_output.jpg"), out[:, :, ::-1])  # convert RGB->BGR
    print("Saved output to outputs/detections/test_output.jpg")
else:
    print("No output image generated.")
