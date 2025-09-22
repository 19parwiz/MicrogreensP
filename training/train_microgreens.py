# AgroTech/training/train_microgreens.py
"""
Train YOLO11 on microgreens dataset.

Run from project root:
    python training/train_microgreens.py
"""

import os
import argparse
from ultralytics import YOLO
from pathlib import Path
import yaml

# Project / paths
ROOT = Path(__file__).resolve().parents[1]   # AgroTech/
YAML_PATH = ROOT / "data" / "microgreens.yaml"
MODELS_DIR = ROOT / "models"

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main(args):
    # Check YAML exists
    assert YAML_PATH.exists(), f"Missing YAML: {YAML_PATH}"
    print("Using YAML:", YAML_PATH)

    # Make sure models dir exists
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Set up training params
    backbone = args.backbone  # e.g. "yolo11n.pt"
    imgsz = args.imgsz
    batch = args.batch
    epochs = args.epochs
    project = str(MODELS_DIR)   # Ultralytics will write runs to PROJECT/NAME

    print(f"Training config: backbone={backbone}, imgsz={imgsz}, batch={batch}, epochs={epochs}")
    print("If you get OOM, reduce batch or imgsz (e.g. imgsz=512, batch=4).")

    model = YOLO(backbone)  # will download backbone if not present

    # run training
    model.train(
        data=str(YAML_PATH),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name=args.name,
        exist_ok=True,
        device=args.device  # e.g. 'cuda' or 'cpu'
    )

    print("Training finished.")
    print("Check:", MODELS_DIR / args.name / "weights")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, default="yolo11n.pt", help="YOLO11 backbone (n/s/m/l)")
    parser.add_argument("--imgsz", type=int, default=640, help="image size")
    parser.add_argument("--batch", type=int, default=8, help="batch size (decrease if OOM)")
    parser.add_argument("--epochs", type=int, default=50, help="epochs")
    parser.add_argument("--name", type=str, default="yolo11_microgreens", help="project run name (saved under models/)")
    parser.add_argument("--device", type=str, default="cuda", help="device, e.g. 'cuda' or 'cpu'")
    args = parser.parse_args()
    main(args)
