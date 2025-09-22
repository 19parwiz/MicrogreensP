# AgroTech/training/prepare_dataset.py
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # project root (AgroTech)
DATA_IMAGES = ROOT / "data" / "microgreens"

def check_split(split_name):
    img_root = DATA_IMAGES / split_name
    print(f"Checking: {img_root}")
    if not img_root.exists():
        print("  MISSING:", img_root)
        return
    classes = sorted([p.name for p in img_root.iterdir() if p.is_dir()])
    print("  Classes:", classes)
    total = 0
    for cls in classes:
        cls_dir = img_root / cls
        imgs = [p for p in cls_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        print(f"    {cls}: {len(imgs)} images")
        total += len(imgs)
    print(f"  Total images in {split_name}: {total}\n")

def check_labels(split_name):
    # YOLO expects labels in parallel to images: one .txt per image (same filename but .txt)
    images_root = DATA_IMAGES / split_name
    if not images_root.exists():
        print(f"Missing images root: {images_root}")
        return
    missing_labels = 0
    for cls_dir in images_root.iterdir():
        if not cls_dir.is_dir(): continue
        for img in cls_dir.iterdir():
            if img.suffix.lower() not in [".jpg", ".jpeg", ".png"]: continue
            label = img.with_suffix(".txt")
            if not label.exists():
                missing_labels += 1
    print(f"Missing label files in {split_name}: {missing_labels}")

if __name__ == "__main__":
    print("=== Dataset quick check ===")
    for s in ("train", "val", "test"):
        check_split(s)
        check_labels(s)
    print("If missing labels > 0, annotate images with LabelImg / Roboflow in YOLO format.")
