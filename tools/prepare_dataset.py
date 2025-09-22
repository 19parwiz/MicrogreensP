import os
import shutil
import random
from pathlib import Path
from PIL import Image

# Paths
root = Path("data/microgreens")
output = root  # we will create train/val/test inside this

# Classes
classes = ["Arugula", "Basil", "Beetroot", "Mangold", "Tarragon"]

# Create output dirs
for split in ["train", "val", "test"]:
    (output / split / "images").mkdir(parents=True, exist_ok=True)
    (output / split / "labels").mkdir(parents=True, exist_ok=True)

# Collect all image-label pairs
pairs = []
for cls in classes:
    class_dir = root / "train" / cls
    for img_file in class_dir.glob("*"):
        if img_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".dng"]:
            label_file = img_file.with_suffix(".txt")
            if label_file.exists():
                pairs.append((img_file, label_file))

print(f"Found {len(pairs)} image-label pairs.")

# Shuffle for splitting
random.shuffle(pairs)
n = len(pairs)
train_split = int(0.8 * n)
val_split = int(0.9 * n)

splits = {
    "train": pairs[:train_split],
    "val": pairs[train_split:val_split],
    "test": pairs[val_split:]
}

# Copy & clean files
counter = 1
for split, items in splits.items():
    for img_file, lbl_file in items:
        # New name
        new_name = f"{counter:05d}.jpg"
        new_lbl = f"{counter:05d}.txt"
        
        # Handle image (convert DNG → JPG if needed)
        if img_file.suffix.lower() == ".dng":
            img = Image.open(img_file)
            img = img.convert("RGB")
            img.save(output / split / "images" / new_name, "JPEG")
        else:
            shutil.copy(img_file, output / split / "images" / new_name)
        
        # Copy label
        shutil.copy(lbl_file, output / split / "labels" / new_lbl)
        
        counter += 1

print(" Dataset prepared in YOLO format!")
