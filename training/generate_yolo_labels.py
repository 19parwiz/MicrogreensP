import os
from PIL import Image

# BASE DIR for microgreens dataset
BASE_DIR = r"D:\AgroTech\data\microgreens"

# splits and classes
splits = ["train", "val", "test"]
class_map = {
    "Arugula": 0,
    "Basil": 1,
    "Beetroot": 2,
    "Mangold": 3,
    "Tarragon": 4
}

# Loop through dataset
for split in splits:
    split_dir = os.path.join(BASE_DIR, split)
    for cls_name, cls_id in class_map.items():
        cls_dir = os.path.join(split_dir, cls_name)
        if not os.path.exists(cls_dir):
            continue
        for img_file in os.listdir(cls_dir):
            if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(cls_dir, img_file)
                txt_path = os.path.splitext(img_path)[0] + ".txt"

                # Get image size
                with Image.open(img_path) as img:
                    w, h = img.size

                # YOLO format for full image box
                x_center = 0.5
                y_center = 0.5
                width = 1.0
                height = 1.0

                with open(txt_path, "w") as f:
                    f.write(f"{cls_id} {x_center} {y_center} {width} {height}\n")

print("YOLO labels generated!")
