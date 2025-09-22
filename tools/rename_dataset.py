import os
import glob

base = "data/microgreens"
splits = ["train", "val", "test"]

for split in splits:
    split_path = os.path.join(base, split)
    for class_name in os.listdir(split_path):
        class_path = os.path.join(split_path, class_name)
        if not os.path.isdir(class_path):
            continue

        # Collect images
        images = sorted(glob.glob(os.path.join(class_path, "*.jpg")) +
                        glob.glob(os.path.join(class_path, "*.png")))

        for idx, img_path in enumerate(images, start=1):
            ext = os.path.splitext(img_path)[1]
            new_name = f"{class_name}_{idx:04d}{ext}"
            new_txt = f"{class_name}_{idx:04d}.txt"

            old_txt = os.path.splitext(img_path)[0] + ".txt"

            # Rename image
            new_img_path = os.path.join(class_path, new_name)
            os.rename(img_path, new_img_path)

            # Rename label if exists
            if os.path.exists(old_txt):
                new_txt_path = os.path.join(class_path, new_txt)
                os.rename(old_txt, new_txt_path)

        print(f"Renamed {len(images)} files in {split}/{class_name}")
