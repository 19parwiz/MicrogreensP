import os
import shutil

base_path = "data/microgreens"
print("Fixing YOLO dataset structure...")

for split in ['train', 'val', 'test']:
    print(f"Processing {split}...")
    
    # Create YOLO structure
    os.makedirs(f"{base_path}/{split}/images", exist_ok=True)
    os.makedirs(f"{base_path}/{split}/labels", exist_ok=True)
    
    # Move images to images folder
    for item in os.listdir(f"{base_path}/{split}"):
        item_path = f"{base_path}/{split}/{item}"
        if os.path.isdir(item_path) and item not in ['images', 'labels']:
            print(f"  Moving images from {item}...")
            
            # Move only image files
            for img in os.listdir(item_path):
                if img.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')):
                    #  FIX: Rename files to include class name to avoid duplicates
                    new_name = f"{item}_{img}"
                    shutil.move(f"{item_path}/{img}", f"{base_path}/{split}/images/{new_name}")
                elif img.endswith('.DNG'):  # Handle DNG files too
                    new_name = f"{item}_{img}"
                    shutil.move(f"{item_path}/{img}", f"{base_path}/{split}/images/{new_name}")
            
            # Check if folder is empty BEFORE removing
            if len(os.listdir(item_path)) == 0:
                os.rmdir(item_path)
                print(f"    Removed empty folder: {item}")
            else:
                print(f"    Folder not empty, keeping: {item}")
                print(f"    Remaining files: {os.listdir(item_path)}")

print(" Done! Dataset now has YOLO structure:")
print(" data/microgreens/train/images/")
print(" data/microgreens/train/labels/") 
print(" data/microgreens/val/images/")
print(" data/microgreens/val/labels/")
print(" data/microgreens/test/images/")
print(" data/microgreens/test/labels/")