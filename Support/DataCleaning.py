import os
import cv2
import shutil

def clean_dataset(data_dir="ResearchData", dump_dir="DumpImages"):
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    broken_count = 0
    watermark_count = 0

    print(f"🧹 Scanning '{data_dir}' (and all sub-folders) for broken images and watermarks...")

    # Using os.walk allows the script to automatically dive into sub-folders like 'abacus'
    for root, dirs, files in os.walk(data_dir):
        # Skip the dump directory if it happens to be inside our scan area
        if dump_dir in root:
            continue

        for filename in files:
            # Only process actual image files
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            file_path = os.path.join(root, filename)
            folder_name = os.path.basename(root)
            
            # --- 1. CHECK FOR BROKEN / CORRUPTED IMAGES ---
            img = cv2.imread(file_path)
            if img is None:
                print(f"🗑️ Deleting corrupted file: {folder_name}/{filename}")
                os.remove(file_path)
                broken_count += 1
                continue 
            
            # --- 2. CHECK FOR WATERMARKS ---
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
            
            white_pixels = cv2.countNonZero(mask)
            total_pixels = img.shape[0] * img.shape[1]
            ratio = white_pixels / total_pixels
            
            if ratio > 0.015: 
                print(f"🚨 Watermark found in {folder_name}/{filename}! Moving to {dump_dir}...")
                
                # Safe naming for the dump folder so files don't overwrite each other
                destination = os.path.join(dump_dir, f"{folder_name}_{filename}")
                counter = 1
                base_name, ext = os.path.splitext(f"{folder_name}_{filename}")
                while os.path.exists(destination):
                    destination = os.path.join(dump_dir, f"{base_name}_v{counter}{ext}")
                    counter += 1
                
                shutil.move(file_path, destination)
                watermark_count += 1

    print("\n=========================================")
    print("✅ DATASET CLEANUP COMPLETE!")
    print(f"🗑️ Broken Images Deleted: {broken_count}")
    print(f"🚨 Watermarked Images Moved: {watermark_count}")
    print("=========================================\n")

if __name__ == "__main__":
    clean_dataset()