import os
import cv2
import shutil

def clean_dataset(data_dir="ResearchData", dump_dir="DumpImages"):
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    broken_count = 0
    watermark_count = 0

    print(f"🧹 Scanning '{data_dir}' for broken images and watermarks...")

    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        
        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                
                # --- 1. CHECK FOR BROKEN / CORRUPTED IMAGES ---
                img = cv2.imread(file_path)
                if img is None:
                    print(f"🗑️ Deleting corrupted file: {folder}/{filename}")
                    os.remove(file_path)
                    broken_count += 1
                    continue # Skip the watermark check and go to the next image
                
                # --- 2. CHECK FOR WATERMARKS ---
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
                
                white_pixels = cv2.countNonZero(mask)
                total_pixels = img.shape[0] * img.shape[1]
                ratio = white_pixels / total_pixels
                
                if ratio > 0.015: 
                    print(f"🚨 Watermark found in {folder}/{filename}! Moving to {dump_dir}...")
                    destination = os.path.join(dump_dir, f"{folder}_{filename}")
                    shutil.move(file_path, destination)
                    watermark_count += 1

    print("\n=========================================")
    print("✅ DATASET CLEANUP COMPLETE!")
    print(f"🗑️ Broken Images Deleted: {broken_count}")
    print(f"🚨 Watermarked Images Moved: {watermark_count}")
    print("=========================================\n")
    print("➡️ You can now run your datasetbalancer.py safely!")

if __name__ == "__main__":
    clean_dataset()