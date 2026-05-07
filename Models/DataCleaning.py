import os
import cv2
import shutil

def clean_watermarked_images(data_dir="Research_Data", dump_dir="DumpImages"):
    # Create the DumpImages folder if it doesn't exist
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)
        print(f"Created folder: {dump_dir}")

    moved_count = 0

    # Scan through all your soil folders (Alluvial, Black, etc.)
    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        
        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                
                # Load the image
                img = cv2.imread(file_path)
                if img is None: continue
                
                # --- WATERMARK DETECTION LOGIC ---
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # Find all pixels that are extremely bright (like white text)
                _, mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
                
                # Calculate the percentage of the image covered by these bright pixels
                white_pixels = cv2.countNonZero(mask)
                total_pixels = img.shape * img.shape
                ratio = white_pixels / total_pixels
                
                # If more than 1.5% of the image is pure white text, consider it watermarked
                if ratio > 0.015: 
                    print(f"🚨 Watermark found in {folder}/{filename}! Moving to DumpImages...")
                    destination = os.path.join(dump_dir, f"{folder}_{filename}")
                    
                    # Physically move the file
                    shutil.move(file_path, destination)
                    moved_count += 1

    print(f"\n✅ Cleanup Complete! Moved {moved_count} images to '{dump_dir}'.")

if __name__ == "__main__":
    clean_watermarked_images()