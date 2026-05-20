import os
import shutil

def flatten_dataset(data_dir="ResearchData"):
    print(f"🔍 Scanning '{data_dir}' to fix folder structures...")
    moved_count = 0
    folders_deleted = 0

    # Go through every soil type (Alluvial, not_soil, etc.)
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        
        if os.path.isdir(class_path):
            # 1. Find all sub-folders (like 'cars', 'leaves') inside this class
            sub_folders = [f for f in os.listdir(class_path) if os.path.isdir(os.path.join(class_path, f))]
            
            # 2. Loop through them and automatically assign a category number (0, 1, 2...)
            for category_idx, folder_name in enumerate(sub_folders):
                folder_path = os.path.join(class_path, folder_name)
                print(f"📂 Found sub-folder: '{folder_name}'. Assigning category ID: {category_idx}...")
                
                # Move every image inside it up one level
                for img_file in os.listdir(folder_path):
                    src = os.path.join(folder_path, img_file)
                    
                    if os.path.isfile(src):
                        # 3. Create your requested naming convention (e.g., "0_0.jpg")
                        safe_name = f"{category_idx}_{img_file}"
                        dst = os.path.join(class_path, safe_name)
                        
                        # Collision protection just in case
                        counter = 1
                        base_name, ext = os.path.splitext(safe_name)
                        while os.path.exists(dst):
                            dst = os.path.join(class_path, f"{base_name}_v{counter}{ext}")
                            counter += 1
                        
                        # Physically move the file
                        shutil.move(src, dst)
                        moved_count += 1
                
                # Delete the empty folder when done
                try:
                    os.rmdir(folder_path)
                    folders_deleted += 1
                except OSError:
                    print(f"⚠️ Could not delete {folder_path} (it might contain non-image files).")

    print("\n=========================================")
    print("✅ DATASET FLATTENED SUCCESSFULLY!")
    print(f"📸 Images recovered & renamed: {moved_count}")
    print(f"🗑️ Weird sub-folders deleted: {folders_deleted}")
    print("=========================================\n")

if __name__ == "__main__":
    flatten_dataset()