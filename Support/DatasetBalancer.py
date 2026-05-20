import os
import cv2
import random
import shutil
import numpy as np
import uuid

class DatasetBalancer:
    def __init__(self, research_data="ResearchData", target_count=600):
        self.research_data = research_data
        self.target_count = target_count
        self.img_size = (256, 256)

    def augment_image(self, img):
        choice = random.choice(['flip_h', 'flip_v', 'bright', 'dark', 'rotate'])
        if choice == 'flip_h': return cv2.flip(img, 1)
        if choice == 'flip_v': return cv2.flip(img, 0)
        if choice == 'bright': return cv2.convertScaleAbs(img, alpha=1.2, beta=10)
        if choice == 'dark': return cv2.convertScaleAbs(img, alpha=0.8, beta=-10)
        if choice == 'rotate':
            M = cv2.getRotationMatrix2D((128, 128), random.randint(1, 359), 1)
            return cv2.warpAffine(img, M, self.img_size)
        return img

    def process_and_balance_all(self):
        """Universal balancer that flattens, trims, and augments ALL folders equally."""
        for folder in os.listdir(self.research_data):
            f_path = os.path.join(self.research_data, folder)
            if not os.path.isdir(f_path): continue

            print(f"\n🔄 Processing folder: '{folder}'...")

            # 1. FLATTEN SUBFOLDERS
            for item in os.listdir(f_path):
                item_path = os.path.join(f_path, item)
                if os.path.isdir(item_path):
                    print(f"   📂 Flattening sub-folder: '{item}'...")
                    for img_file in os.listdir(item_path):
                        src = os.path.join(item_path, img_file)
                        if os.path.isfile(src):
                            dst = os.path.join(f_path, f"{item}_{img_file}")
                            shutil.move(src, dst)
                    try:
                        os.rmdir(item_path)
                    except OSError:
                        pass

            # 2. COUNT AND TRIM EXCESS FIRST (Massive Speedup!)
            current_files = [f for f in os.listdir(f_path) if os.path.isfile(os.path.join(f_path, f))]
            if len(current_files) > self.target_count:
                excess = len(current_files) - self.target_count
                print(f"   ✂️ Too many images! Randomly trimming {excess} images to reach {self.target_count}...")
                to_delete = random.sample(current_files, excess)
                for f in to_delete:
                    try:
                        os.remove(os.path.join(f_path, f))
                    except FileNotFoundError:
                        pass

            # 3. STANDARDIZE NAMES (Only renames the remaining 600 files)
            files = [f for f in os.listdir(f_path) if os.path.isfile(os.path.join(f_path, f)) and not f.startswith('aug_')]
            safe_idx = 0 # Moved outside the loop to stop it from resetting to 0!
            for f in files:
                if not f.startswith('orig_'):
                    while os.path.exists(os.path.join(f_path, f"orig_{safe_idx}.jpg")):
                        safe_idx += 1
                    os.rename(os.path.join(f_path, f), os.path.join(f_path, f"orig_{safe_idx}.jpg"))

            # 4. AUGMENT SHORTAGES
            current_files = [f for f in os.listdir(f_path) if os.path.isfile(os.path.join(f_path, f))]
            if len(current_files) < self.target_count:
                needed = self.target_count - len(current_files)
                print(f"   ⚖️ Balancing: Generating {needed} new augmented images...")
                
                originals = [f for f in os.listdir(f_path) if f.startswith('orig_')]
                if not originals:
                    print(f"   ⚠️ Warning: No original images found to augment!")
                    continue
                    
                for _ in range(needed):
                    src_file = random.choice(originals)
                    img = cv2.imread(os.path.join(f_path, src_file))
                    if img is not None:
                        unique_id = uuid.uuid4().hex[:8]
                        cv2.imwrite(os.path.join(f_path, f"aug_{unique_id}.jpg"), self.augment_image(img))

if __name__ == "__main__":
    balancer = DatasetBalancer(target_count=600)
    balancer.process_and_balance_all()
    print("\n✅ All folders are now flattened, cleaned, and perfectly balanced to 600 images!")