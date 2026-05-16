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

    def fill_not_soil_from_imagenet(self, imagenet_path):
        not_soil_path = os.path.join(self.research_data, "Not_Soil")
        if not os.path.exists(not_soil_path):
            os.makedirs(not_soil_path)

        existing_count = len(os.listdir(not_soil_path))
        needed = self.target_count - existing_count
        if needed <= 0:
            return

        all_available_images = []
        for root, _, files in os.walk(imagenet_path):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    all_available_images.append(os.path.join(root, f))

        if not all_available_images:
            return

        random.shuffle(all_available_images)
        selected_samples = all_available_images[:needed]

        for i, src_path in enumerate(selected_samples):
            img = cv2.imread(src_path)
            if img is not None:
                img = cv2.resize(img, self.img_size)
                cv2.imwrite(os.path.join(not_soil_path, f"imgnet_sample_{uuid.uuid4().hex[:8]}.jpg"), img)

    def balance_soil_folders(self):
        for folder in os.listdir(self.research_data):
            # FIX 1: Ignore case sensitivity!
            if folder.lower() == "not_soil": continue
            f_path = os.path.join(self.research_data, folder)
            if not os.path.isdir(f_path): continue

            # FIX 2: Only rename FILES, completely ignore sub-directories
            files = [f for f in os.listdir(f_path) if os.path.isfile(os.path.join(f_path, f)) and not f.startswith('aug_')]
            for f in files:
                if not f.startswith('orig_'):
                    safe_idx = 0
                    while os.path.exists(os.path.join(f_path, f"orig_{safe_idx}.jpg")):
                        safe_idx += 1
                    os.rename(os.path.join(f_path, f), os.path.join(f_path, f"orig_{safe_idx}.jpg"))

            current_files = [f for f in os.listdir(f_path) if os.path.isfile(os.path.join(f_path, f))]
            if len(current_files) < self.target_count:
                needed = self.target_count - len(current_files)
                print(f"⚖️ Balancing '{folder}': Generating {needed} new images...")
                
                originals = [f for f in os.listdir(f_path) if f.startswith('orig_')]
                if not originals:
                    print(f"⚠️ Warning: No original images found in {folder} to augment!")
                    continue
                    
                for _ in range(needed):
                    src_file = random.choice(originals)
                    img = cv2.imread(os.path.join(f_path, src_file))
                    if img is not None:
                        unique_id = uuid.uuid4().hex[:8]
                        cv2.imwrite(os.path.join(f_path, f"aug_{unique_id}.jpg"), self.augment_image(img))

if __name__ == "__main__":
    # FIX 3: Make sure this points to your massive backup folder OUTSIDE of ResearchData!
    # DO NOT point this to "./ResearchData/not_soil"
    IMAGENET_SOURCE = r"C:\Users\D524-PC\Desktop\Raw_ImageNet_Backup" 
    
    balancer = DatasetBalancer()
    
    if os.path.exists(IMAGENET_SOURCE):
        balancer.fill_not_soil_from_imagenet(IMAGENET_SOURCE)
    else:
        print(f"⚠️ Could not find ImageNet backup at {IMAGENET_SOURCE}")
    
    balancer.balance_soil_folders()
    print("✅ All folders are now perfectly balanced to 600 images!")