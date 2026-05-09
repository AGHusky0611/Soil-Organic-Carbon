import os
import cv2
import random
import shutil
import numpy as np

class DatasetBalancer:
    def __init__(self, research_data="ResearchData", target_count=600):
        """Initializes paths and target image counts for normalization."""
        self.research_data = research_data
        self.target_count = target_count
        self.img_size = (256, 256)

    def augment_image(self, img):
        """Applies random transformations to increase dataset variety."""
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
        """
        Iterates through all subdirectories in the ImageNet path.
        Randomly selects images to reach the target count for the Not_Soil class.
        """
        not_soil_path = os.path.join(self.research_data, "Not_Soil")
        if not os.path.exists(not_soil_path):
            os.makedirs(not_soil_path)

        existing_count = len(os.listdir(not_soil_path))
        needed = self.target_count - existing_count
        if needed <= 0:
            return

        # Automatically iterate through all nested subfolders
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
                # Save with unique index to avoid naming conflicts
                cv2.imwrite(os.path.join(not_soil_path, f"imgnet_sample_{existing_count + i}.jpg"), img)

    def balance_soil_folders(self):
        """Standardizes all soil folders to contain exactly 600 samples."""
        for folder in os.listdir(self.research_data):
            if folder == "Not_Soil": continue
            f_path = os.path.join(self.research_data, folder)
            if not os.path.isdir(f_path): continue

            # Standardize names of original files
            files = [f for f in os.listdir(f_path) if not f.startswith('aug_')]
            for idx, f in enumerate(files):
                if not f.startswith('orig_'):
                    os.rename(os.path.join(f_path, f), os.path.join(f_path, f"orig_{idx}.jpg"))

            current_files = os.listdir(f_path)
            if len(current_files) < self.target_count:
                needed = self.target_count - len(current_files)
                originals = [f for f in os.listdir(f_path) if f.startswith('orig_')]
                for i in range(needed):
                    src_file = random.choice(originals)
                    img = cv2.imread(os.path.join(f_path, src_file))
                    cv2.imwrite(os.path.join(f_path, f"aug_{i}.jpg"), self.augment_image(img))

if __name__ == "__main__":
    # Path to your ImageNet download
    IMAGENET_SOURCE = r"./ResearchData/not_soil" 
    
    balancer = DatasetBalancer()
    
    # Process ImageNet samples first
    if os.path.exists(IMAGENET_SOURCE):
        balancer.fill_not_soil_from_imagenet(IMAGENET_SOURCE)
    
    # Balance remaining classes
    balancer.balance_soil_folders()