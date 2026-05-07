import os
import cv2
import shutil
import numpy as np
from Models.SVM_Calibrator import SoilCalibratorSVM
from Models.CLS_extraction import LabColorExtractor
from Models.XGB_Classification import SoilClassifierXGB

class TrainingManager:
    def __init__(self, data_dir="ResearchData", dump_dir="DumpImages"):
        self.data_dir = data_dir
        self.dump_dir = dump_dir
        self.calibrator = SoilCalibratorSVM()
        self.extractor = LabColorExtractor()
        self.classifier = SoilClassifierXGB(learning_rate=0.1)
        
        self.features_file = "extracted_features.npy"
        self.labels_file = "extracted_labels.npy"

    def clean_dataset(self):
        """Scans the dataset and removes watermarked images before training."""
        if not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir)

        print("🧹 Scanning dataset for watermarks...")
        moved_count = 0

        for folder in os.listdir(self.data_dir):
            folder_path = os.path.join(self.data_dir, folder)
            if os.path.isdir(folder_path):
                for filename in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, filename)
                    
                    img = cv2.imread(file_path)
                    if img is None: continue
                    
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    _, mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
                    
                    # Calculate white pixel ratio
                    white_pixels = cv2.countNonZero(mask)
                    total_pixels = img.shape * img.shape
                    ratio = white_pixels / total_pixels
                    
                    if ratio > 0.015: 
                        print(f"🚨 Watermark found! Moving {filename} to {self.dump_dir}")
                        shutil.move(file_path, os.path.join(self.dump_dir, f"{folder}_{filename}"))
                        moved_count += 1
                        
        if moved_count > 0:
            print(f"✅ Cleaned {moved_count} images. You must delete the .npy files to re-extract features!")
        else:
            print("✅ Dataset is clean. No watermarks found.")

    def process_images_to_data(self):
        """Scans folders and processes images into mathematical features."""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            print(f"⚠️ Created missing folder: {self.data_dir}. Please add images and restart.")
            return np.array([]), np.array([])

        if os.path.exists(self.features_file) and os.path.exists(self.labels_file):
            print("Loading previously extracted features...")
            return np.load(self.features_file), np.load(self.labels_file)

        print("Extracting features from images. This might take a minute...")
        X, y = [], []
        for class_name in os.listdir(self.data_dir):
            class_path = os.path.join(self.data_dir, class_name)
            if os.path.isdir(class_path):
                for file in os.listdir(class_path):
                    img = cv2.imread(os.path.join(class_path, file))
                    if img is not None:
                        img = cv2.resize(img, (128, 128))
                        calibrated = self.calibrator.calibrate(img)
                        features = self.extractor.extract_features(calibrated)
                        X.append(features)
                        y.append(class_name)

        X, y = np.array(X), np.array(y)
        if len(X) > 0:
            np.save(self.features_file, X)
            np.save(self.labels_file, y)
        return X, y

    def run_training_loop(self, epochs=3):
        """Executes the cleaning, feature extraction, and training loop."""
        # 1. Clean the dataset first!
        self.clean_dataset()
        
        # 2. Extract Features
        X, y = self.process_images_to_data()
        
        if len(X) == 0:
            print("No data available to train on.")
            return

        current_lr = 0.1
        print("Starting Continuous Training Loop...")
        
        for epoch in range(1, epochs + 1):
            print(f"\n--- Epoch {epoch} | Learning Rate: {current_lr:.3f} ---")
            self.classifier.update_learning_rate(current_lr)
            accuracy = self.classifier.train(X, y)
            print(f"Accuracy after Epoch {epoch}: {accuracy * 100:.2f}%")
            current_lr = current_lr * 0.8 # Decay learning rate

if __name__ == "__main__":
    manager = TrainingManager()
    manager.run_training_loop(epochs=5)