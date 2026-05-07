import os
import cv2
import numpy as np
from calibrator import SoilCalibratorSVM
from extractor import LabColorExtractor
from classifier import SoilClassifierXGB

class TrainingManager:
    def __init__(self, data_dir="Research_Data"):
        self.data_dir = data_dir
        self.calibrator = SoilCalibratorSVM()
        self.extractor = LabColorExtractor()
        self.classifier = SoilClassifierXGB(learning_rate=0.1)
        
        self.features_file = "extracted_features.npy"
        self.labels_file = "extracted_labels.npy"

    def process_images_to_data(self):
        """Scans folders and processes images. Skips this if already done."""
        if os.path.exists(self.features_file) and os.path.exists(self.labels_file):
            print("Loading previously extracted features (Saves time!)...")
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
        # Save so we don't have to process images again next run
        np.save(self.features_file, X)
        np.save(self.labels_file, y)
        return X, y

    def run_training_loop(self, epochs=3):
        """
        Trains the model multiple times. 
        Decreases the learning rate each epoch to 'fine-tune' the accuracy.
        """
        X, y = self.process_images_to_data()
        
        if len(X) == 0:
            print(f"No images found in {self.data_dir}")
            return

        current_lr = 0.1
        print("Starting Continuous Training Loop...")
        
        for epoch in range(1, epochs + 1):
            print(f"\n--- Epoch {epoch} | Learning Rate: {current_lr:.3f} ---")
            
            # Apply the current learning rate
            self.classifier.update_learning_rate(current_lr)
            
            # Train and get accuracy
            accuracy = self.classifier.train(X, y)
            print(f"Accuracy after Epoch {epoch}: {accuracy * 100:.2f}%")
            
            # Decay the learning rate by 20% for the next loop
            # This helps the model settle into the best weights without bouncing around
            current_lr = current_lr * 0.8

if __name__ == "__main__":
    # Ensure your folders are setup: Research_Data/Red, Research_Data/Noise, etc.
    manager = TrainingManager(data_dir="Research_Data")
    
    # Run the loop 5 times, decaying the learning rate each time
    manager.run_training_loop(epochs=5)