import os
import cv2
import shutil
import numpy as np
import pandas as pd
from datetime import datetime
from ClassificationModels.SVM_Calibrator import SoilCalibratorSVM
from ClassificationModels.CLS_extraction import LabColorExtractor
from ClassificationModels.XGB_Classification import SoilClassifierXGB

class TrainingManager:
    def __init__(self, data_dir="ResearchData", log_file="training_history.xlsx"):
        self.data_dir = data_dir
        self.log_file = log_file
        self.img_size = (128, 128)
        self.calibrator = SoilCalibratorSVM()
        self.extractor = LabColorExtractor()
        self.classifier = SoilClassifierXGB()

    def reset_artifacts(self):
        """Removes previous model and feature files for clean training."""
        for f in ["soil_xgb_model.json", "soil_classes.npy", "extracted_features.npy", "extracted_labels.npy"]:
            if os.path.exists(f): os.remove(f)

    def process_balanced_dataset(self):
        """Extracts features from all folders in ResearchData."""
        X, y = [], []
        for class_name in os.listdir(self.data_dir):
            c_path = os.path.join(self.data_dir, class_name)
            if os.path.isdir(c_path):
                for f in os.listdir(c_path):
                    img = cv2.imread(os.path.join(c_path, f))
                    if img is not None:
                        img = cv2.resize(img, self.img_size)
                        calibrated = self.calibrator.calibrate(img)
                        features = self.extractor.extract_features(calibrated)
                        X.append(features)
                        y.append(class_name)
        return np.array(X), np.array(y)

    def log_to_excel(self, data):
        df = pd.DataFrame([data])
        if os.path.exists(self.log_file):
            df = pd.concat([pd.read_excel(self.log_file), df], ignore_index=True)
        df.to_excel(self.log_file, index=False)

    def execute(self):
        start = datetime.now()
        self.reset_artifacts()
        X, y = self.process_balanced_dataset()
        
        if len(X) == 0: return
        results = self.classifier.train(X, y)
        
        end = datetime.now()
        log = {
            "Start Time": start.strftime("%Y-%m-%d %H:%M:%S"),
            "End Time": end.strftime("%Y-%m-%d %H:%M:%S"),
            "Duration": str(end - start),
            "Samples": len(X),
            "Accuracy": round(results['accuracy'], 4),
            "RMSE": round(results['rmse'], 4),
            "R2": round(results['r2'], 4)
        }
        self.log_to_excel(log)

        print("\n" + "="*50)
        print("METHODOLOGY TRAINING COMPLETE")
        print("-"*50)
        print(f"Total Samples: {len(X)}")
        print(f"Accuracy:      {results['accuracy']*100:.2f}%")
        print(f"RMSE:          {results['rmse']:.4f}")
        print(f"R2 Score:      {results['r2']:.4f}")
        print("="*50 + "\n")

if __name__ == "__main__":
    TrainingManager().execute()