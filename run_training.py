import os
import argparse
import cv2
import shutil
import numpy as np
import pandas as pd
from datetime import datetime
from ClassificationModels.SVM_Calibrator import SoilCalibratorSVM, SoilClassifierSVM
from ClassificationModels.CLS_extraction import LabColorExtractor
from ClassificationModels.XGB_Classification import SoilClassifierXGB, tune_xgb_classification
from SOCModels.XGBoost_NPK import train_xgb_npk

class TrainingManager:
    def __init__(self, data_dir="SoilScanDataset", log_file="training_history.xlsx"):
        self.data_dir = data_dir
        self.log_file = log_file
        self.img_size = (128, 128)
        self.svm_classifier = SoilClassifierSVM()
        self.extractor = LabColorExtractor()
        self.classifier = SoilClassifierXGB()
        self.calibrator = SoilCalibratorSVM()

    def reset_artifacts(self):
        """Removes previous model and feature files for clean training."""
        for f in ["soil_xgb_model.json", "soil_classes.npy", "extracted_features.npy", "extracted_labels.npy"]:
            if os.path.exists(f): os.remove(f)

    def process_balanced_dataset(self, allowed_exts=(".jpg", ".jpeg", ".png", ".bmp")):
        """Extracts features with class balancing and basic file filtering."""
        print(f"[DATA] Scanning dataset folder: {self.data_dir}")
        class_files = {}
        for class_name in os.listdir(self.data_dir):
            c_path = os.path.join(self.data_dir, class_name)
            if os.path.isdir(c_path):
                files = [
                    os.path.join(c_path, f)
                    for f in os.listdir(c_path)
                    if f.lower().endswith(allowed_exts)
                ]
                if files:
                    class_files[class_name] = files

        if not class_files:
            return np.array([]), np.array([])

        # Balance by downsampling to the smallest class
        min_count = min(len(files) for files in class_files.values())
        print(f"[DATA] Balancing to {min_count} samples per class")
        X, y = [], []
        for class_name, files in class_files.items():
            print(f"[DATA] Processing class '{class_name}' ({len(files)} files)")
            sampled = np.random.choice(files, min_count, replace=False)
            for fpath in sampled:
                img = cv2.imread(fpath)
                if img is None:
                    continue
                img = cv2.resize(img, self.img_size)
                calibrated = self.calibrator.calibrate(img)
                features = self.extractor.extract_features(calibrated)
                X.append(features)
                y.append(class_name)

        print(f"[DATA] Final dataset size: {len(X)} samples")
        return np.array(X), np.array(y)

    def log_to_excel(self, data):
        df = pd.DataFrame([data])
        if os.path.exists(self.log_file):
            df = pd.concat([pd.read_excel(self.log_file), df], ignore_index=True)
        df.to_excel(self.log_file, index=False)

    def train_npk_models(self):
        """Train NPK models using the full pipeline: SVM → CLS → XGBoost NPK"""
        print("\n" + "="*50)
        print("STARTING NPK MODEL TRAINING")
        print("="*50)
        
        csv_path = f"{self.data_dir}/micro-dataset.csv"
        
        # This calls XGBoost_NPK training which internally uses SVM + CLS
        results = train_xgb_npk(
            csv_path=csv_path,
            image_base_dir=self.data_dir,
            image_size=(128, 128)
        )
        
        return results
    
    def execute(self, tune_xgb=False, train_npk=True):
        start = datetime.now()
        self.reset_artifacts()
        X, y = self.process_balanced_dataset()
        
        if len(X) == 0: return
        if tune_xgb:
            tuning = tune_xgb_classification(X, y)
            if tuning is None:
                return
            results = {
                "accuracy": 0.0,
                "macro_f1": tuning["best_macro_f1"],
                "weighted_f1": 0.0,
            }
        else:
            results = self.classifier.train(X, y)  # ← XGB_Classification is used
        
        end = datetime.now()
        log = {
            "Start Time": start.strftime("%Y-%m-%d %H:%M:%S"),
            "End Time": end.strftime("%Y-%m-%d %H:%M:%S"),
            "Duration": str(end - start),
            "Samples": len(X),
            "Accuracy": round(results['accuracy'], 4),
            "Macro F1": round(results['macro_f1'], 4),
            "Weighted F1": round(results['weighted_f1'], 4)
        }
        self.log_to_excel(log)

        print("\n" + "="*50)
        print("METHODOLOGY TRAINING COMPLETE")
        print("-"*50)
        print(f"Total Samples: {len(X)}")
        print(f"Accuracy:      {results['accuracy']*100:.2f}%")
        print(f"Macro F1:      {results['macro_f1']:.4f}")
        print(f"Weighted F1:   {results['weighted_f1']:.4f}")
        print("="*50 + "\n")

        conf_mat = results.get("confusion_matrix")
        class_names = results.get("class_names")

        if conf_mat is not None and class_names is not None:
            print("\nConfusion Matrix:")
            header = " " * 12 + " ".join(f"{name:>10}" for name in class_names)
            print(header)
            for i, row in enumerate(conf_mat):
                row_str = " ".join(f"{val:>10d}" for val in row)
                print(f"{class_names[i]:>10}  {row_str}")


            df_cm = pd.DataFrame(conf_mat, index=class_names, columns=class_names)
            df_cm.to_excel("confusion_matrix.xlsx", index=True)

        if not tune_xgb:
            self.svm_classifier.train(X, y)
            print("SVM model trained and saved to soil_svm.pkl")

        if train_npk:
            print("\n[PIPELINE] Training NPK models...")
            self.train_npk_models()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune-xgb", action="store_true", help="Run XGB hyperparameter tuning")
    parser.add_argument("--npk", action="store_true", default=True, help="Train NPK models")
    args = parser.parse_args()
    
    tm = TrainingManager("SoilScanDataset")
    tm.execute(tune_xgb=args.tune_xgb, train_npk=args.npk)