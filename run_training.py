"""
run_training.py
---------------
Entry point for the full NPK training pipeline.

Pipeline
--------
1. Scan SoilScanDataset/ for images listed in micro-dataset.csv
2. BrightnessCalibrator  – corrects field lighting per image
3. LabColorExtractor     – extracts 79-D feature vector per image
4. XGBClassifier × 3    – one DART-regularised model per nutrient (N, P, K)
5. Save calibrated images to processed_images/ (mirrors original structure)
6. Save models + metadata

Usage
-----
    py run_training.py
    py run_training.py --data-dir path/to/SoilScanDataset
    py run_training.py --csv    path/to/micro-dataset.csv
"""

import argparse
import os
from datetime import datetime

import pandas as pd

from SOCModels.PRXGBoost_NPK import train_xgb_npk


class TrainingManager:
    def __init__(
        self,
        data_dir: str = "SoilScanDataset",
        csv_path: str | None = None,
        log_file: str = "training_history.xlsx",
    ):
        self.data_dir  = data_dir
        self.csv_path  = csv_path or os.path.join(data_dir, "micro-dataset.csv")
        self.log_file  = log_file
        self.img_size  = (128, 128)

        # Calibrated images will be saved here, mirroring the original layout
        self.processed_dir = os.path.join(data_dir, "processed_images")

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log(self, data: dict):
        df = pd.DataFrame([data])
        if os.path.exists(self.log_file):
            df = pd.concat([pd.read_excel(self.log_file), df], ignore_index=True)
        df.to_excel(self.log_file, index=False)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def execute(self):
        print("\n" + "=" * 55)
        print("SOIL NPK TRAINING PIPELINE")
        print("=" * 55)
        print(f"Dataset folder : {self.data_dir}")
        print(f"CSV file       : {self.csv_path}")
        print(f"Processed imgs : {self.processed_dir}")
        print("=" * 55 + "\n")

        start = datetime.now()

        results = train_xgb_npk(
            csv_path       = self.csv_path,
            image_base_dir = self.data_dir,
            image_size     = self.img_size,
            processed_dir  = self.processed_dir,
        )

        end = datetime.now()

        if results is None:
            print("\n[PIPELINE] Training failed — check errors above.")
            return

        # ------------------------------------------------------------------
        # Log summary to Excel
        # ------------------------------------------------------------------
        log_row = {
            "Start Time":  start.strftime("%Y-%m-%d %H:%M:%S"),
            "End Time":    end.strftime("%Y-%m-%d %H:%M:%S"),
            "Duration":    str(end - start),
            "N CV Acc":    round(results["N"]["cv_accuracy"], 4),
            "P CV Acc":    round(results["P"]["cv_accuracy"], 4),
            "K CV Acc":    round(results["K"]["cv_accuracy"], 4),
        }
        self._log(log_row)

        # ------------------------------------------------------------------
        # Final summary
        # ------------------------------------------------------------------
        print("\n" + "=" * 55)
        print("PIPELINE COMPLETE")
        print("-" * 55)
        print(f"Duration  : {end - start}")
        for nut in ("N", "P", "K"):
            info = results[nut]
            print(f"  {nut} model : CV Accuracy = {info['cv_accuracy']*100:.2f}%"
                  f"  →  saved to {info['model_path']}")
        print(f"Processed images saved to: {self.processed_dir}")
        print("=" * 55 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train XGBoost NPK classifiers from soil field photos."
    )
    parser.add_argument(
        "--data-dir", default="SoilScanDataset",
        help="Root folder containing images and micro-dataset.csv "
             "(default: SoilScanDataset)"
    )
    parser.add_argument(
        "--csv", default=None,
        help="Path to CSV file. Defaults to <data-dir>/micro-dataset.csv"
    )
    args = parser.parse_args()

    tm = TrainingManager(data_dir=args.data_dir, csv_path=args.csv)
    tm.execute()