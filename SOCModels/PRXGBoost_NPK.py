"""
XGBoost_NPK.py
--------------
Trains three independent XGBoost classifiers to predict soil nutrient
levels (N, P, K) from field photos.

NPK labels are ordinal categories:
    0 = Low  |  1 = Medium  |  2 = High

Pipeline per image
------------------
1. Resize to IMAGE_SIZE
2. BrightnessCalibrator  – corrects field lighting variation
3. LabColorExtractor     – GLCM texture (72-D) + LAB stats (7-D) = 79-D vector
4. XGBClassifier × 3    – one model per nutrient (N, P, K)

Tuning strategy
---------------
- ParameterGrid search over DART hyperparameters (mirrors SOC tuning)
- LeaveOneOut CV for reliable estimates on small datasets
- Best config per nutrient retrained on 100% of data
- Per-nutrient tuning results saved to CSV
"""

import json
import os
import time

import cv2
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import LeaveOneOut, ParameterGrid
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder

from ClassificationModels.CLS_extraction import LabColorExtractor
from ClassificationModels.SVM_Calibrator import BrightnessCalibrator

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_N_MODEL  = "npk_n_model.json"
DEFAULT_P_MODEL  = "npk_p_model.json"
DEFAULT_K_MODEL  = "npk_k_model.json"
DEFAULT_META     = "npk_meta.json"
DEFAULT_IMG_SIZE = (128, 128)

LABEL_MAP = {0: "Low", 1: "Medium", 2: "High"}

# ---------------------------------------------------------------------------
# Parameter grid (mirrors SOC tuning style)
# ---------------------------------------------------------------------------
PARAM_GRID = {
    "n_estimators":     [150, 250, 350],
    "learning_rate":    [0.05, 0.1],
    "max_depth":        [4, 6],
    "rate_drop":        [0.1, 0.2],
    "skip_drop":        [0.5],
    "subsample":        [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8],
}

XGB_FIXED = {
    "booster":          "dart",
    "objective":        "multi:softprob",
    "eval_metric":      "mlogloss",
    "random_state":     42,
    "min_child_weight": 2,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_meta(path: str, img_size: tuple, best_params: dict):
    payload = {
        "image_size":        list(img_size),
        "feature_extractor": "BrightnessCalibrator + LabColorExtractor",
        "booster":           "dart",
        "label_map":         LABEL_MAP,
        "best_params":       best_params,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


# ---------------------------------------------------------------------------
# Predictor (inference)
# ---------------------------------------------------------------------------

class NPKPredictor:
    """Load saved models and predict N / P / K level for a single image."""

    def __init__(
        self,
        model_n_path = DEFAULT_N_MODEL,
        model_p_path = DEFAULT_P_MODEL,
        model_k_path = DEFAULT_K_MODEL,
        meta_path    = DEFAULT_META,
    ):
        for path in (model_n_path, model_p_path, model_k_path):
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model not found: '{path}'")

        self.model_n = xgb.XGBClassifier(); self.model_n.load_model(model_n_path)
        self.model_p = xgb.XGBClassifier(); self.model_p.load_model(model_p_path)
        self.model_k = xgb.XGBClassifier(); self.model_k.load_model(model_k_path)

        self.img_size   = DEFAULT_IMG_SIZE
        self.calibrator = BrightnessCalibrator()
        self.extractor  = LabColorExtractor()

        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            self.img_size = tuple(meta.get("image_size", list(DEFAULT_IMG_SIZE)))

    def predict(self, img_bgr: np.ndarray) -> dict:
        """
        Returns:
            {"N": "Low"|"Medium"|"High", "P": ..., "K": ...}
        """
        img        = cv2.resize(img_bgr, self.img_size)
        calibrated = self.calibrator.calibrate(img)
        feat       = self.extractor.extract_features(calibrated).reshape(1, -1)

        n_idx = int(np.argmax(self.model_n.predict_proba(feat), axis=1))
        p_idx = int(np.argmax(self.model_p.predict_proba(feat), axis=1))
        k_idx = int(np.argmax(self.model_k.predict_proba(feat), axis=1))

        return {
            "N": LABEL_MAP.get(n_idx, str(n_idx)),
            "P": LABEL_MAP.get(p_idx, str(p_idx)),
            "K": LABEL_MAP.get(k_idx, str(k_idx)),
        }


# ---------------------------------------------------------------------------
# Image lookup
# ---------------------------------------------------------------------------

def _build_filename_index(image_base_dir: str) -> dict[str, str]:
    """
    Walk every subfolder of *image_base_dir* and build a dict mapping
    bare filename (without extension) -> absolute path on disk.
    """
    index: dict[str, str] = {}
    for root, _, files in os.walk(image_base_dir):
        for fname in files:
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                key = os.path.splitext(fname.lower())[0]
                if key in index:
                    print(f"[NPK] WARNING: duplicate filename '{fname}' — "
                          f"keeping {os.path.join(root, fname)}")
                index[key] = os.path.join(root, fname)
    return index


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_xgb_npk(
    csv_path        = "SoilScanDataset/micro-dataset.csv",
    image_base_dir  = "SoilScanDataset/",
    image_size      = DEFAULT_IMG_SIZE,
    model_n_path    = DEFAULT_N_MODEL,
    model_p_path    = DEFAULT_P_MODEL,
    model_k_path    = DEFAULT_K_MODEL,
    meta_path       = DEFAULT_META,
    processed_dir   = None,
):
    """
    Full training pipeline.

    1. Load CSV and filter rows with complete N / P / K labels.
    2. Build a filename -> disk-path index by walking image_base_dir.
    3. For each image: resize -> calibrate -> extract features.
    4. ParameterGrid search with LeaveOneOut CV for each nutrient.
    5. Retrain best config on 100% of data and save.
    6. Save per-nutrient tuning results to CSV.

    Returns dict with per-nutrient best CV accuracy, or None on failure.
    """

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    print(f"\n[NPK] Loading dataset from {csv_path} ...")
    df = pd.read_csv(csv_path)
    df_clean = df[df[["n", "p", "k"]].notna().all(axis=1)].copy()
    print(f"[NPK] Total rows: {len(df)} | Rows with complete NPK: {len(df_clean)}")

    if df_clean.empty:
        print("[NPK] ERROR: No rows with complete NPK data.")
        return None

    # ------------------------------------------------------------------
    # 2. Build filename -> disk-path index
    # ------------------------------------------------------------------
    print(f"[NPK] Scanning image folder: {image_base_dir} ...")
    filename_index = _build_filename_index(image_base_dir)
    print(f"[NPK] Found {len(filename_index)} images on disk.")

    # ------------------------------------------------------------------
    # 3. Feature extraction
    # ------------------------------------------------------------------
    calibrator = BrightnessCalibrator()
    extractor  = LabColorExtractor()

    X, y_n, y_p, y_k = [], [], [], []
    skipped = 0

    print(f"[NPK] Extracting features ...")

    for _, row in df_clean.iterrows():
        img_filename = row["image_filename"]
        bare_name    = os.path.splitext(os.path.basename(img_filename))[0].lower()
        img_path     = filename_index.get(bare_name)

        if img_path is None:
            print(f"[NPK] SKIP (not found): {img_filename}")
            skipped += 1
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"[NPK] SKIP (unreadable): {img_path}")
            skipped += 1
            continue

        img        = cv2.resize(img, image_size)
        calibrated = calibrator.calibrate(img)
        features   = extractor.extract_features(calibrated)

        # Save processed image mirroring class-folder structure
        if processed_dir is not None:
            rel_to_base  = os.path.relpath(img_path, image_base_dir)
            class_folder = rel_to_base.split(os.sep)[0]
            out_dir      = os.path.join(processed_dir, class_folder)
            os.makedirs(out_dir, exist_ok=True)
            base         = os.path.splitext(os.path.basename(img_path))[0]
            out_path     = os.path.join(out_dir, f"{base}_processed.jpg")
            cv2.imwrite(out_path, calibrated)

        X.append(features)
        y_n.append(int(row["n"]))
        y_p.append(int(row["p"]))
        y_k.append(int(row["k"]))

    if not X:
        print("[NPK] ERROR: No images could be loaded.")
        return None

    X   = np.array(X,   dtype=np.float64)
    y_n = np.array(y_n, dtype=np.int32)
    y_p = np.array(y_p, dtype=np.int32)
    y_k = np.array(y_k, dtype=np.int32)

    print(f"[NPK] Features shape: {X.shape} | Skipped: {skipped}")

    # ------------------------------------------------------------------
    # 4. ParameterGrid tuning with LeaveOneOut CV
    # ------------------------------------------------------------------
    results = {}
    grid    = list(ParameterGrid(PARAM_GRID))
    all_best_params = {}

    nutrients = [
        ("N", y_n, model_n_path, len(np.unique(y_n))),
        ("P", y_p, model_p_path, len(np.unique(y_p))),
        ("K", y_k, model_k_path, len(np.unique(y_k))),
    ]

    for nutrient, y, model_path, num_class in nutrients:
        print(f"\n{'='*55}")
        print(f"[NPK] Tuning  {nutrient}  classifier  "
              f"({num_class} classes, LOO-CV)")
        print(f"[TUNE] Total configs: {len(grid)}")
        print(f"{'='*55}")

        all_results = []
        best_acc    = -1
        best_params = None

        for params in grid:
            print(f"[TUNE] Running params: {params}")
            start_time = time.time()

            loo        = LeaveOneOut()
            fold_preds = np.zeros(len(y), dtype=np.int32)

            for tr_idx, val_idx in loo.split(X):
                X_tr, X_val = X[tr_idx], X[val_idx]
                y_tr        = y[tr_idx]

                model = xgb.XGBClassifier(
                    num_class=num_class,
                    **params,
                    **XGB_FIXED,
                )
                model.fit(X_tr, y_tr, verbose=False)
                fold_preds[val_idx] = np.argmax(
                    model.predict_proba(X_val), axis=1
                )

            duration = time.time() - start_time
            acc      = accuracy_score(y, fold_preds)
            macro_f1 = f1_score(y, fold_preds, average="macro",    zero_division=0)
            wtd_f1   = f1_score(y, fold_preds, average="weighted", zero_division=0)

            print(f"[XGB] Accuracy: {acc:.4f} | "
                  f"Macro F1: {macro_f1:.4f} | "
                  f"Weighted F1: {wtd_f1:.4f} | "
                  f"Duration: {duration:.2f}s")
            print("-" * 50)

            all_results.append({
                "nutrient":     nutrient,
                "accuracy":     acc,
                "macro_f1":     macro_f1,
                "weighted_f1":  wtd_f1,
                "duration_sec": duration,
                **params,
            })

            if acc > best_acc:
                best_acc    = acc
                best_params = params

        # Save tuning results to CSV (mirrors SOC)
        results_df = pd.DataFrame(all_results)
        csv_out    = f"NPK_{nutrient}_tuning_results.csv"
        results_df.to_csv(csv_out, index=False)
        print(f"\n[NPK] {nutrient} tuning results saved -> {csv_out}")

        # Print best config found
        print(f"[NPK] Best {nutrient} params  : {best_params}")
        print(f"[NPK] Best {nutrient} LOO-CV accuracy: {best_acc:.4f}")

        # --------------------------------------------------------------
        # 5. Retrain on 100% of data with best params
        # --------------------------------------------------------------
        print(f"[NPK] Retraining {nutrient} on full data ...")
        final_model = xgb.XGBClassifier(
            num_class=num_class,
            **best_params,
            **XGB_FIXED,
        )
        final_model.fit(X, y, verbose=False)
        final_model.save_model(model_path)
        print(f"[NPK] Saved {nutrient} model -> {model_path}")

        class_names = [LABEL_MAP[c] for c in range(num_class)]
        preds_full  = np.argmax(final_model.predict_proba(X), axis=1)
        print(classification_report(y, preds_full,
                                    target_names=class_names,
                                    zero_division=0))

        all_best_params[nutrient] = best_params
        results[nutrient] = {
            "cv_accuracy": best_acc,
            "best_params": best_params,
            "model_path":  model_path,
        }

    # ------------------------------------------------------------------
    # 6. Save metadata
    # ------------------------------------------------------------------
    _save_meta(meta_path, image_size, all_best_params)
    print(f"\n[NPK] Metadata saved -> {meta_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 55)
    print("NPK TRAINING COMPLETE")
    print("=" * 55)
    for nut, info in results.items():
        print(f"  {nut}: LOO-CV Accuracy = {info['cv_accuracy']*100:.2f}%  "
              f"->  saved to {info['model_path']}")
        print(f"  {nut}: Best params = {info['best_params']}")

    return results