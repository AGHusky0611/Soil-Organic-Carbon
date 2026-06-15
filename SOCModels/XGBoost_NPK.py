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

Anti-memorisation strategy
--------------------------
- DART booster with rate_drop / skip_drop randomly drops previous trees
- subsample + colsample_bytree drop rows and features per tree
- Early stopping on a held-out validation fold
- Stratified K-Fold cross-validation (5 folds) for reliable estimates
- Final models are retrained on 100 % of data using the best epoch found
"""

import json
import os

import cv2
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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
# XGBoost hyperparameters
# ---------------------------------------------------------------------------
XGB_PARAMS = {
    # --- DART dropout (anti-memorisation) ---
    "booster":       "dart",
    "rate_drop":     0.15,   # randomly drop 15 % of previous trees each round
    "skip_drop":     0.50,   # 50 % chance to skip dropout in a round

    # --- row / feature sub-sampling (additional regularisation) ---
    "subsample":        0.80,
    "colsample_bytree": 0.75,

    # --- tree complexity ---
    "max_depth":        4,
    "min_child_weight": 2,
    "learning_rate":    0.05,

    # --- general ---
    "n_estimators":  300,    # upper bound; early stopping kicks in earlier
    "objective":     "multi:softprob",
    "eval_metric":   "mlogloss",
    "random_state":  42,
    "use_label_encoder": False,
}

N_FOLDS        = 5
EARLY_STOP_RND = 30
TEST_SIZE      = 0.20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_model(num_class: int) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(num_class=num_class, **XGB_PARAMS)


def _print_fold_report(nutrient: str, fold: int, y_true, y_pred, classes):
    acc = accuracy_score(y_true, y_pred)
    print(f"  [Fold {fold}] {nutrient} accuracy: {acc:.4f}")


def _save_meta(path: str, img_size: tuple):
    payload = {
        "image_size":       list(img_size),
        "feature_extractor": "BrightnessCalibrator + LabColorExtractor",
        "booster":           "dart",
        "label_map":         LABEL_MAP,
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

        n_idx = int(self.model_n.predict(feat)[0])
        p_idx = int(self.model_p.predict(feat)[0])
        k_idx = int(self.model_k.predict(feat)[0])

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
    bare filename (e.g. 'atok_paoay_20260213_020214_ed6fc014.jpg')
    → absolute path on disk.

    The CSV stores paths like 'images/atok/paoay/.../filename.jpg' which
    don't match the actual flat layout on disk (Atok/ and LaTrinidad/).
    Matching by filename alone is more robust than trying to translate
    the CSV paths.

    If two files share the same basename (unlikely given the UUID suffix),
    the last one wins and a warning is printed.
    """
    index: dict[str, str] = {}
    for root, _, files in os.walk(image_base_dir):
        for fname in files:
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                key = fname.lower()
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
    processed_dir   = None,          # folder to save calibrated images
):
    """
    Full training pipeline.

    1. Load CSV and filter rows with complete N / P / K labels.
    2. Build a filename → disk-path index by walking image_base_dir.
    3. For each image: resize → calibrate → extract features.
       Optionally save the calibrated image to *processed_dir* mirroring
       the class-folder structure (Atok/ or LaTrinidad/).
    4. Train 3 XGBClassifier models (N, P, K) using Stratified K-Fold
       cross-validation + DART dropout.
    5. Retrain each model on 100 % of data at the median best epoch.
    6. Save models + metadata.

    Returns dict with per-nutrient CV accuracy, or None on failure.
    """

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    print(f"\n[NPK] Loading dataset from {csv_path} …")
    df = pd.read_csv(csv_path)
    df_clean = df[df[["n", "p", "k"]].notna().all(axis=1)].copy()
    print(f"[NPK] Total rows: {len(df)} | Rows with complete NPK: {len(df_clean)}")

    if df_clean.empty:
        print("[NPK] ERROR: No rows with complete NPK data.")
        return None

    # ------------------------------------------------------------------
    # 2. Build filename → disk-path index
    # ------------------------------------------------------------------
    print(f"[NPK] Scanning image folder: {image_base_dir} …")
    filename_index = _build_filename_index(image_base_dir)
    print(f"[NPK] Found {len(filename_index)} images on disk.")

    # ------------------------------------------------------------------
    # 3. Feature extraction
    # ------------------------------------------------------------------
    calibrator = BrightnessCalibrator()
    extractor  = LabColorExtractor()

    X, y_n, y_p, y_k = [], [], [], []
    skipped = 0

    print(f"[NPK] Extracting features …")

    for _, row in df_clean.iterrows():
        img_filename = row["image_filename"]
        # Match by bare filename only — ignores CSV directory prefix
        bare_name = os.path.basename(img_filename).lower()
        img_path  = filename_index.get(bare_name)

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

        # -- save processed image (mirrors class-folder structure) ------
        if processed_dir is not None:
            # Determine which class folder the image lives in
            # e.g. …/SoilScanDataset/Atok/filename.jpg  →  class_folder = "Atok"
            rel_to_base = os.path.relpath(img_path, image_base_dir)
            class_folder = rel_to_base.split(os.sep)[0]   # "Atok" or "LaTrinidad"
            out_dir  = os.path.join(processed_dir, class_folder)
            os.makedirs(out_dir, exist_ok=True)
            base     = os.path.splitext(os.path.basename(img_path))[0]
            out_path = os.path.join(out_dir, f"{base}_processed.jpg")
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
    # 3. Cross-validated training (DART + early stopping)
    # ------------------------------------------------------------------
    results = {}

    nutrients = [
        ("N", y_n, model_n_path, len(np.unique(y_n))),
        ("P", y_p, model_p_path, len(np.unique(y_p))),
        ("K", y_k, model_k_path, len(np.unique(y_k))),
    ]

    for nutrient, y, model_path, num_class in nutrients:
        print(f"\n{'='*55}")
        print(f"[NPK] Training  {nutrient}  classifier  "
              f"({num_class} classes, {N_FOLDS}-fold CV)")
        print(f"{'='*55}")

        skf        = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
        fold_accs  = []
        best_iters = []

        for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), start=1):
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]

            model = _build_model(num_class)
            es    = xgb.callback.EarlyStopping(
                rounds        = EARLY_STOP_RND,
                metric_name   = "mlogloss",
                data_name     = "validation_0",
                save_best     = True,
            )
            model.set_params(callbacks=[es])

            model.fit(
                X_tr, y_tr,
                eval_set = [(X_val, y_val)],
                verbose  = False,
            )

            best_iters.append(model.best_iteration)
            preds = model.predict(X_val)
            _print_fold_report(nutrient, fold, y_val, preds,
                               [LABEL_MAP[c] for c in range(num_class)])
            fold_accs.append(accuracy_score(y_val, preds))

        cv_acc      = float(np.mean(fold_accs))
        median_iter = int(np.median(best_iters))
        print(f"\n[NPK] {nutrient} CV accuracy: {cv_acc:.4f} "
              f"| Median best epoch: {median_iter}")

        # --------------------------------------------------------------
        # 4. Retrain on 100 % of data at the median best epoch
        # --------------------------------------------------------------
        print(f"[NPK] Retraining {nutrient} on full data "
              f"({median_iter} rounds) …")

        final_model = xgb.XGBClassifier(
            num_class    = num_class,
            n_estimators = median_iter if median_iter > 0 else 50,
            **{k: v for k, v in XGB_PARAMS.items()
               if k not in ("n_estimators",)},
        )
        # DART does not support early stopping on full data — train fixed rounds
        final_model.fit(X, y, verbose=False)
        final_model.save_model(model_path)
        print(f"[NPK] Saved {nutrient} model → {model_path}")

        # Confusion matrix on full data (training summary, not a test metric)
        preds_full = final_model.predict(X)
        class_names = [LABEL_MAP[c] for c in range(num_class)]
        print(classification_report(y, preds_full,
                                    target_names=class_names,
                                    zero_division=0))

        results[nutrient] = {
            "cv_accuracy":    cv_acc,
            "median_epoch":   median_iter,
            "fold_accuracies": fold_accs,
            "model_path":     model_path,
        }

    # ------------------------------------------------------------------
    # 5. Save metadata
    # ------------------------------------------------------------------
    _save_meta(meta_path, image_size)
    print(f"\n[NPK] Metadata saved → {meta_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "="*55)
    print("NPK TRAINING COMPLETE")
    print("="*55)
    for nut, info in results.items():
        print(f"  {nut}: CV Accuracy = {info['cv_accuracy']*100:.2f}%  "
              f"(best epoch ≈ {info['median_epoch']})")

    return results