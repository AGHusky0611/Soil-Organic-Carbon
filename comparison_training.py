"""
NP_Training.py
--------------
Compares two XGBoost NPK training pipelines on the same dataset to determine
which image-capture / preprocessing approach produces better classification.

    Pipeline A — PREPROCESSED
        Resize → BrightnessCalibrator → LabColorExtractor (79-D features)

    Pipeline B — RAW PIXELS + PCA
        Resize → flatten BGR pixels → PCA (n_components=50)
        Compresses 49 152-D down to 50-D before XGBoost.
        PCA is fitted once per LOO fold on the training split only — no leakage.

Design
------
Both pipelines are instances of NPKPipeline. The class owns the entire
tuning loop (ParameterGrid × LeaveOneOut CV), so both pipelines are
guaranteed to run through identical code with identical hyperparameters.
The only difference between them is the feature extraction callable
passed into the constructor.

Outputs
-------
    comparison_results.csv        – per-nutrient side-by-side accuracy
    comparison_results.xlsx       – same data as Excel (two sheets)
    NPK_<N/P/K>_<tag>_tuning.csv – per-config LOO log per pipeline
    npk_<n/p/k>_<tag>.json       – retrained final models
"""

import os
import time
from datetime import datetime
from typing import Callable

import cv2
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneOut, ParameterGrid
from sklearn.metrics import accuracy_score, classification_report, f1_score

from ClassificationModels.CLS_extraction import LabColorExtractor
from ClassificationModels.SVM_Calibrator import BrightnessCalibrator

# ---------------------------------------------------------------------------
# Shared config — both pipelines use these exactly
# ---------------------------------------------------------------------------
DEFAULT_IMG_SIZE  = (128, 128)
PCA_COMPONENTS    = 50       # raw-pixel dims after PCA (was 49 152)
LABEL_MAP         = {0: "Low", 1: "Medium", 2: "High"}

PARAM_GRID = {
    "n_estimators":     [50, 100, 150],
    "learning_rate":    [0.01, 0.05, 0.1],
    "max_depth":        [2, 3, 4],
    "rate_drop":        [0.1, 0.3],
    "skip_drop":        [0.5],
    "subsample":        [0.6, 0.7, 0.8],
    "colsample_bytree": [0.5, 0.6, 0.7],
    "reg_alpha":        [0.1, 1.0],
    "reg_lambda":       [1.0, 5.0],
}

XGB_FIXED = {
    "booster":          "dart",
    "objective":        "multi:softprob",
    "eval_metric":      "mlogloss",
    "random_state":     42,
    "min_child_weight": 5,
}


# ---------------------------------------------------------------------------
# Image index helper
# ---------------------------------------------------------------------------

def _build_filename_index(image_base_dir: str) -> dict:
    """
    Walk every subfolder and build:
        bare_filename_lower (no extension) -> absolute path

    os.path.splitext returns a tuple — always index [0] for the name part.
    """
    index = {}
    for root, _, files in os.walk(image_base_dir):
        for fname in files:
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                key = os.path.splitext(fname.lower())[0]
                if key in index:
                    print(f"[IDX] WARNING: duplicate '{fname}' — "
                          f"keeping {os.path.join(root, fname)}")
                index[key] = os.path.join(root, fname)
    return index


# ---------------------------------------------------------------------------
# Feature extraction functions
# ---------------------------------------------------------------------------

def features_preprocessed(
    img_bgr:    np.ndarray,
    image_size: tuple,
    calibrator: BrightnessCalibrator,
    extractor:  LabColorExtractor,
) -> np.ndarray:
    """Resize → BrightnessCalibrator → LabColorExtractor (79-D)."""
    img        = cv2.resize(img_bgr, image_size)
    calibrated = calibrator.calibrate(img)
    return extractor.extract_features(calibrated)


def features_raw_pixels(
    img_bgr:    np.ndarray,
    image_size: tuple,
    **_kwargs,                          # absorbs unused calibrator/extractor args
) -> np.ndarray:
    """Resize → flatten raw BGR pixels, normalised to [0, 1].
    PCA reduction happens after build_features(), not here.
    """
    img = cv2.resize(img_bgr, image_size)
    return img.flatten().astype(np.float64) / 255.0


# ---------------------------------------------------------------------------
# NPKPipeline — single class, both pipelines are instances of this
# ---------------------------------------------------------------------------

class NPKPipeline:
    """
    Encapsulates feature extraction + LOO-CV tuning + final model saving
    for one pipeline variant.

    Parameters
    ----------
    name            : short tag used in filenames and console output
    extract_fn      : callable(img_bgr, image_size, **kwargs) -> np.ndarray
    extract_kwargs  : extra keyword args forwarded to extract_fn
                      (e.g. calibrator / extractor objects for Pipeline A)
    image_size      : (w, h) tuple passed to extract_fn
    """

    def __init__(
        self,
        name:           str,
        extract_fn:     Callable,
        extract_kwargs: dict      = None,
        image_size:     tuple     = DEFAULT_IMG_SIZE,
        pca_components: int       = None,
    ):
        self.name           = name
        self.extract_fn     = extract_fn
        self.extract_kwargs = extract_kwargs or {}
        self.image_size     = image_size
        self.pca_components = pca_components   # if set, PCA is applied per LOO fold
        self.results: dict  = {}               # filled after run()

    # ------------------------------------------------------------------
    # Feature extraction for a list of raw BGR images
    # ------------------------------------------------------------------

    def build_features(self, images: list[np.ndarray]) -> np.ndarray:
        """Extract raw features. PCA (if enabled) is applied inside _tune_nutrient
        per LOO fold on the training split only, to avoid data leakage."""
        feats = [
            self.extract_fn(img, self.image_size, **self.extract_kwargs)
            for img in images
        ]
        return np.array(feats, dtype=np.float64)

    # ------------------------------------------------------------------
    # LOO-CV tuning for a single nutrient
    # ------------------------------------------------------------------

    def _tune_nutrient(
        self,
        X:         np.ndarray,
        y:         np.ndarray,
        nutrient:  str,
        num_class: int,
    ) -> tuple[dict, float, np.ndarray]:
        """
        Grid-search over PARAM_GRID using LeaveOneOut CV.

        Returns best_params, best_loo_accuracy, best_loo_predictions.
        """
        grid        = list(ParameterGrid(PARAM_GRID))
        all_results = []
        best_acc    = -1.0
        best_params = None
        best_preds  = None

        print(f"\n  [{self.name}] Tuning {nutrient} | "
              f"{num_class} classes | {len(grid)} configs")

        for cfg_idx, params in enumerate(grid, start=1):
            pct = cfg_idx / len(grid) * 100
            print(f"\n  {pct:5.1f}% [{cfg_idx}/{len(grid)}] {self.name} | {nutrient} — "
                  f"Running params: {params}")
            t0         = time.time()

            loo        = LeaveOneOut()
            fold_preds = np.zeros(len(y), dtype=np.int32)

            for tr_idx, val_idx in loo.split(X):
                X_tr, X_val = X[tr_idx], X[val_idx]

                # PCA: fit on training split only to avoid leakage
                if self.pca_components:
                    n_comp = min(self.pca_components, X_tr.shape[0],
                                 X_tr.shape[1])
                    pca    = PCA(n_components=n_comp, random_state=42)
                    X_tr   = pca.fit_transform(X_tr)
                    X_val  = pca.transform(X_val)

                clf = xgb.XGBClassifier(
                    num_class=num_class,
                    **params,
                    **XGB_FIXED,
                )
                clf.fit(X_tr, y[tr_idx], verbose=False)
                fold_preds[val_idx] = np.argmax(
                    clf.predict_proba(X_val), axis=1
                )

            duration = time.time() - t0
            acc      = accuracy_score(y, fold_preds)
            macro_f1 = f1_score(y, fold_preds, average="macro",    zero_division=0)
            wtd_f1   = f1_score(y, fold_preds, average="weighted", zero_division=0)

            all_results.append({
                "pipeline":    self.name,
                "nutrient":    nutrient,
                "accuracy":    acc,
                "macro_f1":    macro_f1,
                "weighted_f1": wtd_f1,
                "duration_s":  round(duration, 2),
                **params,
            })

            if acc > best_acc:
                best_acc    = acc
                best_params = params
                best_preds  = fold_preds.copy()

        # Save per-config tuning log
        csv_out = f"NPK_{nutrient}_{self.name}_tuning.csv"
        pd.DataFrame(all_results).to_csv(csv_out, index=False)
        print(f"  [{self.name}] {nutrient} tuning log -> {csv_out}")
        print(f"  [{self.name}] {nutrient} best LOO acc: {best_acc:.4f}")

        return best_params, best_acc, best_preds

    # ------------------------------------------------------------------
    # Main run: tune all three nutrients + save final models
    # ------------------------------------------------------------------

    def run(
        self,
        X:   np.ndarray,
        y_n: np.ndarray,
        y_p: np.ndarray,
        y_k: np.ndarray,
    ) -> dict:
        """
        Tune and retrain N, P, K classifiers.
        Populates and returns self.results.
        """
        print(f"\n{'#'*60}")
        print(f"  PIPELINE : {self.name}")
        print(f"  Features : {X.shape[1]} dims  |  Samples : {X.shape[0]}")
        print(f"{'#'*60}")

        nutrients = [
            ("N", y_n, len(np.unique(y_n))),
            ("P", y_p, len(np.unique(y_p))),
            ("K", y_k, len(np.unique(y_k))),
        ]

        for nutrient, y, num_class in nutrients:

            best_params, best_acc, best_preds = self._tune_nutrient(
                X, y, nutrient, num_class
            )

            # LOO classification report (honest — not training-set predictions)
            class_names = [LABEL_MAP[c] for c in range(num_class)]
            print(f"\n  [{self.name}] {nutrient} — LOO Classification Report:")
            print(classification_report(y, best_preds,
                                        target_names=class_names,
                                        zero_division=0))

            # Retrain final model on 100 % of data with best params
            model_path = f"npk_{nutrient.lower()}_{self.name}.json"
            X_train    = X

            # PCA: fit on full dataset for the saved final model
            if self.pca_components:
                n_comp     = min(self.pca_components, X.shape[0], X.shape[1])
                final_pca  = PCA(n_components=n_comp, random_state=42)
                X_train    = final_pca.fit_transform(X)
                pca_path   = f"npk_{nutrient.lower()}_{self.name}_pca.npz"
                np.savez(pca_path,
                         components=final_pca.components_,
                         mean=final_pca.mean_)
                print(f"  [{self.name}] {nutrient} PCA saved -> {pca_path}")

            final_model = xgb.XGBClassifier(
                num_class=num_class,
                **best_params,
                **XGB_FIXED,
            )
            final_model.fit(X_train, y, verbose=False)
            final_model.save_model(model_path)
            print(f"  [{self.name}] {nutrient} model saved -> {model_path}")

            macro_f1 = f1_score(y, best_preds, average="macro",    zero_division=0)
            wtd_f1   = f1_score(y, best_preds, average="weighted", zero_division=0)

            self.results[nutrient] = {
                "pipeline":     self.name,
                "nutrient":     nutrient,
                "loo_accuracy": round(best_acc,  4),
                "macro_f1":     round(macro_f1,  4),
                "weighted_f1":  round(wtd_f1,    4),
                "feature_dims": n_comp if self.pca_components else X.shape[1],
                "best_params":  str(best_params),
                "model_path":   model_path,
            }

        return self.results


# ---------------------------------------------------------------------------
# Comparison runner
# ---------------------------------------------------------------------------

def run_comparison(
    csv_path       = "SoilScanDataset/micro-dataset.csv",
    image_base_dir = "SoilScanDataset/",
    image_size     = DEFAULT_IMG_SIZE,
    output_csv     = "comparison_results.csv",
    output_xlsx    = "comparison_results.xlsx",
    pipeline       = "both",   # "both" | "pre" | "raw"
):
    """
    1. Load images once.
    2. Build feature matrices for the requested pipeline(s).
    3. Run NPKPipeline instance(s).
    4. Save side-by-side comparison to CSV + Excel (only when pipeline="both").

    pipeline options
    ----------------
    "both"  run Pipeline A (Preprocessed) + Pipeline B (Raw+PCA) and compare
    "pre"   run Pipeline A only
    "raw"   run Pipeline B (Raw+PCA) only
    """

    pipeline = pipeline.lower().strip()
    assert pipeline in ("both", "pre", "raw"), \
        f"--pipeline must be 'both', 'pre', or 'raw', got '{pipeline}'"

    mode_label = {
        "both": "Preprocessed vs Raw Pixels (PCA)",
        "pre":  "Preprocessed only",
        "raw":  "Raw Pixels + PCA only",
    }[pipeline]

    print("\n" + "="*60)
    print(f"  NP_Training — {mode_label}")
    print("="*60)
    print(f"  CSV          : {csv_path}")
    print(f"  Image folder : {image_base_dir}")
    print(f"  Image size   : {image_size}")
    print(f"  Pipeline     : {pipeline}")
    print("="*60)

    # ------------------------------------------------------------------
    # 1. Load CSV
    # ------------------------------------------------------------------
    df       = pd.read_csv(csv_path)
    df_clean = df[df[["n", "p", "k"]].notna().all(axis=1)].copy()
    print(f"\n[NP] Rows total: {len(df)} | Complete NPK: {len(df_clean)}")
    if df_clean.empty:
        print("[NP] ERROR: No rows with complete NPK labels.")
        return None

    # ------------------------------------------------------------------
    # 2. Build image index
    # ------------------------------------------------------------------
    filename_index = _build_filename_index(image_base_dir)
    print(f"[NP] Images found on disk: {len(filename_index)}")

    # ------------------------------------------------------------------
    # 3. Load images + build both feature matrices in one pass
    # ------------------------------------------------------------------
    calibrator = BrightnessCalibrator()
    extractor  = LabColorExtractor()

    raw_images             = []      # store decoded BGR arrays for both pipelines
    y_n, y_p, y_k         = [], [], []
    skipped                = 0

    print("\n[NP] Loading images ...")

    for _, row in df_clean.iterrows():
        img_filename = row["image_filename"]
        bare_name    = os.path.splitext(os.path.basename(img_filename))[0].lower()
        img_path     = filename_index.get(bare_name)

        if img_path is None:
            print(f"  SKIP (not found): {img_filename}")
            skipped += 1
            continue

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"  SKIP (unreadable): {img_path}")
            skipped += 1
            continue

        raw_images.append(img_bgr)
        y_n.append(int(row["n"]))
        y_p.append(int(row["p"]))
        y_k.append(int(row["k"]))

    if not raw_images:
        print("[NP] ERROR: No images loaded.")
        return None

    y_n = np.array(y_n, dtype=np.int32)
    y_p = np.array(y_p, dtype=np.int32)
    y_k = np.array(y_k, dtype=np.int32)
    print(f"[NP] Loaded {len(raw_images)} images (skipped: {skipped})")

    # ------------------------------------------------------------------
    # 4. Instantiate pipelines — only the extract_fn differs
    # ------------------------------------------------------------------
    pipeline_A = NPKPipeline(
        name           = "Preprocessed",
        extract_fn     = features_preprocessed,
        extract_kwargs = {"calibrator": calibrator, "extractor": extractor},
        image_size     = image_size,
    )

    pipeline_B = NPKPipeline(
        name           = "RawPixels",
        extract_fn     = features_raw_pixels,
        image_size     = image_size,
        pca_components = PCA_COMPONENTS,   # 49 152-D -> 50-D per LOO fold
    )

    # ------------------------------------------------------------------
    # 5. Build feature matrices only for requested pipeline(s)
    # ------------------------------------------------------------------
    print("\n[NP] Building feature matrices ...")
    X_A = pipeline_A.build_features(raw_images) if pipeline in ("both", "pre") else None
    X_B = pipeline_B.build_features(raw_images) if pipeline in ("both", "raw") else None
    if X_A is not None:
        print(f"  Pipeline A (Preprocessed) : {X_A.shape[1]} dims")
    if X_B is not None:
        print(f"  Pipeline B (Raw + PCA)    : {X_B.shape[1]} dims raw -> {PCA_COMPONENTS}-D after PCA")

    # ------------------------------------------------------------------
    # 6. Run requested pipeline(s)
    # ------------------------------------------------------------------
    start = datetime.now()

    res_A = pipeline_A.run(X_A, y_n, y_p, y_k) if pipeline in ("both", "pre") else None
    res_B = pipeline_B.run(X_B, y_n, y_p, y_k) if pipeline in ("both", "raw") else None

    end = datetime.now()

    # If only one pipeline was requested, print its summary and return early
    if pipeline != "both":
        res = res_A if pipeline == "pre" else res_B
        tag = "Preprocessed" if pipeline == "pre" else "Raw Pixels + PCA"
        print("\n" + "="*60)
        print(f"  {tag.upper()} TRAINING COMPLETE")
        print("="*60)
        for nut, info in res.items():
            print(f"  {nut}: LOO-CV Accuracy = {info['loo_accuracy']*100:.2f}%"
                  f"  ->  {info['model_path']}")
        print(f"  Duration: {end - start}")
        print("="*60 + "\n")
        return res

    # ------------------------------------------------------------------
    # 7. Build comparison table
    # ------------------------------------------------------------------
    rows = []
    for nutrient in ("N", "P", "K"):
        a = res_A[nutrient]
        b = res_B[nutrient]

        if   a["loo_accuracy"] > b["loo_accuracy"]: winner = "A — Preprocessed"
        elif b["loo_accuracy"] > a["loo_accuracy"]: winner = "B — Raw Pixels"
        else:                                        winner = "Tie"

        rows.append({
            "Nutrient":        nutrient,
            "A_LOO_Accuracy":  a["loo_accuracy"],
            "B_LOO_Accuracy":  b["loo_accuracy"],
            "A_Macro_F1":      a["macro_f1"],
            "B_Macro_F1":      b["macro_f1"],
            "A_Weighted_F1":   a["weighted_f1"],
            "B_Weighted_F1":   b["weighted_f1"],
            "A_Feature_Dims":  a["feature_dims"],
            "B_Feature_Dims":  b["feature_dims"],
            "Winner":          winner,
            "A_Best_Params":   a["best_params"],
            "B_Best_Params":   b["best_params"],
        })

    mean_A = float(np.mean([res_A[n]["loo_accuracy"] for n in ("N", "P", "K")]))
    mean_B = float(np.mean([res_B[n]["loo_accuracy"] for n in ("N", "P", "K")]))

    rows.append({
        "Nutrient":       "OVERALL (mean)",
        "A_LOO_Accuracy": round(mean_A, 4),
        "B_LOO_Accuracy": round(mean_B, 4),
        "Winner":         "A — Preprocessed" if mean_A >= mean_B else "B — Raw Pixels",
    })

    comparison_df = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # 8. Save CSV + Excel
    # ------------------------------------------------------------------
    comparison_df.to_csv(output_csv, index=False)
    print(f"\n[NP] Comparison CSV   -> {output_csv}")

    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        comparison_df.to_excel(writer, sheet_name="Comparison", index=False)
        detail_rows = list(res_A.values()) + list(res_B.values())
        pd.DataFrame(detail_rows).to_excel(writer, sheet_name="Detail", index=False)

    print(f"[NP] Comparison Excel -> {output_xlsx}")

    # ------------------------------------------------------------------
    # 9. Console summary
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("  COMPARISON SUMMARY")
    print("="*60)
    print(f"  {'Nutrient':<16} {'A (Pre)':>10} {'B (Raw)':>10}   Winner")
    print("  " + "-"*54)
    for row in rows:
        a_s = f"{row['A_LOO_Accuracy']:.4f}" if isinstance(row.get("A_LOO_Accuracy"), float) else ""
        b_s = f"{row['B_LOO_Accuracy']:.4f}" if isinstance(row.get("B_LOO_Accuracy"), float) else ""
        print(f"  {row['Nutrient']:<16} {a_s:>10} {b_s:>10}   {row['Winner']}")

    print(f"\n  Duration      : {end - start}")
    print(f"  Overall winner: {'A — Preprocessed' if mean_A >= mean_B else 'B — Raw Pixels'}")
    print("="*60 + "\n")

    return comparison_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare preprocessed vs raw-pixel XGBoost NPK pipelines."
    )
    parser.add_argument("--data-dir",     default="SoilScanDataset")
    parser.add_argument("--csv",          default=None)
    parser.add_argument("--output-csv",   default="comparison_results.csv")
    parser.add_argument("--output-xlsx",  default="comparison_results.xlsx")
    parser.add_argument(
        "--pipeline", default="both",
        choices=["both", "pre", "raw"],
        help=(
            "Which pipeline to run: "
            "'both' (default) runs both and saves comparison, "
            "'pre' runs Preprocessed only, "
            "'raw' runs Raw Pixels + PCA only."
        )
    )
    args = parser.parse_args()

    csv_path = args.csv or os.path.join(args.data_dir, "micro-dataset.csv")

    run_comparison(
        csv_path       = csv_path,
        image_base_dir = args.data_dir,
        output_csv     = args.output_csv,
        output_xlsx    = args.output_xlsx,
        pipeline       = args.pipeline,
    )