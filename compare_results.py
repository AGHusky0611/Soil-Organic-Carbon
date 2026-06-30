"""
compare_results.py
------------------
Runs Pipeline B (Raw Pixels + PCA) and combines its results with
already-completed Pipeline A (Preprocessed) tuning CSVs to produce
a side-by-side comparison without retraining Pipeline A.

Usage
-----
    # Run raw pipeline and compare against existing preprocessed results
    py compare_results.py

    # Custom paths
    py compare_results.py --data-dir path/to/SoilScanDataset
    py compare_results.py --pre-dir  path/to/preprocessed_csvs

Expected Pipeline A CSVs (from NP_Training.py or PRXGBoost_NPK.py)
-------------------------------------------------------------------
    NPK_N_Preprocessed_tuning.csv
    NPK_P_Preprocessed_tuning.csv
    NPK_K_Preprocessed_tuning.csv

Outputs
-------
    NPK_N_RawPixels_tuning.csv   – raw pipeline per-config LOO log
    NPK_P_RawPixels_tuning.csv
    NPK_K_RawPixels_tuning.csv
    npk_n_RawPixels.json         – final retrained raw models
    npk_p_RawPixels.json
    npk_k_RawPixels.json
    comparison_results.csv       – side-by-side accuracy table
    comparison_results.xlsx      – same as Excel (two sheets)
"""

import argparse
import os
import time
from datetime import datetime

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
# Config — must match what Pipeline A was trained with
# ---------------------------------------------------------------------------
DEFAULT_IMG_SIZE = (128, 128)
PCA_COMPONENTS   = 50
LABEL_MAP        = {0: "Low", 1: "Medium", 2: "High"}

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
# Image index
# ---------------------------------------------------------------------------

def _build_filename_index(image_base_dir: str) -> dict:
    index = {}
    for root, _, files in os.walk(image_base_dir):
        for fname in files:
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                key = os.path.splitext(fname.lower())[0]
                if key in index:
                    print(f"[IDX] WARNING: duplicate '{fname}'")
                index[key] = os.path.join(root, fname)
    return index


# ---------------------------------------------------------------------------
# Pipeline A — read best result from existing CSV
# ---------------------------------------------------------------------------

def load_pipeline_a_results(pre_dir: str) -> dict | None:
    """
    Read the best config per nutrient from existing Pipeline A tuning CSVs.
    Returns a dict keyed by nutrient ("N", "P", "K") or None if any CSV
    is missing.
    """
    results = {}
    for nutrient in ("N", "P", "K"):
        csv_path = os.path.join(pre_dir, f"NPK_{nutrient}_Preprocessed_tuning.csv")

        if not os.path.exists(csv_path):
            print(f"[PRE] ERROR: Missing Pipeline A CSV: {csv_path}")
            print(f"      Run NP_Training.py --pipeline pre first, or use --pipeline both.")
            return None

        df      = pd.read_csv(csv_path)
        best    = df.loc[df["accuracy"].idxmax()]
        acc     = float(best["accuracy"])
        macro   = float(best["macro_f1"])
        wtd     = float(best["weighted_f1"])

        # Reconstruct best_params dict from CSV columns
        param_cols = [c for c in df.columns
                      if c not in ("pipeline", "nutrient", "accuracy",
                                   "macro_f1", "weighted_f1", "duration_s")]
        best_params = {c: best[c] for c in param_cols if c in best.index}

        results[nutrient] = {
            "pipeline":     "Preprocessed",
            "nutrient":     nutrient,
            "loo_accuracy": round(acc,   4),
            "macro_f1":     round(macro, 4),
            "weighted_f1":  round(wtd,   4),
            "best_params":  str(best_params),
            "source":       csv_path,
        }
        print(f"[PRE] {nutrient}: best LOO acc = {acc:.4f}  (from {csv_path})")

    return results


# ---------------------------------------------------------------------------
# Pipeline B — train raw pixels + PCA
# ---------------------------------------------------------------------------

def run_pipeline_b(
    raw_images: list,
    y_n: np.ndarray,
    y_p: np.ndarray,
    y_k: np.ndarray,
) -> dict:
    """
    Run full ParameterGrid × LOO-CV tuning for Pipeline B (Raw + PCA).
    Returns per-nutrient results dict.
    """
    grid      = list(ParameterGrid(PARAM_GRID))
    n_configs = len(grid)
    results   = {}

    nutrients = [
        ("N", y_n, len(np.unique(y_n))),
        ("P", y_p, len(np.unique(y_p))),
        ("K", y_k, len(np.unique(y_k))),
    ]

    # Build raw feature matrix once
    print("\n[RAW] Building raw pixel feature matrix ...")
    X_raw = np.array([
        (cv2.resize(img, DEFAULT_IMG_SIZE).flatten().astype(np.float64) / 255.0)
        for img in raw_images
    ])
    print(f"[RAW] Raw feature shape: {X_raw.shape}  ->  PCA to {PCA_COMPONENTS}-D per fold")

    for nutrient, y, num_class in nutrients:
        print(f"\n{'='*60}")
        print(f"[RAW] Tuning {nutrient} | {num_class} classes | {n_configs} configs")
        print(f"{'='*60}")

        all_results = []
        best_acc    = -1.0
        best_params = None
        best_preds  = None

        for cfg_idx, params in enumerate(grid, start=1):
            pct = cfg_idx / n_configs * 100
            print(f"\n  {pct:5.1f}% [{cfg_idx}/{n_configs}] {nutrient} — "
                  f"params: {params}")
            t0 = time.time()

            loo        = LeaveOneOut()
            fold_preds = np.zeros(len(y), dtype=np.int32)

            for tr_idx, val_idx in loo.split(X_raw):
                X_tr, X_val = X_raw[tr_idx], X_raw[val_idx]

                # PCA fitted on training split only — no leakage
                n_comp = min(PCA_COMPONENTS, X_tr.shape[0], X_tr.shape[1])
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

            print(f"  [RAW] Acc: {acc:.4f} | Macro F1: {macro_f1:.4f} | "
                  f"Weighted F1: {wtd_f1:.4f} | {duration:.1f}s")

            all_results.append({
                "pipeline":    "RawPixels",
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

        # Save tuning log
        csv_out = f"NPK_{nutrient}_RawPixels_tuning.csv"
        pd.DataFrame(all_results).to_csv(csv_out, index=False)
        print(f"\n[RAW] {nutrient} tuning log -> {csv_out}")
        print(f"[RAW] {nutrient} best LOO acc: {best_acc:.4f} | params: {best_params}")

        # LOO classification report
        class_names = [LABEL_MAP[c] for c in range(num_class)]
        print(f"\n[RAW] {nutrient} — LOO Classification Report:")
        print(classification_report(y, best_preds,
                                    target_names=class_names,
                                    zero_division=0))

        # Retrain final model on 100% of data
        model_path = f"npk_{nutrient.lower()}_RawPixels.json"
        n_comp     = min(PCA_COMPONENTS, X_raw.shape[0], X_raw.shape[1])
        final_pca  = PCA(n_components=n_comp, random_state=42)
        X_train    = final_pca.fit_transform(X_raw)

        pca_path = f"npk_{nutrient.lower()}_RawPixels_pca.npz"
        np.savez(pca_path,
                 components=final_pca.components_,
                 mean=final_pca.mean_)

        final_model = xgb.XGBClassifier(
            num_class=num_class,
            **best_params,
            **XGB_FIXED,
        )
        final_model.fit(X_train, y, verbose=False)
        final_model.save_model(model_path)
        print(f"[RAW] {nutrient} model saved  -> {model_path}")
        print(f"[RAW] {nutrient} PCA saved    -> {pca_path}")

        macro_f1 = f1_score(y, best_preds, average="macro",    zero_division=0)
        wtd_f1   = f1_score(y, best_preds, average="weighted", zero_division=0)

        results[nutrient] = {
            "pipeline":     "RawPixels",
            "nutrient":     nutrient,
            "loo_accuracy": round(best_acc,  4),
            "macro_f1":     round(macro_f1,  4),
            "weighted_f1":  round(wtd_f1,    4),
            "best_params":  str(best_params),
        }

    return results


# ---------------------------------------------------------------------------
# Comparison table builder
# ---------------------------------------------------------------------------

def build_comparison(
    res_A: dict,
    res_B: dict,
    output_csv:  str,
    output_xlsx: str,
):
    rows = []
    for nutrient in ("N", "P", "K"):
        a = res_A[nutrient]
        b = res_B[nutrient]

        if   a["loo_accuracy"] > b["loo_accuracy"]: winner = "A — Preprocessed"
        elif b["loo_accuracy"] > a["loo_accuracy"]: winner = "B — Raw Pixels"
        else:                                        winner = "Tie"

        rows.append({
            "Nutrient":       nutrient,
            "A_LOO_Accuracy": a["loo_accuracy"],
            "B_LOO_Accuracy": b["loo_accuracy"],
            "A_Macro_F1":     a["macro_f1"],
            "B_Macro_F1":     b["macro_f1"],
            "A_Weighted_F1":  a["weighted_f1"],
            "B_Weighted_F1":  b["weighted_f1"],
            "Winner":         winner,
            "A_Best_Params":  a["best_params"],
            "B_Best_Params":  b["best_params"],
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
    comparison_df.to_csv(output_csv, index=False)
    print(f"\n[CMP] Comparison CSV   -> {output_csv}")

    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        comparison_df.to_excel(writer, sheet_name="Comparison", index=False)
        detail = list(res_A.values()) + list(res_B.values())
        pd.DataFrame(detail).to_excel(writer, sheet_name="Detail", index=False)
    print(f"[CMP] Comparison Excel -> {output_xlsx}")

    # Console summary
    print("\n" + "="*60)
    print("  COMPARISON SUMMARY")
    print("="*60)
    print(f"  {'Nutrient':<16} {'A (Pre)':>10} {'B (Raw)':>10}   Winner")
    print("  " + "-"*54)
    for row in rows:
        a_s = f"{row['A_LOO_Accuracy']:.4f}" if isinstance(row.get("A_LOO_Accuracy"), float) else ""
        b_s = f"{row['B_LOO_Accuracy']:.4f}" if isinstance(row.get("B_LOO_Accuracy"), float) else ""
        print(f"  {row['Nutrient']:<16} {a_s:>10} {b_s:>10}   {row['Winner']}")

    overall = "A — Preprocessed" if mean_A >= mean_B else "B — Raw Pixels"
    print(f"\n  Overall winner: {overall}")
    print("="*60 + "\n")

    return comparison_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    csv_path:    str,
    image_dir:   str,
    pre_dir:     str,
    output_csv:  str,
    output_xlsx: str,
):
    print("\n" + "="*60)
    print("  compare_results.py")
    print("  Pipeline A: load from existing CSVs")
    print("  Pipeline B: Raw Pixels + PCA (train now)")
    print("="*60)

    # ------------------------------------------------------------------
    # 1. Load Pipeline A results from existing CSVs
    # ------------------------------------------------------------------
    print("\n[PRE] Loading Pipeline A results from CSVs ...")
    res_A = load_pipeline_a_results(pre_dir)
    if res_A is None:
        return

    # ------------------------------------------------------------------
    # 2. Load images
    # ------------------------------------------------------------------
    print(f"\n[NP] Loading dataset from {csv_path} ...")
    df       = pd.read_csv(csv_path)
    df_clean = df[df[["n", "p", "k"]].notna().all(axis=1)].copy()
    print(f"[NP] Rows: {len(df)} | Complete NPK: {len(df_clean)}")

    filename_index = _build_filename_index(image_dir)
    print(f"[NP] Images on disk: {len(filename_index)}")

    raw_images        = []
    y_n, y_p, y_k    = [], [], []
    skipped           = 0

    for _, row in df_clean.iterrows():
        bare     = os.path.splitext(os.path.basename(row["image_filename"]))[0].lower()
        img_path = filename_index.get(bare)

        if img_path is None:
            print(f"  SKIP (not found): {row['image_filename']}")
            skipped += 1
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"  SKIP (unreadable): {img_path}")
            skipped += 1
            continue

        raw_images.append(img)
        y_n.append(int(row["n"]))
        y_p.append(int(row["p"]))
        y_k.append(int(row["k"]))

    if not raw_images:
        print("[NP] ERROR: No images loaded.")
        return

    y_n = np.array(y_n, dtype=np.int32)
    y_p = np.array(y_p, dtype=np.int32)
    y_k = np.array(y_k, dtype=np.int32)
    print(f"[NP] Loaded {len(raw_images)} images (skipped: {skipped})")

    # ------------------------------------------------------------------
    # 3. Train Pipeline B
    # ------------------------------------------------------------------
    start = datetime.now()
    res_B = run_pipeline_b(raw_images, y_n, y_p, y_k)
    end   = datetime.now()
    print(f"\n[RAW] Pipeline B done in {end - start}")

    # ------------------------------------------------------------------
    # 4. Build comparison
    # ------------------------------------------------------------------
    build_comparison(res_A, res_B, output_csv, output_xlsx)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Train Pipeline B (Raw Pixels + PCA) and compare against "
            "existing Pipeline A (Preprocessed) tuning CSV results."
        )
    )
    parser.add_argument(
        "--data-dir", default="SoilScanDataset",
        help="Root folder containing images and micro-dataset.csv"
    )
    parser.add_argument(
        "--csv", default=None,
        help="Path to CSV. Defaults to <data-dir>/micro-dataset.csv"
    )
    parser.add_argument(
        "--pre-dir", default=".",
        help="Folder containing existing NPK_N/P/K_Preprocessed_tuning.csv files "
             "(default: current directory)"
    )
    parser.add_argument("--output-csv",  default="comparison_results.csv")
    parser.add_argument("--output-xlsx", default="comparison_results.xlsx")
    args = parser.parse_args()

    main(
        csv_path    = args.csv or os.path.join(args.data_dir, "micro-dataset.csv"),
        image_dir   = args.data_dir,
        pre_dir     = args.pre_dir,
        output_csv  = args.output_csv,
        output_xlsx = args.output_xlsx,
    )