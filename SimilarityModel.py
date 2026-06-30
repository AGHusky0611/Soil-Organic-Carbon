"""
pipeline_similarity.py
======================
Compares preprocessed soil images from two pipeline folders:
  - AgriCapture  : crop + center only
  - TerraScan    : BrightnessCalibrator → white balance → denoise → 256×256

Upgraded to measure Similarity, Image Quality, Feature Counts, and generate Visual Diffs.


Dependencies
------------
    pip install opencv-python scikit-image numpy scipy scikit-learn matplotlib


README USAGE:
Visualizer:
python SimilarityModel.py SoilScanDataset ProcessedSoilScanDataset --output-csv results.csv --visualize

CLI | CSV
python SimilarityModel.py SoilScanDataset ProcessedSoilScanDataset --output-csv results.csv

OUTPUT:
./Visual_comparisons/  (if --visualize)
./results.csv  (if --output-csv)

"""

import os
import re
import csv
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.distance import cosine
from skimage.metrics import structural_similarity as ssim
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy
import matplotlib.pyplot as plt


# ── UUID extractor ─────────────────────────────────────────────────────────────
_UUID_RE = re.compile(r"_([0-9a-f]{8})(?:_processed)?(?:\.[^.]+)?$", re.IGNORECASE)

def _extract_uuid(filename: str) -> str | None:
    m = _UUID_RE.search(Path(filename).stem + "." + Path(filename).suffix)
    if not m:
        m = _UUID_RE.search(filename)
    return m.group(1).lower() if m else None


# ── Core metric & Quality functions ────────────────────────────────────────────

def _load_bgr(path: str) -> np.ndarray | None:
    """Load image as BGR uint8, stripping alpha if present."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    return img


def _soil_mask(bgr: np.ndarray, bg: str = "auto") -> np.ndarray:
    """
    Boolean mask of non-background pixels.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if bg == "auto":
        corners = [gray[0,0], gray[0,-1], gray[-1,0], gray[-1,-1]]
        bg = "black" if np.mean(corners) < 128 else "white"
    if bg == "black":
        return gray > 15
    return gray < 240


def _resize(bgr: np.ndarray, size: int = 256) -> np.ndarray:
    return cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)


def _glcm_vector(gray: np.ndarray) -> np.ndarray:
    """72-D GLCM feature vector (3 distances × 4 angles × 6 props)."""
    g = np.clip((gray // 32).astype(np.uint8), 0, 7)
    feats = []
    for d in [1, 2, 3]:
        for a in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]:
            m = graycomatrix(g, [d], [a], levels=8, symmetric=True, normed=True)
            for prop in ["contrast", "dissimilarity", "homogeneity",
                         "energy", "correlation", "ASM"]:
                feats.append(graycoprops(m, prop)[0, 0])
    return np.array(feats, dtype=np.float64)


def _cls_features(bgr: np.ndarray) -> np.ndarray:
    """
    Full 79-D CLS feature vector matching LabColorExtractor.extract_features().
    Steps: white balance → denoise → 256×256 → GLCM (72) + LAB stats (7)
    """
    # White balance (Gray World)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    avg_a, avg_b = np.mean(lab[:, :, 1]), np.mean(lab[:, :, 2])
    lab[:, :, 1] -= (avg_a - 128) * (lab[:, :, 0] / 255.0) * 1.1
    lab[:, :, 2] -= (avg_b - 128) * (lab[:, :, 0] / 255.0) * 1.1
    wb = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

    # Denoise + standardise
    wb = cv2.GaussianBlur(wb, (3, 3), 0)
    wb = cv2.resize(wb, (256, 256))
    wb = cv2.normalize(wb, None, 0, 255, cv2.NORM_MINMAX)

    gray = cv2.cvtColor(wb, cv2.COLOR_BGR2GRAY)
    glcm = _glcm_vector(gray)

    lab2 = cv2.cvtColor(wb, cv2.COLOR_BGR2LAB)
    ml, ma, mb = cv2.mean(lab2)[:3]
    sl = np.std(lab2[:, :, 0])
    sa = np.std(lab2[:, :, 1])
    sb = np.std(lab2[:, :, 2])
    lap = cv2.Laplacian(gray, cv2.CV_64F).var()

    return np.hstack([glcm, [ml, ma, mb, sl, sa, sb, lap]])


def _color_hist(bgr: np.ndarray, mask: np.ndarray, bins: int = 32) -> np.ndarray:
    pixels = (bgr.astype(np.float32) / 255.0)[mask]
    hists = [np.histogram(pixels[:, c], bins=bins, range=(0, 1), density=True)[0]
             for c in range(3)]
    return np.concatenate(hists)


def _lab_stats(bgr: np.ndarray, mask: np.ndarray) -> dict:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    px = lab[mask]
    return {
        "L_mean": float(px[:, 0].mean()), "L_std": float(px[:, 0].std()),
        "A_mean": float(px[:, 1].mean()), "A_std": float(px[:, 1].std()),
        "B_mean": float(px[:, 2].mean()), "B_std": float(px[:, 2].std()),
    }

def _get_quality_metrics(gray: np.ndarray) -> dict:
    """Measures sharpness and information content (entropy)."""
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    entropy = shannon_entropy(gray)
    return {"sharpness": float(laplacian_var), "entropy": float(entropy)}

def _count_keypoints(gray: np.ndarray) -> int:
    """Uses ORB to count distinct structural features (corners/edges)."""
    orb = cv2.ORB_create()
    keypoints = orb.detect(gray, None)
    return len(keypoints)


# ── Main class ─────────────────────────────────────────────────────────────────

class PipelineSimilarityAnalyzer:
    """
    Compare AgriCapture vs TerraScan preprocessed image folders.
    """

    SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    CSV_FIELDS = [
        "uuid", "agricapture_file", "terrascan_file",
        "ssim", "hist_cosine", "glcm_cosine", "cls_cosine",
        "L_mean_ac", "L_mean_ts", "L_delta",
        "A_mean_ac", "A_mean_ts", "A_delta",
        "B_mean_ac", "B_mean_ts", "B_delta",
        "sharpness_ac", "sharpness_ts", 
        "entropy_ac", "entropy_ts",
        "kp_count_ac", "kp_count_ts",
        "similarity_verdict", "quality_winner"
    ]

    def __init__(
        self,
        agricapture_dir: str | Path,
        terrascan_dir: str | Path,
        output_csv: str | Path = "similarity_results.csv",
        resize: int = 256,
        visualize: bool = False
    ):
        self.agricapture_dir = Path(agricapture_dir)
        self.terrascan_dir   = Path(terrascan_dir)
        self.output_csv      = Path(output_csv)
        self.resize          = resize
        self.visualize       = visualize
        self.results: list[dict] = []
        
        if self.visualize:
            self.vis_dir = Path("visual_comparisons")
            self.vis_dir.mkdir(exist_ok=True)

    # ── File discovery & matching ──────────────────────────────────────────────

    def _index_folder(self, folder: Path) -> dict[str, Path]:
        """Return {uuid: filepath} for all supported images in folder."""
        index = {}
        
        if not folder.exists():
            print(f"  [ERROR] Directory not found: {folder}")
            return index

        for f in folder.rglob('*'):
            if not f.is_file() or f.suffix.lower() not in self.SUPPORTED_EXT:
                continue
            uid = _extract_uuid(f.name)
            if uid:
                index[uid] = f
            else:
                print(f"  [WARN] Could not extract UUID from: {f.name}")
        return index

    def _match_pairs(self) -> list[tuple[str, Path, Path]]:
        ac_index = self._index_folder(self.agricapture_dir)
        ts_index = self._index_folder(self.terrascan_dir)

        ac_uuids = set(ac_index)
        ts_uuids = set(ts_index)
        matched  = ac_uuids & ts_uuids

        if ac_uuids - ts_uuids:
            print(f"  [INFO] {len(ac_uuids - ts_uuids)} AgriCapture file(s) with no TerraScan match — skipped.")
        if ts_uuids - ac_uuids:
            print(f"  [INFO] {len(ts_uuids - ac_uuids)} TerraScan file(s) with no AgriCapture match — skipped.")

        return [(uid, ac_index[uid], ts_index[uid]) for uid in sorted(matched)]

    # ── Visual Diff ───────────────────────────────────────────────────────────

    def _generate_visual_diff(self, uuid: str, ac_r: np.ndarray, ts_r: np.ndarray, metrics: dict):
        """Creates a side-by-side image with a difference heatmap."""
        gray_ac = cv2.cvtColor(ac_r, cv2.COLOR_BGR2GRAY)
        gray_ts = cv2.cvtColor(ts_r, cv2.COLOR_BGR2GRAY)
        
        # Calculate absolute difference
        diff = cv2.absdiff(gray_ac, gray_ts)
        diff_heatmap = cv2.applyColorMap(diff, cv2.COLORMAP_JET)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        axes[0].imshow(cv2.cvtColor(ac_r, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f"AgriCap\nFeatures: {metrics['kp_count_ac']} | Ent: {metrics['entropy_ac']:.2f}")
        axes[0].axis('off')

        axes[1].imshow(cv2.cvtColor(ts_r, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f"TerraScan\nFeatures: {metrics['kp_count_ts']} | Ent: {metrics['entropy_ts']:.2f}")
        axes[1].axis('off')

        axes[2].imshow(cv2.cvtColor(diff_heatmap, cv2.COLOR_BGR2RGB))
        axes[2].set_title(f"Difference Heatmap\nWinner: {metrics['quality_winner']}")
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig(self.vis_dir / f"{uuid}_comparison.png")
        plt.close(fig)

    # ── Per-pair analysis ─────────────────────────────────────────────────────

    def _analyze_pair(self, uuid: str, ac_path: Path, ts_path: Path) -> dict | None:
        ac_bgr = _load_bgr(ac_path)
        ts_bgr = _load_bgr(ts_path)
        if ac_bgr is None or ts_bgr is None:
            print(f"  [ERROR] Could not load images for UUID {uuid} — skipped.")
            return None

        # Resize to common resolution
        ac_r = _resize(ac_bgr, self.resize)
        ts_r = _resize(ts_bgr, self.resize)

        # Masks (auto-detect background colour)
        mask_ac = _soil_mask(ac_r, "auto")
        mask_ts = _soil_mask(ts_r, "auto")
        combined = mask_ac & mask_ts

        # Fallback: if combined mask is too sparse, use full image
        if combined.sum() < 100:
            combined = np.ones((self.resize, self.resize), dtype=bool)

        # ── Core Original Metrics ─────────────────────────────────────────────
        gray_ac = cv2.cvtColor(ac_r, cv2.COLOR_BGR2GRAY)
        gray_ts = cv2.cvtColor(ts_r, cv2.COLOR_BGR2GRAY)

        ssim_val, _ = ssim(gray_ac, gray_ts, full=True)

        h_ac = _color_hist(ac_r, combined)
        h_ts = _color_hist(ts_r, combined)
        hist_cos = float(1 - cosine(h_ac, h_ts))

        gv_ac = _glcm_vector(gray_ac)
        gv_ts = _glcm_vector(gray_ts)
        glcm_cos = float(1 - cosine(gv_ac, gv_ts))

        cls_ac = _cls_features(ac_r)
        cls_ts = _cls_features(ts_r)
        cls_cos = float(1 - cosine(cls_ac, cls_ts))

        lab_ac = _lab_stats(ac_r, combined)
        lab_ts = _lab_stats(ts_r, combined)

        # ── New Quality & Feature Metrics ─────────────────────────────────────
        qual_ac, qual_ts = _get_quality_metrics(gray_ac), _get_quality_metrics(gray_ts)
        kp_ac, kp_ts = _count_keypoints(gray_ac), _count_keypoints(gray_ts)

        # ── Verdicts ──────────────────────────────────────────────────────────
        # Similarity Verdict
        if cls_cos >= 0.995 and glcm_cos >= 0.95:
            sim_verdict = "EQUIVALENT"
        elif cls_cos >= 0.97:
            sim_verdict = "CLOSE"
        elif cls_cos >= 0.90:
            sim_verdict = "MODERATE_DIVERGENCE"
        else:
            sim_verdict = "HIGH_DIVERGENCE"

        # Quality Winner Verdict
        if qual_ts["entropy"] > qual_ac["entropy"] and kp_ts > kp_ac:
            winner = "TerraScan (More Data)"
        elif qual_ac["entropy"] > qual_ts["entropy"] and kp_ac > kp_ts:
            winner = "AgriCap (TerraScan lost data)"
        else:
            winner = "Mixed / Unclear"

        metrics = {
            "uuid": uuid,
            "agricapture_file": ac_path.name,
            "terrascan_file":   ts_path.name,
            "ssim":       round(float(ssim_val), 4),
            "hist_cosine": round(hist_cos, 4),
            "glcm_cosine": round(glcm_cos, 4),
            "cls_cosine":  round(cls_cos,  4),
            "L_mean_ac": round(lab_ac["L_mean"], 2),
            "L_mean_ts": round(lab_ts["L_mean"], 2),
            "L_delta":   round(abs(lab_ac["L_mean"] - lab_ts["L_mean"]), 2),
            "A_mean_ac": round(lab_ac["A_mean"], 2),
            "A_mean_ts": round(lab_ts["A_mean"], 2),
            "A_delta":   round(abs(lab_ac["A_mean"] - lab_ts["A_mean"]), 2),
            "B_mean_ac": round(lab_ac["B_mean"], 2),
            "B_mean_ts": round(lab_ts["B_mean"], 2),
            "B_delta":   round(abs(lab_ac["B_mean"] - lab_ts["B_mean"]), 2),
            "sharpness_ac": round(qual_ac["sharpness"], 2),
            "sharpness_ts": round(qual_ts["sharpness"], 2),
            "entropy_ac": round(qual_ac["entropy"], 4),
            "entropy_ts": round(qual_ts["entropy"], 4),
            "kp_count_ac": kp_ac,
            "kp_count_ts": kp_ts,
            "similarity_verdict": sim_verdict,
            "quality_winner": winner,
        }

        if self.visualize:
            self._generate_visual_diff(uuid, ac_r, ts_r, metrics)

        return metrics

    # ── Console report ────────────────────────────────────────────────────────

    def _print_report(self):
        n = len(self.results)
        if n == 0:
            print("\nNo results to report.")
            return

        sep = "─" * 68

        print(f"\n{'═'*68}")
        print(f"  PIPELINE SIMILARITY REPORT  |  {n} image pair(s)")
        print(f"{'═'*68}")

        for r in self.results:
            print(f"\n  UUID : {r['uuid']}")
            print(f"  AC   : {r['agricapture_file']}")
            print(f"  TS   : {r['terrascan_file']}")
            print(f"  {sep}")
            print(f"  {'Metric':<28} {'Score':>8}   {'Interpretation'}")
            print(f"  {sep}")
            print(f"  {'Pixel-level SSIM':<28} {r['ssim']:>8.4f}   {_interp('ssim', r['ssim'])}")
            print(f"  {'Color histogram cosine':<28} {r['hist_cosine']:>8.4f}   {_interp('hist', r['hist_cosine'])}")
            print(f"  {'GLCM texture cosine':<28} {r['glcm_cosine']:>8.4f}   {_interp('tex',  r['glcm_cosine'])}")
            print(f"  {'CLS feature vector cosine':<28} {r['cls_cosine']:>8.4f}   {_interp('cls',  r['cls_cosine'])}")
            print(f"  {sep}")
            print(f"  LAB channel deltas  (soil pixels only)")
            print(f"    L (luminance): AC={r['L_mean_ac']:6.2f}  TS={r['L_mean_ts']:6.2f}  Δ={r['L_delta']:5.2f}")
            print(f"    A (green-red): AC={r['A_mean_ac']:6.2f}  TS={r['A_mean_ts']:6.2f}  Δ={r['A_delta']:5.2f}")
            print(f"    B (blue-yel): AC={r['B_mean_ac']:6.2f}  TS={r['B_mean_ts']:6.2f}  Δ={r['B_delta']:5.2f}")
            print(f"  {sep}")
            print(f"  Quality & Features")
            print(f"    Entropy      : AC={r['entropy_ac']:.2f}  TS={r['entropy_ts']:.2f}")
            print(f"    ORB Features : AC={r['kp_count_ac']}     TS={r['kp_count_ts']}")
            print(f"  {sep}")
            print(f"  Sim. Verdict : {r['similarity_verdict']}")
            print(f"  Winner       : {r['quality_winner']}")

        # ── Aggregate summary ─────────────────────────────────────────────────
        if n > 1:
            def avg(key): return sum(r[key] for r in self.results) / n
            verdicts = [r["similarity_verdict"] for r in self.results]
            winners = [r["quality_winner"] for r in self.results]

            print(f"\n{'═'*68}")
            print(f"  AGGREGATE SUMMARY  ({n} pairs)")
            print(f"{'═'*68}")
            print(f"  Avg SSIM              : {avg('ssim'):.4f}")
            print(f"  Avg CLS feature vec   : {avg('cls_cosine'):.4f}")
            print(f"  Avg ORB Features      : AC={avg('kp_count_ac'):.0f} | TS={avg('kp_count_ts'):.0f}")
            print(f"  Verdicts              : {', '.join(sorted(set(verdicts)))}")
            print(f"  Top Winner            : {max(set(winners), key=winners.count)}")
            print(f"{'═'*68}\n")

    # ── CSV writer ────────────────────────────────────────────────────────────

    def _save_csv(self):
        try:
            with open(self.output_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.CSV_FIELDS)
                writer.writeheader()
                writer.writerows(self.results)
            print(f"\n  ✓ Results saved → {self.output_csv.resolve()}")
        except PermissionError:
            print(f"\n  [ERROR] Permission denied: '{self.output_csv.name}'.")
            print("  Make sure the file is not currently open in another program (like Excel) and try again.")

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self) -> list[dict]:
        """
        Run the full analysis pipeline.
        """
        print(f"\n  AgriCapture dir : {self.agricapture_dir}")
        print(f"  TerraScan  dir  : {self.terrascan_dir}")
        print(f"  Resize          : {self.resize}×{self.resize}")
        if self.visualize:
            print(f"  Visuals         : Enabled (saving to {self.vis_dir}/)")

        pairs = self._match_pairs()
        if not pairs:
            print("\n  [ERROR] No matched pairs found. Check your directories and filenames.")
            return []

        print(f"\n  Matched {len(pairs)} pair(s). Running analysis...\n")

        self.results = []
        for i, (uuid, ac_path, ts_path) in enumerate(pairs, 1):
            t0 = time.perf_counter()
            result = self._analyze_pair(uuid, ac_path, ts_path)
            elapsed = time.perf_counter() - t0
            if result:
                self.results.append(result)
                print(f"  [{i}/{len(pairs)}] {uuid}  Sim={result['cls_cosine']:.4f}  Winner={result['quality_winner']}  ({elapsed:.2f}s)")

        self._print_report()
        self._save_csv()
        return self.results


# ── Interpretation helpers ─────────────────────────────────────────────────────

def _interp(kind: str, val: float) -> str:
    if kind == "ssim":
        if val >= 0.80: return "High pixel similarity"
        if val >= 0.50: return "Moderate — size/exposure differ"
        return "Low — expected across pipelines"
    if kind == "hist":
        if val >= 0.90: return "Very similar color distribution"
        if val >= 0.70: return "Moderate — brightness shift present"
        return "Large color divergence"
    if kind in ("tex", "cls"):
        if val >= 0.995: return "Near-identical ✓"
        if val >= 0.97:  return "Very close"
        if val >= 0.90:  return "Moderate divergence"
        return "High divergence — investigate"
    return ""


# ── CLI convenience ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare AgriCapture vs TerraScan preprocessed image folders."
    )
    parser.add_argument("agricapture_dir", nargs='?', default=None, help="Path to AgriCapture output folder")
    parser.add_argument("terrascan_dir",   help="Path to TerraScan output folder")
    parser.add_argument("--output-csv",    default="similarity_results.csv",
                        help="CSV output path (default: similarity_results.csv)")
    parser.add_argument("--resize",        type=int, default=256,
                        help="Resize resolution for pixel metrics (default: 256)")
    parser.add_argument("--visualize",     action="store_true", 
                        help="Generate side-by-side images with difference heatmaps")
    args = parser.parse_args()

    analyzer = PipelineSimilarityAnalyzer(
        agricapture_dir=args.agricapture_dir,
        terrascan_dir=args.terrascan_dir,
        output_csv=args.output_csv,
        resize=args.resize,
        visualize=args.visualize
    )
    analyzer.run()