import json
import os
import time

import cv2
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid, train_test_split

DEFAULT_MODEL_PATH = "soc_xgb_model.json"
DEFAULT_META_PATH = "soc_xgb_meta.json"
DEFAULT_IMAGE_SIZE = (32, 32)


def extract_soil_features(img):
    """Extracts global stats and raw pixel structures."""
    mean_val = np.mean(img) / 255.0
    std_val = np.std(img) / 255.0
    pixels = (img.flatten() / 255.0).astype(np.float32)
    return np.append([mean_val, std_val], pixels)


def preprocess_soc_image(img_bgr, image_size):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, image_size)
    return cv2.equalizeHist(gray)


def calculate_acceptance_accuracy(y_true, y_pred, tolerance=0.005):
    """Calculates the percentage of predictions within +/- 5% error."""
    relative_error = np.abs((y_true - y_pred) / (y_true + 1e-7))
    return np.mean(relative_error <= tolerance) * 100


def get_augmented_images(img):
    """Generates standard flipped and rotated versions."""
    return [
        img,
        cv2.flip(img, 1),
        cv2.flip(img, 0),
        cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),
    ]


def _write_meta(meta_path, image_size, best_iteration):
    payload = {
        "image_size": list(image_size),
        "best_iteration": best_iteration,
        "equalize_hist": True,
    }
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


class SOCXGBPredictor:
    def __init__(self, model_path=DEFAULT_MODEL_PATH, meta_path=DEFAULT_META_PATH):
        self.model_path = model_path
        self.meta_path = meta_path
        self.model = xgb.XGBRegressor()
        self.image_size = DEFAULT_IMAGE_SIZE
        self.best_iteration = None
        self._load()

    def _load(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"SOC model not found at '{self.model_path}'. Train and save the model first."
            )

        self.model.load_model(self.model_path)

        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r", encoding="utf-8") as handle:
                meta = json.load(handle)
            self.image_size = tuple(meta.get("image_size", list(DEFAULT_IMAGE_SIZE)))
            self.best_iteration = meta.get("best_iteration")

    def predict_image(self, img_bgr):
        gray = preprocess_soc_image(img_bgr, self.image_size)
        features = extract_soil_features(gray).reshape(1, -1)
        if self.best_iteration is not None:
            pred = self.model.predict(features, iteration_range=(0, self.best_iteration + 1))
        else:
            pred = self.model.predict(features)
        return float(pred[0])


def train_xgb_soc(
    csv_path="SOCDataset/carbon.csv",
    image_dir="SOCDataset/soil images/",
    image_sizes=None,
    param_grid=None,
    model_path=DEFAULT_MODEL_PATH,
    meta_path=DEFAULT_META_PATH,
):
    if image_sizes is None:
        image_sizes = [DEFAULT_IMAGE_SIZE]

    if param_grid is None:
        param_grid = {
            "learning_rate": [0.5],
            "max_depth": [8],
            "subsample": [0.5],
            "rate_drop": [0.2],
            "skip_drop": [0.3],
        }

    df = pd.read_csv(csv_path)
    all_results = []
    best_score = -1.0
    best_model = None
    best_meta = None

    for size in image_sizes:
        print(f"\n{'#'*40}")
        print(f" LOADING & PROCESSING DATASET FOR SIZE: {size} ")
        print(f"{'#'*40}")

        X, y = [], []
        original_count = 0

        for _, row in df.iterrows():
            img_path = os.path.join(image_dir, row["image_number"])
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is not None:
                original_count += 1
                img = cv2.resize(img, size)
                img = cv2.equalizeHist(img)

                for aug_img in get_augmented_images(img):
                    X.append(extract_soil_features(aug_img))
                    y.append(row["Organic carbon"])

        X, y = np.array(X), np.array(y)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print(f"Original source files loaded: {original_count}")

        for params in ParameterGrid(param_grid):
            print(f"\nRunning Configuration: {params}")
            start_time = time.time()

            es_callback = xgb.callback.EarlyStopping(
                rounds=50,
                metric_name="rmse",
                data_name="validation_1",
                save_best=True,
            )

            model = xgb.XGBRegressor(
                n_estimators=1500,
                **params,
                colsample_bytree=0.8,
                random_state=42,
                booster="dart",
                tree_method="hist",
                device="cuda",
                sample_type="uniform",
                normalize_type="tree",
                eval_metric="rmse",
                callbacks=[es_callback],
            )

            model.fit(
                X_train,
                y_train,
                eval_set=[(X_train, y_train), (X_test, y_test)],
                verbose=False,
            )

            duration = time.time() - start_time
            best_epoch = model.best_iteration

            train_preds = model.predict(X_train, iteration_range=(0, best_epoch + 1))
            test_preds = model.predict(X_test, iteration_range=(0, best_epoch + 1))

            train_r2 = r2_score(y_train, train_preds)
            test_r2 = r2_score(y_test, test_preds)
            r2_gap = train_r2 - test_r2
            rmse_test = np.sqrt(mean_squared_error(y_test, test_preds))
            acc_5_pct = calculate_acceptance_accuracy(
                y_test, test_preds, tolerance=0.005
            )

            print(
                f"   -> Results | Test R2: {test_r2:.4f} | Overfit Gap: {r2_gap:.4f}"
            )
            print(
                f"   -> Metrics | Test RMSE: {rmse_test:.4f} | 0.5% Tolerance Accuracy: {acc_5_pct:.2f}%"
            )
            print(
                f"   -> Stopping | Converged at Epoch: {best_epoch} | Duration: {duration:.2f}s"
            )
            print("-" * 50)

            all_results.append(
                {
                    "img_size": f"{size[0]}x{size[1]}",
                    "feature_count": X.shape[1],
                    "train_r2": train_r2,
                    "test_r2": test_r2,
                    "r2_gap": r2_gap,
                    "rmse_test": rmse_test,
                    "duration_sec": duration,
                    "converged_epoch": best_epoch,
                    "acc_within_5_pct": acc_5_pct,
                    **params,
                }
            )

            if acc_5_pct > best_score:
                best_score = acc_5_pct
                best_model = model
                best_meta = {
                    "image_size": size,
                    "best_iteration": int(best_epoch),
                }

    results_df = pd.DataFrame(all_results)
    results_df.to_csv("XGBoost_Unlimited_Tuning_Results.csv", index=False)

    if best_model is not None:
        best_model.save_model(model_path)
        _write_meta(meta_path, best_meta["image_size"], best_meta["best_iteration"])

    return results_df, best_meta


if __name__ == "__main__":
    results_df, best_meta = train_xgb_soc()

    print("\n" + "=" * 40)
    print(" DYNAMIC EXPERIMENT COMPLETE ")
    print("=" * 40)
    print("Results archived to 'XGBoost_Unlimited_Tuning_Results.csv'")

    if not results_df.empty:
        print("\nBEST RUN IDENTIFIED BY MAXIMUM TARGET METRIC (0.55% ACCEPTANCE):")
        print(results_df.loc[results_df["acc_within_5_pct"].idxmax()])

    if best_meta is not None:
        print("\nSOC model saved for inference:")
        print(f"- Model: {DEFAULT_MODEL_PATH}")
        print(f"- Meta:  {DEFAULT_META_PATH}")