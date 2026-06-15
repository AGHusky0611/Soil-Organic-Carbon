# XGBoost_NPK.py
import json
import os
import time
import cv2
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from pathlib import Path

from ClassificationModels.CLS_extraction import LabColorExtractor
from ClassificationModels.SVM_Calibrator import SoilCalibratorSVM

DEFAULT_NPK_N_MODEL = "npk_n_model.json"
DEFAULT_NPK_P_MODEL = "npk_p_model.json"
DEFAULT_NPK_K_MODEL = "npk_k_model.json"
DEFAULT_NPK_META = "npk_meta.json"
DEFAULT_IMAGE_SIZE = (128, 128)


def _write_meta(meta_path, image_size):
    """Write metadata for NPK models"""
    payload = {
        "image_size": list(image_size),
        "equalize_hist": False,
        "feature_extractor": "LabColorExtractor + SoilCalibratorSVM",
    }
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


class NPKXGBPredictor:
    """Load and predict N, P, K from images"""
    
    def __init__(
        self,
        model_n_path=DEFAULT_NPK_N_MODEL,
        model_p_path=DEFAULT_NPK_P_MODEL,
        model_k_path=DEFAULT_NPK_K_MODEL,
        meta_path=DEFAULT_NPK_META,
    ):
        self.model_n_path = model_n_path
        self.model_p_path = model_p_path
        self.model_k_path = model_k_path
        self.meta_path = meta_path
        
        self.model_n = xgb.XGBRegressor()
        self.model_p = xgb.XGBRegressor()
        self.model_k = xgb.XGBRegressor()
        
        self.image_size = DEFAULT_IMAGE_SIZE
        self.calibrator = SoilCalibratorSVM()
        self.extractor = LabColorExtractor()
        
        self._load()
    
    def _load(self):
        """Load all three models and metadata"""
        if not os.path.exists(self.model_n_path):
            raise FileNotFoundError(f"NPK N model not found at '{self.model_n_path}'")
        if not os.path.exists(self.model_p_path):
            raise FileNotFoundError(f"NPK P model not found at '{self.model_p_path}'")
        if not os.path.exists(self.model_k_path):
            raise FileNotFoundError(f"NPK K model not found at '{self.model_k_path}'")
        
        self.model_n.load_model(self.model_n_path)
        self.model_p.load_model(self.model_p_path)
        self.model_k.load_model(self.model_k_path)
        
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r", encoding="utf-8") as handle:
                meta = json.load(handle)
            self.image_size = tuple(meta.get("image_size", list(DEFAULT_IMAGE_SIZE)))
    
    def predict_image(self, img_bgr):
        """Predict N, P, K from a single image"""
        # Resize
        img = cv2.resize(img_bgr, self.image_size)
        
        # Calibrate
        calibrated = self.calibrator.calibrate(img)
        
        # Extract features
        features = self.extractor.extract_features(calibrated).reshape(1, -1)
        
        # Predict
        n_pred = float(self.model_n.predict(features)[0])
        p_pred = float(self.model_p.predict(features)[0])
        k_pred = float(self.model_k.predict(features)[0])
        
        return {"N": n_pred, "P": p_pred, "K": k_pred}


def train_xgb_npk(
    csv_path="SoilScanDataset/micro-dataset.csv",
    image_base_dir="SoilScanDataset/",
    image_size=(128, 128),
    param_grid=None,
    model_n_path=DEFAULT_NPK_N_MODEL,
    model_p_path=DEFAULT_NPK_P_MODEL,
    model_k_path=DEFAULT_NPK_K_MODEL,
    meta_path=DEFAULT_NPK_META,
):
    """
    Train XGBoost regressors for N, P, K prediction from images
    
    Args:
        csv_path: Path to micro-dataset.csv
        image_base_dir: Base directory for images
        image_size: Image resize target
        param_grid: XGBoost hyperparameters
        model_*_path: Output paths for models
        meta_path: Output path for metadata
    """
    
    if param_grid is None:
        param_grid = {
            "learning_rate": [0.01, 0.05],
            "max_depth": [3, 4, 5],
            "subsample": [0.8, 0.9, 1.0],
            "colsample_bytree": [0.6, 0.7],
            "min_child_weight": [1, 2, 3],
            "reg_lambda": [0.1, 1.0],
        }
    
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Filter rows with valid NPK values
    df_clean = df[df[['n', 'p', 'k']].notna().all(axis=1)].copy()
    print(f"Total rows in CSV: {len(df)}")
    print(f"Rows with complete NPK data: {len(df_clean)}")
    
    if len(df_clean) == 0:
        print("ERROR: No rows with complete NPK data!")
        return None
    
    print("\nNPK Value Statistics:")
    print(df_clean[['n', 'p', 'k']].describe())
    
    # Initialize components
    calibrator = SoilCalibratorSVM()
    extractor = LabColorExtractor()
    
    # Extract features
    print(f"\nExtracting features from images...")
    X, y_n, y_p, y_k = [], [], [], []
    loaded_count = 0
    
    processed_images_dir = os.path.join(image_base_dir, "processed_images")
    
    for idx, row in df_clean.iterrows():
        img_filename = row['image_filename']
        img_path = os.path.join(image_base_dir, img_filename)
        
        if not os.path.exists(img_path):
            continue
        
        try:
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            img = cv2.resize(img, image_size)
            calibrated = calibrator.calibrate(img)
            features = extractor.extract_features(calibrated)
            
            # === NEW: Save processed image ===
            # Extract directory structure from original path
            rel_dir = os.path.dirname(img_filename)
            processed_output_dir = os.path.join(processed_images_dir, rel_dir)
            os.makedirs(processed_output_dir, exist_ok=True)
            
            # Create processed filename
            base_name = os.path.splitext(os.path.basename(img_filename))[0]
            processed_filename = f"{base_name}_processed.jpg"
            processed_path = os.path.join(processed_output_dir, processed_filename)
            
            # Save preprocessed image (calibrated version)
            cv2.imwrite(processed_path, calibrated)
            # === END NEW ===
            
            X.append(features)
            y_n.append(float(row['n']))
            y_p.append(float(row['p']))
            y_k.append(float(row['k']))
            loaded_count += 1
            
            if loaded_count % 10 == 0:
                print(f"  Processed: {loaded_count} images")
        
        except Exception as e:
            print(f"  Error processing {img_path}: {e}")
            continue
    
    X = np.array(X)
    y_n = np.array(y_n)
    y_p = np.array(y_p)
    y_k = np.array(y_k)
    
    print(f"Successfully loaded and processed: {loaded_count} images")
    print(f"Feature shape: {X.shape}")
    
    if len(X) == 0:
        print("ERROR: No images processed!")
        return None
    
    # Train/test split
    X_train, X_test, y_n_train, y_n_test = train_test_split(
        X, y_n, test_size=0.2, random_state=42
    )
    _, _, y_p_train, y_p_test = train_test_split(
        X, y_p, test_size=0.2, random_state=42
    )
    _, _, y_k_train, y_k_test = train_test_split(
        X, y_k, test_size=0.2, random_state=42
    )
    
    print(f"\nTrain/Test split: {len(X_train)}/{len(X_test)}")
    
    # Train models
    results = {}
    
    for npk_name, y_train, y_test, model_path in [
        ('N', y_n_train, y_n_test, model_n_path),
        ('P', y_p_train, y_p_test, model_p_path),
        ('K', y_k_train, y_k_test, model_k_path),
    ]:
        print(f"\n{'='*50}")
        print(f"Training XGBoost for {npk_name} prediction")
        print(f"{'='*50}")
        print(f"Target stats - Min: {y_test.min():.4f}, Max: {y_test.max():.4f}, Mean: {y_test.mean():.4f}")
        
        best_model = None
        best_score = -np.inf
        best_params = None
        all_results = []
        
        for params in [p for p in [dict(next(iter(param_grid.items())) for _ in range(5))]]:
            # Use simpler param grid for faster training
            params_simplified = {
                'learning_rate': 0.05,
                'max_depth': 4,
                'subsample': 0.9,
                'colsample_bytree': 0.7,
                'min_child_weight': 2,
                'reg_lambda': 1.0,
            }
            
            print(f"\nTraining with params: {params_simplified}")
            start_time = time.time()
            
            es_callback = xgb.callback.EarlyStopping(
                rounds=30,
                metric_name='rmse',
                data_name='validation_1',
                save_best=True,
            )
            
            model = xgb.XGBRegressor(
                n_estimators=200,
                **params_simplified,
                random_state=42,
                booster='gbtree',
                tree_method='hist',
                eval_metric='rmse',
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
            
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
            mae_test = mean_absolute_error(y_test, y_test_pred)
            
            print(f"  Test R²: {test_r2:.4f} | Train R²: {train_r2:.4f}")
            print(f"  Test RMSE: {rmse_test:.4f} | MAE: {mae_test:.4f}")
            print(f"  Converged at epoch: {best_epoch} | Duration: {duration:.2f}s")
            
            all_results.append({
                'npk': npk_name,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'rmse': rmse_test,
                'mae': mae_test,
                'epoch': best_epoch,
                **params_simplified,
            })
            
            if test_r2 > best_score:
                best_score = test_r2
                best_model = model
                best_params = params_simplified
        
        # Save best model
        if best_model is not None:
            best_model.save_model(model_path)
            print(f"\nSaved {npk_name} model to {model_path}")
            results[npk_name] = {
                'test_r2': best_score,
                'model_path': model_path,
                'params': best_params,
            }
    
    # Write metadata
    _write_meta(meta_path, image_size)
    print(f"\nSaved metadata to {meta_path}")
    
    print("\n" + "="*50)
    print("NPK MODEL TRAINING COMPLETE")
    print("="*50)
    for npk_name, info in results.items():
        print(f"{npk_name}: Test R² = {info['test_r2']:.4f}")
    
    return results


if __name__ == "__main__":
    results = train_xgb_npk()
    
    if results:
        print("\nModels saved:")
        print(f"  - N model: {DEFAULT_NPK_N_MODEL}")
        print(f"  - P model: {DEFAULT_NPK_P_MODEL}")
        print(f"  - K model: {DEFAULT_NPK_K_MODEL}")
        print(f"  - Metadata: {DEFAULT_NPK_META}")