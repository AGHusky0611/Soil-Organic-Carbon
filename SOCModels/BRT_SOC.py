import pandas as pd
import numpy as np
import cv2
import os
import time
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import mean_squared_error, r2_score

# 1. SETTINGS & PATHS
csv_path = 'SOC Dataset/carbon.csv'
image_dir = 'SOC Dataset/soil images/'
df = pd.read_csv(csv_path)

# --- NEW: List of image sizes to test ---
image_sizes = [(32, 32), (64, 64), (128, 128), (512, 512)] 

param_grid = {
    'n_estimators': [500, 1000], 
    'learning_rate': [0.5, 0.05, 0.01, 0.005, 0.001], 
    'max_depth': [3, 5],           
    'subsample': [0.8]
}

# 2. HELPER FUNCTIONS
def extract_soil_features(img):
    mean_val = np.mean(img) / 255.0
    std_val = np.std(img) / 255.0
    pixels = (img.flatten() / 255.0).astype(np.float32)
    return np.append([mean_val, std_val], pixels)

def calculate_acceptance_accuracy(y_true, y_pred, tolerance=0.05):
    relative_error = np.abs((y_true - y_pred) / (y_true + 1e-7))
    return np.mean(relative_error <= tolerance) * 100

def get_augmented_images(img):
    return [img, cv2.flip(img, 1), cv2.flip(img, 0), cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)]

# 3. EXPERIMENT LOOP
all_results = []

for size in image_sizes:
    print(f"\n{'#'*30}")
    print(f"PROCESSING IMAGE SIZE: {size}")
    print(f"{'#'*30}")
    
    X, y = [], []
    
    # Reload and resize images for the current "size" iteration
    for index, row in df.iterrows():
        img_path = os.path.join(image_dir, row['image_number'])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, size)
            for aug_img in get_augmented_images(img):
                X.append(extract_soil_features(aug_img))
                y.append(row['Organic carbon'])

    X, y = np.array(X), np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Features for this size: {X.shape[1]}")

    # 4. HYPERPARAMETER LOOP
    for params in ParameterGrid(param_grid):
        print(f"Testing Config: {params}")
        start_time = time.time()
        
        model = GradientBoostingRegressor(**params, random_state=42)
        model.fit(X_train, y_train)
        
        duration = time.time() - start_time
        
        # Calculate Metrics
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)
        
        train_r2 = r2_score(y_train, train_preds)
        test_r2 = r2_score(y_test, test_preds)
        r2_gap = train_r2 - test_r2
        rmse_test = np.sqrt(mean_squared_error(y_test, test_preds))
        acc_5_pct = calculate_acceptance_accuracy(y_test, test_preds)
        
        print(f"   -> R2 Test: {test_r2:.4f} | Gap: {r2_gap:.4f} | 5% Acc: {acc_5_pct:.2f}%")
        
        # 5. LOG EVERYTHING
        all_results.append({
            'img_size': f"{size[0]}x{size[1]}",
            'feature_count': X.shape[1],
            'train_r2': train_r2,
            'test_r2': test_r2,
            'r2_gap': r2_gap,
            'rmse_test': rmse_test,
            'duration_sec': duration,
            'acc_within_5_pct': acc_5_pct,
            **params
        })

# 6. SAVE RESULTS
results_df = pd.DataFrame(all_results)
results_df.to_csv('BRT_MultiSize_Results.csv', index=False)
print("\n--- ALL TESTS COMPLETE ---")