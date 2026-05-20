import pandas as pd
import numpy as np
import cv2
import os
import time
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import mean_squared_error, r2_score

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

# ==========================================
# 1. SETTINGS & UTILITIES
# ==========================================
csv_path = 'SOC Dataset/carbon.csv'
image_dir = 'SOC Dataset/soil images/'
df = pd.read_csv(csv_path)

# image_sizes = [(32, 32), (64, 64), (128, 128), (256, 256), (512,512)] # Experimment 2.1
image_sizes = [(32, 32), (48, 48), (64, 64), (128, 128)]

# Experiment 1
# param_grid = {
#     'learning_rate': [0.1, 0.05, 0.01, 0.005, 0.001], 
#     'max_depth': [5],                
#     'subsample': [0.8, 0.9, 1.0],
#     'rate_drop': [0.05, 0.1, 0.2],   
#     'skip_drop': [0.5, 0.3, 0.2, 0.1]               
# }

# Experiment 2: Added ALpha to address the high overfit gap
# param_grid = {
#     'learning_rate': [0.005, 0.003, 0.001], 
#     'subsample': [0.9, 1.0],         # Best performers used almost all data
#     'rate_drop': [0.2, 0.3, 0.4],    # Increased strength
#     'skip_drop': [0.1, 0.05],        # Force dropout almost 100% of the time
#     'alpha': [0.01, 0.1]             # Add this to your SGDRegressor settings
# }

# Best in terms of R^2 Train
param_grid = {
    'learning_rate': [0.003], 
    'subsample': [0.9],         # Best performers used almost all data
    'rate_drop': [0.2],    # Increased strength
    'skip_drop': [0.05],        # Force dropout almost 100% of the time
    'alpha': [0.01]             # Add this to your SGDRegressor settings
}

MAX_EPOCHS = 1500
EARLY_STOPPING_PATIENCE = 50

def extract_soil_features(img):
    mean_val = np.mean(img) / 255.0
    std_val = np.std(img) / 255.0
    pixels = (img.flatten() / 255.0).astype(np.float32)
    return np.append([mean_val, std_val], pixels)

def calculate_acceptance_accuracy(y_true, y_pred, tolerance=0.005):
    relative_error = np.abs((y_true - y_pred) / (y_true + 1e-7))
    return np.mean(relative_error <= tolerance) * 100

def get_augmented_images(img):
    return [img, cv2.flip(img, 1), cv2.flip(img, 0), cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)]

# ==========================================
# 2. CORE PROCESSING LOOP
# ==========================================
all_results = []

for size in image_sizes:
    X, y = [], []
    for index, row in df.iterrows():
        img_path = os.path.join(image_dir, row['image_number'])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.equalizeHist(cv2.resize(img, size))
            for aug_img in get_augmented_images(img):
                X.append(extract_soil_features(aug_img))
                y.append(row['Organic carbon'])
                
    X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(y), test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_s = torch.tensor(scaler.fit_transform(X_train), dtype=torch.float32).to(device)
    X_test_s = torch.tensor(scaler.transform(X_test), dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

    for params in ParameterGrid(param_grid):
        print(f"\nConfig: {params}")
        
        model = nn.Linear(X_train_s.shape[1], 1).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=params['learning_rate'])
        
        best_rmse = float('inf')
        patience_counter = 0
        start_time = time.time()

        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            
            # --- Subsampling ---
            if params['subsample'] < 1.0:
                perm = torch.randperm(X_train_s.size(0))
                idx = perm[:int(X_train_s.size(0) * params['subsample'])]
                X_batch, y_batch = X_train_s[idx], y_train_t[idx]
            else:
                X_batch, y_batch = X_train_s, y_train_t

            # --- DART Logic ---
            if np.random.rand() >= params['skip_drop']:
                mask = (torch.rand_like(X_batch) >= params['rate_drop']).float() / (1.0 - params['rate_drop'])
                X_batch = X_batch * mask

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = torch.mean(torch.clamp(torch.abs(outputs - y_batch) - 0.01, min=0))
            loss.backward()
            optimizer.step()

            # Validation Check
            model.eval()
            with torch.no_grad():
                preds = model(X_test_s)
                val_rmse = torch.sqrt(torch.mean((preds - y_test_t)**2)).item()

            if val_rmse < best_rmse:
                best_rmse, best_epoch = val_rmse, epoch
                patience_counter = 0
                state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1

            if patience_counter >= EARLY_STOPPING_PATIENCE: break

        duration = time.time() - start_time
        model.load_state_dict(state_dict)
        model.eval()
        
        # --- CALCULATE FINAL DIAGNOSTICS ---
        with torch.no_grad():
            final_test_preds = model(X_test_s).cpu().numpy().flatten()
            final_train_preds = model(X_train_s).cpu().numpy().flatten()

        train_r2 = r2_score(y_train, final_train_preds)
        test_r2 = r2_score(y_test, final_test_preds)
        r2_gap = train_r2 - test_r2
        acc_5 = calculate_acceptance_accuracy(y_test, final_test_preds, 0.005)

        print(f"   -> Results | Test R2: {test_r2:.4f} | Overfit Gap: {r2_gap:.4f}")
        print(f"   -> Metrics | Test RMSE: {best_rmse:.4f} | 0.5% Tolerance Accuracy: {acc_5:.2f}%")
        print(f"   -> Stopping | Converged at Epoch: {best_epoch} | Duration: {duration:.2f}s")
        print("-" * 50)
        
        # --- STORAGE ---
        all_results.append({
            'img_size': f"{size[0]}x{size[1]}",
            'feature_count': X_train_s.shape[1],
            'train_r2': train_r2,
            'test_r2': test_r2,
            'r2_gap': r2_gap,
            'rmse_test': best_rmse,
            'duration_sec': duration,
            'converged_epoch': best_epoch,
            'acc_within_5_pct': acc_5,
            **params
        })

results_df = pd.DataFrame(all_results)

# Final Save
output_filename = 'SVR_CUDA_Overfitting_Results.csv'
results_df.to_csv(output_filename, index=False)

print("\n" + "="*40)
print(" SVR CUDA EXPERIMENT COMPLETE ")
print("="*40)
print(f"Results successfully archived to '{output_filename}'")

# Check if we actually have results to avoid index errors
if not results_df.empty:
    print("\nBEST RUN IDENTIFIED BY MAXIMUM TARGET METRIC (0.5% ACCEPTANCE):")
    # Finding the row with the highest accuracy
    best_run = results_df.loc[results_df['acc_within_5_pct'].idxmax()]
    print(best_run)
    
    # Quick Overfitting Insight for the best run
    print(f"\nDiagnostic for Best Run:")
    print(f" -> Training R2: {best_run['train_r2']:.4f}")
    print(f" -> Testing R2:  {best_run['test_r2']:.4f}")
    print(f" -> Overfit Gap: {best_run['r2_gap']:.4f}")
else:
    print("\nNo results were recorded. Check your image paths and dataframe.")