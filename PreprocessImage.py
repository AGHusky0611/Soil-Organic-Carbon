import cv2
from pathlib import Path

# FIX: Explicitly tell Python to look inside the SOCModels folder
from ClassificationModels.SVM_Calibrator import BrightnessCalibrator
from ClassificationModels.CLS_extraction import LabColorExtractor

def process_single_image(image_path, calibrator, extractor):
    """
    Executes the brightness calibration (SVM) and feature standardization (CLS).
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return None
        
    # 1. Apply Brightness Calibration (Your "SVM" module)
    calibrated_img = calibrator.calibrate(img)
    
    # 2. Apply White Balance, Denoising, and Standardization (Your "CLS" module)
    final_img = extractor.preprocess_image(calibrated_img)
    
    return final_img

def mirror_and_process(source_dir, dest_dir):
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)

    print("Initializing Pipeline Engines...")
    calibrator = BrightnessCalibrator(target_intensity=128)
    extractor = LabColorExtractor()

    valid_extensions = {'.jpg', '.jpeg', '.png'}
    processed_count = 0
    
    print(f"\nScanning '{source_path}' for images...")

    for file_path in source_path.rglob('*'):
        if file_path.suffix.lower() in valid_extensions:
            
            # Map paths to maintain folder structure
            relative_path = file_path.relative_to(source_path)
            target_file_path = dest_path / relative_path
            
            target_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Execute processing
            processed_img = process_single_image(file_path, calibrator, extractor)
            
            if processed_img is not None:
                cv2.imwrite(str(target_file_path), processed_img)
                processed_count += 1
                print(f"Processed: {relative_path}")
            else:
                print(f"ERROR: Could not read {file_path}")

    print(f"\nPipeline complete! Successfully processed and saved {processed_count} images.")

if __name__ == "__main__":
    # --- CONFIGURATION ---
    # Update to your exact folder names
    SOURCE_DIRECTORY = "SoilScanDataset" 
    DESTINATION_DIRECTORY = "ProcessedSoilScanDataset"
    
    mirror_and_process(SOURCE_DIRECTORY, DESTINATION_DIRECTORY)