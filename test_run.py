import cv2
import numpy as np
import os
import xgboost as xgb
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from Models.SVM_Calibrator import SoilCalibratorSVM
from Models.CLS_extraction import LabColorExtractor

def test_image():
    # 1. Open a Windows File Picker dialog to choose an image
    Tk().withdraw() 
    image_path = askopenfilename(title="Select a Soil Image to Test", 
                                 filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
    
    if not image_path:
        print("No image selected. Exiting...")
        return

    print(f"\n🔍 Analyzing: {os.path.basename(image_path)}...")

    # 2. Check if the AI brain exists
    if not os.path.exists("soil_xgb_model.json"):
        print("❌ Error: You need to run run.py to train the model first!")
        return

    # 3. Load Modules & AI Model
    calibrator = SoilCalibratorSVM()
    extractor = LabColorExtractor()
    model = xgb.XGBClassifier()
    model.load_model("soil_xgb_model.json")
    class_names = np.load("soil_classes.npy", allow_pickle=True)

    # 4. Process the selected image
    # Robust loading for Windows
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    
    if img is None:
        print(f"❌ Error: Could not load image at {image_path}")
        return

    img = cv2.resize(img, (128, 128))
    
    # --- THESE ARE THE MISSING LINES ---
    calibrated = calibrator.calibrate(img)
    features = extractor.extract_features(calibrated)
    features_2d = features.reshape(1, -1) 
    # -----------------------------------

    # 5. Get Prediction and Confidence Score
    probabilities = model.predict_proba(features_2d)
    max_confidence = np.max(probabilities)
    best_guess_index = np.argmax(probabilities)
    result = class_names[best_guess_index]

    # 6. Your Custom Output Logic
    print("=========================================")
    if max_confidence < 0.50:
        print("Name of soil: Unknown")
        print(f"(Confidence was too low: {max_confidence*100:.2f}%)")
    elif result == "Noise":
        print("Name of soil: Invalid")
        print("(Not a soil or heavy interference detected)")
    else:
        print(f"Name of soil: {result}")
        print(f"(Confidence: {max_confidence*100:.2f}%)")
    print("=========================================\n")

if __name__ == "__main__":
    while True:
        test_image()
        again = input("Do you want to test another image? (y/n): ")
        if again.lower() != 'y':
            break