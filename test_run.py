import cv2
import numpy as np
import os
import xgboost as xgb
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from ClassificationModels.SVM_Calibrator import SoilCalibratorSVM
from ClassificationModels.CLS_extraction import LabColorExtractor
from SOCModels.XGBoost_SOC import SOCXGBPredictor

class SoilApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Soil Carbon Analysis")
        self.root.geometry("500x700")
        self.root.configure(bg="#2c3e50")

        self.calibrator = SoilCalibratorSVM()
        self.extractor = LabColorExtractor()
        self.model = xgb.XGBClassifier()
        self.soc_predictor = None
        
        if os.path.exists("soil_xgb_model.json"):
            self.model.load_model("soil_xgb_model.json")
            self.classes = np.load("soil_classes.npy", allow_pickle=True)
        if os.path.exists("soc_xgb_model.json"):
            try:
                self.soc_predictor = SOCXGBPredictor()
            except Exception as exc:
                messagebox.showwarning("SOC Model", f"SOC model load failed: {exc}")
        
        self.setup_ui()

    def setup_ui(self):
        tk.Label(self.root, text="SOIL ANALYSIS SYSTEM", font=("Arial", 16, "bold"), bg="#2c3e50", fg="white").pack(pady=20)
        self.display = tk.Label(self.root, bg="#34495e", width=40, height=15)
        self.display.pack(pady=10)
        
        tk.Button(self.root, text="LOAD IMAGE", command=self.load_img, width=20).pack(pady=5)
        tk.Button(self.root, text="ANALYZE", command=self.process, width=20, bg="#27ae60", fg="white").pack(pady=5)
        
        self.res_text = tk.Label(self.root, text="Result: ---", font=("Arial", 14), bg="#2c3e50", fg="white")
        self.res_text.pack(pady=20)
        self.img_path = None

    def load_img(self):
        path = filedialog.askopenfilename()
        if path:
            self.img_path = path
            img = Image.open(path).resize((300, 250))
            img_tk = ImageTk.PhotoImage(img)
            self.display.config(image=img_tk)
            self.display.image = img_tk

    def process(self):
        if not self.img_path: return
        raw = np.fromfile(self.img_path, dtype=np.uint8)
        img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        
        resized = cv2.resize(img, (128, 128))
        calibrated = self.calibrator.calibrate(resized)
        features = self.extractor.extract_features(calibrated).reshape(1, -1)
        
        probs = self.model.predict_proba(features)
        conf = np.max(probs)
        pred = self.classes[np.argmax(probs)]

        # Rejection logic for non-soil or low confidence
        if pred == "Not_Soil" or conf < 0.75:
            self.res_text.config(text="REJECTED: NOT SOIL", fg="#e74c3c")
        else:
            soc_text = "SOC: model not loaded"
            if self.soc_predictor is not None:
                soc_value = self.soc_predictor.predict_image(img)
                soc_text = f"SOC: {soc_value:.4f}"
            self.res_text.config(
                text=f"{pred} ({conf*100:.1f}%)\n{soc_text}", fg="#2ecc71"
            )

if __name__ == "__main__":
    root = tk.Tk()
    SoilApp(root)
    root.mainloop()