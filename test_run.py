import cv2
import numpy as np
import os
import xgboost as xgb
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
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
        self.detail_text = tk.Label(
            self.root,
            text="",
            font=("Arial", 11),
            bg="#2c3e50",
            fg="#bdc3c7",
            justify="left"
        )
        self.detail_text.pack(pady=5)

        self.conf_bar = ttk.Progressbar(self.root, length=300, mode="determinate")
        self.conf_bar.pack(pady=6)
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
        if not self.img_path:
            return

        raw = np.fromfile(self.img_path, dtype=np.uint8)
        img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if img is None:
            messagebox.showerror("Image Error", "Could not read image. Try a different file.")
            return

        resized = cv2.resize(img, (128, 128))
        calibrated = self.calibrator.calibrate(resized)
        features = self.extractor.extract_features(calibrated).reshape(1, -1)

        probs = self.model.predict_proba(features)[0]
        top_idx = np.argsort(probs)[::-1][:3]
        top_classes = self.classes[top_idx]
        top_probs = probs[top_idx]

        pred = top_classes[0]
        conf = float(top_probs[0])
        margin = float(top_probs[0] - top_probs[1]) if len(top_probs) > 1 else conf

        # Update confidence bar
        self.conf_bar["value"] = conf * 100

        # Rejection logic
        if pred == "Not_Soil" or conf < 0.75:
            self.res_text.config(text="REJECTED: NOT SOIL", fg="#e74c3c")
            self.detail_text.config(text="")
            return

        # SOC prediction
        soc_text = "SOC: model not loaded"
        soc_detail = ""
        if self.soc_predictor is not None:
            soc_value = float(self.soc_predictor.predict_image(img))

            # Assume model outputs % SOC
            soc_percent = soc_value / 10.0
            soc_gkg = soc_value * 10.0

            # Categorize
            if soc_percent < 1.0:
                soc_cat = "Low"
                soc_explain = "Low organic carbon; fertility and structure may be limited."
            elif soc_percent <= 2.5:
                soc_cat = "Medium"
                soc_explain = "Moderate organic carbon; typical for many cultivated soils."
            else:
                soc_cat = "High"
                soc_explain = "High organic carbon; generally good structure and fertility."

            soc_text = f"SOC: {soc_percent:.3f}% ({soc_gkg:.1f} g/kg)"
            soc_detail = f"Category: {soc_cat}\n{soc_explain}"

        # Build detail lines
        top3_lines = [
            f"1) {top_classes[0]} - {top_probs[0]*100:.1f}%",
            f"2) {top_classes[1]} - {top_probs[1]*100:.1f}%",
            f"3) {top_classes[2]} - {top_probs[2]*100:.1f}%"
        ]

        warn_text = ""
        if margin < 0.10:
            warn_text = "WARNING: Low confidence margin"

        self.res_text.config(
            text=f"{pred} ({conf*100:.1f}%)\n{soc_text}",
            fg="#2ecc71"
        )
        self.detail_text.config(
            text="\n".join(
                top3_lines + [f"Margin: {margin:.3f}", warn_text, soc_detail]
            ).strip()
        )

if __name__ == "__main__":
    root = tk.Tk()
    SoilApp(root)
    root.mainloop()