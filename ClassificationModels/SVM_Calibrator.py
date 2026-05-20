import cv2
import numpy as np

class SoilCalibratorSVM:
    def __init__(self, target_intensity=128):
        self.target_intensity = target_intensity

    def calibrate(self, img_bgr):
        if img_bgr is None:
            return None
            
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        current_mean = np.mean(l)
        offset = self.target_intensity - current_mean
        
        l_calibrated = cv2.add(l, np.array([offset], dtype=np.float64))
        l_calibrated = np.clip(l_calibrated, 0, 255).astype(np.uint8)
        
        calibrated_lab = cv2.merge((l_calibrated, a, b))
        return cv2.cvtColor(calibrated_lab, cv2.COLOR_LAB2BGR)