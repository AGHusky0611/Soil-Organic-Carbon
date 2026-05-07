import cv2
import numpy as np
from skimage.feature import local_binary_pattern

class LabColorExtractor:
    def __init__(self, lbp_radius=3, lbp_points=24):
        self.lbp_radius = lbp_radius
        self.lbp_points = lbp_points

    def extract_features(self, img_calibrated):
        if img_calibrated is None:
            return None
            
        lab = cv2.cvtColor(img_calibrated, cv2.COLOR_BGR2LAB)
        mean_l, mean_a, mean_b = cv2.mean(lab)[:3]
        std_l, std_a, std_b = np.std(lab[:,:,0]), np.std(lab[:,:,1]), np.std(lab[:,:,2])
        
        gray = cv2.cvtColor(img_calibrated, cv2.COLOR_BGR2GRAY)
        
        lbp = local_binary_pattern(gray, self.lbp_points, self.lbp_radius, method='uniform')
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, self.lbp_points + 3), range=(0, self.lbp_points + 2))
        lbp_hist = hist.astype("float")
        lbp_hist /= (lbp_hist.sum() + 1e-7)
        
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        return np.hstack([mean_l, mean_a, mean_b, std_l, std_a, std_b, lbp_hist, laplacian_var])