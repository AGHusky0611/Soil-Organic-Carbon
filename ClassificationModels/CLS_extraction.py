import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from scipy import ndimage

class LabColorExtractor:
    def __init__(self, lbp_radius=3, lbp_points=24):
        self.lbp_radius = lbp_radius
        self.lbp_points = lbp_points

    def _white_balance(self, img):
        """Apply white balance using Gray World assumption"""
        result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        avg_a = np.mean(result[:, :, 1])
        avg_b = np.mean(result[:, :, 2])
        result[:, :, 1] -= ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result[:, :, 2] -= ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result = np.clip(result, 0, 255).astype(np.uint8)
        return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

    def _denoise(self, img):
        """Apply light Gaussian blur to remove camera noise"""
        return cv2.GaussianBlur(img, (3, 3), 0)

    def _dimensional_standardization(self, img):
        """Normalize image dimensions and intensity"""
        # Ensure fixed size
        standardized = cv2.resize(img, (256, 256))
        # Normalize intensity range
        standardized = cv2.normalize(standardized, None, 0, 255, cv2.NORM_MINMAX)
        return standardized

    def preprocess_image(self, img):
        """
        Preprocess image for better visual quality
        
        Steps:
        1. White balance correction
        2. Light denoising (Gaussian blur)
        3. Dimensional standardization
        """
        if img is None:
            return None
        
        wb_img = self._white_balance(img)
        denoised = self._denoise(wb_img)
        standardized = self._dimensional_standardization(denoised)
        return standardized

    def _extract_glcm_features(self, gray):
        """Extract GLCM (Gray Level Co-occurrence Matrix) texture features"""
        # Reduce bit depth for GLCM computation
        gray_reduced = (gray / 32).astype(np.uint8)
        
        # Compute GLCM for multiple directions
        distances = [1, 2, 3]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        glcm_features = []
        
        for distance in distances:
            for angle in angles:
                glcm = graycomatrix(
                    gray_reduced, 
                    distances=[distance], 
                    angles=[angle], 
                    levels=8, 
                    symmetric=True, 
                    normed=True
                )
                
                # REVISION: Removed the 'glcm = glcm[:, :, 0, 0]' slice so it remains 4D
                
                # Extract GLCM properties
                contrast = graycoprops(glcm, 'contrast')
                dissimilarity = graycoprops(glcm, 'dissimilarity')
                homogeneity = graycoprops(glcm, 'homogeneity')
                energy = graycoprops(glcm, 'energy')
                correlation = graycoprops(glcm, 'correlation')
                asm = graycoprops(glcm, 'asm')
                
                glcm_features.extend([
                    contrast, dissimilarity, homogeneity, 
                    energy, correlation, asm
                ])
        
        return np.array(glcm_features)

    def extract_features(self, img_input):
        """
        Complete preprocessing and feature extraction pipeline
        
        Steps:
        1. White balance correction
        2. Denoising
        3. Dimensional standardization
        4. GLCM texture feature extraction
        """
        if img_input is None:
            return None
        
        # 1. White balance
        wb_img = self._white_balance(img_input)
        
        # 2. Denoise
        denoised = self._denoise(wb_img)
        
        # 3. Dimensional standardization
        standardized = self._dimensional_standardization(denoised)
        
        # Convert to grayscale for GLCM
        gray = cv2.cvtColor(standardized, cv2.COLOR_BGR2GRAY)
        
        # 4. Extract GLCM features
        glcm_features = self._extract_glcm_features(gray)
        
        # Also extract basic LAB statistics for complementary information
        lab = cv2.cvtColor(standardized, cv2.COLOR_BGR2LAB)
        mean_l, mean_a, mean_b = cv2.mean(lab)[:3]
        std_l, std_a, std_b = np.std(lab[:,:,0]), np.std(lab[:,:,1]), np.std(lab[:,:,2])
        
        # Laplacian variance for sharpness
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Combine all features
        lab_features = np.array([mean_l, mean_a, mean_b, std_l, std_a, std_b, laplacian_var])
        
        return np.hstack([glcm_features, lab_features])