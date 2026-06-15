import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops


class LabColorExtractor:
    def __init__(self, lbp_radius=3, lbp_points=24):
        self.lbp_radius = lbp_radius
        self.lbp_points = lbp_points

    # ------------------------------------------------------------------
    # Preprocessing helpers
    # ------------------------------------------------------------------

    def _white_balance(self, img):
        """Apply white balance using the Gray World assumption."""
        result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        avg_a = np.mean(result[:, :, 1])
        avg_b = np.mean(result[:, :, 2])
        result[:, :, 1] -= (avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1
        result[:, :, 2] -= (avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1
        result = np.clip(result, 0, 255).astype(np.uint8)
        return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

    def _denoise(self, img):
        """Apply a light Gaussian blur to suppress camera noise."""
        return cv2.GaussianBlur(img, (3, 3), 0)

    def _dimensional_standardization(self, img):
        """Resize to 256×256 and normalise pixel intensity to [0, 255]."""
        standardized = cv2.resize(img, (256, 256))
        standardized = cv2.normalize(standardized, None, 0, 255, cv2.NORM_MINMAX)
        return standardized

    # ------------------------------------------------------------------
    # Public preprocessing entry-point
    # ------------------------------------------------------------------

    def preprocess_image(self, img):
        """
        Preprocess an image for display / quality inspection.

        Steps:
            1. White balance correction (Gray World)
            2. Light denoising (Gaussian blur)
            3. Dimensional standardisation (resize + intensity normalisation)

        Returns:
            Preprocessed BGR image, or None if *img* is None.
        """
        if img is None:
            return None
        wb_img      = self._white_balance(img)
        denoised    = self._denoise(wb_img)
        standardized = self._dimensional_standardization(denoised)
        return standardized

    # ------------------------------------------------------------------
    # Feature extraction helpers
    # ------------------------------------------------------------------

    def _extract_glcm_features(self, gray):
        """
        Extract GLCM texture features from a greyscale image.

        Iterates over 3 distances × 4 angles = 12 (distance, angle) pairs.
        For each pair, 6 GLCM properties are computed, yielding a flat
        1-D vector of length 72.

        FIX: graycoprops() returns an ndarray of shape (n_dist, n_angle).
        Each value is extracted as a Python scalar via [0, 0] so the
        feature list stays flat and np.array() produces a 1-D vector.
        """
        # Quantise to 8 levels (values 0-7) for GLCM efficiency
        gray_reduced = (gray // 32).astype(np.uint8)
        gray_reduced = np.clip(gray_reduced, 0, 7)   # guard against edge-case 256/32=8

        distances = [1, 2, 3]
        angles    = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

        glcm_features = []

        for distance in distances:
            for angle in angles:
                glcm = graycomatrix(
                    gray_reduced,
                    distances=[distance],
                    angles=[angle],
                    levels=8,
                    symmetric=True,
                    normed=True,
                )
                # graycoprops returns shape (n_distances, n_angles) = (1, 1) here.
                # Extract the scalar with [0, 0] so the list stays 1-D.
                glcm_features.extend([
                    graycoprops(glcm, 'contrast')[0, 0],
                    graycoprops(glcm, 'dissimilarity')[0, 0],
                    graycoprops(glcm, 'homogeneity')[0, 0],
                    graycoprops(glcm, 'energy')[0, 0],
                    graycoprops(glcm, 'correlation')[0, 0],
                    graycoprops(glcm, 'ASM')[0, 0],
                ])

        return np.array(glcm_features, dtype=np.float64)   # shape: (72,)

    # ------------------------------------------------------------------
    # Public feature-extraction entry-point
    # ------------------------------------------------------------------

    def extract_features(self, img_input):
        """
        Full preprocessing + feature extraction pipeline.

        Steps:
            1. White balance correction
            2. Denoising
            3. Dimensional standardisation
            4. GLCM texture feature extraction  → 72-D vector
            5. LAB colour statistics            →  7-D vector

        Returns:
            Flat 1-D float64 ndarray of length 79, or None if *img_input*
            is None.
        """
        if img_input is None:
            return None

        # 1-3. Preprocess
        wb_img       = self._white_balance(img_input)
        denoised     = self._denoise(wb_img)
        standardized = self._dimensional_standardization(denoised)

        # Convert to greyscale for GLCM
        gray = cv2.cvtColor(standardized, cv2.COLOR_BGR2GRAY)

        # 4. GLCM features  → shape (72,)
        glcm_features = self._extract_glcm_features(gray)

        # 5. LAB colour statistics
        lab = cv2.cvtColor(standardized, cv2.COLOR_BGR2LAB)
        mean_l, mean_a, mean_b = cv2.mean(lab)[:3]
        std_l  = np.std(lab[:, :, 0])
        std_a  = np.std(lab[:, :, 1])
        std_b  = np.std(lab[:, :, 2])

        # Laplacian variance as a sharpness proxy
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        lab_features = np.array(
            [mean_l, mean_a, mean_b, std_l, std_a, std_b, laplacian_var],
            dtype=np.float64,
        )   # shape: (7,)

        # Concatenate → shape (79,)
        return np.hstack([glcm_features, lab_features])