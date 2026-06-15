import cv2
import numpy as np


class BrightnessCalibrator:
    """
    Corrects for lighting differences between photos taken in varying
    field conditions by shifting the LAB L-channel mean to a fixed
    target intensity.

    Formerly misnamed 'SoilCalibratorSVM' — no SVM is involved.
    """

    def __init__(self, target_intensity: int = 128):
        self.target_intensity = target_intensity

    def calibrate(self, img_bgr: np.ndarray) -> np.ndarray | None:
        """
        Shift the luminance of *img_bgr* so its mean L value equals
        ``target_intensity``.

        Args:
            img_bgr: Input BGR image (uint8).

        Returns:
            Calibrated BGR image, or None if input is None.
        """
        if img_bgr is None:
            return None

        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        offset = self.target_intensity - np.mean(l)
        l_cal = np.clip(l.astype(np.float32) + offset, 0, 255).astype(np.uint8)

        calibrated_lab = cv2.merge((l_cal, a, b))
        return cv2.cvtColor(calibrated_lab, cv2.COLOR_LAB2BGR)