"""Frame preprocessing: low-light / night enhancement."""
import cv2
import numpy as np


class NightEnhancer:
    """CLAHE-based low-light enhancement on the L channel of LAB."""

    def __init__(self, clip_limit: float = 2.5, tile_grid: int = 8):
        self.clahe = cv2.createCLAHE(
            clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid)
        )

    def enhance(self, frame: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l2 = self.clahe.apply(l)
        merged = cv2.merge((l2, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    @staticmethod
    def is_low_light(frame: np.ndarray, threshold: float = 60.0) -> bool:
        """Cheap brightness check via grayscale mean."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(gray.mean()) < threshold
