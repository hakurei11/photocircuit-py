"""TFLite-backed element classifier.

Port of ``MLController.java``. Loads the same .tflite model shipped with the
Android app; crops each detected rect from the thresholded image, letterboxes
it onto a black 120x120 canvas, and reads the winning class.
"""

from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import numpy as np

from .classes import CLASSES
from .image_processing import Rect, resize_image_by_width, resize_image_by_height

__all__ = ["CircuitElement", "Classifier", "default_model_path"]

# Path to the model used by the Android app, relative to this repo.
_REPO_MODEL = (
    Path(__file__).resolve().parent.parent.parent
    / "PhotoCircuit" / "PhotoCircuit" / "app" / "src" / "main" / "ml"
    / "classification_model_final_c_prob.tflite"
)


def default_model_path() -> Path:
    return _REPO_MODEL


class CircuitElement:
    """One detected element: its rect, class probabilities, and best guess."""

    def __init__(self, probabilities: np.ndarray, rect: Rect):
        self.probabilities = probabilities
        self.rect = rect
        self.best_guess_index = int(np.argmax(probabilities)) if probabilities is not None else -1

    @property
    def best_guess(self) -> str:
        return CLASSES[self.best_guess_index] if 0 <= self.best_guess_index < len(CLASSES) else ""


def _preprocess(img: np.ndarray) -> np.ndarray:
    """Letterbox onto black 120x120, float32 — MLController.preprocessImage."""
    h, w = img.shape[:2]
    if w >= h:
        resized = resize_image_by_width(img, 120)
    else:
        resized = resize_image_by_height(img, 120)
    canvas = np.zeros((120, 120), dtype=np.uint8)
    rh, rw = resized.shape[:2]
    y0 = (120 - rh) // 2
    x0 = (120 - rw) // 2
    canvas[y0:y0 + rh, x0:x0 + rw] = resized
    return canvas.astype(np.float32)


class Classifier:
    def __init__(self, model_path: Optional[Path] = None):
        from ai_edge_litert.interpreter import Interpreter

        path = Path(model_path) if model_path is not None else default_model_path()
        if not path.exists():
            raise FileNotFoundError(
                f"TFLite model not found: {path}. Pass model_path=... or place the "
                "model somewhere reachable."
            )
        self._interp = Interpreter(model_path=str(path))
        self._interp.allocate_tensors()
        self._inp = self._interp.get_input_details()[0]
        self._out = self._interp.get_output_details()[0]
        expected_in = (1, 120, 120, 1)
        if tuple(self._inp["shape"]) != expected_in:
            raise ValueError(f"Unexpected model input shape {self._inp['shape']}, want {expected_in}")

    def classify_patch(self, patch: np.ndarray) -> np.ndarray:
        """Classify one grayscale patch; returns the class probability vector."""
        x = _preprocess(patch)[np.newaxis, :, :, np.newaxis]
        self._interp.set_tensor(self._inp["index"], x)
        self._interp.invoke()
        return self._interp.get_tensor(self._out["index"])[0]

    def classify(self, image: np.ndarray, rects: Sequence[Rect]) -> List[CircuitElement]:
        """Classify each rect crop of `image` (the thresholded image)."""
        out = []
        for r in rects:
            crop = image[r.y:r.y + r.h, r.x:r.x + r.w]
            if crop.size == 0:
                out.append(CircuitElement(np.zeros(len(CLASSES), np.float32), r))
                continue
            probs = self.classify_patch(crop)
            out.append(CircuitElement(probs, r))
        return out
