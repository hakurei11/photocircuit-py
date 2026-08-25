"""End-to-end pipeline: photo -> rects -> classifications -> circuit -> TikZ.

Python equivalent of the FirstFragment -> SecondFragment flow in the Android
app (minus the camera and the termbin upload, which the CLI doesn't need).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from .circuit_model import Circuit
from .classification import Classifier, CircuitElement
from .image_processing import (
    ImageProcessingResult,
    Rect,
    delete_elements,
    detect_lines,
    process_image,
    resize_image_by_height,
    resize_image_by_width,
)

__all__ = ["PipelineResult", "PhotocircuitPipeline", "run"]

_TARGET_LONG_EDGE = 800  # matches FirstFragment's resize before processing


@dataclass
class PipelineResult:
    processing: ImageProcessingResult
    elements: List[CircuitElement]
    lines: np.ndarray
    circuit: Circuit
    circuitikz: str


class PhotocircuitPipeline:
    def __init__(self, classifier: Optional[Classifier] = None):
        self._classifier = classifier

    @classmethod
    def with_default_model(cls, model_path: Optional[Path] = None) -> "PhotocircuitPipeline":
        return cls(Classifier(model_path))

    def run(self, img_bgr: np.ndarray, landscape: bool = True) -> PipelineResult:
        """Process one BGR image and return every intermediate stage.

        Args:
            img_bgr: image as read by cv2.imread (BGR).
            landscape: orientation hint for the pre-resize, matching the
                Android app's rotateCamera toggle. Portrait photos taken with
                the back camera arrive rotated 90°, hence the default True.
        """
        if img_bgr is None:
            raise ValueError("img_bgr is None")

        # FirstFragment scales the long edge to 800 before processing.
        h, w = img_bgr.shape[:2]
        if landscape:
            mat = resize_image_by_width(img_bgr, _TARGET_LONG_EDGE) if w >= h \
                else resize_image_by_height(img_bgr, _TARGET_LONG_EDGE)
        else:
            mat = resize_image_by_height(img_bgr, _TARGET_LONG_EDGE)

        processing = process_image(mat)

        # SecondFragment: classify patches on the thresholded image.
        if self._classifier is None:
            elements = [
                CircuitElement(np.zeros(1, np.float32), r)
                for r in processing.circuit_element_rects
            ]
        else:
            elements = self._classifier.classify(
                processing.thresholded_image, processing.circuit_element_rects)

        # Line detection on the thinned image minus the element patches.
        image_without_elements = processing.thinned_image.copy()
        delete_elements(image_without_elements, [e.rect for e in elements])
        lines = detect_lines(image_without_elements)

        circuit = Circuit.from_lines_and_elements(lines, elements)
        circuit.invert_y(processing.thresholded_image.shape[0])
        circuit.straighten()

        return PipelineResult(processing, elements, lines, circuit, circuit.to_circuitikz())


def run(image_path, model_path: Optional[Path] = None, landscape: bool = True) -> PipelineResult:
    """Convenience entry point: path in, full result out."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return PhotocircuitPipeline.with_default_model(model_path).run(img, landscape=landscape)
