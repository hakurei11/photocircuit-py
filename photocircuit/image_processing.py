"""Classical OpenCV pipeline that detects where circuit elements sit.

Port of ``ImageProcessor.java`` / ``elem_detector.ipynb``:

    bilateral filter -> adaptive threshold -> closing -> thinning
    -> pruning (light/heavy) -> endpoints + curved-line residue
    -> dilated "patches" -> bounding rects (area-filtered)

Everything except the per-element classification happens here; the classifier
only labels the rects this module finds.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

__all__ = ["Rect", "ImageProcessingResult", "process_image", "detect_lines", "delete_elements"]


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    w: int
    h: int

    @property
    def area(self) -> int:
        return self.w * self.h


@dataclass
class ImageProcessingResult:
    circuit_element_rects: List[Rect] = field(default_factory=list)
    thresholded_image: Optional[np.ndarray] = None
    thinned_image: Optional[np.ndarray] = None


# --- 3x3 hit-or-miss kernels (CV_8S semantics: 1 fg, 0 don't-care, -1 bg) ---
# Expanded to all 4 rotations, same as the Java static initializer.
# OpenCV's MORPH_HITMISS accepts these int8 kernels directly (1=foreground,
# -1=background, 0=don't care); do NOT re-encode them.

def _all_rotations(kernels: np.ndarray) -> np.ndarray:
    out = np.zeros((kernels.shape[0] * 4, 3, 3), dtype=np.int8)
    for i in range(kernels.shape[0]):
        for rot in range(4):
            out[i * 4 + rot] = np.rot90(kernels[i], rot)
    return out


_KERNELS_THINNING = _all_rotations(np.array([
    [[0, -1, -1],
     [1, 1, -1],
     [0, 1, 0]],
    [[-1, -1, -1],
     [0, 1, 0],
     [1, 1, 1]],
], dtype=np.int8))

_KERNELS_PRUNING = _all_rotations(np.array([
    [[-1, -1, -1],
     [-1, 1, -1],
     [-1, 0, 0]],
    [[-1, -1, -1],
     [-1, 1, -1],
     [0, 0, -1]],
], dtype=np.int8))

_KERNELS_STRAIGHT = _all_rotations(np.array([
    [[0, 0, 0],
     [1, 1, 1],
     [0, 0, 0]],
    [[0, 1, 0],
     [0, 1, 0],
     [0, 1, 0]],
], dtype=np.int8))


def _iterative_reverse_hitmiss(src: np.ndarray, kernels, num_iterations: Optional[int] = None) -> np.ndarray:
    """Thin `src` by repeatedly removing hit-or-miss matches.

    Mirrors ImageProcessor.iterativeReverseHitmiss: runs to a fixed point
    when num_iterations is None, else exactly num_iterations times.
    """
    dst = src.copy()
    i = 0
    while num_iterations is None or i < num_iterations:
        tmp = dst.copy()
        for k in kernels:
            hitmiss = cv2.morphologyEx(tmp, cv2.MORPH_HITMISS, k)
            tmp = cv2.subtract(tmp, hitmiss)
        if num_iterations is None and np.array_equal(dst, tmp):
            break
        dst = tmp
        i += 1
    return dst


def _dilation(src: np.ndarray, num_iterations: int = 1) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=np.uint8)
    res = src.copy()
    for _ in range(num_iterations):
        res = cv2.dilate(res, kernel)
    return res


def resize_image_by_width(img: np.ndarray, new_width: int) -> np.ndarray:
    new_height = int(new_width * (img.shape[0] / img.shape[1]))
    return cv2.resize(img, (new_width, new_height))


def resize_image_by_height(img: np.ndarray, new_height: int) -> np.ndarray:
    new_width = int(new_height * (img.shape[1] / img.shape[0]))
    return cv2.resize(img, (new_width, new_height))


def process_image(img_in: np.ndarray) -> ImageProcessingResult:
    """Detect element bounding rects in a grayscale-ish circuit photo.

    Args:
        img_in: BGR or grayscale image (any size); treated like
            ImageProcessor.processImage's Mat input.

    Returns:
        rects + intermediate images (thresholded for classification,
        thinned for line detection).
    """
    if img_in.ndim == 3:
        img = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
    else:
        img = img_in

    img_filtered = cv2.bilateralFilter(img, 35, 10, 10)

    avg_color = int(img_filtered.mean())
    if avg_color > 100:
        img_filtered = cv2.bitwise_not(img_filtered)

    img_thresh = cv2.adaptiveThreshold(
        img_filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, -20)

    # Closing
    img_closed = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, np.ones((8, 8), np.uint8))

    # Thinning
    img_thinned = _iterative_reverse_hitmiss(img_closed, _KERNELS_THINNING)

    # Light pruning (for output / line detection)
    img_pruned_light = _iterative_reverse_hitmiss(img_thinned, _KERNELS_PRUNING, 5)

    # Heavy pruning (for endpoints)
    img_pruned = _iterative_reverse_hitmiss(img_thinned, _KERNELS_PRUNING, 22)

    # Get curved lines
    pre_curved_lines = _iterative_reverse_hitmiss(img_pruned, _KERNELS_STRAIGHT)
    curved_lines = cv2.GaussianBlur(pre_curved_lines, (101, 101), 0).astype(np.float32) * 2
    curved_lines = cv2.threshold(curved_lines, 6, 255, cv2.THRESH_BINARY)[1]
    curved_lines = _dilation(curved_lines.astype(np.uint8), 12)

    # Get endpoints
    endpoints = cv2.subtract(img_thinned, img_pruned)
    endpoints = cv2.GaussianBlur(endpoints, (101, 101), 0).astype(np.float32) * 2
    endpoints = cv2.threshold(endpoints, 4, 255, cv2.THRESH_BINARY)[1].astype(np.uint8)
    curved_lines = _dilation(curved_lines, 2)

    # Get patches
    elem_patches = _dilation(cv2.add(endpoints, curved_lines), 2)

    contours, _ = cv2.findContours(elem_patches, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = int(0.005 * img.shape[1] * img.shape[0])
    rects = [
        Rect(x, y, w, h)
        for (x, y, w, h) in (cv2.boundingRect(c) for c in contours)
        if w * h >= min_area
    ]

    return ImageProcessingResult(rects, img_thresh, img_pruned_light)


def detect_lines(image: np.ndarray) -> np.ndarray:
    """LSD line segments; returns an (N, 4) float array (x1, y1, x2, y2).

    (OpenCV 5.x's Python binding returns (N, 4); 4.x returns (N, 1, 4).)
    """
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_NONE, 1, 0.6, 2.0, 70)
    lines = lsd.detect(image)[0]
    if lines is None:
        return np.zeros((0, 4), dtype=np.float32)
    return lines.reshape(-1, 4)


def delete_elements(img: np.ndarray, rects) -> None:
    """Black out element rects in-place so line detection only sees wires."""
    for r in rects:
        img[r.y:r.y + r.h, r.x:r.x + r.w] = 0
