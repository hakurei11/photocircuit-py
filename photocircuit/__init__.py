"""Photograph a hand-drawn circuit schematic, get CircuiTikZ code.

Python port of the PhotoCircuit Android app's recognition pipeline.
"""

from .classes import CLASSES
from .image_processing import (
    ImageProcessingResult,
    Rect,
    process_image,
    detect_lines,
    delete_elements,
)
from .classification import Classifier, CircuitElement, default_model_path
from .circuit_model import Circuit, Node, Segment, Element

__version__ = "0.1.0"

__all__ = [
    "CLASSES",
    "Classifier",
    "Circuit",
    "CircuitElement",
    "Element",
    "ImageProcessingResult",
    "Node",
    "Rect",
    "Segment",
    "default_model_path",
    "delete_elements",
    "detect_lines",
    "process_image",
]
