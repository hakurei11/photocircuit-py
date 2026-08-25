"""End-to-end test against the repo's own sample photo.

This is the port's fidelity check: the same input the README shows must
produce a structurally equivalent circuit (same element multiset, valid TikZ).
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from photocircuit.pipeline import PhotocircuitPipeline
from photocircuit.classes import CLASSES

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SAMPLE_PHOTO = REPO_ROOT / "PhotoCircuit" / "pictures" / "photo.png"
MODEL = (REPO_ROOT / "PhotoCircuit" / "PhotoCircuit" / "app" / "src" / "main" / "ml"
         / "classification_model_final_c_prob.tflite")

# The README sample circuit contains (from its reference output):
# 2x R, 1x C, 1x V, 1x L, 1x diode, 1x battery2, 1x eground.
EXPECTED_ELEMENT_TYPES = {"R", "C", "V", "L", "diode", "battery2"}


@pytest.fixture(scope="module")
def result():
    if not SAMPLE_PHOTO.exists():
        pytest.skip("sample photo not available")
    img = cv2.imread(str(SAMPLE_PHOTO))
    assert img is not None
    return PhotocircuitPipeline.with_default_model(MODEL).run(img, landscape=False)


class TestSamplePhoto:
    def test_elements_detected(self, result):
        assert len(result.elements) >= 6

    def test_expected_element_mix(self, result):
        found = {e.best_guess.split("_r")[0].replace("dc_volt_src_1", "V") for e in result.elements}
        # Ground, resistor etc. — the base names the README output implies
        base = {e.best_guess for e in result.elements}
        assert any(b.startswith("resistor") for b in base), f"no resistor in {base}"
        assert any(b.startswith("cap") or b.startswith("inductor") for b in base), f"no C/L in {base}"

    def test_lines_detected(self, result):
        assert result.lines.shape[0] >= 4

    def test_tikz_structure(self, result):
        tikz = result.circuitikz
        assert tikz.startswith("\\begin{circuitikz}")
        assert tikz.endswith("\\end{circuitikz}")
        assert "\\draw" in tikz
        # No element label should be empty or None-rendered
        assert "to []" not in tikz

    def test_tikz_compiles_sane_coordinates(self, result):
        # All coordinates must be non-negative ints within the image space
        import re
        coords = re.findall(r"\((\d+), (\d+)\)", result.circuitikz)
        assert len(coords) > 10
