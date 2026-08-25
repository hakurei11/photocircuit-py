"""Tests for the image-processing layer (synthetic fixtures, no photos needed)."""

import numpy as np
import cv2
import pytest

from photocircuit.image_processing import (
    Rect, process_image, detect_lines, delete_elements,
    _KERNELS_THINNING, _KERNELS_PRUNING, _KERNELS_STRAIGHT)


class TestHitMissKernels:
    def test_kernels_expanded_to_4_rotations(self):
        assert _KERNELS_THINNING.shape == (8, 3, 3)
        assert _KERNELS_PRUNING.shape == (8, 3, 3)
        assert _KERNELS_STRAIGHT.shape == (8, 3, 3)
        assert _KERNELS_THINNING.dtype == np.int8

    def test_thinning_removes_a_two_px_bar_to_one(self):
        # A 2px-wide horizontal bar should thin to a single line.
        img = np.zeros((9, 9), np.uint8)
        img[3:5, 1:8] = 255
        from photocircuit.image_processing import _iterative_reverse_hitmiss
        thinned = _iterative_reverse_hitmiss(img, _KERNELS_THINNING)
        # one row of 7 should remain
        row_sums = thinned.sum(axis=1)
        assert (row_sums == 255 * 7).sum() == 1


class TestProcessImageSynthetic:
    def _schematic(self, size=(600, 800)):
        """Draw a resistor zigzag + wires on white, mimicking a photo."""
        img = np.full(size, 255, np.uint8)
        # Two horizontal wires meeting a zigzag in the middle
        cv2.line(img, (100, 300), (300, 300), 0, 4)
        cv2.line(img, (500, 300), (700, 300), 0, 4)
        # Zigzag (resistor-like) between 300 and 500
        pts = [(300, 300), (330, 270), (370, 330), (410, 270), (450, 330), (470, 300), (500, 300)]
        for p, q in zip(pts, pts[1:]):
            cv2.line(img, p, q, 0, 4)
        return img

    def test_finds_at_least_one_patch(self):
        result = process_image(self._schematic())
        assert len(result.circuit_element_rects) >= 1
        assert result.thresholded_image is not None
        assert result.thinned_image is not None

    def test_rect_lands_on_zigzag_region(self):
        result = process_image(self._schematic())
        rects = result.circuit_element_rects
        # Some detected rect must overlap the zigzag x-range (300..500)
        overlapping = [r for r in rects if r.x < 500 and r.x + r.w > 300]
        assert overlapping, f"no rect overlaps zigzag: {rects}"

    def test_tiny_specks_filtered_by_min_area(self):
        img = np.full((600, 800), 255, np.uint8)
        cv2.circle(img, (400, 300), 3, 0, -1)  # tiny dot
        result = process_image(img)
        assert result.circuit_element_rects == []


class TestDetectLines:
    def test_detects_long_straight_wire(self):
        img = np.zeros((400, 600), np.uint8)
        cv2.line(img, (50, 200), (550, 200), 255, 3)
        lines = detect_lines(img)
        assert lines.shape[0] >= 1
        assert lines.shape[1] == 4  # (N, 4)
        # At least one detected line should be roughly horizontal, long
        best = max(lines.tolist(), key=lambda l: abs(l[2] - l[0]) + abs(l[3] - l[1]))
        assert abs(best[3] - best[1]) < 10  # near-horizontal

    def test_empty_image_gives_zero_lines(self):
        img = np.zeros((100, 100), np.uint8)
        assert detect_lines(img).shape[0] == 0


class TestDeleteElements:
    def test_rects_blacked_out(self):
        img = np.full((100, 100), 255, np.uint8)
        delete_elements(img, [Rect(10, 10, 20, 20)])
        assert img[10:30, 10:30].sum() == 0
        assert img[0:10, :].sum() > 0
