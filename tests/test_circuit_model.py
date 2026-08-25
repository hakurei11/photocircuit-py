"""Tests for the circuit graph model — the pure-logic layer.

These cover the behavior the Java original had no tests for: node merging,
branch splitting, dedup, element attachment, straightening, and TikZ
serialization (including the rotation-encoded ",invert" logic).
"""

import numpy as np
import pytest

from photocircuit.circuit_model import Circuit, Element, Node, Segment
from photocircuit.classes import CLASSES
from photocircuit.classification import CircuitElement
from photocircuit.image_processing import Rect


def _lines(*segments):
    """Build an (N,4) LSD-style array from (x1,y1,x2,y2) tuples."""
    return np.array([list(s) for s in segments], dtype=np.float32)


class TestNodeMerging:
    def test_close_endpoints_merge_into_one_node(self):
        # Two segments whose endpoints are within 30px -> shared node.
        lines = _lines((0, 0, 100, 0), (105, 3, 200, 0))
        circuit = Circuit.from_lines_and_elements(lines, [])
        # 4 endpoints, but the two middle ones merge -> 3 nodes
        assert len(circuit.nodes) == 3
        assert len(circuit.segments) == 2

    def test_far_endpoints_stay_separate(self):
        lines = _lines((0, 0, 100, 0), (200, 0, 300, 0))
        circuit = Circuit.from_lines_and_elements(lines, [])
        assert len(circuit.nodes) == 4

    def test_identical_duplicate_segments_removed(self):
        lines = _lines((0, 0, 100, 0), (2, 2, 98, 0))  # near-identical
        circuit = Circuit.from_lines_and_elements(lines, [])
        assert len(circuit.segments) == 1


class TestBranchSplitting:
    def test_crossing_segment_splits_at_interior_node(self):
        # Horizontal wire with a T-branch node in the middle.
        # The long segment (0,0)-(300,0) must split at (150,2).
        lines = _lines((0, 0, 300, 0), (150, 2, 150, 100))
        circuit = Circuit.from_lines_and_elements(lines, [])
        # After split: (0,0)-(150,2), (150,2)-(300,0), plus the vertical = 3
        assert len(circuit.segments) == 3
        branch = [n for n in circuit.nodes if abs(n.x - 150) < 10 and abs(n.y - 2) < 10]
        assert len(branch) == 1
        assert len(branch[0].segments) == 3  # junction of 3 wires

    def test_no_split_when_no_interior_nodes(self):
        lines = _lines((0, 0, 100, 0), (200, 0, 300, 0))
        circuit = Circuit.from_lines_and_elements(lines, [])
        assert len(circuit.segments) == 2


class TestElementAttachment:
    def test_element_attaches_to_nodes_on_its_edges(self):
        # Wire (100,100)-(200,100); rect spans x[100,200] y[75,125].
        # (100,100) sits in the rect's left band, (200,100) in the right band.
        lines = _lines((100, 100, 200, 100))
        circuit = Circuit.from_lines_and_elements(lines, [])
        rect = Rect(100, 75, 100, 50)
        ce = CircuitElement(np.array([0.0]), rect)
        ce.best_guess_index = CLASSES.index("resistor_r0")
        Circuit._add_circuit_elements_to_segments(circuit.segments, circuit.nodes, [ce])
        elements = [s for s in circuit.segments if isinstance(s, Element)]
        assert len(elements) == 1
        assert elements[0].node1 is not None
        assert elements[0].node2 is not None
        assert elements[0].type_as_string == "resistor_r0"

    def test_ground_renders_eground(self):
        lines = _lines((100, 100, 200, 100))
        circuit = Circuit.from_lines_and_elements(lines, [])
        rect = Rect(100, 75, 100, 50)
        ce = CircuitElement(np.array([0.0]), rect)
        ce.best_guess_index = CLASSES.index("gnd_1")
        Circuit._add_circuit_elements_to_segments(circuit.segments, circuit.nodes, [ce])
        elements = [s for s in circuit.segments if isinstance(s, Element)]
        assert "eground" in elements[0].get_circuitikz_draw()


class TestStraightening:
    def test_column_nodes_snap_to_shared_x(self):
        # Two vertical chains at slightly different x should each get an
        # averaged x; each chain's nodes share one x afterwards.
        lines = _lines((0, 0, 2, 100), (5, 0, 3, 100))
        circuit = Circuit.from_lines_and_elements(lines, [])
        circuit.straighten()
        xs = {n.x for n in circuit.nodes}
        # 4 nodes in 2 chains -> at most 2 distinct x values
        assert len(xs) <= 2

    def test_y_flipped_by_invert_y(self):
        lines = _lines((0, 0, 100, 0))
        circuit = Circuit.from_lines_and_elements(lines, [])
        heights = [n.y for n in circuit.nodes]
        circuit.invert_y(100)
        assert all(n.y == 100 - h for n, h in zip(circuit.nodes, heights))


class TestCircuiTikZOutput:
    def test_wire_serializes_as_short(self):
        lines = _lines((0, 0, 100, 0))
        circuit = Circuit.from_lines_and_elements(lines, [])
        circuit.invert_y(0)
        tikz = circuit.to_circuitikz()
        assert "\\begin{circuitikz}" in tikz
        assert "to [short]" in tikz
        assert "\\end{circuitikz}" in tikz

    def test_element_invert_variants(self):
        """The r0..r3 rotation suffix decides `,invert` — spot-check diodes.

        diode is "consistent": forwards + r0/r1 -> no invert, r2/r3 -> invert.
        """
        def make(cls_name, node1, node2):
            e = Element(node1, node2, CLASSES.index(cls_name))
            return e

        n1, n2 = Node(0, 0), Node(100, 0)  # horizontal, node1 left of node2
        assert "invert" not in make("diode_r0", n1, n2).get_circuitikz_label()
        assert "invert" in make("diode_r2", n1, n2).get_circuitikz_label()
        # reversed direction flips the outcome
        assert "invert" in make("diode_r0", n2, n1).get_circuitikz_label()
        assert "invert" not in make("diode_r2", n2, n1).get_circuitikz_label()

    def test_battery_is_inconsistent_class(self):
        n1, n2 = Node(0, 0), Node(100, 0)
        assert "invert" in make_label("battery_r0", n1, n2)
        assert "invert" not in make_label("battery_r2", n1, n2)


def make_label(cls_name, node1, node2):
    return Element(node1, node2, CLASSES.index(cls_name)).get_circuitikz_label()


class TestSegmentBasics:
    def test_vertical_vs_horizontal(self):
        a, b, c = Node(0, 0), Node(10, 100), Node(100, 10)
        assert Segment(a, b).is_vertical()
        assert Segment(a, c).is_horizontal()
        assert Segment(a, None).is_vertical()  # dangling defaults vertical

    def test_other_node(self):
        a, b = Node(0, 0), Node(1, 1)
        s = Segment(a, b)
        assert s.get_other_node(a) is b
        assert s.get_other_node(b) is a

    def test_dangling_segment_draw_skipped(self):
        # Element with node1=None returns None (skipped in output)
        e = Element(None, Node(0, 0), 0)
        assert e.get_circuitikz_draw() is None


class TestRegressionAgainstJava:
    """Golden behavior: the README's example output shape."""

    def test_readme_example_elements_parse(self):
        # Every non-ground class must produce a valid TikZ label.
        # gnd_1 is rendered as `node[eground]{}` by get_circuitikz_draw,
        # never via get_circuitikz_label, so it is excluded here.
        valid = {"sV", "C", "L", "R", "battery", "battery2", "V", "I", "cI", "cV", "diode"}
        for i, name in enumerate(CLASSES):
            if name == "gnd_1":
                continue
            e = Element(Node(0, 0), Node(100, 0), i)
            label = e.get_circuitikz_label()
            assert label != "", f"{name} produced empty label"
            base = label.replace(",invert", "")
            assert base in valid, f"{name} -> unexpected label {label!r}"
