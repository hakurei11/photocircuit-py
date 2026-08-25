"""Circuit graph model: nodes, wire segments, and classified elements.

Port of the ``circuit_model`` Java package (Node.java, Segment.java,
Element.java, Circuit.java). LSD line segments become a Node/Segment graph,
branch points split segments, classified element rects attach as Elements,
then the graph is straightened onto an axis-aligned grid and serialized to
CircuiTikZ.
"""

from typing import List, Optional

import numpy as np

from .classes import CLASSES

__all__ = ["Node", "Segment", "Element", "Circuit"]


class Node:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
        self.segments: List["Segment"] = []

    def add_segment(self, segment: "Segment") -> None:
        self.segments.append(segment)

    def remove_segment(self, segment: "Segment") -> bool:
        try:
            self.segments.remove(segment)
            return True
        except ValueError:
            return False

    def get_node_on_direction(self, dir: int) -> Optional["Node"]:
        """dir: 0 right, 1 top, 2 left, 3 bottom (image coordinates)."""
        s = self.get_segment_on_direction(dir)
        if s is None:
            return None
        return s.get_other_node(self)

    def get_segment_on_direction(self, dir: int) -> Optional["Segment"]:
        for s in self.segments:
            other = s.get_other_node(self)
            if other is None:
                continue
            if s.is_vertical():
                found_dir = 3 if other.y > self.y else 1
            else:
                found_dir = 0 if other.x > self.x else 2
            if found_dir == dir:
                return s
        return None

    def __eq__(self, other):
        return isinstance(other, Node) and self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))

    def __repr__(self):
        return f"({self.x}, {self.y})"


class Segment:
    """A wire between two nodes. `node2` may be None (dangling end)."""

    def __init__(self, node1: Optional[Node], node2: Optional[Node]):
        self.node1 = node1
        self.node2 = node2

    def get_other_node(self, node: Node) -> Optional[Node]:
        if node is self.node1:
            return self.node2
        return self.node1

    def is_vertical(self) -> bool:
        if self.node1 is None or self.node2 is None:
            return True
        return abs(self.node1.x - self.node2.x) < abs(self.node1.y - self.node2.y)

    def is_horizontal(self) -> bool:
        return not self.is_vertical()

    def get_circuitikz_label(self) -> str:
        return "short"

    def get_circuitikz_draw(self) -> str:
        n1, n2 = self.node1, self.node2
        s1 = f"({n1.x}, {n1.y})"
        s2 = f"({n2.x}, {n2.y})"
        return f"\\draw {s1} to [short] {s2};\n"

    def __eq__(self, other):
        return isinstance(other, Segment) and self.node1 == other.node1 and self.node2 == other.node2

    def __hash__(self):
        return hash((self.node1, self.node2))


class Element(Segment):
    """A classified circuit element connecting two nodes (node2 may be None,
    e.g. a ground symbol with only one terminal)."""

    def __init__(self, node1: Optional[Node], node2: Optional[Node], type_index: int):
        super().__init__(node1, node2)
        self.type_index = type_index

    @property
    def type_as_string(self) -> str:
        return CLASSES[self.type_index]

    def _is_forwards(self) -> bool:
        if self.is_vertical():
            return self.node1.y < self.node2.y
        return self.node1.x < self.node2.x

    def _invert_if_needed(self, in_str: str, rot: str, is_consistent: bool) -> str:
        # rot is the rotation suffix "r0".."r3"; Java uses rot.charAt(1),
        # i.e. the digit char.
        digit = rot[1] if len(rot) > 1 else rot
        is_actually_forwards = (is_consistent and self._is_forwards()) or (
            not is_consistent and not self._is_forwards())
        needs_invert = (
            (is_actually_forwards and digit in ("2", "3"))
            or (not is_actually_forwards and digit in ("0", "1"))
        )
        return in_str + ",invert" if needs_invert else in_str

    def get_circuitikz_label(self) -> str:
        t = self.type_as_string
        if t.startswith("ac_src"):
            return "sV"
        if t.startswith("battery"):
            return self._invert_if_needed("battery", t.split("_")[1], False)
        if t.startswith("cap"):
            return "C"
        if t.startswith("curr_src"):
            return self._invert_if_needed("I", t.split("_")[2], True)
        if t.startswith("dc_volt_src_1"):
            return self._invert_if_needed("V", t.split("_")[4], False)
        if t.startswith("dc_volt_src_2"):
            return self._invert_if_needed("battery2", t.split("_")[4], False)
        if t.startswith("dep_curr_src"):
            return self._invert_if_needed("cI", t.split("_")[3], True)
        if t.startswith("dep_volt"):
            return self._invert_if_needed("cV", t.split("_")[2], False)
        if t.startswith("diode"):
            return self._invert_if_needed("diode", t.split("_")[1], True)
        if t.startswith("inductor"):
            return "L"
        if t.startswith("resistor"):
            return "R"
        return ""

    def get_circuitikz_draw(self) -> Optional[str]:
        if self.node1 is None:
            return None
        n1 = f"({self.node1.x}, {self.node1.y})"
        if self.type_as_string.startswith("gnd_1") or self.node2 is None:
            return f"\\draw {n1} node[eground]{{}};\n"
        n2 = f"({self.node2.x}, {self.node2.y})"
        return f"\\draw {n1} to [{self.get_circuitikz_label()}] {n2};\n"

    def __eq__(self, other):
        return isinstance(other, Element) and super().__eq__(other) and self.type_index == other.type_index

    def __hash__(self):
        return hash((self.node1, self.node2, self.type_index))


def _points_are_close(x1: int, y1: int, x2: int, y2: int, threshold: int = 30) -> bool:
    return abs(x1 - x2) <= threshold and abs(y1 - y2) <= threshold


def _point_by_rect(x: int, y: int, rect) -> int:
    """Return which edge band of `rect` (x, y) falls in.

    0 right, 1 top, 2 left, 3 bottom, -1 none. `rect` is a photocircuit
    Rect (x, y, w, h). Bands are `threshold` wide, centered on each edge —
    mirrors Circuit.pointByRect.
    """
    threshold = 25
    half = threshold // 2
    rx, ry, rw, rh = rect.x, rect.y, rect.w, rect.h
    bands = [
        (rx + rw - half, ry, threshold, rh),          # right
        (rx, ry - half, rw, threshold),                # top
        (rx - half, ry, threshold, rh),                # left
        (rx, ry + rh - half, rw, threshold),           # bottom
    ]
    for dir_, (bx, by, bw, bh) in enumerate(bands):
        if bx <= x < bx + bw and by <= y < by + bh:
            return dir_
    return -1


class Circuit:
    def __init__(self, segments: List[Segment], nodes: List[Node]):
        self.segments = segments
        self.nodes = nodes

    @classmethod
    def from_lines_and_elements(cls, lines: np.ndarray, circuit_elements) -> "Circuit":
        """Build the graph from LSD `lines` ((N,4) x1,y1,x2,y2) and
        classified CircuitElements."""
        nodes: List[Node] = []
        segments: List[Segment] = []

        for i in range(lines.shape[0]):
            x1, y1, x2, y2 = (int(v) for v in lines[i])
            n1 = None
            n2 = None
            for n in nodes:
                if _points_are_close(x1, y1, n.x, n.y):
                    n1 = n
                if _points_are_close(x2, y2, n.x, n.y):
                    n2 = n
            if n1 is None:
                n1 = Node(x1, y1)
                nodes.append(n1)
            if n2 is None:
                n2 = Node(x2, y2)
                nodes.append(n2)
            seg = Segment(n1, n2)
            n1.add_segment(seg)
            n2.add_segment(seg)
            segments.append(seg)

        cls._split_segments_at_branches(segments, nodes)
        segments = cls._remove_duplicate_segments(segments)
        cls._add_circuit_elements_to_segments(segments, nodes, circuit_elements)

        return cls(segments, nodes)

    @staticmethod
    def _split_segments_at_branches(segments: List[Segment], nodes: List[Node]) -> None:
        n_segments = len(segments)
        i = 0
        while i < n_segments:
            s = segments[i]
            node1, node2 = s.node1, s.node2

            x1 = min(node1.x, node2.x)
            x2 = max(node1.x, node2.x)
            y1 = min(node1.y, node2.y)
            y2 = max(node1.y, node2.y)

            if s.is_vertical():
                delta_x = x2 - x1
                if delta_x < 10:
                    x1 = x1 - (10 - delta_x) // 2
                    x2 = x2 + (10 - delta_x) // 2
            else:
                delta_y = y2 - y1
                if delta_y < 10:
                    y1 = y1 - (10 - delta_y) // 2
                    y2 = y2 + (10 - delta_y) // 2

            colliding = [n for n in nodes
                         if n is not node1 and n is not node2
                         and x1 < n.x < x2 and y1 < n.y < y2]

            if s.is_vertical():
                colliding.sort(key=lambda n: n.y)
                if node1.y > node2.y:
                    node1, node2 = node2, node1
            else:
                colliding.sort(key=lambda n: n.x)
                if node1.x > node2.x:
                    node1, node2 = node2, node1

            if colliding:
                chain = [node1] + colliding + [node2]
                for j in range(len(chain) - 1):
                    first, second = chain[j], chain[j + 1]
                    new_seg = Segment(first, second)
                    if j == 0:
                        node1.remove_segment(s)
                    if j == len(chain) - 2:
                        node2.remove_segment(s)
                    segments.append(new_seg)
                    first.add_segment(new_seg)
                    second.add_segment(new_seg)
                segments.pop(i)
                n_segments -= 1
            else:
                i += 1

    @staticmethod
    def _remove_duplicate_segments(in_segments: List[Segment]) -> List[Segment]:
        out: List[Segment] = []
        for s1 in in_segments:
            found = False
            for s2 in out:
                if (s1.node1 == s2.node1 and s1.node2 == s2.node2) or \
                   (s1.node1 == s2.node2 and s1.node2 == s2.node1):
                    found = True
                    break
            if not found:
                out.append(s1)
        return out

    @staticmethod
    def _add_circuit_elements_to_segments(segments, nodes, circuit_elements) -> None:
        for circuit_element in circuit_elements:
            rect = circuit_element.rect
            n1 = None
            n2 = None
            for n in nodes:
                direction = _point_by_rect(n.x, n.y, rect)
                if direction != -1:
                    if n1 is None:
                        n1 = n
                    else:
                        n2 = n
                        break
            element = Element(n1, n2, circuit_element.best_guess_index)
            if n1 is not None:
                n1.add_segment(element)
            if n2 is not None:
                n2.add_segment(element)
            segments.append(element)

    def invert_y(self, height: int) -> None:
        """Flip to math coordinates (y up) before serializing."""
        for n in self.nodes:
            n.y = height - n.y

    def straighten_single_dim(self, dimension: int) -> None:
        """Snap nodes onto rows (dimension=0) / columns (dimension=1).

        Walks each chain of connected nodes along the perpendicular axis and
        averages their perpendicular coordinate.
        """
        explored: List[Node] = []
        sorted_nodes = sorted(
            self.nodes,
            key=(lambda n: n.x) if dimension == 0 else (lambda n: n.y),
        )
        direction = 0 if dimension == 0 else 3

        for n in sorted_nodes:
            if n in explored:
                continue
            connected: List[Node] = []
            next_node = n
            total = 0
            while next_node is not None:
                total += next_node.y if dimension == 0 else next_node.x
                connected.append(next_node)
                explored.append(next_node)
                next_node = next_node.get_node_on_direction(direction)

            average = total // len(connected)
            for nn in connected:
                if dimension == 0:
                    nn.y = average
                else:
                    nn.x = average

    def straighten(self) -> None:
        self.straighten_single_dim(0)
        self.straighten_single_dim(1)

    def to_circuitikz(self) -> str:
        parts = ["\\begin{circuitikz}[american,x=0.01cm,y=0.01cm]\n"]
        for segment in self.segments:
            draw = segment.get_circuitikz_draw()
            if draw is not None:
                parts.append(draw)
        parts.append("\\end{circuitikz}")
        return "".join(parts)
