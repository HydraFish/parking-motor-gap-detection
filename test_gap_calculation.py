import unittest

from process_parking_video import Detection, calculate_gaps


FRAME_WIDTH = 1000
FRAME_HEIGHT = 100


def detection_at(x: float, y: float = 50.0) -> Detection:
    return Detection(x - 5, y - 10, x + 5, y, 0.9)


def gap_profile() -> dict:
    return {
        "gap_tolerance": 0.10,
        "rows": [
            {
                "id": "TEST_ROW",
                "polygon": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                "axis": [1.0, 0.0],
                "boundary_start": [0.1, 0.5],
                "boundary_end": [0.9, 0.5],
                "edge_margin_pitch": 0.5,
                "reference_pitch": 0.2,
            }
        ],
    }


class GapCalculationTests(unittest.TestCase):
    def calculate(self, xs: list[float]):
        return calculate_gaps(
            [detection_at(x) for x in xs],
            gap_profile(),
            FRAME_WIDTH,
            FRAME_HEIGHT,
        )[0]

    def test_internal_gap_still_works(self) -> None:
        gaps = self.calculate([300, 700])
        internal = [gap for gap in gaps if gap.gap_type == "internal"]
        self.assertEqual(len(internal), 1)
        self.assertEqual(internal[0].capacity, 1)

    def test_edge_gaps_are_calculated(self) -> None:
        gaps = self.calculate([500])
        capacities = {gap.gap_type: gap.capacity for gap in gaps}
        self.assertEqual(capacities["edge_start"], 1)
        self.assertEqual(capacities["edge_end"], 1)

    def test_empty_row_has_capacity(self) -> None:
        gaps = self.calculate([])
        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0].gap_type, "empty_row")
        self.assertEqual(gaps[0].capacity, 4)

    def test_full_row_has_zero_capacity(self) -> None:
        gaps = self.calculate([200, 400, 600, 800])
        self.assertEqual(sum(gap.capacity for gap in gaps), 0)


if __name__ == "__main__":
    unittest.main()
