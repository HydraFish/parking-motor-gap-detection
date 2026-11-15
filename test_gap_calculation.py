import unittest

from process_parking_video import Detection, calculate_gaps


FRAME_WIDTH = 1000
FRAME_HEIGHT = 100


def detection_at(x: float, y: float = 50.0) -> Detection:
    return Detection(x - 5, y - 10, x + 5, y, 0.9)


def gap_profile(
    *,
    start_visible: bool = True,
    end_visible: bool = True,
    axis: list[float] | None = None,
    boundary_start: list[float] | None = None,
    boundary_end: list[float] | None = None,
    polygon: list[list[float]] | None = None,
    reference_pitch: float = 0.2,
) -> dict:
    return {
        "gap_tolerance": 0.10,
        "rows": [
            {
                "id": "TEST_ROW",
                "polygon": polygon
                or [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                "axis": axis or [1.0, 0.0],
                "boundary_start": boundary_start or [0.1, 0.5],
                "boundary_end": boundary_end or [0.9, 0.5],
                "boundary_visibility": {
                    "start": start_visible,
                    "end": end_visible,
                },
                "edge_margin_pitch": 0.5,
                "reference_pitch": reference_pitch,
            }
        ],
    }


class GapCalculationTests(unittest.TestCase):
    def calculate(
        self, xs: list[float], profile: dict | None = None, y: float = 50.0
    ):
        return calculate_gaps(
            [detection_at(x, y) for x in xs],
            profile or gap_profile(),
            FRAME_WIDTH,
            FRAME_HEIGHT,
        )[0]

    def calculate_with_counts(
        self, xs: list[float], profile: dict | None = None, y: float = 50.0
    ):
        return calculate_gaps(
            [detection_at(x, y) for x in xs],
            profile or gap_profile(),
            FRAME_WIDTH,
            FRAME_HEIGHT,
        )

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

    def test_empty_row_with_both_boundaries_visible_has_capacity(self) -> None:
        gaps = self.calculate([])
        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0].gap_type, "empty_row")
        self.assertEqual(gaps[0].capacity, 4)

    def test_full_row_has_zero_capacity(self) -> None:
        gaps = self.calculate([200, 400, 600, 800])
        self.assertEqual(sum(gap.capacity for gap in gaps), 0)

    def test_motor_before_boundary_is_not_a_member(self) -> None:
        gaps, row_counts = self.calculate_with_counts([50])
        self.assertEqual(row_counts["TEST_ROW"], 0)
        self.assertEqual(
            [(gap.gap_type, gap.capacity) for gap in gaps], [("empty_row", 4)]
        )

    def test_motor_after_boundary_is_not_a_member(self) -> None:
        gaps, row_counts = self.calculate_with_counts([950])
        self.assertEqual(row_counts["TEST_ROW"], 0)
        self.assertEqual(
            [(gap.gap_type, gap.capacity) for gap in gaps], [("empty_row", 4)]
        )

    def test_motor_outside_polygon_is_not_a_member(self) -> None:
        profile = gap_profile(
            polygon=[[0.0, 0.0], [1.0, 0.0], [1.0, 0.6], [0.0, 0.6]],
            boundary_start=[0.1, 0.3],
            boundary_end=[0.9, 0.3],
        )
        gaps, row_counts = self.calculate_with_counts([500], profile, y=80.0)
        self.assertEqual(row_counts["TEST_ROW"], 0)
        self.assertEqual(
            [(gap.gap_type, gap.capacity) for gap in gaps], [("empty_row", 4)]
        )

    def test_reversed_axis_and_boundaries(self) -> None:
        profile = gap_profile(
            axis=[-1.0, 0.0],
            boundary_start=[0.9, 0.5],
            boundary_end=[0.1, 0.5],
        )
        gaps, row_counts = self.calculate_with_counts([300, 700], profile)
        self.assertEqual(row_counts["TEST_ROW"], 2)
        self.assertEqual(
            [(gap.gap_type, gap.capacity) for gap in gaps],
            [("edge_start", 0), ("internal", 1), ("edge_end", 0)],
        )

    def test_asymmetric_edge_start(self) -> None:
        gaps = self.calculate([700])
        capacities = {gap.gap_type: gap.capacity for gap in gaps}
        self.assertEqual(capacities["edge_start"], 2)
        self.assertEqual(capacities["edge_end"], 0)

    def test_asymmetric_edge_end(self) -> None:
        gaps = self.calculate([300])
        capacities = {gap.gap_type: gap.capacity for gap in gaps}
        self.assertEqual(capacities["edge_start"], 0)
        self.assertEqual(capacities["edge_end"], 2)

    def test_positive_intervals_are_not_double_counted(self) -> None:
        gaps = self.calculate([300, 500], gap_profile(reference_pitch=0.1))
        positive = [gap for gap in gaps if gap.capacity > 0]

        self.assertEqual(
            [(gap.gap_type, gap.capacity) for gap in positive],
            [("edge_start", 1), ("internal", 1), ("edge_end", 3)],
        )
        self.assertEqual(sum(gap.capacity for gap in positive), 5)
        self.assertEqual(positive[0].end, positive[1].start)
        self.assertEqual(positive[1].end, positive[2].start)
        self.assertLessEqual(positive[0].end[0], positive[1].start[0])
        self.assertLessEqual(positive[1].end[0], positive[2].start[0])

    def test_invisible_start_boundary_omits_edge_start(self) -> None:
        profile = gap_profile(start_visible=False, end_visible=True)
        gaps = self.calculate([700], profile)
        self.assertEqual([gap.gap_type for gap in gaps], ["edge_end"])

    def test_invisible_end_boundary_omits_edge_end(self) -> None:
        profile = gap_profile(start_visible=True, end_visible=False)
        gaps = self.calculate([300], profile)
        self.assertEqual([gap.gap_type for gap in gaps], ["edge_start"])

    def test_empty_row_with_one_invisible_boundary_is_not_reported(self) -> None:
        profile = gap_profile(start_visible=True, end_visible=False)
        gaps, row_counts = self.calculate_with_counts([], profile)
        self.assertEqual(row_counts["TEST_ROW"], 0)
        self.assertEqual(gaps, [])

    def test_equal_boundaries_raise_clear_error(self) -> None:
        profile = gap_profile(
            boundary_start=[0.5, 0.5],
            boundary_end=[0.5, 0.5],
        )
        with self.assertRaisesRegex(ValueError, "tidak boleh sama"):
            self.calculate([], profile)

    def test_boundary_without_length_along_axis_raises_clear_error(self) -> None:
        profile = gap_profile(
            boundary_start=[0.5, 0.2],
            boundary_end=[0.5, 0.8],
        )
        with self.assertRaisesRegex(ValueError, "panjang positif sepanjang axis"):
            self.calculate([], profile)


if __name__ == "__main__":
    unittest.main()
