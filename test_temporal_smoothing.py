import unittest

from process_parking_video import GapResult, TemporalGapSmoother


FRAME_WIDTH = 1000
FRAME_HEIGHT = 100


def row_config(row_id: str, y: float) -> dict:
    return {
        "id": row_id,
        "polygon": [[0.0, y - 0.1], [1.0, y - 0.1], [1.0, y + 0.1], [0.0, y + 0.1]],
        "axis": [1.0, 0.0],
        "boundary_start": [0.0, y],
        "boundary_end": [1.0, y],
        "boundary_visibility": {"start": True, "end": True},
        "edge_margin_pitch": 0.5,
        "reference_pitch": 0.1,
    }


def smoothing_profile(
    *,
    enabled: bool = True,
    activation_seconds: float = 1.5,
    release_seconds: float = 0.75,
    rows: list[dict] | None = None,
) -> dict:
    return {
        "temporal_smoothing": {
            "enabled": enabled,
            "activation_seconds": activation_seconds,
            "release_seconds": release_seconds,
            "match_overlap_threshold": 0.5,
            "capacity_window": 5,
        },
        "rows": rows or [row_config("ROW_A", 0.25)],
    }


def gap(
    *,
    row_id: str = "ROW_A",
    gap_type: str = "internal",
    start_x: float = 200.0,
    end_x: float = 600.0,
    y: float = 25.0,
    capacity: int = 1,
) -> GapResult:
    return GapResult(
        row_id=row_id,
        gap_type=gap_type,
        start=(start_x, y),
        end=(end_x, y),
        distance=abs(end_x - start_x),
        capacity=capacity,
    )


class TemporalGapSmootherTests(unittest.TestCase):
    def make_smoother(self, profile: dict | None = None) -> TemporalGapSmoother:
        return TemporalGapSmoother(
            profile or smoothing_profile(), FRAME_WIDTH, FRAME_HEIGHT
        )

    def confirm_gap(self, smoother: TemporalGapSmoother, value: GapResult) -> None:
        for timestamp in (0.0, 0.5, 1.0, 1.5):
            smoother.update([value], timestamp)

    def test_brief_gap_is_not_confirmed(self) -> None:
        smoother = self.make_smoother()
        self.assertEqual(smoother.update([gap()], 0.0), [])
        self.assertEqual(smoother.update([], 0.5), [])
        self.assertEqual(smoother.update([gap()], 2.0), [])

    def test_gap_persisting_for_activation_time_is_confirmed(self) -> None:
        smoother = self.make_smoother()
        stable: list[GapResult] = []
        for timestamp in (0.0, 0.5, 1.0, 1.5):
            stable = smoother.update([gap()], timestamp)
        self.assertEqual(len(stable), 1)
        self.assertEqual(stable[0].capacity, 1)

    def test_confirmed_gap_survives_short_loss(self) -> None:
        smoother = self.make_smoother()
        self.confirm_gap(smoother, gap())
        stable = smoother.update([], 2.0)
        self.assertEqual(len(stable), 1)
        self.assertEqual(stable[0].gap_type, "internal")

    def test_confirmed_gap_is_removed_after_release_time(self) -> None:
        smoother = self.make_smoother()
        self.confirm_gap(smoother, gap())
        self.assertEqual(smoother.update([], 2.26), [])

    def test_different_rows_do_not_match(self) -> None:
        profile = smoothing_profile(
            rows=[row_config("ROW_A", 0.25), row_config("ROW_B", 0.75)]
        )
        smoother = self.make_smoother(profile)
        self.assertEqual(smoother.update([gap(row_id="ROW_A")], 0.0), [])
        for timestamp in (0.5, 1.0, 1.5):
            stable = smoother.update(
                [gap(row_id="ROW_B", y=75.0)], timestamp
            )
        self.assertEqual(stable, [])
        stable = smoother.update([gap(row_id="ROW_B", y=75.0)], 2.0)
        self.assertEqual(len(stable), 1)
        self.assertEqual(stable[0].row_id, "ROW_B")

    def test_different_gap_types_do_not_match(self) -> None:
        smoother = self.make_smoother()
        self.assertEqual(smoother.update([gap(gap_type="internal")], 0.0), [])
        for timestamp in (0.5, 1.0, 1.5):
            stable = smoother.update([gap(gap_type="edge_start")], timestamp)
        self.assertEqual(stable, [])
        stable = smoother.update([gap(gap_type="edge_start")], 2.0)
        self.assertEqual(len(stable), 1)
        self.assertEqual(stable[0].gap_type, "edge_start")

    def test_multiple_gaps_are_not_double_counted(self) -> None:
        smoother = self.make_smoother()
        values = [gap(start_x=100.0, end_x=300.0), gap(start_x=500.0, end_x=700.0)]
        stable: list[GapResult] = []
        for timestamp in (0.0, 0.5, 1.0, 1.5):
            stable = smoother.update(values, timestamp)
        self.assertEqual(len(stable), 2)
        self.assertEqual(sum(item.capacity for item in stable), 2)
        self.assertEqual({item.start[0] for item in stable}, {100.0, 500.0})

    def test_median_capacity_stabilizes_changes(self) -> None:
        smoother = self.make_smoother()
        stable: list[GapResult] = []
        for timestamp, capacity in zip(
            (0.0, 0.4, 0.8, 1.2, 1.5), (1, 3, 2, 1, 2)
        ):
            stable = smoother.update([gap(capacity=capacity)], timestamp)
        self.assertEqual(len(stable), 1)
        self.assertEqual(stable[0].capacity, 2)

    def test_disabled_mode_returns_raw_p3_results_immediately(self) -> None:
        smoother = self.make_smoother(smoothing_profile(enabled=False))
        raw = [gap(capacity=2), gap(start_x=700.0, end_x=800.0, capacity=0)]
        self.assertEqual(smoother.update(raw, 0.0), raw)

    def test_activation_uses_seconds_not_number_of_updates(self) -> None:
        smoother = self.make_smoother()
        stable: list[GapResult] = []
        for index in range(20):
            stable = smoother.update([gap()], index * 0.05)
        self.assertEqual(stable, [])
        self.assertEqual(smoother.update([gap()], 1.49), [])
        self.assertEqual(len(smoother.update([gap()], 1.5)), 1)


if __name__ == "__main__":
    unittest.main()
