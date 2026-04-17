"""Sanity checks for the nearest-neighbour timestamp matcher."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.time_align import TimeAligner, find_closest


def test_empty():
    assert find_closest([], 5.0) is None
    assert TimeAligner([]).find_closest(5.0) is None


def test_before_first():
    data = [(10.0, "a"), (20.0, "b"), (30.0, "c")]
    assert find_closest(data, 5.0) == (10.0, "a")


def test_after_last():
    data = [(10.0, "a"), (20.0, "b"), (30.0, "c")]
    assert find_closest(data, 100.0) == (30.0, "c")


def test_exact_match():
    data = [(10.0, "a"), (20.0, "b"), (30.0, "c")]
    assert find_closest(data, 20.0) == (20.0, "b")


def test_picks_nearer_neighbour():
    data = [(10.0, "a"), (20.0, "b")]
    # 11 is 1s from a, 9s from b → a wins
    assert find_closest(data, 11.0) == (10.0, "a")
    # 19 is 9s from a, 1s from b → b wins
    assert find_closest(data, 19.0) == (20.0, "b")


def test_three_stream_alignment():
    """Synthetic 1 Hz / 5 Hz / 10 Hz streams — each anchor finds a plausible neighbour."""
    slow = [(float(t), f"slow_{t}") for t in range(0, 10)]                    # 1 Hz
    medium = [(t * 0.2, f"med_{t}") for t in range(0, 50)]                    # 5 Hz
    fast = [(t * 0.1, f"fast_{t}") for t in range(0, 100)]                    # 10 Hz

    medium_aligner = TimeAligner(medium)
    fast_aligner = TimeAligner(fast)

    for ts, _ in slow:
        m = medium_aligner.find_closest(ts)
        f = fast_aligner.find_closest(ts)
        assert abs(m[0] - ts) <= 0.1, f"medium gap too large at {ts}: {m}"
        assert abs(f[0] - ts) <= 0.05, f"fast gap too large at {ts}: {f}"


def test_max_gap_warns(caplog):
    import logging
    sparse = [(0.0, "a"), (100.0, "b")]
    aligner = TimeAligner(sparse, max_gap_seconds=1.0)
    with caplog.at_level(logging.WARNING):
        aligner.find_closest(50.0)   # 50s from both → should warn
    assert any("Timestamp mismatch" in rec.message for rec in caplog.records)
