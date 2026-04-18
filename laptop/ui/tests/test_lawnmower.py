"""Sanity checks on the lawnmower pattern generator."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from lib.lawnmower import generate, total_length


def test_basic_25m_square_5m_rows():
    wps = generate(0, 25, 0, 25, row_spacing=5.0, buffer=2.5)
    # Buffer = 2.5 → x range 2.5..22.5, y range 2.5..22.5.
    # Rows at y = 2.5, 7.5, 12.5, 17.5, 22.5 → 5 rows, 2 waypoints each = 10.
    assert len(wps) == 10
    assert wps[0] == (2.5, 2.5)
    assert wps[1] == (22.5, 2.5)
    assert wps[2] == (22.5, 7.5)   # S-turn
    assert wps[3] == (2.5, 7.5)


def test_single_row():
    wps = generate(0, 10, 0, 5.5, row_spacing=5.0, buffer=2.5)
    # y range = 2.5..3.0 (just 0.5 m), so only one row at y=2.5.
    assert len(wps) == 2
    assert wps == [(2.5, 2.5), (7.5, 2.5)]


def test_buffer_larger_than_x_range_raises():
    with pytest.raises(ValueError):
        generate(0, 4, 0, 25, row_spacing=5.0, buffer=2.5)


def test_buffer_larger_than_y_range_raises():
    with pytest.raises(ValueError):
        generate(0, 25, 0, 4, row_spacing=5.0, buffer=2.5)


def test_zero_spacing_raises():
    with pytest.raises(ValueError):
        generate(0, 25, 0, 25, row_spacing=0.0, buffer=2.5)


def test_total_length_matches_expectations():
    wps = generate(0, 25, 0, 25, row_spacing=5.0, buffer=2.5)
    # 5 rows of 20m each = 100m, plus 4 transitions of 5m each = 20m => 120m.
    assert abs(total_length(wps) - 120.0) < 0.01


def test_odd_number_of_rows_still_has_s_shape():
    wps = generate(0, 20, 0, 15, row_spacing=5.0, buffer=2.5)
    # y = 2.5, 7.5, 12.5 → 3 rows → 6 waypoints.
    assert len(wps) == 6
    assert wps[0] == (2.5, 2.5)    # row 0 west → east
    assert wps[1] == (17.5, 2.5)
    assert wps[2] == (17.5, 7.5)   # row 1 east → west
    assert wps[3] == (2.5, 7.5)
    assert wps[4] == (2.5, 12.5)   # row 2 west → east again
    assert wps[5] == (17.5, 12.5)
