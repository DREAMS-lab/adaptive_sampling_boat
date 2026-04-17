"""Nearest-neighbour timestamp matching across sensor streams.

This is the "frequency combiner" — given a target timestamp (e.g. from the
slow sonde stream) find the closest sample in a faster stream (GPS or sonar).
"""

from __future__ import annotations

import bisect
import logging
from typing import List, Optional, Sequence, Tuple

log = logging.getLogger(__name__)


class TimeAligner:
    """Binary-search lookup table for nearest-neighbour timestamp matching.

    If `max_gap_seconds` is set, warns once when the nearest neighbour is
    further away than that — the old code silently paired samples regardless
    of how far apart they were in time.
    """

    def __init__(self, data: Sequence[Tuple], max_gap_seconds: Optional[float] = None):
        self._data = list(data)
        self._timestamps = [row[0] for row in self._data]
        self._max_gap = max_gap_seconds
        self._warned_large_gap = False

    def __len__(self) -> int:
        return len(self._data)

    def find_closest(self, target: float) -> Optional[Tuple]:
        if not self._data:
            return None

        pos = bisect.bisect_left(self._timestamps, target)
        if pos == 0:
            match = self._data[0]
        elif pos == len(self._data):
            match = self._data[-1]
        else:
            before = self._data[pos - 1]
            after = self._data[pos]
            match = before if target - before[0] < after[0] - target else after

        if self._max_gap is not None and not self._warned_large_gap:
            gap = abs(match[0] - target)
            if gap > self._max_gap:
                log.warning(
                    "Timestamp mismatch exceeds %.2fs (gap=%.2fs). "
                    "Boat/laptop clocks may be out of sync.",
                    self._max_gap, gap,
                )
                self._warned_large_gap = True

        return match


def find_closest(data_list: List[Tuple], target: float) -> Optional[Tuple]:
    """Stateless shim for one-off lookups. Prefer TimeAligner for many queries."""
    return TimeAligner(data_list).find_closest(target)
