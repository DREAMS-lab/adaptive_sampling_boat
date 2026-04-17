"""Parse a raw #DATA line from the YSI EXO sonde into a structured dict."""

from __future__ import annotations

import logging
from typing import Optional

log = logging.getLogger(__name__)


# Column layout of the sonde output. Index 2 is a VOID field we skip.
# Some sonde firmwares omit turbidity (11 fields instead of 12).
SONDE_FIELDS = [
    ("Date", 0),
    ("Time", 1),
    ("Temp deg C", 3),
    ("pH units", 4),
    ("Depth m", 5),
    ("SpCond uS/cm", 6),
    ("HDO sat", 7),
    ("HDO mg/L", 8),
    ("Chl ug/L", 9),
    ("CDOM ppb", 10),
    ("Turb NTU", 11),
]


def parse_sonde_line(timestamp: float, raw: str) -> Optional[dict]:
    """Parse one whitespace-separated sonde record.

    Returns a dict with `timestamp` and the parsed fields, or None if the line
    is malformed. Logs a warning on unexpected field counts instead of silently
    skipping (the old code simply dropped malformed lines).
    """
    values = [v.strip() for v in raw.split() if v]

    if len(values) not in (11, 12):
        log.warning("Unexpected sonde field count: %d (expected 11 or 12). Raw: %r",
                    len(values), raw)
        return None

    record: dict = {"timestamp": timestamp}
    for name, idx in SONDE_FIELDS:
        if idx < len(values):
            record[name] = values[idx]
    return record


def has_turbidity(record: dict) -> bool:
    """Whether this record came from a 12-field sonde output."""
    return "Turb NTU" in record
