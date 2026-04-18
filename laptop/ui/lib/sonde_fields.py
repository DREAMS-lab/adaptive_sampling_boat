"""Friendly display mapping for sonde rows.

Loads parse_sonde_line from laptop/post_processing/lib/sonde_parser.py by
absolute file path — sharing a sibling `lib/` package name makes regular
imports conflict.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Optional


_PARSER_PATH = (
    Path(__file__).resolve().parents[2]
    / "post_processing"
    / "lib"
    / "sonde_parser.py"
)

_spec = importlib.util.spec_from_file_location("_sonde_parser_shared", _PARSER_PATH)
_mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
assert _spec and _spec.loader
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]
parse_sonde_line = _mod.parse_sonde_line


# Order matters — the UI renders them in this order.
DISPLAY_ORDER = [
    ("Date", "Date"),
    ("Time", "Time"),
    ("Temperature (°C)", "Temp deg C"),
    ("pH", "pH units"),
    ("Depth (m)", "Depth m"),
    ("Conductivity (µS/cm)", "SpCond uS/cm"),
    ("Dissolved O₂ (%Sat)", "HDO sat"),
    ("Dissolved O₂ (mg/L)", "HDO mg/L"),
    ("Chlorophyll (µg/L)", "Chl ug/L"),
    ("CDOM (ppb)", "CDOM ppb"),
    ("Turbidity (NTU)", "Turb NTU"),
]


def display_fields(raw: str, timestamp: float = 0.0) -> Optional[list[tuple[str, str]]]:
    """Parse a raw /sonde_data line into a list of (label, value) pairs.

    Returns None if the line doesn't parse.
    """
    record = parse_sonde_line(timestamp, raw)
    if record is None:
        return None
    return [(label, record.get(key, "—")) for label, key in DISPLAY_ORDER]
