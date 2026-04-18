"""Launch MAVROS in a terminal window so logs are visible and Ctrl-C works.

We spawn a terminal (konsole / gnome-terminal / xterm / xfce4-terminal,
whichever is on PATH) that runs `ros2 launch mavros px4.launch ...` directly.
The user sees live output, and Ctrl-C in that window shuts MAVROS down the
same way it would from any other terminal.

The UI's Start/Stop buttons are convenience wrappers:
  Start → opens the terminal (or refuses if MAVROS is already running)
  Stop  → SIGINTs the mavros_node process and the terminal

PID tracking uses pgrep/pkill on the process name since the terminal itself
is the subprocess parent we own, not MAVROS directly.
"""

from __future__ import annotations

import logging
import os
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

DEFAULT_FCU = "/dev/ttyUSB0:57600"
DEFAULT_GCS = "udp://@localhost:14550"
MAVROS_PROC_NAME = "mavros_node"

# Terminal emulators to try, in priority order, with the flag for "run command".
TERMINAL_CANDIDATES = [
    ("konsole", "-e"),
    ("gnome-terminal", "--"),
    ("xfce4-terminal", "-e"),
    ("xterm", "-e"),
]


def _mavros_is_running() -> bool:
    r = subprocess.run(["pgrep", "-f", MAVROS_PROC_NAME],
                       capture_output=True, text=True)
    return r.returncode == 0 and bool(r.stdout.strip())


def _find_terminal() -> Optional[tuple[str, str]]:
    for exe, flag in TERMINAL_CANDIDATES:
        if shutil.which(exe):
            return exe, flag
    return None


class MavrosLauncher:
    """Owns the terminal window that runs MAVROS."""

    def __init__(self) -> None:
        self._term: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

    def is_running(self) -> bool:
        return _mavros_is_running()

    def start(
        self,
        fcu_url: str = DEFAULT_FCU,
        gcs_url: str = DEFAULT_GCS,
    ) -> tuple[bool, str]:
        """Open a terminal window running `ros2 launch mavros px4.launch …`."""
        with self._lock:
            if _mavros_is_running():
                return False, "MAVROS already running — Stop first"

            term = _find_terminal()
            if term is None:
                return False, "no terminal emulator found (install konsole or xterm)"
            exe, flag = term

            # The inner bash script: run MAVROS and keep the terminal open
            # after it exits so the user can read the final log.
            inner = (
                f"echo '=== MAVROS launch ===' && "
                f"echo 'fcu_url={fcu_url}' && "
                f"echo 'gcs_url={gcs_url}' && "
                f"echo '--- Ctrl-C in this window stops MAVROS ---' && echo && "
                f"ros2 launch mavros px4.launch "
                f"fcu_url:={fcu_url} gcs_url:={gcs_url}; "
                f"echo; echo '[MAVROS exited. This window will close in 3s]'; sleep 3"
            )

            # konsole + gnome-terminal need "bash -c" expressed slightly differently.
            if exe == "gnome-terminal":
                cmd = [exe, "--", "bash", "-c", inner]
            else:
                cmd = [exe, flag, "bash", "-c", inner]

            log.info("spawning MAVROS terminal: %s", " ".join(cmd[:3]))
            try:
                self._term = subprocess.Popen(cmd, preexec_fn=os.setsid)
            except Exception as e:
                return False, f"terminal spawn failed: {e}"

        # Wait up to 5s for mavros_node to appear
        for _ in range(25):
            time.sleep(0.2)
            if _mavros_is_running():
                return True, f"MAVROS running (check the terminal window)"
        return False, "MAVROS didn't start within 5s — check the terminal for errors"

    def stop(self, timeout: float = 5.0) -> tuple[bool, str]:
        """SIGINT mavros_node and close the terminal window."""
        # Polite interrupt first — ros2 launch catches this and tears down cleanly.
        if _mavros_is_running():
            subprocess.run(["pkill", "-INT", "-f", MAVROS_PROC_NAME],
                           capture_output=True)
            t0 = time.monotonic()
            while _mavros_is_running() and time.monotonic() - t0 < timeout:
                time.sleep(0.2)
            if _mavros_is_running():
                subprocess.run(["pkill", "-KILL", "-f", MAVROS_PROC_NAME],
                               capture_output=True)
        # Close the terminal window too
        with self._lock:
            term = self._term
            self._term = None
        if term is not None and term.poll() is None:
            try:
                os.killpg(os.getpgid(term.pid), signal.SIGTERM)
            except Exception:
                pass
        return True, "stopped"
