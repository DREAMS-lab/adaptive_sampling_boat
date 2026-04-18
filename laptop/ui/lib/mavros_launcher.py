"""Launch and manage a MAVROS subprocess owned by the UI.

MAVROS is `ros2 launch mavros px4.launch ...` — it stays alive as long as the
process runs. We spawn it via subprocess.Popen, capture stdout+stderr to a
log file, and kill the whole process group on stop.

Killing with os.killpg is important because `ros2 launch` spawns children
(mavros_node, parameter loaders) that would linger if we only killed the
parent PID.
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

LOG_PATH = Path("/tmp/karin_mavros.log")

# Try these in order; first one on PATH wins.
TERMINAL_CANDIDATES = [
    ("konsole", ["-e", "bash", "-c"]),
    ("gnome-terminal", ["--", "bash", "-c"]),
    ("xfce4-terminal", ["-e", "bash -c"]),
    ("xterm", ["-e", "bash", "-c"]),
]


def _open_log_terminal() -> Optional[subprocess.Popen]:
    """Open a terminal window tailing LOG_PATH. Returns the Popen or None."""
    for exe, args in TERMINAL_CANDIDATES:
        if shutil.which(exe):
            cmd = [exe] + args + [f"echo '=== MAVROS log ({LOG_PATH}) ==='; tail -F {LOG_PATH}"]
            log.info("opening log terminal: %s", exe)
            try:
                return subprocess.Popen(cmd, preexec_fn=os.setsid)
            except Exception as e:
                log.warning("terminal %s spawn failed: %s", exe, e)
    log.warning("no terminal emulator found on PATH")
    return None


class MavrosLauncher:
    """Owns the MAVROS child process. Safe to call start()/stop() repeatedly."""

    def __init__(self) -> None:
        self._proc: Optional[subprocess.Popen] = None
        self._term_proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

    # ------------------------------- api ------------------------------- #

    def is_running(self) -> bool:
        with self._lock:
            return self._proc is not None and self._proc.poll() is None

    def start(
        self,
        fcu_url: str = DEFAULT_FCU,
        gcs_url: str = DEFAULT_GCS,
    ) -> tuple[bool, str]:
        """Start `ros2 launch mavros px4.launch ...`. Idempotent (stops first)."""
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return False, "already running"

            cmd = [
                "ros2", "launch", "mavros", "px4.launch",
                f"fcu_url:={fcu_url}",
                f"gcs_url:={gcs_url}",
            ]
            log.info("starting MAVROS: %s", " ".join(cmd))
            try:
                logf = LOG_PATH.open("w")
                self._proc = subprocess.Popen(
                    cmd,
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                    preexec_fn=os.setsid,       # new process group so we can killpg
                )
            except FileNotFoundError:
                return False, "ros2 not on PATH — is /opt/ros/humble sourced?"
            except Exception as e:
                return False, f"spawn failed: {e}"

            # Spin up a terminal window tailing the log so the user can see output.
            self._term_proc = _open_log_terminal()

        # Give it a moment to blow up if wrong args
        time.sleep(1.0)
        if not self.is_running():
            return False, f"MAVROS exited immediately — check {LOG_PATH}"
        msg = f"MAVROS running (pid {self._proc.pid})"
        if self._term_proc is None:
            msg += f"; no terminal found, log at {LOG_PATH}"
        return True, msg

    def stop(self, timeout: float = 5.0) -> tuple[bool, str]:
        """Kill the launch process group (SIGINT first, SIGKILL if hung)."""
        with self._lock:
            proc = self._proc
            term = self._term_proc
            self._proc = None
            self._term_proc = None

        # Close log terminal too
        if term is not None and term.poll() is None:
            try:
                os.killpg(os.getpgid(term.pid), signal.SIGTERM)
            except Exception:
                pass

        if proc is None or proc.poll() is not None:
            return True, "was not running"

        pgid = os.getpgid(proc.pid)

        # Polite SIGINT (ros2 launch catches this and shuts down children).
        os.killpg(pgid, signal.SIGINT)
        try:
            proc.wait(timeout=timeout)
            return True, "stopped cleanly"
        except subprocess.TimeoutExpired:
            pass

        # Hung — escalate.
        os.killpg(pgid, signal.SIGKILL)
        try:
            proc.wait(timeout=2.0)
            return True, "force-killed"
        except subprocess.TimeoutExpired:
            return False, "process still running after SIGKILL (manual cleanup needed)"

    def log_tail(self, lines: int = 20) -> str:
        if not LOG_PATH.exists():
            return "(no log yet)"
        try:
            with LOG_PATH.open() as f:
                content = f.readlines()
            return "".join(content[-lines:])
        except Exception as e:
            return f"(log read error: {e})"
