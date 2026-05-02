"""Alert system: beep + frame snapshot + event log (consumed by Flask API)."""
from __future__ import annotations

import json
import os
import platform
import subprocess
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Deque, Dict, List, Optional

import cv2


@dataclass
class Event:
    ts: float
    iso_time: str
    score: float
    threshold: float
    is_night: bool
    reasons: List[str]
    image_path: str
    components: Dict[str, float] = field(default_factory=dict)


def _beep(freq_hz: int, duration_ms: int) -> None:
    """Best-effort cross-platform beep. Never raises."""
    try:
        if platform.system() == "Windows":
            import winsound

            winsound.Beep(freq_hz, duration_ms)
            return
        # Linux / macOS: try `beep`, then printf BEL as final fallback.
        if subprocess.run(
            ["which", "beep"], capture_output=True
        ).returncode == 0:
            subprocess.run(
                ["beep", "-f", str(freq_hz), "-l", str(duration_ms)],
                capture_output=True,
                timeout=2,
            )
            return
        # ASCII BEL — works in most terminals
        print("\a", end="", flush=True)
    except Exception:
        pass


class AlertManager:
    def __init__(self, cfg: dict):
        a = cfg["alert"]
        self.cooldown = float(a["cooldown_seconds"])
        self.save_dir = a["save_dir"]
        self.beep_enabled = bool(a["beep"])
        self.beep_freq = int(a["beep_freq_hz"])
        self.beep_dur = int(a["beep_duration_ms"])
        os.makedirs(self.save_dir, exist_ok=True)
        self._last_alert_ts: float = 0.0
        self._events: Deque[Event] = deque(maxlen=200)
        self._lock = threading.Lock()
        self._log_path = os.path.join(self.save_dir, "events.jsonl")

    def in_cooldown(self) -> bool:
        return (time.time() - self._last_alert_ts) < self.cooldown

    def trigger(self, frame, risk) -> Optional[Event]:
        if self.in_cooldown():
            return None
        ts = time.time()
        fname = time.strftime("event_%Y%m%d_%H%M%S", time.localtime(ts)) + ".jpg"
        path = os.path.join(self.save_dir, fname)
        try:
            cv2.imwrite(path, frame)
        except Exception as e:
            print(f"[alert] failed to save frame: {e}")

        ev = Event(
            ts=ts,
            iso_time=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts)),
            score=risk.score,
            threshold=risk.threshold,
            is_night=risk.is_night,
            reasons=list(risk.reasons),
            image_path=path,
            components=dict(risk.components),
        )

        with self._lock:
            self._events.append(ev)
            try:
                with open(self._log_path, "a") as f:
                    f.write(json.dumps(asdict(ev)) + "\n")
            except Exception:
                pass

        self._last_alert_ts = ts
        if self.beep_enabled:
            threading.Thread(
                target=_beep, args=(self.beep_freq, self.beep_dur), daemon=True
            ).start()

        print(
            f"[ALERT] score={risk.score} thr={risk.threshold} "
            f"night={risk.is_night} reasons={risk.reasons} -> {path}"
        )
        return ev

    def recent(self, limit: int = 50) -> List[dict]:
        with self._lock:
            return [asdict(e) for e in list(self._events)[-limit:]]
