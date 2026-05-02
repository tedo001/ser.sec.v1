"""
Detection pipeline as a thread-safe class.
Start/stop the loop, expose latest frame + state for the Flask app.
"""
from __future__ import annotations

import threading
import time
from typing import Optional

import cv2
import numpy as np

from src.alerts import AlertManager
from src.detector import YoloDetector
from src.preprocess import NightEnhancer
from src.risk import RiskScorer, RiskBreakdown
from src.tracker import CentroidTracker
from src.utils import draw_overlay


class PipelineState:
    """Shared mutable state between the pipeline thread and the Flask app."""

    def __init__(self):
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._risk: Optional[RiskBreakdown] = None
        self._fps: float = 0.0
        self._running: bool = False
        self._track_count: int = 0
        self._loiter_count: int = 0
        self._total_detections: int = 0

    # --- writers (pipeline thread) ---

    def set_frame(self, frame: np.ndarray) -> None:
        with self._lock:
            self._frame = frame.copy()

    def set_risk(self, risk: RiskBreakdown) -> None:
        with self._lock:
            self._risk = risk

    def set_fps(self, fps: float) -> None:
        with self._lock:
            self._fps = round(fps, 1)

    def set_status(self, running: bool, tracks: int, loiters: int, dets: int) -> None:
        with self._lock:
            self._running = running
            self._track_count = tracks
            self._loiter_count = loiters
            self._total_detections = dets

    # --- readers (Flask threads) ---

    def get_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def snapshot(self) -> dict:
        with self._lock:
            r = self._risk
            return {
                "running": self._running,
                "fps": self._fps,
                "tracks": self._track_count,
                "loiters": self._loiter_count,
                "total_detections": self._total_detections,
                "risk": {
                    "score": round(r.score, 3) if r else 0.0,
                    "threshold": round(r.threshold, 2) if r else 0.8,
                    "triggered": r.triggered if r else False,
                    "is_night": r.is_night if r else False,
                    "reasons": list(r.reasons) if r else [],
                    "components": dict(r.components) if r else {},
                },
            }


class SurveillancePipeline:
    def __init__(self, cfg: dict, alert_manager: AlertManager):
        self.cfg = cfg
        self.alerts = alert_manager
        self.state = PipelineState()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def restart(self, cfg: dict) -> None:
        self.cfg = cfg
        self.stop()
        self.start()

    # ------------------------------------------------------------------
    def _loop(self) -> None:
        cfg = self.cfg
        detector = YoloDetector(
            primary=cfg["model"]["primary"],
            fallback=cfg["model"]["fallback"],
            conf=cfg["model"]["conf_threshold"],
            iou=cfg["model"]["iou_threshold"],
            device=cfg["model"]["device"],
            imgsz=cfg["model"]["imgsz"],
            custom_weapons=cfg["model"].get("custom_weapons") or None,
            custom_mask=cfg["model"].get("custom_mask") or None,
        )
        tracker = CentroidTracker()
        scorer = RiskScorer(cfg)
        enhancer = NightEnhancer(
            clip_limit=cfg["preprocess"]["clahe_clip"],
            tile_grid=cfg["preprocess"]["clahe_grid"],
        )
        night_enhance = bool(cfg["preprocess"]["night_enhance"])
        loiter_sec = float(cfg["risk"]["loiter_seconds"])

        src = cfg["source"]["input"]
        if isinstance(src, str) and src.isdigit():
            src = int(src)
        cap = cv2.VideoCapture(src)

        if not cap.isOpened():
            print(f"[pipeline] cannot open source: {src}")
            return

        self.state.set_status(True, 0, 0, 0)

        fps_t0 = time.time()
        frames = 0
        total_dets = 0

        while not self._stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                # Loop video files; give up on live streams
                if isinstance(src, str) and not src.startswith("rtsp"):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            low_light = NightEnhancer.is_low_light(frame)
            if night_enhance and low_light:
                frame = enhancer.enhance(frame)

            detections = detector.detect(frame)
            tracks = tracker.update(detections)
            loitering = tracker.loitering_track_ids("person", loiter_sec)
            risk = scorer.score(detections, loitering, low_light_hint=low_light)

            if risk.triggered:
                self.alerts.trigger(frame.copy(), risk)

            draw_overlay(frame, detections, tracks, risk)
            self.state.set_frame(frame)
            self.state.set_risk(risk)

            total_dets += len(detections)
            frames += 1
            if frames % 15 == 0:
                fps = frames / (time.time() - fps_t0)
                self.state.set_fps(fps)
                self.state.set_status(True, len(tracks), len(loitering), total_dets)

        cap.release()
        self.state.set_status(False, 0, 0, total_dets)
        print("[pipeline] stopped")
