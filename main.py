"""
AI Surveillance System - main entrypoint.

Usage:
    python main.py                       # uses config.yaml
    python main.py --source 0            # webcam
    python main.py --source video.mp4    # file
    python main.py --source rtsp://...   # CCTV stream
    python main.py --no-display          # headless (Raspberry Pi)
"""
from __future__ import annotations

import argparse
import sys
import time

import cv2

from src.alerts import AlertManager
from src.detector import YoloDetector
from src.preprocess import NightEnhancer
from src.risk import RiskScorer
from src.server import run_in_background
from src.tracker import CentroidTracker
from src.utils import draw_overlay, load_config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--source", default=None, help="override video source")
    p.add_argument("--no-display", action="store_true")
    return p.parse_args()


def open_source(src):
    # Webcam index given as a string
    if isinstance(src, str) and src.isdigit():
        src = int(src)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[error] cannot open source: {src}")
        sys.exit(1)
    return cap


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.source is not None:
        cfg["source"]["input"] = args.source

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
    alerts = AlertManager(cfg)
    enhancer = NightEnhancer(
        clip_limit=cfg["preprocess"]["clahe_clip"],
        tile_grid=cfg["preprocess"]["clahe_grid"],
    )

    if cfg["server"].get("enabled", True):
        run_in_background(alerts, cfg["server"]["host"], cfg["server"]["port"])

    cap = open_source(cfg["source"]["input"])
    loiter_seconds = float(cfg["risk"]["loiter_seconds"])
    night_enhance = bool(cfg["preprocess"]["night_enhance"])
    show = not args.no_display

    fps_t0 = time.time()
    frames = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[info] stream ended")
                break

            low_light = NightEnhancer.is_low_light(frame)
            if night_enhance and low_light:
                frame = enhancer.enhance(frame)

            detections = detector.detect(frame)
            tracks = tracker.update(detections)
            loitering = tracker.loitering_track_ids("person", loiter_seconds)
            risk = scorer.score(detections, loitering, low_light_hint=low_light)

            if risk.triggered:
                alerts.trigger(frame.copy(), risk)

            if show:
                draw_overlay(frame, detections, tracks, risk)
                cv2.imshow("AI Surveillance", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frames += 1
            if frames % 30 == 0:
                dt = time.time() - fps_t0
                print(f"[fps] {frames / dt:.1f}")
    finally:
        cap.release()
        if show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
