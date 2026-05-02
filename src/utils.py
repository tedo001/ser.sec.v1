"""Shared helpers."""
from __future__ import annotations

import cv2
import yaml


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def draw_overlay(frame, detections, tracks, risk) -> None:
    """Draw boxes, track IDs, and risk HUD on the frame in-place."""
    for d in detections:
        color = (0, 255, 0)
        lab = d.label.lower()
        if lab == "person":
            color = (255, 200, 0)
        if lab in {"knife", "scissors", "gun", "pistol", "rifle", "crowbar", "rod", "hammer"}:
            color = (0, 0, 255)
        cv2.rectangle(
            frame, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), color, 2
        )
        cv2.putText(
            frame,
            f"{d.label} {d.conf:.2f}",
            (int(d.x1), int(d.y1) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )

    for tid, t in tracks.items():
        x1, y1, x2, y2 = [int(v) for v in t.bbox]
        cv2.putText(
            frame,
            f"ID{tid} {t.age_seconds:.1f}s",
            (x1, y2 + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )

    hud_color = (0, 0, 255) if risk.triggered else (50, 220, 50)
    label = "ALERT" if risk.triggered else "OK"
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(
        frame,
        f"[{label}] risk={risk.score:.2f}/thr={risk.threshold:.2f} "
        f"night={risk.is_night} reasons={','.join(risk.reasons) or '-'}",
        (8, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        hud_color,
        1,
    )
