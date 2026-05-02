"""YOLO detection wrapper. Tries YOLO26, falls back to YOLOv8."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    cls_id: int
    label: str

    @property
    def bbox(self) -> tuple:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def center(self) -> tuple:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


class YoloDetector:
    """
    Wraps an Ultralytics YOLO model. Loads YOLO26 weights if present,
    otherwise falls back to YOLOv8n. Optionally fuses extra heads
    (custom mask / weapon detectors) by running them in parallel.
    """

    def __init__(
        self,
        primary: str = "yolo26.pt",
        fallback: str = "yolov8n.pt",
        conf: float = 0.35,
        iou: float = 0.45,
        device: str = "cpu",
        imgsz: int = 640,
        custom_weapons: Optional[str] = None,
        custom_mask: Optional[str] = None,
    ) -> None:
        from ultralytics import YOLO

        self.conf = conf
        self.iou = iou
        self.device = device
        self.imgsz = imgsz

        weights = primary if self._weights_available(primary) else fallback
        print(f"[detector] Loading model: {weights}")
        self.model = YOLO(weights)

        self.weapon_model = (
            YOLO(custom_weapons) if custom_weapons and os.path.exists(custom_weapons) else None
        )
        self.mask_model = (
            YOLO(custom_mask) if custom_mask and os.path.exists(custom_mask) else None
        )

    @staticmethod
    def _weights_available(path: str) -> bool:
        # Ultralytics auto-downloads known weight names; for unknown names
        # require a local file.
        if os.path.exists(path):
            return True
        known = {"yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"}
        return path in known

    def _run(self, model, frame: np.ndarray) -> List[Detection]:
        results = model.predict(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            imgsz=self.imgsz,
            verbose=False,
        )
        out: List[Detection] = []
        if not results:
            return out
        r = results[0]
        names = r.names
        if r.boxes is None:
            return out
        for box in r.boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            out.append(Detection(x1, y1, x2, y2, conf, cls_id, str(names[cls_id])))
        return out

    def detect(self, frame: np.ndarray) -> List[Detection]:
        dets = self._run(self.model, frame)
        if self.weapon_model is not None:
            dets.extend(self._run(self.weapon_model, frame))
        if self.mask_model is not None:
            dets.extend(self._run(self.mask_model, frame))
        return dets
