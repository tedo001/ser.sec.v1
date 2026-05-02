"""
Lightweight centroid-IoU tracker.

A full SORT/DeepSORT implementation is overkill for an edge-device
surveillance use case; a simple IoU + centroid tracker is fast and
good enough to detect loitering (same person staying too long).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


def iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    track_id: int
    bbox: Tuple[float, float, float, float]
    label: str
    first_seen: float
    last_seen: float
    hits: int = 1
    misses: int = 0
    history: List[Tuple[float, float]] = field(default_factory=list)

    @property
    def age_seconds(self) -> float:
        return self.last_seen - self.first_seen


class CentroidTracker:
    def __init__(self, iou_threshold: float = 0.3, max_misses: int = 15):
        self.iou_threshold = iou_threshold
        self.max_misses = max_misses
        self._next_id = 1
        self.tracks: Dict[int, Track] = {}

    def update(self, detections) -> Dict[int, Track]:
        now = time.time()
        det_boxes = [d.bbox for d in detections]
        det_labels = [d.label for d in detections]

        unmatched_dets = set(range(len(detections)))
        unmatched_trks = set(self.tracks.keys())

        # Greedy IoU matching
        pairs = []
        for tid, trk in self.tracks.items():
            for di in range(len(detections)):
                if det_labels[di] != trk.label:
                    continue
                score = iou(trk.bbox, det_boxes[di])
                if score >= self.iou_threshold:
                    pairs.append((score, tid, di))
        pairs.sort(reverse=True)

        for _, tid, di in pairs:
            if tid in unmatched_trks and di in unmatched_dets:
                trk = self.tracks[tid]
                trk.bbox = det_boxes[di]
                trk.last_seen = now
                trk.hits += 1
                trk.misses = 0
                cx = (det_boxes[di][0] + det_boxes[di][2]) / 2.0
                cy = (det_boxes[di][1] + det_boxes[di][3]) / 2.0
                trk.history.append((cx, cy))
                if len(trk.history) > 64:
                    trk.history = trk.history[-64:]
                unmatched_trks.discard(tid)
                unmatched_dets.discard(di)

        # New tracks for unmatched detections
        for di in unmatched_dets:
            tid = self._next_id
            self._next_id += 1
            cx = (det_boxes[di][0] + det_boxes[di][2]) / 2.0
            cy = (det_boxes[di][1] + det_boxes[di][3]) / 2.0
            self.tracks[tid] = Track(
                track_id=tid,
                bbox=det_boxes[di],
                label=det_labels[di],
                first_seen=now,
                last_seen=now,
                history=[(cx, cy)],
            )

        # Age out missed tracks
        to_delete = []
        for tid in unmatched_trks:
            self.tracks[tid].misses += 1
            if self.tracks[tid].misses > self.max_misses:
                to_delete.append(tid)
        for tid in to_delete:
            del self.tracks[tid]

        return self.tracks

    def loitering_track_ids(self, person_label: str, seconds: float) -> List[int]:
        return [
            tid
            for tid, t in self.tracks.items()
            if t.label == person_label and t.age_seconds >= seconds
        ]
