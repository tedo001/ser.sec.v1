"""Risk scoring engine."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List


@dataclass
class RiskBreakdown:
    score: float = 0.0
    components: Dict[str, float] = field(default_factory=dict)
    threshold: float = 0.8
    is_night: bool = False
    triggered: bool = False
    reasons: List[str] = field(default_factory=list)


class RiskScorer:
    def __init__(self, cfg: dict):
        self.w = cfg["risk"]["weights"]
        self.t_day = cfg["risk"]["threshold_day"]
        self.t_night = cfg["risk"]["threshold_night"]
        self.night_start = cfg["risk"]["night_start_hour"]
        self.night_end = cfg["risk"]["night_end_hour"]
        self.weapon_classes = {c.lower() for c in cfg.get("weapon_classes", [])}
        self.mask_classes = {c.lower() for c in cfg.get("mask_classes", [])}
        self.force_night = cfg["source"].get("force_night", False)

    def is_night(self, low_light_hint: bool = False) -> bool:
        if self.force_night or low_light_hint:
            return True
        h = datetime.now().hour
        if self.night_start <= self.night_end:
            return self.night_start <= h < self.night_end
        # wrap-around (e.g. 19 -> 6)
        return h >= self.night_start or h < self.night_end

    def score(self, detections, loitering_ids, low_light_hint: bool = False) -> RiskBreakdown:
        rb = RiskBreakdown()
        rb.is_night = self.is_night(low_light_hint)
        rb.threshold = self.t_night if rb.is_night else self.t_day

        labels = [d.label.lower() for d in detections]
        person = any(l == "person" for l in labels)
        mask = any(l in self.mask_classes for l in labels)
        weapon = any(l in self.weapon_classes for l in labels)

        if person:
            rb.score += self.w["person"]
            rb.components["person"] = self.w["person"]
            rb.reasons.append("person detected")
        if rb.is_night:
            rb.score += self.w["night"]
            rb.components["night"] = self.w["night"]
            rb.reasons.append("night-time")
        if mask:
            rb.score += self.w["mask"]
            rb.components["mask"] = self.w["mask"]
            rb.reasons.append("face mask")
        if weapon:
            rb.score += self.w["weapon"]
            rb.components["weapon"] = self.w["weapon"]
            rb.reasons.append("weapon/tool")
        if loitering_ids:
            rb.score += self.w["loitering"]
            rb.components["loitering"] = self.w["loitering"]
            rb.reasons.append(f"loitering x{len(loitering_ids)}")

        rb.score = round(min(rb.score, 1.5), 3)
        rb.triggered = rb.score >= rb.threshold
        return rb
