from __future__ import annotations

import threading
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ROI:
    id: str
    x: int
    y: int
    w: int
    h: int

    def to_dict(self) -> Dict:
        return asdict(self)


class ROIRegistry:
    """
    Thread-safe ROI registry with monotonic IDs (roi-1, roi-2, ...).
    Stores ROI coordinates in stream pixel space (natural image size).
    Provides CRUD and bounds-clamping vs current frame size.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._rois: Dict[str, ROI] = {}

    # ---------- Helpers ----------

    @staticmethod
    def _clamp_rect(x: int, y: int, w: int, h: int, frame_wh: Optional[Tuple[int, int]]) -> Tuple[int, int, int, int]:
        # Ensure integers and sane sizes
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        if w < 1: w = 1
        if h < 1: h = 1

        # Enforce non-negative origin first
        if x < 0:
            w += x  # reduce width by overshoot
            x = 0
        if y < 0:
            h += y  # reduce height by overshoot
            y = 0
        if w < 1: w = 1
        if h < 1: h = 1

        if frame_wh:
            fw, fh = frame_wh
            # Right/bottom clamp
            if x >= fw:  # push inside
                x = max(0, fw - 1)
                w = 1
            if y >= fh:
                y = max(0, fh - 1)
                h = 1

            # Shrink to fit
            if x + w > fw:
                w = max(1, fw - x)
            if y + h > fh:
                h = max(1, fh - y)

        return x, y, w, h

    def _allocate_id(self) -> str:
        """Find the lowest available ROI ID (e.g., roi-1, roi-2, ...)."""
        i = 1
        while True:
            rid = f"roi-{i}"
            if rid not in self._rois:
                return rid
            i += 1

    # ---------- CRUD ----------

    def list(self) -> List[ROI]:
        with self._lock:
            return list(self._rois.values())

    def list_dicts(self) -> List[Dict]:
        return [r.to_dict() for r in self.list()]

    def get(self, roi_id: str) -> Optional[ROI]:
        with self._lock:
            return self._rois.get(str(roi_id))

    def get_dict(self, roi_id: str) -> Optional[Dict]:
        r = self.get(roi_id)
        return r.to_dict() if r else None

    def create(self, x: int, y: int, w: int, h: int, frame_wh: Optional[Tuple[int, int]]) -> ROI:
        with self._lock:
            rid = self._allocate_id()
            x, y, w, h = self._clamp_rect(x, y, w, h, frame_wh)
            roi = ROI(id=rid, x=x, y=y, w=w, h=h)
            self._rois[rid] = roi
            return roi

    def update(self, roi_id: str, x: int, y: int, w: int, h: int, frame_wh: Optional[Tuple[int, int]]) -> Optional[ROI]:
        with self._lock:
            if roi_id not in self._rois:
                return None
            x, y, w, h = self._clamp_rect(x, y, w, h, frame_wh)
            roi = ROI(id=roi_id, x=x, y=y, w=w, h=h)
            self._rois[roi_id] = roi
            return roi

    def delete(self, roi_id: str) -> bool:
        with self._lock:
            return self._rois.pop(roi_id, None) is not None

    # ---------- Utilities ----------

    def clamp_all(self, frame_wh: Optional[Tuple[int, int]]) -> None:
        """Clamp all ROIs to a new frame size (e.g., when resolution changes)."""
        if not frame_wh:
            return
        with self._lock:
            updated: Dict[str, ROI] = {}
            for rid, r in self._rois.items():
                x, y, w, h = self._clamp_rect(r.x, r.y, r.w, r.h, frame_wh)
                updated[rid] = ROI(id=rid, x=x, y=y, w=w, h=h)
            self._rois = updated
