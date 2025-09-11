from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class RoiMetric:
    id: str
    sum_gray: int          # Background-corrected sum over ROI
    value_per_ms: float    # Exposure-normalized value (per ms)
    raw_sum: int           # Raw sum over ROI (no background subtraction)
    bg_mean: float         # Mean of background ring around ROI

    def to_dict(self) -> Dict:
        return asdict(self)


class MetricsComputer:
    """
    Computes per-frame ROI integrations on a single shared gray frame.
    - Applies background subtraction using a ring around each ROI.
    - Applies exposure normalization (per ms).
    - Maintains rolling FPS.
    - Remembers max integration value per ROI.
    Thread-safe snapshot access via an internal lock.
    """

    def __init__(self, fps_window_seconds: float = 2.0):
        self._lock = threading.Lock()
        self._fps_times: deque[float] = deque()
        self._fps_window = float(max(0.5, fps_window_seconds))

        # Per-ROI persistent state
        self._y_max_integral: Dict[str, float] = {}

        self._snapshot: Dict = {
            "timestamp": 0.0,
            "fps": 0.0,
            "exposure_us": 0,
            "rois": [],  # list[dict]
            "y_max_integral": {}, # dict[str, float]
        }

    # ---------- Helpers ----------

    @staticmethod
    def _safe_crop(gray: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
        h_img, w_img = gray.shape[:2]
        x0 = max(0, min(int(x), w_img))
        y0 = max(0, min(int(y), h_img))
        x1 = max(x0 + 1, min(int(x + w), w_img))
        y1 = max(y0 + 1, min(int(y + h), h_img))
        return gray[y0:y1, x0:x1]

    @staticmethod
    def _background_mean(gray: np.ndarray, x: int, y: int, w: int, h: int, t: int = 2) -> float:
        """
        Compute the mean pixel value in a boundary ring of thickness t around the ROI.
        The background region is the expanded rectangle minus the inner ROI.
        """
        h_img, w_img = gray.shape[:2]

        # Expanded rectangle around ROI by thickness t
        x0e = max(0, int(x) - int(t))
        y0e = max(0, int(y) - int(t))
        x1e = min(w_img, int(x) + int(w) + int(t))
        y1e = min(h_img, int(y) + int(h) + int(t))
        if x1e <= x0e or y1e <= y0e:
            return 0.0

        ext = gray[y0e:y1e, x0e:x1e]

        # Inner ROI coords relative to ext
        x0i = max(0, int(x) - x0e)
        y0i = max(0, int(y) - y0e)
        x1i = max(x0i + 1, min(x0i + int(w), ext.shape[1]))
        y1i = max(y0i + 1, min(y0i + int(h), ext.shape[0]))

        sum_ext = float(np.sum(ext, dtype=np.uint64))
        cnt_ext = float(ext.size)

        if x1i > x0i and y1i > y0i:
            inner = ext[y0i:y1i, x0i:x1i]
            sum_in = float(np.sum(inner, dtype=np.uint64))
            cnt_in = float(inner.size)
        else:
            sum_in = 0.0
            cnt_in = 0.0

        cnt_bg = max(0.0, cnt_ext - cnt_in)
        if cnt_bg <= 0.0:
            return 0.0
        return (sum_ext - sum_in) / cnt_bg

    def _update_fps(self, now: float) -> float:
        self._fps_times.append(now)
        # Drop samples older than window
        cutoff = now - self._fps_window
        while self._fps_times and self._fps_times[0] < cutoff:
            self._fps_times.popleft()

        n = len(self._fps_times)
        if n <= 1:
            return 0.0
        duration = self._fps_times[-1] - self._fps_times[0]
        if duration <= 0:
            return 0.0
        return (n - 1) / duration

    # ---------- Public API ----------

    def update(
        self,
        gray: Optional[np.ndarray],
        exposure_us: int,
        rois: Optional[Sequence[object]],
    ) -> None:
        """
        Update metrics snapshot using the latest gray frame, exposure, and ROIs.

        - gray: Grayscale image of shape (H, W). dtype uint8 preferred, but any numeric dtype is accepted.
        - exposure_us: Current exposure in microseconds. If 0 or missing, normalization uses 1 to avoid div-by-zero.
        - rois: Sequence of ROI-like items, each with attributes or keys: id, x, y, w, h.
        """
        now = time.time()
        fps = self._update_fps(now)

        roi_metrics: List[RoiMetric] = []
        rois_iter: Sequence[Any] = rois if rois is not None else []

        for r in rois_iter:
            # Support both dataclass/attribute-style and dict-style ROIs
            rid = getattr(r, "id", None) if hasattr(r, "id") else (r.get("id") if isinstance(r, dict) else None)
            x = getattr(r, "x", None) if hasattr(r, "x") else (r.get("x") if isinstance(r, dict) else None)
            y = getattr(r, "y", None) if hasattr(r, "y") else (r.get("y") if isinstance(r, dict) else None)
            w = getattr(r, "w", None) if hasattr(r, "w") else (r.get("w") if isinstance(r, dict) else None)
            h = getattr(r, "h", None) if hasattr(r, "h") else (r.get("h") if isinstance(r, dict) else None)
            if rid is None or x is None or y is None or w is None or h is None:
                continue

            s_raw = 0
            bg = 0.0
            if isinstance(gray, np.ndarray) and gray.size > 0:
                crop = self._safe_crop(gray, int(x), int(y), int(w), int(h))
                if crop.size > 0:
                    # Use uint64 to avoid overflow on big ROIs
                    s_raw = int(np.sum(crop, dtype=np.uint64))
                bg = float(self._background_mean(gray, int(x), int(y), int(w), int(h), t=2))

            area = max(1, int(w) * int(h))
            s_corr = int(max(0.0, float(s_raw) - bg * float(area)))

            exp_us = max(1, int(exposure_us))
            v_ms = (float(s_corr) / float(exp_us)) * 1000.0

            roi_metrics.append(
                RoiMetric(
                    id=str(rid),
                    sum_gray=int(s_corr),
                    value_per_ms=float(v_ms),
                    raw_sum=int(s_raw),
                    bg_mean=float(bg),
                )
            )

        with self._lock:
            # Update per-ROI max values
            for m in roi_metrics:
                # Initialize with first value, then track max
                current_max = self._y_max_integral.get(m.id, m.value_per_ms)
                self._y_max_integral[m.id] = max(current_max, m.value_per_ms)

            self._snapshot = {
                "timestamp": now,
                "fps": float(fps),
                "exposure_us": int(exposure_us),
                "rois": [m.to_dict() for m in roi_metrics],
                "y_max_integral": dict(self._y_max_integral),
            }

    def get_snapshot(self) -> Dict:
        with self._lock:
            # Return a deep enough copy for safe iteration
            snap = dict(self._snapshot)
            snap["rois"] = [dict(r) for r in self._snapshot.get("rois", [])]
            snap["y_max_integral"] = dict(self._snapshot.get("y_max_integral", {}))
            return snap

    def reset_max_integral(self, rid: str) -> None:
        with self._lock:
            # Find current value for this ROI from last snapshot
            current_value = 0.0
            for r in self._snapshot.get("rois", []):
                if r.get("id") == rid:
                    current_value = r.get("value_per_ms", 0.0)
                    break
            # Reset max to current value
            if rid in self._y_max_integral:
                self._y_max_integral[rid] = current_value

    def remove_roi_metrics(self, rid: str) -> None:
        with self._lock:
            self._y_max_integral.pop(rid, None)
