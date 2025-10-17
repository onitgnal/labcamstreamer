import io
import csv
import json
import re
import logging
import threading
import time
import zipfile
import argparse
import atexit
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Generator, Optional, Tuple, List
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request, send_file
from beam_analysis import analyze_beam

from camera_service import CameraService, apply_colormap_to_gray, apply_colormap_to_bgr
from metrics import MetricsComputer
from models.caustic import (
    CausticManager,
    CausticPoint,
    CausticRadiiSource,
    convert_length_to_meters,
    convert_meters_to_length,
)
from roi import ROIRegistry

# ----- App setup -----
app = Flask(__name__, template_folder="templates", static_folder="static", static_url_path="/static")

# ----- Logging Setup -----
def setup_logging(dev_mode=False):
    log_level = logging.DEBUG if dev_mode else logging.INFO
    app.logger.setLevel(log_level)

    if not dev_mode:
        return

    log_file = "app_debug.log"

    # Clear log file on startup
    with open(log_file, "w"):
        pass

    handler = logging.FileHandler(log_file)
    handler.setLevel(log_level)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # The Flask app's logger is configured here.
    # Any component holding a reference to app.logger will see the updates.
    app.logger.addHandler(handler)
    app.logger.info("Application starting up...")
    app.logger.info("Development mode enabled.")

# ----- Global services -----
# Instantiate services globally. The logger will be configured later in the main block.
cam_service = CameraService(logger=app.logger)
roi_registry = ROIRegistry()
metrics = MetricsComputer()
_plot_lock = threading.Lock()
caustic_manager = CausticManager()

CAUSTIC_CACHE_DIR = Path("caustic_cache")
CAUSTIC_CACHE_DIR.mkdir(exist_ok=True)

EXPORTS_DIR = Path("exports")
EXPORTS_DIR.mkdir(exist_ok=True)

CAUSTIC_AUTOSAVE_DIR = EXPORTS_DIR / "caustic_autosave"
CAUSTIC_AUTOSAVE_DIR.mkdir(parents=True, exist_ok=True)
caustic_manager.set_autosave_dir(CAUSTIC_AUTOSAVE_DIR)

# ----- Beam analysis options (global) -----
_beam_opts_lock = threading.Lock()
_beam_opts = {
    "pixel_size": None,              # Optional float (physical units per pixel)
    "compute": "both",              # one of: both, second, gauss, none
    "clip_negatives": "none",       # one of: none, zero, otsu
    "angle_clip_mode": "otsu",      # one of: same, none, zero, otsu
    "background_subtraction": True,  # bool
    "rotation": "auto",             # one of: auto, fixed
    "fixed_angle": None,             # degrees (float) when rotation == fixed
}


def _get_beam_options_copy() -> Dict[str, object]:
    with _beam_opts_lock:
        return dict(_beam_opts)


class BeamAnalysisManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: Dict[str, Dict[str, object]] = {}
        self._frame_ts: float = 0.0
        self._compute_mode: str = "none"
        self._pixel_size: Optional[float] = None
        self._min_interval: float = 0.3
        self._max_workers: Optional[int] = None
        self._pool: Optional[ProcessPoolExecutor] = None
        self._last_signature: Tuple = ()
        self._create_pool()
        atexit.register(self._shutdown_pool)

    def _shutdown_pool(self) -> None:
        try:
            if self._pool:
                self._pool.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass

    def _create_pool(self) -> None:
        if self._pool:
            try:
                self._pool.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
        self._pool = ProcessPoolExecutor(max_workers=self._max_workers)

    def update(self, gray: Optional[np.ndarray], rois: Optional[Tuple], opts: Dict[str, object]) -> None:
        frame_ts = time.time()
        compute = str(opts.get("compute", "both") or "both").lower()
        if compute not in {"both", "second", "gauss", "none"}:
            compute = "both"

        clip_mode = str(opts.get("clip_negatives", "none") or "none").lower()
        if clip_mode not in {"none", "zero", "otsu"}:
            clip_mode = "none"

        angle_clip = str(opts.get("angle_clip_mode", "otsu") or "otsu").lower()
        angle_clip_param: Optional[str]
        if angle_clip == "same":
            angle_clip_param = None
        elif angle_clip in {"none", "zero", "otsu"}:
            angle_clip_param = angle_clip
        else:
            angle_clip_param = "otsu"

        bg_sub = bool(opts.get("background_subtraction", True))
        rotation_mode = str(opts.get("rotation", "auto") or "auto").lower()
        fixed_angle_deg = opts.get("fixed_angle")
        if rotation_mode == "fixed":
            try:
                rot_angle = 0.0 if fixed_angle_deg in (None, "") else float(fixed_angle_deg) * np.pi / 180.0
            except Exception:
                rot_angle = 0.0
        else:
            rot_angle = None
        compute_gauss = compute in ("both", "gauss")

        try:
            pixel_size_val = float(opts.get("pixel_size")) if opts.get("pixel_size") not in (None, "") else None
        except Exception:
            pixel_size_val = None

        entries: Dict[str, Dict[str, object]] = {}
        rois_list = list(rois) if rois else []
        images: List[np.ndarray] = []
        roi_ids: List[str] = []

        signature = tuple(
            (
                getattr(roi, "id", ""),
                int(getattr(roi, "x", 0)),
                int(getattr(roi, "y", 0)),
                int(getattr(roi, "w", 0)),
                int(getattr(roi, "h", 0)),
            )
            for roi in rois_list
        )

        with self._lock:
            previous_data = {k: dict(v) for k, v in self._data.items()}
            last_signature = getattr(self, "_last_signature", ())
            last_ts = self._frame_ts
            last_compute = self._compute_mode

        now = frame_ts
        should_analyze = compute != "none"
        if should_analyze and signature == last_signature and compute == last_compute:
            if now - last_ts < self._min_interval:
                should_analyze = False

        if gray is not None and rois_list:
            for roi in rois_list:
                rid = getattr(roi, "id", None)
                if not rid:
                    continue
                crop = MetricsComputer._safe_crop(gray, roi.x, roi.y, roi.w, roi.h)
                if crop.size == 0:
                    entries[rid] = {
                        "roi_gray": None,
                        "result": None,
                        "frame_ts": frame_ts,
                        "pixel_size": pixel_size_val,
                    }
                    continue
                crop_copy = crop.copy()
                entries[rid] = {
                    "roi_gray": crop_copy,
                    "result": None,
                    "frame_ts": frame_ts,
                    "pixel_size": pixel_size_val,
                }
                if should_analyze:
                    images.append(crop_copy.astype(np.float64))
                    roi_ids.append(rid)
                else:
                    prev_entry = previous_data.get(rid)
                    if prev_entry:
                        entries[rid]["result"] = prev_entry.get("result")

        analysis_kwargs = {
            "clip_negatives": clip_mode,
            "angle_clip_mode": angle_clip_param,
            "background_subtraction": bg_sub,
            "rotation_angle": rot_angle,
            "compute_gaussian": compute_gauss,
            "compute_iso": compute in ("both", "second"),
        }

        if should_analyze and images:
            payloads = [(img, dict(analysis_kwargs)) for img in images]
            results: List[Optional[Dict[str, object]]] = []
            if not self._pool:
                self._create_pool()
            pool_attempted = False
            if self._pool:
                try:
                    pool_attempted = True
                    results = list(self._pool.map(_beam_worker, payloads))
                except (BrokenProcessPool, RuntimeError) as exc:
                    app.logger.warning(f"beam_analysis pool unavailable, recreating: {exc}")
                    self._create_pool()
                    if self._pool:
                        try:
                            results = list(self._pool.map(_beam_worker, payloads))
                        except Exception as inner_exc:
                            app.logger.warning(f"beam_analysis pool retry failed: {inner_exc}")
                            results = []
                    else:
                        results = []
                except Exception as exc:
                    app.logger.warning(f"beam_analysis pool failure, switching to sequential: {exc}")
                    results = []
            if not results:
                # Sequential fallback (covers no pool case or failures)
                results = [
                    _beam_worker(payload)
                    for payload in payloads
                ]

            for rid, res in zip(roi_ids, results):
                if rid in entries:
                    entries[rid]["result"] = res

        with self._lock:
            self._data = entries
            if should_analyze and images:
                self._frame_ts = frame_ts
            self._compute_mode = compute
            self._pixel_size = pixel_size_val
            self._last_signature = signature

    def get(self, rid: str) -> Optional[Dict[str, object]]:
        with self._lock:
            entry = self._data.get(rid)
            if not entry:
                return None
            return {
                "result": entry.get("result"),
                "roi_gray": entry.get("roi_gray"),
                "frame_ts": entry.get("frame_ts", self._frame_ts),
                "compute": self._compute_mode,
                "pixel_size": entry.get("pixel_size", self._pixel_size),
            }

    def remove(self, rid: str) -> None:
        with self._lock:
            self._data.pop(rid, None)


beam_manager = BeamAnalysisManager()

# ----- Per-ROI intensity tracking -----
_roi_intensity_lock = threading.Lock()
_roi_peak_intensity: Dict[str, float] = {}
_roi_cuts_peak: Dict[str, Tuple[float, float]] = {}


def _update_roi_peak_intensity(rid: str, arr: Optional[np.ndarray]) -> float:
    if arr is None or getattr(arr, "size", 0) == 0:
        with _roi_intensity_lock:
            stored = _roi_peak_intensity.get(rid)
            if stored is None:
                app.logger.debug(f"ROI {rid}: peak requested with no history; returning default 1.0")
                return 1.0
            return stored
    a = np.asarray(arr, dtype=np.float32)
    if a.size == 0:
        with _roi_intensity_lock:
            stored = _roi_peak_intensity.get(rid)
            if stored is None:
                app.logger.debug(f"ROI {rid}: zero-sized array provided; returning default 1.0")
                return 1.0
            return stored
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    peak_now = float(np.max(a))
    if not np.isfinite(peak_now):
        peak_now = 0.0
    peak_now = max(0.0, peak_now)
    with _roi_intensity_lock:
        prev = _roi_peak_intensity.get(rid)
        base = prev if prev is not None else 0.0
        peak = max(base, peak_now, 1.0)
        _roi_peak_intensity[rid] = peak
        if prev is None:
            app.logger.debug(f"ROI {rid}: initializing peak intensity to {peak:.3f}")
        elif peak > prev:
            app.logger.debug(f"ROI {rid}: peak intensity updated from {prev:.3f} to {peak:.3f}")
        return peak


def _get_roi_peak_intensity(rid: str) -> float:
    with _roi_intensity_lock:
        return _roi_peak_intensity.get(rid, 1.0)


def _update_roi_cuts_peak(rid: str, Ix: np.ndarray, Iy: np.ndarray) -> Tuple[float, float]:
    max_x = float(np.nanmax(Ix)) if getattr(Ix, "size", 0) else 0.0
    max_y = float(np.nanmax(Iy)) if getattr(Iy, "size", 0) else 0.0
    max_x = max(0.0, max_x)
    max_y = max(0.0, max_y)
    with _roi_intensity_lock:
        prev_x, prev_y = _roi_cuts_peak.get(rid, (None, None))
        base_x = prev_x if prev_x is not None else 0.0
        base_y = prev_y if prev_y is not None else 0.0
        peak_x = max(base_x, max_x, 1.0)
        peak_y = max(base_y, max_y, 1.0)
        _roi_cuts_peak[rid] = (peak_x, peak_y)
        if prev_x is None or prev_y is None:
            app.logger.debug(f"ROI {rid}: initializing cuts peaks to ({peak_x:.3f}, {peak_y:.3f})")
        else:
            changed = []
            if peak_x > prev_x:
                changed.append(f"Ix {prev_x:.3f}->{peak_x:.3f}")
            if peak_y > prev_y:
                changed.append(f"Iy {prev_y:.3f}->{peak_y:.3f}")
            if changed:
                joined = '; '.join(changed)
                app.logger.debug(f"ROI {rid}: cuts peaks updated {joined}")
        return peak_x, peak_y


def _get_roi_cuts_peak(rid: str) -> Tuple[float, float]:
    with _roi_intensity_lock:
        return _roi_cuts_peak.get(rid, (1.0, 1.0))


def _reset_roi_peak_history(rid: str) -> None:
    with _roi_intensity_lock:
        had_peak = rid in _roi_peak_intensity
        had_cuts = rid in _roi_cuts_peak
        _roi_peak_intensity.pop(rid, None)
        _roi_cuts_peak.pop(rid, None)
    app.logger.debug(f"ROI {rid}: peak history reset (intensity={had_peak}, cuts={had_cuts})")


def _normalize_to_u8(arr: np.ndarray, peak: float) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    hi = max(1.0, float(peak))
    norm = np.clip(a / hi, 0.0, 1.0)
    return (norm * 255.0).astype(np.uint8)


def _beam_worker(payload: Tuple[np.ndarray, Dict[str, object]]) -> Optional[Dict[str, object]]:
    img, kwargs = payload
    try:
        return analyze_beam(img, **kwargs)
    except Exception as exc:
        app.logger.warning(f"beam_analysis worker error: {exc}")
        return None



def _get_roi_gray_image(rid: str, *, entry: Optional[Dict[str, object]] = None) -> Optional[np.ndarray]:
    """Return the latest grayscale ROI image for ``rid`` as float32."""
    if entry is None:
        entry = beam_manager.get(rid)
    roi_gray = None
    if entry:
        roi_gray = entry.get("roi_gray")
        if isinstance(roi_gray, np.ndarray) and getattr(roi_gray, "size", 0) > 0:
            return np.asarray(roi_gray, dtype=np.float32)
    r = roi_registry.get(rid)
    if r is None:
        return None
    gray = cam_service.get_latest_gray()
    if gray is None or getattr(gray, "size", 0) == 0:
        gray = cam_service.get_latest_gray(raw=True)
    if gray is None or getattr(gray, "size", 0) == 0:
        return None
    h_img, w_img = gray.shape[:2]
    x0 = max(0, min(int(r.x), w_img - 1))
    y0 = max(0, min(int(r.y), h_img - 1))
    x1 = max(x0 + 1, min(int(r.x + r.w), w_img))
    y1 = max(y0 + 1, min(int(r.y + r.h), h_img))
    roi_crop = gray[y0:y1, x0:x1]
    if roi_crop.size == 0:
        return None
    return np.asarray(roi_crop, dtype=np.float32).copy()


def _get_roi_raw_gray_image(rid: str) -> Optional[np.ndarray]:
    """Return the latest raw grayscale ROI (no background subtraction)."""
    r = roi_registry.get(rid)
    if r is None:
        return None
    raw = cam_service.get_latest_gray(raw=True)
    if raw is None or getattr(raw, "size", 0) == 0:
        return None
    h_img, w_img = raw.shape[:2]
    x0 = max(0, min(int(r.x), w_img - 1))
    y0 = max(0, min(int(r.y), h_img - 1))
    x1 = max(x0 + 1, min(int(r.x + r.w), w_img))
    y1 = max(y0 + 1, min(int(r.y + r.h), h_img))
    roi_crop = raw[y0:y1, x0:x1]
    if roi_crop.size == 0:
        return None
    return np.asarray(roi_crop).copy()


def _compose_roi_profile_image(rid: str) -> Optional[np.ndarray]:
    entry = beam_manager.get(rid)
    compute_mode = str(entry.get("compute") if entry else "none").lower()
    result = entry.get("result") if entry else None

    roi_gray = _get_roi_gray_image(rid, entry=entry)
    if roi_gray is None:
        return None

    compute = compute_mode or "none"
    cm = cam_service.get_colormap()
    base_peak = _update_roi_peak_intensity(rid, roi_gray)

    if compute == "none" or not result:
        roi_u8 = _normalize_to_u8(roi_gray, base_peak)
        return apply_colormap_to_gray(roi_u8, cm)

    try:
        proc = result.get("img_for_spec") if isinstance(result, dict) else None
        if proc is None or not isinstance(proc, np.ndarray) or proc.size == 0:
            proc = np.asarray(roi_gray, dtype=np.float32)
        else:
            proc = np.asarray(proc, dtype=np.float32)
        if proc.ndim == 3:
            proc = np.mean(proc, axis=-1)
        proc = np.nan_to_num(proc, nan=0.0, posinf=0.0, neginf=0.0)

        peak = _update_roi_peak_intensity(rid, proc)
        roi_u8 = _normalize_to_u8(proc, peak)
        img = apply_colormap_to_gray(roi_u8, cm)

        cy = float(result.get("cy", 0.0))
        cx = float(result.get("cx", 0.0))
        rx_iso = float(result.get("rx_iso", 0.0))
        ry_iso = float(result.get("ry_iso", 0.0))
        theta = float(result.get("theta", 0.0))
        fit_x = result.get("gauss_fit_x") if isinstance(result, dict) else None
        fit_y = result.get("gauss_fit_y") if isinstance(result, dict) else None
        gauss_rx = (fit_x or {}).get("radius")
        gauss_ry = (fit_y or {}).get("radius")

        major_r, minor_r, ang = rx_iso, ry_iso, theta
        if ry_iso > rx_iso:
            major_r, minor_r = ry_iso, rx_iso
            ang = theta + (np.pi * 0.5)
        c = float(np.cos(ang))
        s = float(np.sin(ang))
        if gauss_rx is not None and gauss_ry is not None:
            g_major, g_minor = (float(gauss_rx), float(gauss_ry))
            if ry_iso > rx_iso:
                g_major, g_minor = float(gauss_ry), float(gauss_rx)
        else:
            g_major = g_minor = None

        h_img, w_img = img.shape[:2]
        t = np.linspace(0.0, 2.0 * np.pi, 361, dtype=np.float32)

        major_color, minor_color = _COLOR_X_ISO, _COLOR_Y_ISO
        if ry_iso > rx_iso:
            major_color, minor_color = _COLOR_Y_ISO, _COLOR_X_ISO

        if compute in ("both", "second"):
            ex = cx + major_r * np.cos(t) * c - minor_r * np.sin(t) * s
            ey = cy + major_r * np.cos(t) * s + minor_r * np.sin(t) * c
            ellipse_pts = np.stack([ex, ey], axis=1)
            ellipse_pts = np.nan_to_num(ellipse_pts, nan=0.0)
            ellipse_pts[:, 0] = np.clip(ellipse_pts[:, 0], 0, w_img - 1)
            ellipse_pts[:, 1] = np.clip(ellipse_pts[:, 1], 0, h_img - 1)
            ellipse_pts = np.round(ellipse_pts).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [ellipse_pts], True, _COLOR_ISO_ELLIPSE, 2, cv2.LINE_AA)

            major_pt1 = (int(np.clip(round(cx - major_r * c), 0, w_img - 1)), int(np.clip(round(cy - major_r * s), 0, h_img - 1)))
            major_pt2 = (int(np.clip(round(cx + major_r * c), 0, w_img - 1)), int(np.clip(round(cy + major_r * s), 0, h_img - 1)))
            minor_pt1 = (int(np.clip(round(cx + minor_r * s), 0, w_img - 1)), int(np.clip(round(cy - minor_r * c), 0, h_img - 1)))
            minor_pt2 = (int(np.clip(round(cx - minor_r * s), 0, w_img - 1)), int(np.clip(round(cy + minor_r * c), 0, h_img - 1)))
            cv2.line(img, major_pt1, major_pt2, major_color, 2, cv2.LINE_AA)
            cv2.line(img, minor_pt1, minor_pt2, minor_color, 2, cv2.LINE_AA)

        if compute in ("both", "gauss") and g_major is not None and g_minor is not None:
            gx = cx + g_major * np.cos(t) * c - g_minor * np.sin(t) * s
            gy = cy + g_major * np.cos(t) * s + g_minor * np.sin(t) * c
            g_pts = np.stack([gx, gy], axis=1)
            g_pts = np.nan_to_num(g_pts, nan=0.0)
            g_pts[:, 0] = np.clip(g_pts[:, 0], 0, w_img - 1)
            g_pts[:, 1] = np.clip(g_pts[:, 1], 0, h_img - 1)
            g_pts = np.round(g_pts).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [g_pts], True, _COLOR_GAUSS, 2, cv2.LINE_AA)

        target_h = 240.0
        scale = target_h / max(1.0, float(h_img))
        scaled_w = max(1, int(round(w_img * scale)))
        scaled_h = int(round(target_h))
        return cv2.resize(img, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
    except Exception as exc:
        app.logger.warning(f"compose profile error for ROI {rid}: {exc}", exc_info=True)
        return None

_PLOT_BG = (29, 21, 17)
_PLOT_AXIS = (82, 86, 96)
_PLOT_TEXT_PRI = (230, 232, 236)
_PLOT_TEXT_SEC = (190, 195, 205)
_COLOR_X_ISO = (226, 144, 74)
_COLOR_Y_ISO = (194, 227, 80)
_COLOR_GAUSS = (0, 220, 255)
_COLOR_ISO_ELLIPSE = (255, 255, 255)

def _draw_cuts_panel(
    panel: np.ndarray,
    positions: np.ndarray,
    values: np.ndarray,
    *,
    title: str,
    axis_label: str,
    line_color: Tuple[int, int, int],
    iso_center: float,
    iso_radius: float,
    gauss_amplitude: float,
    gauss_center: float,
    gauss_radius: float,
    pixel_size: Optional[float],
    value_max: float,
) -> None:
    h, w = panel.shape[:2]
    top, bottom, left, right = 28, 32, 36, 18
    plot_w = max(1, w - left - right)
    plot_h = max(1, h - top - bottom)

    origin = (left, h - bottom)
    cv2.line(panel, origin, (w - right, h - bottom), _PLOT_AXIS, 1, cv2.LINE_AA)
    cv2.line(panel, origin, (left, top), _PLOT_AXIS, 1, cv2.LINE_AA)

    cv2.putText(panel, title, (12, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _PLOT_TEXT_PRI, 1, cv2.LINE_AA)
    cv2.putText(panel, axis_label, (left, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _PLOT_TEXT_SEC, 1, cv2.LINE_AA)

    pos = np.asarray(positions, dtype=np.float32).reshape(-1)
    vals = np.asarray(values, dtype=np.float32).reshape(-1)
    if pos.size == 0 or vals.size == 0:
        return

    pos = np.nan_to_num(pos, nan=0.0, posinf=0.0, neginf=0.0)
    vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)

    base = float(np.min(pos)) if pos.size else 0.0
    span = float(np.max(pos) - base) if pos.size else 0.0
    if not np.isfinite(base):
        base = 0.0
    if not np.isfinite(span) or span < 1e-6:
        span = 1.0

    pos_norm = np.clip((pos - base) / span, 0.0, 1.0)
    x_coords = left + np.clip(np.round(pos_norm * (plot_w - 1)).astype(np.int32), 0, plot_w - 1)

    max_val = max(1.0, float(value_max))
    val_norm = np.clip(vals / max_val, 0.0, 1.0)
    y_coords = (h - bottom) - np.clip(np.round(val_norm * (plot_h - 1)).astype(np.int32), 0, plot_h - 1)

    pts = np.column_stack((x_coords, y_coords)).astype(np.int32)
    if pts.shape[0] >= 2:
        cv2.polylines(panel, [pts.reshape(-1, 1, 2)], False, line_color, 2, cv2.LINE_AA)
    elif pts.shape[0] == 1:
        cv2.circle(panel, tuple(pts[0]), 2, line_color, -1, cv2.LINE_AA)

    def _pos_to_x(value: float) -> int:
        if not np.isfinite(value):
            value = base
        ratio = (value - base) / span
        px = left + int(round(ratio * (plot_w - 1)))
        return max(left, min(px, w - right - 1))

    if iso_radius > 0.0:
        lx = _pos_to_x(iso_center - iso_radius)
        rx = _pos_to_x(iso_center + iso_radius)
        cv2.line(panel, (lx, top), (lx, h - bottom), line_color, 1, cv2.LINE_AA)
        cv2.line(panel, (rx, top), (rx, h - bottom), line_color, 1, cv2.LINE_AA)

    if gauss_radius > 0.0:
        lx = _pos_to_x(gauss_center - gauss_radius)
        rx = _pos_to_x(gauss_center + gauss_radius)
        cv2.line(panel, (lx, top), (lx, h - bottom), _COLOR_GAUSS, 1, cv2.LINE_AA)
        cv2.line(panel, (rx, top), (rx, h - bottom), _COLOR_GAUSS, 1, cv2.LINE_AA)

    if gauss_radius > 0.0 and gauss_amplitude > 0.0:
        samples = np.linspace(base, base + span, 256, dtype=np.float32)
        gauss_vals = gauss_amplitude * np.exp(-2.0 * ((samples - gauss_center) / max(gauss_radius, 1e-6)) ** 2)
        gauss_vals = np.nan_to_num(gauss_vals, nan=0.0, posinf=0.0, neginf=0.0)
        sample_norm = np.clip((samples - base) / span, 0.0, 1.0)
        gauss_norm = np.clip(gauss_vals / max_val, 0.0, 1.0)
        gx = left + np.clip(np.round(sample_norm * (plot_w - 1)).astype(np.int32), 0, plot_w - 1)
        gy = (h - bottom) - np.clip(np.round(gauss_norm * (plot_h - 1)).astype(np.int32), 0, plot_h - 1)
        g_pts = np.column_stack((gx, gy)).astype(np.int32)
        if g_pts.shape[0] >= 2:
            cv2.polylines(panel, [g_pts.reshape(-1, 1, 2)], False, _COLOR_GAUSS, 1, cv2.LINE_AA)




def _compose_roi_cuts_image(rid: str) -> Optional[np.ndarray]:
    entry = beam_manager.get(rid)
    compute_mode = str(entry.get("compute") if entry else "none").lower()
    result = entry.get("result") if entry else None
    pixel_size_val = entry.get("pixel_size") if entry else None
    draw_iso = compute_mode in ("both", "second")
    try:
        pixel_size_val = float(pixel_size_val)
    except (TypeError, ValueError):
        pixel_size_val = None

    if compute_mode == "none" or result is None:
        app.logger.debug(f"compose cuts skipping for ROI {rid}: compute={compute_mode}, has_result={bool(result)}")
        return None

    try:
        x_positions, Ix = result["Ix_spectrum"]
        y_positions, Iy = result["Iy_spectrum"]
        Ix = np.asarray(Ix, dtype=np.float32)
        Iy = np.asarray(Iy, dtype=np.float32)
        peak_x, peak_y = _update_roi_cuts_peak(rid, Ix, Iy)

        fit_x = result.get("gauss_fit_x") or {}
        fit_y = result.get("gauss_fit_y") or {}
        cx_iso = float(result.get("cx_iso", result.get("cx", 0.0)))
        cy_iso = float(result.get("cy_iso", result.get("cy", 0.0)))
        rx_iso = float(result.get("rx_iso", 0.0)) if draw_iso else 0.0
        ry_iso = float(result.get("ry_iso", 0.0)) if draw_iso else 0.0
        gauss_cx = float(fit_x.get("centre", cx_iso))
        gauss_cy = float(fit_y.get("centre", cy_iso))
        gauss_rx = float(fit_x.get("radius", 0.0)) if fit_x else 0.0
        gauss_ry = float(fit_y.get("radius", 0.0)) if fit_y else 0.0
        amp_x = float(fit_x.get("amplitude", 0.0)) if fit_x else 0.0
        amp_y = float(fit_y.get("amplitude", 0.0)) if fit_y else 0.0

        unit_suffix = '' if pixel_size_val is not None else ' px'
        rx_iso_disp = rx_iso * (pixel_size_val or 1.0)
        ry_iso_disp = ry_iso * (pixel_size_val or 1.0)
        gauss_rx_disp = gauss_rx * (pixel_size_val or 1.0)
        gauss_ry_disp = gauss_ry * (pixel_size_val or 1.0)

        title_x = f"Ix | ISO:{rx_iso_disp:.1e} | G:{gauss_rx_disp:.1e}{unit_suffix}"
        title_y = f"Iy | ISO:{ry_iso_disp:.1e} | G:{gauss_ry_disp:.1e}{unit_suffix}"

        canvas = np.full((240, 480, 3), _PLOT_BG, dtype=np.uint8)
        with _plot_lock:
            _draw_cuts_panel(
                canvas[:, :240],
                np.asarray(x_positions, dtype=np.float32),
                Ix,
                title=title_x,
                axis_label="x (local px)",
                line_color=_COLOR_X_ISO,
                iso_center=cx_iso,
                iso_radius=rx_iso,
                gauss_amplitude=amp_x,
                gauss_center=gauss_cx,
                gauss_radius=gauss_rx,
                pixel_size=pixel_size_val,
                value_max=peak_x,
            )
            _draw_cuts_panel(
                canvas[:, 240:],
                np.asarray(y_positions, dtype=np.float32),
                Iy,
                title=title_y,
                axis_label="y (local px)",
                line_color=_COLOR_Y_ISO,
                iso_center=cy_iso,
                iso_radius=ry_iso,
                gauss_amplitude=amp_y,
                gauss_center=gauss_cy,
                gauss_radius=gauss_ry,
                pixel_size=pixel_size_val,
                value_max=peak_y,
            )
            cv2.line(canvas, (240, 24), (240, 240 - 24), _PLOT_AXIS, 1, cv2.LINE_AA)
        return canvas
    except Exception as exc:
        app.logger.warning(f"compose cuts error for ROI {rid}: {exc}", exc_info=True)
        return None




# Background metrics loop
def _metrics_loop():
    last_size: Optional[Tuple[int, int]] = None
    while True:
        try:
            cam_service.wait_for_frame_signal(timeout=0.5)
            raw_gray = cam_service.get_latest_gray(raw=True)
            gray_processed = cam_service.get_latest_gray()
            if (gray_processed is None or getattr(gray_processed, "size", 0) == 0) and raw_gray is not None:
                gray_processed = raw_gray.copy()
            if gray_processed is None or getattr(gray_processed, "size", 0) == 0:
                continue
            exp = cam_service.get_exposure_us()
            size = cam_service.get_frame_size()
            if size != last_size:
                app.logger.info(f"Frame size changed to {size}. Clamping ROIs.")
                roi_registry.clamp_all(size)
                last_size = size
            rois = roi_registry.list()
            metrics.update(gray_processed, exp, rois, raw_gray=raw_gray)
            opts = _get_beam_options_copy()
            beam_manager.update(gray_processed, rois, opts)
        except Exception as e:
            app.logger.error(f"Exception in metrics loop: {e}", exc_info=True)
            time.sleep(0.1)

_metrics_thread = threading.Thread(target=_metrics_loop, daemon=True)
_metrics_thread.start()


_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")

def _sanitize_filename(value: str, fallback: str) -> str:
    safe = _SAFE_FILENAME_RE.sub('_', value).strip('._-')
    return safe or fallback


def _json_default(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# ----- Helpers -----
def mjpeg_response(gen: Generator[bytes, None, None]) -> Response:
    return Response(gen, mimetype="multipart/x-mixed-replace; boundary=frame")

def _encode_jpeg(img: np.ndarray) -> Optional[bytes]:
    ok, buf = cv2.imencode(".jpg", img)
    return buf.tobytes() if ok else None

def _encode_png(img: np.ndarray) -> Optional[bytes]:
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes() if ok else None

def _encode_bmp(img: np.ndarray) -> Optional[bytes]:
    ok, buf = cv2.imencode('.bmp', img)
    return buf.tobytes() if ok else None


def _prepare_gray_for_bmp(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if arr is None:
        return None
    img = np.asarray(arr)
    if img.ndim == 3 and img.shape[2] == 1:
        img = img[:, :, 0]
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    if img.ndim != 2:
        return None
    if img.dtype in (np.uint8, np.uint16):
        return img
    if np.issubdtype(img.dtype, np.integer):
        img = np.clip(img, 0, None)
        max_val = int(np.max(img)) if img.size else 0
        if max_val <= 255:
            return img.astype(np.uint8)
        return np.clip(img, 0, 65535).astype(np.uint16)
    if np.issubdtype(img.dtype, np.floating):
        img = np.clip(img, 0.0, None)
        max_val = float(np.max(img)) if img.size else 0.0
        if max_val <= 255.0:
            return np.rint(img).astype(np.uint8)
        return np.rint(np.clip(img, 0.0, 65535.0)).astype(np.uint16)
    return None


def _encode_gray_bmp(arr: Optional[np.ndarray]) -> Optional[bytes]:
    prepared = _prepare_gray_for_bmp(arr)
    if prepared is None:
        return None
    return _encode_bmp(prepared)


def _safe_roi_fs_name(rid: str) -> str:
    return _sanitize_filename(rid, rid.replace(':', '_').replace('/', '_'))


def _zip_write_npy(zf: zipfile.ZipFile, arcname: str, arr: np.ndarray) -> None:
    buf = io.BytesIO()
    np.save(buf, arr)
    zf.writestr(arcname, buf.getvalue())


def _placeholder(text: str, size=(640, 360)) -> np.ndarray:
    w, h = size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(img, text, (20, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def _caustic_point_dir(point_id: str) -> Path:
    path = CAUSTIC_CACHE_DIR / point_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_png_image(img: Optional[np.ndarray], path: Path) -> Optional[str]:
    if img is None:
        return None
    png_bytes = _encode_png(img)
    if not png_bytes:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png_bytes)
    return str(path)


def _save_raw_roi_bmp(arr: Optional[np.ndarray], path: Path) -> Optional[str]:
    if arr is None:
        return None
    data = np.asarray(arr)
    if data.size == 0:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    encode_input = data
    if data.dtype != np.uint8:
        float_data = np.asarray(data, dtype=np.float32)
        float_data = np.nan_to_num(float_data, nan=0.0, posinf=0.0, neginf=0.0)
        encode_input = np.clip(np.rint(float_data), 0.0, 255.0).astype(np.uint8)
    if encode_input.ndim == 3 and encode_input.shape[2] == 1:
        encode_input = encode_input[:, :, 0]
    ok, buf = cv2.imencode(".bmp", encode_input)
    if not ok:
        return None
    path.write_bytes(buf.tobytes())
    return str(path)


def _clean_float(value: Optional[float]) -> Optional[float]:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    return v


def _extract_pixel_size_m(entry: Optional[Dict[str, object]]) -> Optional[float]:
    if not entry:
        return None
    try:
        value = entry.get("pixel_size")
    except AttributeError:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) and result > 0.0 else None


def _serialize_caustic_point(pt: CausticPoint, display_unit: str) -> Dict[str, object]:
    z_display = convert_meters_to_length(pt.z_m, display_unit)
    return {
        "id": pt.point_id,
        "roi_id": pt.roi_id,
        "timestamp_iso": pt.timestamp_iso,
        "z_m": pt.z_m,
        "z_display": z_display,
        "position_unit": display_unit,
        "position_unit_at_capture": pt.position_unit_at_capture,
        "wavelength_nm": pt.wavelength_nm,
        "pixel_size_m": _clean_float(pt.pixel_size_m),
        "radii": {
            "gauss_1e2": {
                "wx": _clean_float(pt.wx_1e2),
                "wy": _clean_float(pt.wy_1e2),
            },
            "moment_2sigma": {
                "wx": _clean_float(pt.wx_2sigma),
                "wy": _clean_float(pt.wy_2sigma),
            },
        },
        "images": {
            "profile": f"/caustic/image/{pt.point_id}/profile" if pt.profile_img_path else None,
            "cut_x": f"/caustic/image/{pt.point_id}/cut_x" if pt.cut_x_img_path else None,
            "cut_y": f"/caustic/image/{pt.point_id}/cut_y" if pt.cut_y_img_path else None,
            "raw": f"/caustic/image/{pt.point_id}/raw" if pt.raw_roi_img_path else None,
        },
    }


def _format_fit_results_for_api(fits: Dict[str, Dict[str, float]], display_unit: str) -> Dict[str, object]:
    formatted: Dict[str, object] = {}

    for axis in ("x", "y"):
        data = fits.get(axis) if isinstance(fits, dict) else None
        if not isinstance(data, dict):
            continue
        sigma = data.get("sigma") if isinstance(data.get("sigma"), dict) else {}

        def _sigma_val(key: str) -> Optional[float]:
            return _clean_float(sigma.get(key)) if sigma else None

        w0_m = _clean_float(data.get("w0_m"))
        z0_m = _clean_float(data.get("z0_m"))
        zr_m = _clean_float(data.get("zR_prime_m"))

        sigma_w0 = _sigma_val("w0_m")
        sigma_z0 = _sigma_val("z0_m")
        sigma_zr = _sigma_val("zR_prime_m")
        sigma_m2 = _sigma_val("M2")

        formatted[axis] = {
            "w0_m": w0_m,
            "w0_um": w0_m * 1e6 if w0_m is not None else None,
            "z0_m": z0_m,
            "z0_display": convert_meters_to_length(z0_m, display_unit) if z0_m is not None else None,
            "zR_prime_m": zr_m,
            "zR_prime_display": convert_meters_to_length(zr_m, display_unit) if zr_m is not None else None,
            "M2": _clean_float(data.get("M2")),
            "sigma": {
                "w0_m": sigma_w0,
                "w0_um": sigma_w0 * 1e6 if sigma_w0 is not None else None,
                "z0_m": sigma_z0,
                "z0_display": convert_meters_to_length(sigma_z0, display_unit) if sigma_z0 is not None else None,
                "zR_prime_m": sigma_zr,
                "zR_prime_display": convert_meters_to_length(sigma_zr, display_unit) if sigma_zr is not None else None,
                "M2": sigma_m2,
            },
        }
    return formatted


def _caustic_state_payload() -> Dict[str, object]:
    config = caustic_manager.config_snapshot()
    display_unit = str(config.get("position_unit", "mm"))
    return {
        "config": config,
        "points": [
            _serialize_caustic_point(pt, display_unit)
            for pt in caustic_manager.list_points()
        ],
        "series": caustic_manager.get_plot_series(),
        "fits": _format_fit_results_for_api(caustic_manager.fit_results(), display_unit),
    }

# ----- Routes -----

@app.before_request
def log_request_info():
    if request.path in ['/log/js', '/metrics']:
        return
    app.logger.debug(f"Request: {request.method} {request.path}")
    if request.data and request.path != '/log/js':
        try:
            decoded_data = request.data.decode('utf-8')
            app.logger.debug(f"Request data: {decoded_data}")
        except UnicodeDecodeError:
            app.logger.debug(f"Request data: <binary data of length {len(request.data)}>")

@app.route("/")
def index():
    return render_template("index.html")

# JS logger endpoint
@app.route("/log/js", methods=["POST"])
def log_js_message():
    msg = request.get_json(silent=True) or {}
    level = msg.get("level", "info").upper()
    log_message = f"[JS-{level}] {msg.get('message', '')}"
    if 'data' in msg:
        log_message += f" | data: {json.dumps(msg['data'])}"
    log_func = getattr(app.logger, level.lower(), app.logger.info)
    log_func(log_message)
    return jsonify(success=True)

# Video stream
@app.route("/video_feed")
def video_feed():
    return mjpeg_response(cam_service.gen_overview_mjpeg())

# Exposure GET/POST
@app.route("/exposure", methods=["GET", "POST"])
def exposure():
    if request.method == "GET":
        try:
            val = cam_service.get_exposure_us()
            return jsonify({"value": int(val)})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    try:
        data = request.get_json(silent=True) or {}
        req_val = int(data.get("value"))
    except Exception:
        return jsonify({"error": "Invalid value"}), 400
    try:
        applied = cam_service.set_exposure_us(req_val)
        return jsonify({"value": int(applied)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/exposure/auto", methods=["POST"])
def exposure_auto():
    data = request.get_json(silent=True) or {}
    target_fraction = 0.7
    tolerance = 0.05
    if "target_fraction" in data:
        try:
            target_fraction = float(data["target_fraction"])
        except (TypeError, ValueError):
            return jsonify({"error": "Invalid target_fraction"}), 400
    if "tolerance" in data:
        try:
            tolerance = float(data["tolerance"])
        except (TypeError, ValueError):
            return jsonify({"error": "Invalid tolerance"}), 400
    target_fraction = max(0.05, min(0.95, target_fraction))
    tolerance = max(0.005, min(0.3, tolerance))
    try:
        result = cam_service.auto_adjust_exposure(
            target_fraction=target_fraction,
            tolerance=tolerance,
        )
        return jsonify(result)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 409
    except Exception as exc:
        app.logger.error(f"Auto exposure failed: {exc}", exc_info=True)
        return jsonify({"error": str(exc)}), 500

# Colormap GET/POST
@app.route("/colormap", methods=["GET", "POST"])
def colormap():
    if request.method == "GET":
        return jsonify({"value": cam_service.get_colormap()})
    data = request.get_json(silent=True) or {}
    value = str(data.get("value", "")).lower()
    try:
        applied = cam_service.set_colormap(value)
        return jsonify({"value": applied})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Camera List
@app.route("/cameras")
def list_cameras():
    cameras = cam_service.list_available_cameras()
    return jsonify(cameras)

# Beam analysis options GET/POST
@app.route("/beam_options", methods=["GET", "POST"])
def beam_options():
    global _beam_opts
    if request.method == "GET":
        with _beam_opts_lock:
            return jsonify(dict(_beam_opts))
    data = request.get_json(silent=True) or {}
    # Validate and update known keys only
    allowed_compute = {"both", "second", "gauss", "none"}
    allowed_clip = {"none", "zero", "otsu"}
    allowed_angle_clip = {"same", "none", "zero", "otsu"}
    allowed_rotation = {"auto", "fixed"}
    with _beam_opts_lock:
        if "pixel_size" in data:
            try:
                v = data["pixel_size"]
                _beam_opts["pixel_size"] = None if v in (None, "", "null") else float(v)
            except Exception:
                pass
        if "compute" in data and str(data["compute"]) in allowed_compute:
            _beam_opts["compute"] = str(data["compute"])
        if "clip_negatives" in data:
            val = data["clip_negatives"]
            if isinstance(val, bool):
                val = "zero" if val else "none"
            val = str(val).lower()
            if val in allowed_clip:
                _beam_opts["clip_negatives"] = val
        if "angle_clip_mode" in data and str(data["angle_clip_mode"]).lower() in allowed_angle_clip:
            _beam_opts["angle_clip_mode"] = str(data["angle_clip_mode"]).lower()
        if "background_subtraction" in data:
            _beam_opts["background_subtraction"] = bool(data["background_subtraction"])
        if "rotation" in data and str(data["rotation"]).lower() in allowed_rotation:
            _beam_opts["rotation"] = str(data["rotation"]).lower()
        if "fixed_angle" in data:
            try:
                v = data["fixed_angle"]
                _beam_opts["fixed_angle"] = None if v in (None, "", "null") else float(v)
            except Exception:
                pass
        return jsonify(dict(_beam_opts))

# Background Subtraction
@app.route("/background_subtraction", methods=["POST"])
def background_subtraction():
    data = request.get_json(silent=True) or {}
    enabled = bool(data.get("enabled", False))
    if enabled:
        num_frames = int(data.get("num_frames", 10))
        try:
            cam_service.start_background_subtraction(num_frames)
            return jsonify({"ok": True})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        try:
            cam_service.stop_background_subtraction()
            return jsonify({"ok": True})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

# Camera On/Off
@app.route("/camera", methods=["POST"])
def camera_toggle():
    data = request.get_json(silent=True) or {}
    enabled = bool(data.get("enabled", False))
    camera_id = data.get("camera_id")
    err = None
    try:
        if enabled:
            app.logger.info(f"Camera START requested for ID: {camera_id}")
            cam_service.start(camera_id)
        else:
            app.logger.info("Camera STOP requested.")
            cam_service.stop()
    except Exception as e:
        err = str(e)
        app.logger.error(f"Failed to toggle camera: {e}", exc_info=True)
    return jsonify({"enabled": cam_service.is_running(), "error": err})

# ROI CRUD
@app.route("/rois", methods=["GET", "POST"])
def rois():
    if request.method == "GET":
        return jsonify(roi_registry.list_dicts())
    data = request.get_json(silent=True) or {}
    try:
        x, y, w, h = int(data["x"]), int(data["y"]), int(data["w"]), int(data["h"])
    except Exception:
        return jsonify({"error": "Invalid ROI"}), 400
    roi = roi_registry.create(x, y, w, h, cam_service.get_frame_size())
    return jsonify(roi.to_dict())

@app.route("/rois/<rid>", methods=["GET", "PUT", "DELETE"])
def rois_item(rid: str):
    if request.method == "GET":
        r = roi_registry.get_dict(rid)
        return jsonify(r) if r else (jsonify({"error": "Not found"}), 404)

    if request.method == "DELETE":
        ok = roi_registry.delete(rid)
        if ok:
            metrics.remove_roi_metrics(rid)
            beam_manager.remove(rid)
            _reset_roi_peak_history(rid)
        return jsonify({"deleted": bool(ok)})

    data = request.get_json(silent=True) or {}
    try:
        x, y, w, h = int(data["x"]), int(data["y"]), int(data["w"]), int(data["h"])
    except Exception:
        return jsonify({"error": "Invalid ROI"}), 400
    r = roi_registry.update(rid, x, y, w, h, cam_service.get_frame_size())
    return jsonify(r.to_dict()) if r else (jsonify({"error": "Not found"}), 404)

@app.route("/roi/<rid>/reset_max_values", methods=["POST"])
def roi_reset_max_values(rid: str):
    metrics.reset_max_values(rid)
    _reset_roi_peak_history(rid)
    return jsonify({"ok": True})

# Metrics polling
@app.route("/metrics")
def metrics_route():
    return jsonify(metrics.get_snapshot())


# ----- Caustic endpoints -----
@app.route("/caustic/state")
def caustic_state():
    return jsonify(_caustic_state_payload())


@app.route("/caustic/config", methods=["POST"])
def caustic_config():
    data = request.get_json(silent=True) or {}

    if "wavelength_nm" in data:
        try:
            wavelength = float(data["wavelength_nm"])
        except (TypeError, ValueError):
            return jsonify({"error": "Invalid wavelength"}), 400
        if wavelength <= 0.0:
            return jsonify({"error": "Wavelength must be positive"}), 400
        caustic_manager.set_wavelength_nm(wavelength)

    if "position_unit" in data:
        unit = str(data["position_unit"] or "").strip()
        if unit:
            try:
                caustic_manager.set_position_unit(unit)
            except ValueError as exc:
                return jsonify({"error": str(exc)}), 400

    if "radii_source" in data:
        try:
            source = CausticRadiiSource.from_label(str(data["radii_source"]))
            caustic_manager.set_radii_source(source)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

    return jsonify(_caustic_state_payload())


@app.route("/caustic/add", methods=["POST"])
def caustic_add_point():
    data = request.get_json(silent=True) or {}
    roi_id = data.get("roi_id")
    if not roi_id:
        return jsonify({"error": "roi_id is required"}), 400

    try:
        z_value = float(data.get("z"))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid z value"}), 400

    config = caustic_manager.config_snapshot()
    try:
        z_m = convert_length_to_meters(z_value, str(config.get("position_unit", "mm")))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    entry = beam_manager.get(roi_id)
    result = entry.get("result") if entry else None
    if not isinstance(result, dict):
        return jsonify({"error": "ROI analysis unavailable"}), 404

    gauss_x = result.get("gauss_fit_x") or {}
    gauss_y = result.get("gauss_fit_y") or {}

    try:
        wx_1e2 = float(gauss_x.get("radius"))
        wy_1e2 = float(gauss_y.get("radius"))
        wx_2sigma = float(result.get("rx_iso"))
        wy_2sigma = float(result.get("ry_iso"))
    except (TypeError, ValueError):
        return jsonify({"error": "Beam radii unavailable"}), 400

    if wx_1e2 <= 0.0 or wy_1e2 <= 0.0:
        return jsonify({"error": "Gaussian radii unavailable"}), 400
    if wx_2sigma <= 0.0 or wy_2sigma <= 0.0:
        return jsonify({"error": "Second-moment radii unavailable"}), 400

    point_id = caustic_manager.generate_point_id()
    point_dir = _caustic_point_dir(point_id)

    profile_img = _compose_roi_profile_image(roi_id)
    profile_path = _save_png_image(profile_img, point_dir / "profile.png") or ""

    cut_x_path = ""
    cut_y_path = ""
    cuts_img = _compose_roi_cuts_image(roi_id)
    if cuts_img is not None:
        try:
            h, w = cuts_img.shape[:2]
            mid = max(1, w // 2)
            cut_x_img = cuts_img[:, :mid].copy()
            cut_y_img = cuts_img[:, mid:].copy()
        except Exception:
            cut_x_img = cuts_img
            cut_y_img = cuts_img
        if cut_x_img.size == 0:
            cut_x_img = cuts_img
        if cut_y_img.size == 0:
            cut_y_img = cuts_img
        cut_x_path = _save_png_image(cut_x_img, point_dir / "cut_x.png") or ""
        cut_y_path = _save_png_image(cut_y_img, point_dir / "cut_y.png") or ""

    roi_gray = _get_roi_gray_image(roi_id, entry=entry)
    raw_roi = _get_roi_raw_gray_image(roi_id)
    raw_source = raw_roi if raw_roi is not None else roi_gray
    raw_path = None
    if raw_source is not None:
        z_unit = str(config.get("position_unit", "mm") or "").strip()
        sign_prefix = "neg_" if z_value < 0 else ""
        magnitude = abs(z_value)
        z_numeric = f"{magnitude:.6g}"
        suffix_parts = [sign_prefix + z_numeric if sign_prefix else z_numeric]
        if z_unit:
            suffix_parts.append(z_unit)
        z_suffix = "_".join(part for part in suffix_parts if part)
        z_slug = _SAFE_FILENAME_RE.sub('_', z_suffix).strip('_') or "z"
        raw_filename = f"raw_{z_slug}.bmp"
        raw_path = _save_raw_roi_bmp(raw_source, point_dir / raw_filename)
    pixel_size_m = _extract_pixel_size_m(entry)

    timestamp_iso = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    point = CausticPoint(
        point_id=point_id,
        roi_id=str(roi_id),
        timestamp_iso=timestamp_iso,
        z_m=z_m,
        position_unit_at_capture=str(config.get("position_unit", "mm")),
        wavelength_nm=float(config.get("wavelength_nm", 1030.0)),
        pixel_size_m=pixel_size_m,
        wx_1e2=wx_1e2,
        wy_1e2=wy_1e2,
        wx_2sigma=wx_2sigma,
        wy_2sigma=wy_2sigma,
        profile_img_path=profile_path,
        cut_x_img_path=cut_x_path,
        cut_y_img_path=cut_y_path,
        raw_roi_img_path=raw_path,
    )
    caustic_manager.add_point(point)

    state = _caustic_state_payload()
    state["last_added_point_id"] = point_id
    return jsonify(state)


@app.route("/caustic/<point_id>", methods=["DELETE"])
def caustic_remove_point(point_id: str):
    if not point_id:
        return jsonify({"deleted": False}), 400
    removed = caustic_manager.remove_point(point_id)
    state = _caustic_state_payload()
    state["deleted"] = bool(removed)
    return jsonify(state)


@app.route("/caustic/fit", methods=["POST"])
def caustic_fit():
    try:
        caustic_manager.compute_m2_fit()
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(_caustic_state_payload())


@app.route("/caustic/save", methods=["POST"])
def caustic_save():
    base_param = (request.args.get("base", "") or "").strip()
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    fallback = f"caustic_{timestamp}"
    base = _sanitize_filename(base_param, fallback)

    target_dir = EXPORTS_DIR / base
    result = caustic_manager.export_dataset(target_dir, clean=True)

    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        export_root = Path(result["path"])
        base_folder = base
        for file_path in export_root.rglob("*"):
            if file_path.is_file():
                arcname = f"{base_folder}/{file_path.relative_to(export_root)}"
                zf.write(file_path, arcname)
    mem.seek(0)

    download_name = f"{base}.zip"
    response = send_file(mem, mimetype="application/zip", as_attachment=True, download_name=download_name)
    response.headers["X-Download-Filename"] = download_name
    return response


@app.route("/caustic/image/<point_id>/<kind>")
def caustic_image(point_id: str, kind: str):
    point = caustic_manager.get_point(point_id)
    if not point:
        return jsonify({"error": "Not found"}), 404
    path_map = {
        "profile": point.profile_img_path,
        "cut_x": point.cut_x_img_path,
        "cut_y": point.cut_y_img_path,
        "raw": point.raw_roi_img_path,
    }
    path_str = path_map.get(kind)
    if not path_str:
        return jsonify({"error": "Image unavailable"}), 404
    fs_path = Path(path_str)
    if not fs_path.exists():
        return jsonify({"error": "Image unavailable"}), 404
    suffix = fs_path.suffix.lower()
    mimetype = "image/bmp" if suffix == ".bmp" else "image/png"
    return send_file(fs_path, mimetype=mimetype)


# ROI mini-streams (~10 Hz)
def _roi_stream_frames(rid: str) -> Generator[bytes, None, None]:
    boundary = b"--frame\r\n"
    while True:
        r = roi_registry.get(rid)
        gray = cam_service.get_latest_gray()
        if r is None or gray is None or gray.size == 0:
            img = _placeholder("ROI or frame unavailable", (256, 256))
            jpg = _encode_jpeg(img)
            if jpg:
                yield boundary + b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            time.sleep(0.1)
            continue
        h_img, w_img = gray.shape[:2]
        x0 = max(0, min(int(r.x), w_img - 1))
        y0 = max(0, min(int(r.y), h_img - 1))
        x1 = max(x0 + 1, min(int(r.x + r.w), w_img))
        y1 = max(y0 + 1, min(int(r.y + r.h), h_img))
        roi_gray = gray[y0:y1, x0:x1]
        if roi_gray.size == 0:
            img = _placeholder("ROI out of bounds", (256, 256))
        else:
            peak = _update_roi_peak_intensity(rid, roi_gray)
            roi_u8 = _normalize_to_u8(roi_gray, peak)
            cm = cam_service.get_colormap()
            img = apply_colormap_to_gray(roi_u8, cm)
        jpg = _encode_jpeg(img)
        if jpg:
            yield boundary + b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
        time.sleep(0.1)

@app.route("/roi_feed/<rid>")
def roi_feed(rid: str):
    return mjpeg_response(_roi_stream_frames(rid))

# ----- Per-ROI Plot Feeds -----

def _roi_profile_frames(rid: str) -> Generator[bytes, None, None]:
    boundary = b"--frame\r\n"
    while True:
        img = _compose_roi_profile_image(rid)
        if img is None:
            img = _placeholder("ROI unavailable", (256, 256))
        jpg = _encode_jpeg(img)
        if jpg:
            yield boundary + b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
        time.sleep(0.1)

@app.route("/roi_profile_feed/<rid>")
def roi_profile_feed(rid: str):
    return mjpeg_response(_roi_profile_frames(rid))


@app.route("/roi_profile_image/<rid>")
def roi_profile_image(rid: str):
    img = _compose_roi_profile_image(rid)
    if img is None:
        img = _placeholder("ROI unavailable", (256, 256))
    png = _encode_png(img)
    if png:
        return Response(png, mimetype="image/png")
    return Response(status=500)

def _roi_cuts_frames(rid: str) -> Generator[bytes, None, None]:
    boundary = b"--frame\r\n"
    while True:
        img = _compose_roi_cuts_image(rid)
        if img is None:
            img = _placeholder("Cuts unavailable", (320, 200))
        jpg = _encode_jpeg(img)
        if jpg:
            yield boundary + b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
        time.sleep(0.2)

@app.route("/roi_cuts_feed/<rid>")
def roi_cuts_feed(rid: str):
    return mjpeg_response(_roi_cuts_frames(rid))


@app.route("/roi_cuts_image/<rid>")
def roi_cuts_image(rid: str):
    img = _compose_roi_cuts_image(rid)
    if img is None:
        img = _placeholder("Cuts unavailable", (320, 200))
    png = _encode_png(img)
    if png:
        return Response(png, mimetype="image/png")
    return Response(status=500)


@app.route("/roi_profile_save/<rid>")
def roi_profile_save(rid: str):
    base_param = (request.args.get("base", "") or "").strip()
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    fallback = f"{rid}_profile_{timestamp}"
    base = _sanitize_filename(base_param, fallback)

    entry = beam_manager.get(rid)
    roi_gray = _get_roi_gray_image(rid, entry=entry)
    if roi_gray is None or roi_gray.size == 0:
        return jsonify({"error": "ROI data unavailable"}), 404

    peak = _update_roi_peak_intensity(rid, roi_gray)
    roi_u8 = _normalize_to_u8(roi_gray, peak)
    bmp_bytes = _encode_bmp(roi_u8)
    if not bmp_bytes:
        return jsonify({"error": "Failed to encode BMP"}), 500

    overlay = _compose_roi_profile_image(rid)
    if overlay is None:
        cm = cam_service.get_colormap()
        overlay = apply_colormap_to_gray(roi_u8, cm)
    png_bytes = _encode_png(overlay)
    if not png_bytes:
        return jsonify({"error": "Failed to encode PNG"}), 500

    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{base}_gray.bmp", bmp_bytes)
        zf.writestr(f"{base}_overlay.png", png_bytes)
    mem.seek(0)
    return send_file(mem, mimetype="application/zip", as_attachment=True, download_name=f"{base}.zip")


def _build_roi_cuts_report(rid: str) -> Tuple[Optional[Dict[str, object]], Optional[str], int]:
    entry = beam_manager.get(rid)
    if not entry:
        return None, "ROI data unavailable", 404
    result = entry.get("result") if entry else None
    if not isinstance(result, dict):
        return None, "No analysis result for ROI", 404

    spectrum_x = result.get("Ix_spectrum")
    spectrum_y = result.get("Iy_spectrum")
    if not spectrum_x or not spectrum_y:
        return None, "Spectra unavailable", 404

    x_positions, Ix_vals = spectrum_x
    y_positions, Iy_vals = spectrum_y
    x_positions = np.asarray(x_positions, dtype=np.float64)
    y_positions = np.asarray(y_positions, dtype=np.float64)
    Ix_vals = np.asarray(Ix_vals, dtype=np.float64)
    Iy_vals = np.asarray(Iy_vals, dtype=np.float64)
    if x_positions.size == 0 or y_positions.size == 0:
        return None, "Empty spectra", 404

    cuts_img = _compose_roi_cuts_image(rid)
    if cuts_img is None:
        return None, "Cuts image unavailable", 404
    png_bytes = _encode_png(cuts_img)
    if not png_bytes:
        return None, "Failed to encode cuts image", 500

    total_x, mean_x, var_x, std_x = _spectrum_stats(x_positions, Ix_vals)
    total_y, mean_y, var_y, std_y = _spectrum_stats(y_positions, Iy_vals)

    pixel_size_val = entry.get("pixel_size") if entry else None
    try:
        pixel_size = float(pixel_size_val) if pixel_size_val is not None else None
    except (TypeError, ValueError):
        pixel_size = None

    theta = float(result.get("theta", 0.0))
    theta_deg = float(np.degrees(theta))
    rx_iso = float(result.get("rx_iso", 0.0))
    ry_iso = float(result.get("ry_iso", 0.0))
    cx_iso = float(result.get("cx_iso", result.get("cx", 0.0)))
    cy_iso = float(result.get("cy_iso", result.get("cy", 0.0)))
    cx = float(result.get("cx", 0.0))
    cy = float(result.get("cy", 0.0))

    fit_x = result.get("gauss_fit_x") or {}
    fit_y = result.get("gauss_fit_y") or {}

    gauss_amp_x = float(fit_x.get("amplitude", 0.0)) if fit_x else 0.0
    gauss_amp_y = float(fit_y.get("amplitude", 0.0)) if fit_y else 0.0
    gauss_center_x = float(fit_x.get("centre", mean_x)) if fit_x else mean_x
    gauss_center_y = float(fit_y.get("centre", mean_y)) if fit_y else mean_y
    gauss_rad_x = float(fit_x.get("radius", 0.0)) if fit_x else 0.0
    gauss_rad_y = float(fit_y.get("radius", 0.0)) if fit_y else 0.0

    csv_buf = io.StringIO()
    writer = csv.writer(csv_buf)
    writer.writerow(["axis", "position_px", "value"])
    for pos, val in zip(x_positions, Ix_vals):
        writer.writerow(["x", f"{pos:.6g}", f"{val:.6g}"])
    for pos, val in zip(y_positions, Iy_vals):
        writer.writerow(["y", f"{pos:.6g}", f"{val:.6g}"])
    spectra_bytes = csv_buf.getvalue().encode("utf-8")

    analysis_lines = [
        f"ROI: {rid}",
        f"Generated: {datetime.utcnow().isoformat(timespec='seconds')}Z",
        "",
        "Second moment summary:",
        f"  Ix total: {total_x:.6g}",
        f"  Ix mean: {mean_x:.6g} px{_format_physical(mean_x, pixel_size)}",
        f"  Ix variance: {var_x:.6g} px^2",
        f"  Ix stddev: {std_x:.6g} px{_format_physical(std_x, pixel_size)}",
        f"  Iy total: {total_y:.6g}",
        f"  Iy mean: {mean_y:.6g} px{_format_physical(mean_y, pixel_size)}",
        f"  Iy variance: {var_y:.6g} px^2",
        f"  Iy stddev: {std_y:.6g} px{_format_physical(std_y, pixel_size)}",
        "",
        "Iso ellipse (second moment):",
        f"  Centre ISO: ({cx_iso:.6g}, {cy_iso:.6g}) px",
        f"  Radii ISO: rx={rx_iso:.6g} px{_format_physical(rx_iso, pixel_size)}, ry={ry_iso:.6g} px{_format_physical(ry_iso, pixel_size)}",
        f"  Theta: {theta_deg:.3f} deg",
        "",
        "Centroid (raw):",
        f"  ({cx:.6g}, {cy:.6g}) px",
        "",
        "Gaussian fits:",
        f"  Ix amplitude: {gauss_amp_x:.6g}",
        f"  Ix centre: {gauss_center_x:.6g} px{_format_physical(gauss_center_x, pixel_size)}",
        f"  Ix radius: {gauss_rad_x:.6g} px{_format_physical(gauss_rad_x, pixel_size)}",
        f"  Iy amplitude: {gauss_amp_y:.6g}",
        f"  Iy centre: {gauss_center_y:.6g} px{_format_physical(gauss_center_y, pixel_size)}",
        f"  Iy radius: {gauss_rad_y:.6g} px{_format_physical(gauss_rad_y, pixel_size)}",
    ]
    iterations = result.get("iterations")
    if iterations is not None:
        analysis_lines.append("")
        analysis_lines.append(f"Iterations: {int(iterations)}")

    analysis_text = "\n".join(analysis_lines) + "\n"

    tsv_lines = [f"# {line}" if line else "#" for line in analysis_lines]
    tsv_lines.append("")
    tsv_lines.append("axis\tposition_px\tvalue")
    for pos, val in zip(x_positions, Ix_vals):
        tsv_lines.append(f"x\t{pos:.9g}\t{val:.9g}")
    for pos, val in zip(y_positions, Iy_vals):
        tsv_lines.append(f"y\t{pos:.9g}\t{val:.9g}")
    ascii_text = "\n".join(tsv_lines) + "\n"

    report = {
        "entry": entry,
        "result": result,
        "cuts_image": cuts_img,
        "png_bytes": png_bytes,
        "spectra_csv": spectra_bytes,
        "analysis_text": analysis_text.encode("utf-8"),
        "spectra_tsv": ascii_text.encode("utf-8"),
        "analysis_lines": analysis_lines,
    }
    return report, None, 200


def _format_physical(value: float, pixel_size: Optional[float]) -> str:
    if pixel_size is None:
        return ""
    return f" ({value * pixel_size:.6g} units)"


def _spectrum_stats(positions: np.ndarray, values: np.ndarray) -> Tuple[float, float, float, float]:
    pos = np.asarray(positions, dtype=np.float64).reshape(-1)
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    total = float(vals.sum())
    if total <= 0.0 or pos.size == 0:
        mean = float(pos.mean()) if pos.size else 0.0
        variance = 0.0
    else:
        mean = float((pos * vals).sum() / total)
        variance = float(((pos - mean) ** 2 * vals).sum() / max(total, 1e-12))
    stddev = float(np.sqrt(max(variance, 0.0)))
    return total, mean, variance, stddev


@app.route("/roi_cuts_save/<rid>")
def roi_cuts_save(rid: str):
    base_param = (request.args.get("base", "") or "").strip()
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    fallback = f"{rid}_cuts_{timestamp}"
    base = _sanitize_filename(base_param, fallback)

    report, error, status = _build_roi_cuts_report(rid)
    if report is None:
        return jsonify({"error": error}), status

    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{base}_cuts.png", report["png_bytes"])
        zf.writestr(f"{base}_spectra.csv", report["spectra_csv"])
        zf.writestr(f"{base}_analysis.txt", report["analysis_text"])
        zf.writestr(f"{base}_spectra.tsv", report["spectra_tsv"])
    mem.seek(0)
    return send_file(mem, mimetype="application/zip", as_attachment=True, download_name=f"{base}.zip")

# Save bundle (JSON + PNG inside a ZIP)
@app.route("/save_bundle")
def save_bundle():
    base_param = (request.args.get("base", "") or "").strip()
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    fallback = f"snapshot_{timestamp}"
    base = _sanitize_filename(base_param, fallback)

    raw_gray = cam_service.get_latest_gray(raw=True)
    processed_gray = cam_service.get_latest_gray()
    background_frame = cam_service.get_background_frame()
    background_enabled = cam_service.is_background_subtraction_enabled()

    colormap = cam_service.get_colormap()

    live_bgr = cam_service.get_latest_bgr()
    if live_bgr is None:
        live_png = _encode_png(_placeholder("No frame", (640, 480)))
        live_colored = None
    else:
        live_colored = apply_colormap_to_bgr(live_bgr, colormap)
        live_png = _encode_png(live_colored)

    raw_gray_bmp = _encode_gray_bmp(raw_gray)
    background_bmp = _encode_gray_bmp(background_frame)

    frame_size = cam_service.get_frame_size()
    exposure_us = cam_service.get_exposure_us()
    camera_id = cam_service.get_camera_id()
    cam_running = cam_service.is_running()

    metrics_snapshot = metrics.get_snapshot()
    beam_options = _get_beam_options_copy()
    rois = roi_registry.list()
    rois_dicts = [roi.to_dict() for roi in rois]

    snapshot_iso = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    settings_payload: Dict[str, object] = {
        "snapshot": {
            "name": base,
            "timestamp_iso": snapshot_iso,
        },
        "camera": {
            "id": camera_id,
            "running": cam_running,
            "frame_size": {"width": frame_size[0], "height": frame_size[1]} if frame_size else None,
            "exposure_us": int(exposure_us),
            "colormap": colormap,
        },
        "background_subtraction": {
            "enabled": background_enabled,
            "frame_present": background_frame is not None,
        },
        "beam_analysis": beam_options,
        "metrics": metrics_snapshot,
        "rois": rois_dicts,
    }

    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Settings and configuration
        settings_bytes = json.dumps(settings_payload, indent=2, default=_json_default).encode("utf-8")
        zf.writestr("settings/settings.json", settings_bytes)

        # Camera-level imagery
        if raw_gray_bmp:
            zf.writestr("images/raw/latest_raw_gray.bmp", raw_gray_bmp)
        if raw_gray is not None:
            _zip_write_npy(zf, "images/raw/latest_raw_gray.npy", np.asarray(raw_gray))

        if background_bmp:
            zf.writestr("images/raw/averaged_background.bmp", background_bmp)
        if background_frame is not None:
            _zip_write_npy(zf, "images/raw/averaged_background.npy", np.asarray(background_frame))

        if live_png:
            zf.writestr("images/live_view.png", live_png)
        if live_colored is not None:
            _zip_write_npy(zf, "images/live_view_colored.npy", np.asarray(live_colored))

        # ROI assets
        for roi in rois:
            rid = roi.id
            roi_entry = beam_manager.get(rid)
            safe_rid = _safe_roi_fs_name(rid)
            roi_dir = f"rois/{safe_rid}"

            roi_meta = roi.to_dict()
            roi_meta.update({
                "frame_timestamp": roi_entry.get("frame_ts") if roi_entry else None,
                "analysis_available": bool(roi_entry and roi_entry.get("result")),
                "pixel_size": roi_entry.get("pixel_size") if roi_entry else None,
            })
            zf.writestr(f"{roi_dir}/{safe_rid}_meta.json", json.dumps(roi_meta, indent=2, default=_json_default).encode("utf-8"))

            roi_gray = _get_roi_gray_image(rid, entry=roi_entry)
            if roi_gray is not None and getattr(roi_gray, "size", 0) > 0:
                roi_gray_bmp = _encode_gray_bmp(roi_gray)
                if roi_gray_bmp:
                    zf.writestr(f"{roi_dir}/{safe_rid}_gray.bmp", roi_gray_bmp)
                _zip_write_npy(zf, f"{roi_dir}/{safe_rid}_gray.npy", np.asarray(roi_gray))

            overlay = _compose_roi_profile_image(rid)
            if overlay is None and roi_gray is not None and getattr(roi_gray, "size", 0) > 0:
                arr = np.asarray(roi_gray, dtype=np.float32)
                peak = float(np.max(arr)) if arr.size else 1.0
                if peak <= 0.0:
                    peak = 1.0
                roi_u8 = _normalize_to_u8(arr, peak)
                overlay = apply_colormap_to_gray(roi_u8, colormap)
            if overlay is not None:
                overlay_png = _encode_png(overlay)
                if overlay_png:
                    zf.writestr(f"{roi_dir}/{safe_rid}_profile.png", overlay_png)

            report, _, _ = _build_roi_cuts_report(rid)
            if report:
                zf.writestr(f"{roi_dir}/{safe_rid}_cuts.png", report["png_bytes"])
                zf.writestr(f"{roi_dir}/{safe_rid}_cuts.tsv", report["spectra_tsv"])
                zf.writestr(f"{roi_dir}/{safe_rid}_cuts.csv", report["spectra_csv"])
                zf.writestr(f"{roi_dir}/{safe_rid}_cuts_analysis.txt", report["analysis_text"])

        with tempfile.TemporaryDirectory() as tmp_dir:
            export_dir = Path(tmp_dir) / "caustic"
            caustic_manager.export_dataset(export_dir, clean=True)
            for file_path in export_dir.rglob("*"):
                if file_path.is_file():
                    arcname = f"caustic/{file_path.relative_to(export_dir)}"
                    zf.write(file_path, arcname)

    mem.seek(0)
    return send_file(mem, mimetype="application/zip", as_attachment=True, download_name=f"{base}.zip")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask-based camera streamer.")
    parser.add_argument("--dev", action="store_true", help="Enable development mode (logging to file).")
    parser.add_argument("--camera_id", type=str, help="Default camera ID to use.")
    args = parser.parse_args()

    port = 5000

    # Setup logging first. cam_service was already instantiated with app.logger,
    # so it will pick up this configuration.
    setup_logging(dev_mode=args.dev)

    # Set the default camera ID from command-line args, if provided.
    if args.camera_id:
        cam_service.set_default_camera_id(args.camera_id)

    app.logger.info(f"Starting server on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)

