import io
import json
import logging
import threading
import time
import zipfile
import argparse
import atexit
from datetime import datetime
from typing import Dict, Generator, Optional, Tuple, List
from concurrent.futures import ProcessPoolExecutor

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, Response, jsonify, render_template, request, send_file
from beam_analysis import analyze_beam

from camera_service import CameraService, apply_colormap_to_gray, apply_colormap_to_bgr
from metrics import MetricsComputer
from roi import ROIRegistry

# ----- App setup -----
app = Flask(__name__, template_folder="templates", static_folder="static", static_url_path="/static")

# ----- Logging Setup -----
def setup_logging(dev_mode=False):
    log_level = logging.DEBUG if dev_mode else logging.INFO
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
    app.logger.setLevel(log_level)
    app.logger.info("Application starting up...")
    if dev_mode:
        app.logger.info("Development mode enabled.")

# ----- Global services -----
# Instantiate services globally. The logger will be configured later in the main block.
cam_service = CameraService(logger=app.logger)
roi_registry = ROIRegistry()
metrics = MetricsComputer()
_plot_lock = threading.Lock()

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
        self._pool = ProcessPoolExecutor()
        atexit.register(self._shutdown_pool)

    def _shutdown_pool(self) -> None:
        try:
            self._pool.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass

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
                if compute != "none":
                    images.append(crop_copy.astype(np.float64))
                    roi_ids.append(rid)

        analysis_kwargs = {
            "clip_negatives": clip_mode,
            "angle_clip_mode": angle_clip_param,
            "background_subtraction": bg_sub,
            "rotation_angle": rot_angle,
            "compute_gaussian": compute_gauss,
        }

        if compute != "none" and images:
            payloads = [(img, dict(analysis_kwargs)) for img in images]
            try:
                results = list(self._pool.map(_beam_worker, payloads))
            except Exception as exc:
                app.logger.warning(f"beam_analysis pool failure, falling back sequential: {exc}")
                results = []
                for payload in payloads:
                    results.append(_beam_worker(payload))

            for rid, res in zip(roi_ids, results):
                if rid in entries:
                    entries[rid]["result"] = res

        with self._lock:
            self._data = entries
            self._frame_ts = frame_ts
            self._compute_mode = compute
            self._pixel_size = pixel_size_val

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


def _beam_worker(payload: Tuple[np.ndarray, Dict[str, object]]) -> Optional[Dict[str, object]]:
    img, kwargs = payload
    try:
        return analyze_beam(img, **kwargs)
    except Exception as exc:
        app.logger.warning(f"beam_analysis worker error: {exc}")
        return None

# Background metrics loop
def _metrics_loop():
    last_size: Optional[Tuple[int, int]] = None
    while True:
        try:
            cam_service.wait_for_frame_signal(timeout=0.5)
            gray = cam_service.get_latest_gray()
            exp = cam_service.get_exposure_us()
            size = cam_service.get_frame_size()
            if size != last_size:
                app.logger.info(f"Frame size changed to {size}. Clamping ROIs.")
                roi_registry.clamp_all(size)
                last_size = size
            rois = roi_registry.list()
            metrics.update(gray, exp, rois)
            opts = _get_beam_options_copy()
            beam_manager.update(gray, rois, opts)
        except Exception as e:
            app.logger.error(f"Exception in metrics loop: {e}", exc_info=True)
            time.sleep(0.1)

_metrics_thread = threading.Thread(target=_metrics_loop, daemon=True)
_metrics_thread.start()

# ----- Helpers -----
def mjpeg_response(gen: Generator[bytes, None, None]) -> Response:
    return Response(gen, mimetype="multipart/x-mixed-replace; boundary=frame")

def _encode_jpeg(img: np.ndarray) -> Optional[bytes]:
    ok, buf = cv2.imencode(".jpg", img)
    return buf.tobytes() if ok else None

def _encode_png(img: np.ndarray) -> Optional[bytes]:
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes() if ok else None

def _placeholder(text: str, size=(640, 360)) -> np.ndarray:
    w, h = size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(img, text, (20, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return img

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
    return jsonify({"ok": True})

# Metrics polling
@app.route("/metrics")
def metrics_route():
    return jsonify(metrics.get_snapshot())


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
            lo, hi = np.percentile(roi_gray, (1, 99))
            if hi <= lo:
                hi = lo + 1.0
            norm = np.clip((roi_gray.astype(np.float32) - float(lo)) / float(hi - lo), 0.0, 1.0)
            roi_u8 = (norm * 255.0).astype(np.uint8)
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
        r = roi_registry.get(rid)
        entry = beam_manager.get(rid)
        compute_mode = str(entry.get("compute") if entry else "none").lower()
        result = entry.get("result") if entry else None
        roi_gray = entry.get("roi_gray") if entry else None
        if r is None:
            img = _placeholder("ROI unavailable", (256, 256))
        else:
            if roi_gray is None or getattr(roi_gray, "size", 0) == 0:
                gray = cam_service.get_latest_gray()
                if gray is not None and gray.size > 0:
                    h_img, w_img = gray.shape[:2]
                    x0 = max(0, min(int(r.x), w_img - 1))
                    y0 = max(0, min(int(r.y), h_img - 1))
                    x1 = max(x0 + 1, min(int(r.x + r.w), w_img))
                    y1 = max(y0 + 1, min(int(r.y + r.h), h_img))
                    roi_crop = gray[y0:y1, x0:x1]
                    roi_gray = roi_crop.copy() if roi_crop.size > 0 else None
            if roi_gray is None or getattr(roi_gray, "size", 0) == 0:
                img = _placeholder("ROI out of bounds", (256, 256))
            else:
                compute = compute_mode or "none"
                if compute == "none" or result is None:
                    lo, hi = np.percentile(roi_gray, (1, 99))
                    if hi <= lo:
                        hi = lo + 1.0
                    norm = np.clip((roi_gray.astype(np.float32) - float(lo)) / float(hi - lo), 0.0, 1.0)
                    roi_u8 = (norm * 255.0).astype(np.uint8)
                    cm = cam_service.get_colormap()
                    img = apply_colormap_to_gray(roi_u8, cm)
                else:
                    try:
                        proc = result.get("img_for_spec")
                        if proc is None or not isinstance(proc, np.ndarray) or proc.size == 0:
                            proc = roi_gray.astype(np.float32)
                        proc = np.asarray(proc, dtype=np.float32)
                        if proc.ndim == 3:
                            proc = np.mean(proc, axis=-1)
                        proc = np.nan_to_num(proc, nan=0.0, posinf=0.0, neginf=0.0)

                        cy = float(result.get("cy", 0.0))
                        cx = float(result.get("cx", 0.0))
                        rx_iso = float(result.get("rx_iso", 0.0))
                        ry_iso = float(result.get("ry_iso", 0.0))
                        theta = float(result.get("theta", 0.0))
                        fit_x = result.get("gauss_fit_x")
                        fit_y = result.get("gauss_fit_y")
                        gauss_rx = (fit_x or {}).get("radius")
                        gauss_ry = (fit_y or {}).get("radius")

                        lo = float(np.percentile(proc, 1))
                        hi = float(np.percentile(proc, 99))
                        if not np.isfinite(lo):
                            lo = float(np.min(proc)) if proc.size else 0.0
                        if not np.isfinite(hi):
                            hi = float(np.max(proc)) if proc.size else 1.0
                        if hi <= lo:
                            hi = lo + 1.0
                        norm = np.clip((proc - lo) / (hi - lo), 0.0, 1.0)
                        roi_u8 = (norm * 255.0).astype(np.uint8)
                        cm = cam_service.get_colormap()
                        img = apply_colormap_to_gray(roi_u8, cm)

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

                        if compute in ("both", "second"):
                            ex = cx + major_r * np.cos(t) * c - minor_r * np.sin(t) * s
                            ey = cy + major_r * np.cos(t) * s + minor_r * np.sin(t) * c
                            ellipse_pts = np.stack([ex, ey], axis=1)
                            ellipse_pts = np.nan_to_num(ellipse_pts, nan=0.0)
                            ellipse_pts[:, 0] = np.clip(ellipse_pts[:, 0], 0, w_img - 1)
                            ellipse_pts[:, 1] = np.clip(ellipse_pts[:, 1], 0, h_img - 1)
                            ellipse_pts = np.round(ellipse_pts).astype(np.int32).reshape(-1, 1, 2)
                            cv2.polylines(img, [ellipse_pts], True, (255, 255, 255), 2, cv2.LINE_AA)

                            major_pt1 = (int(np.clip(round(cx - major_r * c), 0, w_img - 1)), int(np.clip(round(cy - major_r * s), 0, h_img - 1)))
                            major_pt2 = (int(np.clip(round(cx + major_r * c), 0, w_img - 1)), int(np.clip(round(cy + major_r * s), 0, h_img - 1)))
                            minor_pt1 = (int(np.clip(round(cx + minor_r * s), 0, w_img - 1)), int(np.clip(round(cy - minor_r * c), 0, h_img - 1)))
                            minor_pt2 = (int(np.clip(round(cx - minor_r * s), 0, w_img - 1)), int(np.clip(round(cy + minor_r * c), 0, h_img - 1)))
                            cv2.line(img, major_pt1, major_pt2, (255, 255, 0), 2, cv2.LINE_AA)
                            cv2.line(img, minor_pt1, minor_pt2, (255, 0, 255), 2, cv2.LINE_AA)

                        if compute in ("both", "gauss") and g_major is not None and g_minor is not None:
                            gx = cx + g_major * np.cos(t) * c - g_minor * np.sin(t) * s
                            gy = cy + g_major * np.cos(t) * s + g_minor * np.sin(t) * c
                            g_pts = np.stack([gx, gy], axis=1)
                            g_pts = np.nan_to_num(g_pts, nan=0.0)
                            g_pts[:, 0] = np.clip(g_pts[:, 0], 0, w_img - 1)
                            g_pts[:, 1] = np.clip(g_pts[:, 1], 0, h_img - 1)
                            g_pts = np.round(g_pts).astype(np.int32).reshape(-1, 1, 2)
                            cv2.polylines(img, [g_pts], True, (0, 255, 255), 2, cv2.LINE_AA)

                        target_h = 240.0
                        scale = target_h / max(1.0, float(h_img))
                        scaled_w = max(1, int(round(w_img * scale)))
                        scaled_h = int(round(target_h))
                        img = cv2.resize(img, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
                    except Exception as e:
                        app.logger.warning(f"beam profile plot error for ROI {rid}: {e}", exc_info=True)
                        img = _placeholder("Analysis plot error", (256, 256))
        jpg = _encode_jpeg(img)
        if jpg:
            yield boundary + b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
        time.sleep(0.1)

@app.route("/roi_profile_feed/<rid>")
def roi_profile_feed(rid: str):
    return mjpeg_response(_roi_profile_frames(rid))

def _roi_cuts_frames(rid: str) -> Generator[bytes, None, None]:
    boundary = b"--frame\r\n"
    while True:
        r = roi_registry.get(rid)
        entry = beam_manager.get(rid)
        compute_mode = str(entry.get("compute") if entry else "none").lower()
        result = entry.get("result") if entry else None
        pixel_size_val = entry.get("pixel_size") if entry else None
        try:
            pixel_size_val = float(pixel_size_val)
        except (TypeError, ValueError):
            pixel_size_val = None

        if r is None:
            img = _placeholder("ROI unavailable", (320, 200))
        else:
            if compute_mode == "none":
                img = _placeholder("Analysis disabled", (320, 200))
            elif result is None:
                img = _placeholder("Analysis pending", (320, 200))
            else:
                try:
                    x_positions, Ix = result["Ix_spectrum"]
                    y_positions, Iy = result["Iy_spectrum"]
                    fit_x = result.get("gauss_fit_x") or {}
                    fit_y = result.get("gauss_fit_y") or {}
                    cx_iso = float(result.get("cx", 0.0))
                    cy_iso = float(result.get("cy", 0.0))
                    rx_iso = float(result.get("rx_iso", 0.0))
                    ry_iso = float(result.get("ry_iso", 0.0))
                    gauss_cx = float(fit_x.get("centre", cx_iso))
                    gauss_cy = float(fit_y.get("centre", cy_iso))
                    gauss_rx = float(fit_x.get("radius", 0.0)) if fit_x else 0.0
                    gauss_ry = float(fit_y.get("radius", 0.0)) if fit_y else 0.0

                    with _plot_lock:
                        fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.2), dpi=100)
                        axx, axy = axes

                        axx.plot(x_positions, Ix, color="#4a90e2", label="Ix")
                        if compute_mode in ("both", "gauss") and fit_x:
                            A = fit_x["amplitude"]
                            xc = gauss_cx
                            rx = gauss_rx
                            xx = np.linspace(x_positions.min(), x_positions.max(), 400)
                            yy = A * np.exp(-2.0 * ((xx - xc) / max(rx, 1e-6)) ** 2)
                            axx.plot(xx, yy, color="orange", linestyle="--", label="Gauss")
                       
                        if rx_iso > 0.0:
                            axx.axvline(cx_iso - rx_iso, color="#00ffaa", linestyle=":", linewidth=1.2, label="2nd moment radius")
                            axx.axvline(cx_iso + rx_iso, color="#00ffaa", linestyle=":", linewidth=1.2, label=None)
                        if gauss_rx > 0.0:
                            axx.axvline(gauss_cx - gauss_rx, color="#ffaa00", linestyle="-.", linewidth=1.2, label="Gauss radius")
                            axx.axvline(gauss_cx + gauss_rx, color="#ffaa00", linestyle="-.", linewidth=1.2, label=None)
                        ann_lines = []
                        if rx_iso > 0.0:
                            if pixel_size_val:
                                msg = f"ISO: {rx_iso * pixel_size_val:.3f}"
                            else:
                                msg = f"ISO: {rx_iso:.2f} px"
                            ann_lines.append(msg)
                        if gauss_rx > 0.0:
                            if pixel_size_val:
                                msg = f"Gauss: {gauss_rx * pixel_size_val:.3f}"
                            else:
                                msg = f"Gauss: {gauss_rx:.2f} px"
                            ann_lines.append(msg)
                        if ann_lines:
                            axx.text(0.02, 0.95, ' | '.join(ann_lines), transform=axx.transAxes, fontsize=8, va='top', ha='left', bbox=dict(facecolor='black', alpha=0.35, pad=4))
                        handles, labels = axx.get_legend_handles_labels()
                        if handles:
                            unique = {}
                            ordered = []
                            for h, l in zip(handles, labels):
                                if not l or l in unique:
                                    continue
                                unique[l] = h
                                ordered.append((h, l))
                            if ordered:
                                axx.legend([h for h, _ in ordered], [l for _, l in ordered], loc="best", fontsize=8)
                        axx.set_title("Ix cuts", fontsize=10)
                        axx.set_xlabel("x (px)", fontsize=9)
                        axx.set_ylabel("a.u.", fontsize=9)
                        axx.tick_params(labelsize=8)

                        axy.plot(y_positions, Iy, color="#50e3c2", label="Iy")
                        if compute_mode in ("both", "gauss") and fit_y:
                            A = fit_y["amplitude"]
                            yc = gauss_cy
                            ry = gauss_ry
                            yy = np.linspace(y_positions.min(), y_positions.max(), 400)
                            zz = A * np.exp(-2.0 * ((yy - yc) / max(ry, 1e-6)) ** 2)
                            axy.plot(yy, zz, color="orange", linestyle="--", label="Gauss")
                       
                        if ry_iso > 0.0:
                            axy.axvline(cy_iso - ry_iso, color="#00ffaa", linestyle=":", linewidth=1.2, label="2nd moment radius")
                            axy.axvline(cy_iso + ry_iso, color="#00ffaa", linestyle=":", linewidth=1.2, label=None)
                        if gauss_ry > 0.0:
                            axy.axvline(gauss_cy - gauss_ry, color="#ffaa00", linestyle="-.", linewidth=1.2, label="Gauss radius")
                            axy.axvline(gauss_cy + gauss_ry, color="#ffaa00", linestyle="-.", linewidth=1.2, label=None)
                        ann_lines_y = []
                        if ry_iso > 0.0:
                            if pixel_size_val:
                                msg = f"ISO: {ry_iso * pixel_size_val:.3f}"
                            else:
                                msg = f"ISO: {ry_iso:.2f} px"
                            ann_lines_y.append(msg)
                        if gauss_ry > 0.0:
                            if pixel_size_val:
                                msg = f"Gauss: {gauss_ry * pixel_size_val:.3f}"
                            else:
                                msg = f"Gauss: {gauss_ry:.2f} px"
                            ann_lines_y.append(msg)
                        if ann_lines_y:
                            axy.text(0.02, 0.95, ' | '.join(ann_lines_y), transform=axy.transAxes, fontsize=8, va='top', ha='left', bbox=dict(facecolor='black', alpha=0.35, pad=4))
                        handles_y, labels_y = axy.get_legend_handles_labels()
                        if handles_y:
                            unique_y = {}
                            ordered_y = []
                            for h, l in zip(handles_y, labels_y):
                                if not l or l in unique_y:
                                    continue
                                unique_y[l] = h
                                ordered_y.append((h, l))
                            if ordered_y:
                                axy.legend([h for h, _ in ordered_y], [l for _, l in ordered_y], loc="best", fontsize=8)
                        axy.set_title("Iy cuts", fontsize=10)
                        axy.set_xlabel("y (px)", fontsize=9)
                        axy.set_ylabel("a.u.", fontsize=9)
                        axy.tick_params(labelsize=8)

                    fig.tight_layout(pad=0.6)
                    fig.canvas.draw()
                    canvas = fig.canvas
                    w, h = canvas.get_width_height()
                    if hasattr(canvas, "buffer_rgba"):
                        buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
                        rgb = buf[..., :3].copy()
                    else:
                        rgb = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
                    plt.close(fig)
                    base_img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    target_h = 480
                    target_w = target_h * 2
                    img = cv2.resize(base_img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                except Exception as e:
                    app.logger.warning(f"cuts plot error for ROI {rid}: {e}", exc_info=True)
                    img = _placeholder("Cuts plot error", (320, 200))
        jpg = _encode_jpeg(img)
        if jpg:
            yield boundary + b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
        time.sleep(0.2)

@app.route("/roi_cuts_feed/<rid>")
def roi_cuts_feed(rid: str):
    return mjpeg_response(_roi_cuts_frames(rid))

# Save bundle (JSON + PNG inside a ZIP)
@app.route("/save_bundle")
def save_bundle():
    base = request.args.get("base", "").strip()
    if not base:
        return jsonify({"error": "Missing base"}), 400
    bgr = cam_service.get_latest_bgr()
    size = cam_service.get_frame_size()
    exp = cam_service.get_exposure_us()
    cm = cam_service.get_colormap()
    snap = metrics.get_snapshot()
    rois = roi_registry.list_dicts()
    payload: Dict = {
        "timestamp_iso": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "camera_id": cam_service.get_camera_id(),
        "frame_size": {"width": size[0], "height": size[1]} if size else {"width": 0, "height": 0},
        "exposure_us": int(exp),
        "colormap": cm,
        "fps": float(snap.get("fps", 0.0)),
        "rois": rois,
    }
    json_bytes = json.dumps(payload, indent=2).encode("utf-8")
    if bgr is None:
        png_bytes = _encode_png(_placeholder("No frame", (640, 480)))
    else:
        png_bytes = _encode_png(bgr)
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{base}.json", json_bytes)
        if png_bytes:
            zf.writestr(f"{base}.png", png_bytes)
    mem.seek(0)
    return send_file(mem, mimetype="application/zip", as_attachment=True, download_name=f"{base}.zip")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask-based camera streamer.")
    parser.add_argument("--dev", action="store_true", help="Enable development mode (logging to file, port 5001).")
    parser.add_argument("--camera_id", type=str, help="Default camera ID to use.")
    args = parser.parse_args()

    port = 5001 if args.dev else 5000

    # Setup logging first. cam_service was already instantiated with app.logger,
    # so it will pick up this configuration.
    setup_logging(dev_mode=args.dev)

    # Set the default camera ID from command-line args, if provided.
    if args.camera_id:
        cam_service.set_default_camera_id(args.camera_id)

    app.logger.info(f"Starting server on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
