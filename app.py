import io
import json
import logging
import threading
import time
import zipfile
import argparse
from datetime import datetime
from typing import Dict, Generator, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, Response, jsonify, render_template, request, send_file

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

    app.logger.addHandler(handler)
    app.logger.setLevel(log_level)
    app.logger.info("Application starting up...")
    if dev_mode:
        app.logger.info("Development mode enabled.")

# Global services
cam_service = CameraService()
roi_registry = ROIRegistry()
metrics = MetricsComputer()
_plot_lock = threading.Lock()

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
    # Reduce log spam by ignoring frequent requests and logging others at DEBUG level
    if request.path in ['/log/js', '/metrics']:
        return

    app.logger.debug(f"Request: {request.method} {request.path}")
    if request.data and request.path != '/log/js':
        try:
            # Attempt to decode as utf-8, but don't fail if it's binary data
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

# Keep existing MJPEG endpoint
@app.route("/video_feed")
def video_feed():
    return mjpeg_response(cam_service.gen_overview_mjpeg())

# Exposure GET/POST
@app.route("/exposure", methods=["GET", "POST"])
def exposure():
    # ... (omitted for brevity, no changes)
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
    # ... (omitted for brevity, no changes)
    if request.method == "GET":
        return jsonify({"value": cam_service.get_colormap()})
    data = request.get_json(silent=True) or {}
    value = str(data.get("value", "")).lower()
    try:
        applied = cam_service.set_colormap(value)
        return jsonify({"value": applied})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Camera On/Off
@app.route("/camera", methods=["POST"])
def camera_toggle():
    # ... (omitted for brevity, no changes)
    data = request.get_json(silent=True) or {}
    enabled = bool(data.get("enabled", False))
    err = None
    try:
        if enabled:
            app.logger.info("Camera START requested.")
            cam_service.start()
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
        roi_list = roi_registry.list_dicts()
        app.logger.debug(f"GET /rois: Returning {len(roi_list)} ROIs.")
        return jsonify(roi_list)

    # POST create
    data = request.get_json(silent=True) or {}
    app.logger.info(f"POST /rois: Creating ROI with data: {data}")
    try:
        x = int(data.get("x")); y = int(data.get("y")); w = int(data.get("w")); h = int(data.get("h"))
    except Exception:
        app.logger.warning("Invalid ROI data received for creation.")
        return jsonify({"error": "Invalid ROI"}), 400

    roi = roi_registry.create(x, y, w, h, cam_service.get_frame_size())
    app.logger.info(f"ROI created successfully: {roi.to_dict()}")
    return jsonify(roi.to_dict())

@app.route("/rois/<rid>", methods=["GET", "PUT", "DELETE"])
def rois_item(rid: str):
    if request.method == "GET":
        r = roi_registry.get_dict(rid)
        if not r:
            return jsonify({"error": "Not found"}), 404
        return jsonify(r)

    if request.method == "DELETE":
        app.logger.info(f"DELETE /rois/{rid}: Deleting ROI.")
        ok = roi_registry.delete(rid)
        if ok:
            app.logger.info(f"ROI {rid} deleted from registry.")
            metrics.remove_roi_metrics(rid)
            app.logger.info(f"Metrics for ROI {rid} removed.")
        else:
            app.logger.warning(f"Attempted to delete non-existent ROI {rid}.")
        return jsonify({"deleted": bool(ok)})

    # PUT update
    data = request.get_json(silent=True) or {}
    app.logger.info(f"PUT /rois/{rid}: Updating ROI with data: {data}")
    try:
        x = int(data.get("x"))
        y = int(data.get("y"))
        w = int(data.get("w"))
        h = int(data.get("h"))
    except (TypeError, ValueError):
        app.logger.warning(f"Invalid ROI data received for update on ROI {rid}.")
        return jsonify({"error": "Invalid ROI"}), 400

    r = roi_registry.update(rid, x, y, w, h, cam_service.get_frame_size())
    if not r:
        return jsonify({"error": "Not found"}), 404
    return jsonify(r.to_dict())

@app.route("/roi/<rid>/reset_max", methods=["POST"])
def roi_reset_max(rid: str):
    app.logger.info(f"POST /roi/{rid}/reset_max: Resetting max for ROI.")
    metrics.reset_max_integral(rid)
    return jsonify({"ok": True})

# ... (rest of the file is the same, omitted for brevity) ...

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
        gray = cam_service.get_latest_gray()
        if r is None or gray is None or gray.size == 0:
            img = _placeholder("ROI or frame unavailable", (256, 256))
        else:
            snap = metrics.get_snapshot()
            y_max_map = snap.get("y_max_integral", {})
            y_max_integral = y_max_map.get(rid, 0.0)
            exposure_us = snap.get("exposure_us", 1)
            area = max(1, r.w * r.h)
            per_pixel_ylim = (1.1 * y_max_integral) / area
            min_vmax_per_ms = 32.0 / max(1, exposure_us) * 1000.0
            vmax = max(per_pixel_ylim, min_vmax_per_ms)
            h_img, w_img = gray.shape[:2]
            x0 = max(0, min(int(r.x), w_img - 1))
            y0 = max(0, min(int(r.y), h_img - 1))
            x1 = max(x0 + 1, min(int(r.x + r.w), w_img))
            y1 = max(y0 + 1, min(int(r.y + r.h), h_img))
            roi_gray = gray[y0:y1, x0:x1]
            if roi_gray.size == 0:
                img = _placeholder("ROI out of bounds", (256, 256))
            else:
                roi_per_ms = roi_gray.astype(np.float32) / max(1, exposure_us) * 1000.0
                norm = np.clip(roi_per_ms / max(1e-6, vmax), 0, 1)
                roi_u8 = (norm * 255.0).astype(np.uint8)
                cm = cam_service.get_colormap()
                img = apply_colormap_to_gray(roi_u8, cm)
        jpg = _encode_jpeg(img)
        if jpg:
            yield boundary + b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
        time.sleep(0.1)

@app.route("/roi_profile_feed/<rid>")
def roi_profile_feed(rid: str):
    return mjpeg_response(_roi_profile_frames(rid))

# Save bundle (JSON + PNG inside a ZIP)
@app.route("/save_bundle")
def save_bundle():
    # ... (omitted for brevity, no changes)
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
    args = parser.parse_args()

    port = 5001 if args.dev else 5000
    setup_logging(dev_mode=args.dev)

    app.logger.info(f"Starting server on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
