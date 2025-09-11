import io
import json
import threading
import time
import zipfile
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

# Global services
cam_service = CameraService()  # camera_id from env CAMERA_ID or first camera
roi_registry = ROIRegistry()
metrics = MetricsComputer()

# Background metrics loop
def _metrics_loop():
    last_size: Optional[Tuple[int, int]] = None
    while True:
        try:
            # Wait for a frame signal to avoid busy loop
            cam_service.wait_for_frame_signal(timeout=0.5)
            gray = cam_service.get_latest_gray()
            exp = cam_service.get_exposure_us()
            # Clamp ROIs to current frame size if changed
            size = cam_service.get_frame_size()
            if size != last_size:
                roi_registry.clamp_all(size)
                last_size = size
            rois = roi_registry.list()
            metrics.update(gray, exp, rois)
        except Exception:
            # Keep thread alive even if camera/exposure feature is unavailable
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

@app.route("/")
def index():
    return render_template("index.html")

# Keep existing MJPEG endpoint
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

    # POST
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

# Camera On/Off
@app.route("/camera", methods=["POST"])
def camera_toggle():
    data = request.get_json(silent=True) or {}
    enabled = bool(data.get("enabled", False))
    err = None
    try:
        if enabled:
            cam_service.start()
        else:
            cam_service.stop()
    except Exception as e:
        err = str(e)
    # Always return 200 with current state so UI can reflect reality; include error if any
    return jsonify({"enabled": cam_service.is_running(), "error": err})

# ROI CRUD
@app.route("/rois", methods=["GET", "POST"])
def rois():
    if request.method == "GET":
        return jsonify(roi_registry.list_dicts())
    # POST create
    data = request.get_json(silent=True) or {}
    try:
        x = int(data.get("x")); y = int(data.get("y")); w = int(data.get("w")); h = int(data.get("h"))
    except Exception:
        return jsonify({"error": "Invalid ROI"}), 400
    roi = roi_registry.create(x, y, w, h, cam_service.get_frame_size())
    return jsonify(roi.to_dict())

@app.route("/rois/<rid>", methods=["GET", "PUT", "DELETE"])
def rois_item(rid: str):
    if request.method == "GET":
        r = roi_registry.get_dict(rid)
        if not r:
            return jsonify({"error": "Not found"}), 404
        return jsonify(r)
    if request.method == "DELETE":
        ok = roi_registry.delete(rid)
        return jsonify({"deleted": bool(ok)})
    # PUT update
    data = request.get_json(silent=True) or {}
    try:
        x = int(data.get("x")); y = int(data.get("y")); w = int(data.get("w")); h = int(data.get("h"))
    except Exception:
        return jsonify({"error": "Invalid ROI"}), 400
    r = roi_registry.update(rid, x, y, w, h, cam_service.get_frame_size())
    if not r:
        return jsonify({"error": "Not found"}), 404
    return jsonify(r.to_dict())

# Metrics polling
@app.route("/metrics")
def metrics_route():
    return jsonify(metrics.get_snapshot())

# Bar plot MJPEG (~5 Hz)
def _bar_frames() -> Generator[bytes, None, None]:
    boundary = b"--frame\r\n"
    while True:
        snap = metrics.get_snapshot()
        rois = snap.get("rois", [])
        exp = int(snap.get("exposure_us", 0))
        fps = float(snap.get("fps", 0.0))

        if not rois:
            img = _placeholder("Add ROIs to see bar plot", (640, 240))
            jpg = _encode_jpeg(img)
            if jpg:
                yield boundary + b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            time.sleep(0.2)  # ~5 Hz
            continue

        labels = [r["id"] for r in rois]
        values = [float(r["value_per_ms"]) for r in rois]

        fig, ax = plt.subplots(figsize=(6.4, 2.4), dpi=100)
        ax.bar(labels, values, color="#4a90e2")
        ax.set_ylabel("integration / ms")
        ax.set_title(f"Exposure: {exp} µs   FPS: {fps:.1f}")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.canvas.print_png(buf)
        plt.close(fig)
        img = cv2.imdecode(np.frombuffer(buf.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR)
        jpg = _encode_jpeg(img) if img is not None else None
        if jpg:
            yield boundary + b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
        time.sleep(0.2)  # ~5 Hz

@app.route("/bar_feed")
def bar_feed():
    return mjpeg_response(_bar_frames())

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

        # Crop safely
        h_img, w_img = gray.shape[:2]
        x0 = max(0, min(int(r.x), w_img - 1))
        y0 = max(0, min(int(r.y), h_img - 1))
        x1 = max(x0 + 1, min(int(r.x + r.w), w_img))
        y1 = max(y0 + 1, min(int(r.y + r.h), h_img))
        roi_gray = gray[y0:y1, x0:x1]
        if roi_gray.size == 0:
            img = _placeholder("ROI out of bounds", (256, 256))
        else:
            # Percentile normalization 1-99
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
        time.sleep(0.1)  # ~10 Hz

@app.route("/roi_feed/<rid>")
def roi_feed(rid: str):
    return mjpeg_response(_roi_stream_frames(rid))

# Save bundle (JSON + PNG inside a ZIP)
@app.route("/save_bundle")
def save_bundle():
    base = request.args.get("base", "").strip()
    if not base:
        return jsonify({"error": "Missing base"}), 400

    # Capture consistent snapshot
    bgr = cam_service.get_latest_bgr()
    size = cam_service.get_frame_size()
    exp = cam_service.get_exposure_us()
    cm = cam_service.get_colormap()
    snap = metrics.get_snapshot()
    rois = roi_registry.list_dicts()

    # JSON
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

    # PNG of current raw overview frame (BGR)
    if bgr is None:
        png_bytes = _encode_png(_placeholder("No frame", (640, 480)))
    else:
        png_bytes = _encode_png(bgr)

    # ZIP in-memory
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{base}.json", json_bytes)
        if png_bytes:
            zf.writestr(f"{base}.png", png_bytes)
    mem.seek(0)

    return send_file(mem, mimetype="application/zip", as_attachment=True, download_name=f"{base}.zip")

if __name__ == "__main__":
    # Do not auto-start camera; user toggles it from UI
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
