# MJPEG web streamer with exposure (integration time) slider and colormap selector.
# Exposure slider range: 1000..50000 (µs), step: 1000.
# Colormaps: grey, jet, cubiczero.
# Visit http://localhost:5000 or http://HOST_IP:5000

import threading
from queue import Queue
import numpy as np
import cv2
from flask import Flask, Response, jsonify, render_template_string, request
from vmbpy import *

CAMERA_ID = "DEV_000F31F42C02"
opencv_display_format = PixelFormat.Bgr8

app = Flask(__name__)

# Locks for thread-safe access from HTTP handlers while streaming
_exposure_lock = threading.Lock()
_colormap_lock = threading.Lock()

# State
_colormap = "grey"  # default: grey | jet | cubiczero

class Handler:
    def __init__(self):
        self.display_queue = Queue(maxsize=2)

    def get_image(self):
        return self.display_queue.get(True)

    def __call__(self, cam: Camera, stream: Stream, frame: Frame):
        if frame.get_status() == FrameStatus.Complete:
            # Convert to BGR8 if needed for OpenCV/JPEG
            if frame.get_pixel_format() == opencv_display_format:
                display = frame
            else:
                display = frame.convert_pixel_format(opencv_display_format)
            # Put numpy BGR image into queue
            try:
                self.display_queue.put(display.as_opencv_image(), True)
            except Exception:
                pass
        # Re-queue the frame for next acquisition
        cam.queue_frame(frame)

handler = Handler()

def setup_camera(cam: Camera):
    with cam:
        # Optional auto features if supported
        try:
            cam.ExposureAuto.set("Continuous")
        except (AttributeError, VmbFeatureError):
            pass

        try:
            cam.BalanceWhiteAuto.set("Continuous")
        except (AttributeError, VmbFeatureError):
            pass

        # Optimize GigE packet size if available
        try:
            stream = cam.get_streams()[0]
            stream.GVSPAdjustPacketSize.run()
            while not stream.GVSPAdjustPacketSize.is_done():
                pass
        except (AttributeError, VmbFeatureError):
            pass

def setup_pixel_format(cam: Camera):
    # Prefer native BGR8; otherwise select a convertible format
    cam_formats = cam.get_pixel_formats()

    if opencv_display_format in cam_formats:
        cam.set_pixel_format(opencv_display_format)
        return

    cam_color_formats = intersect_pixel_formats(cam_formats, COLOR_PIXEL_FORMATS)
    convertible_color = [f for f in cam_color_formats if opencv_display_format in f.get_convertible_formats()]
    if convertible_color:
        cam.set_pixel_format(convertible_color[0])
        return

    cam_mono_formats = intersect_pixel_formats(cam_formats, MONO_PIXEL_FORMATS)
    convertible_mono = [f for f in cam_mono_formats if opencv_display_format in f.get_convertible_formats()]
    if convertible_mono:
        cam.set_pixel_format(convertible_mono[0])
        return

    raise RuntimeError("Camera does not support an OpenCV compatible format (BGR8).")

def _get_exposure_feature(cam: Camera):
    # Try common exposure feature names in microseconds
    for name in ("ExposureTime", "ExposureTimeAbs"):
        try:
            feat = cam.get_feature_by_name(name)
            _ = feat.get()
            return feat
        except (VmbFeatureError, AttributeError):
            continue
    raise VmbFeatureError("No exposure time feature found on this camera.")

def get_exposure_us(cam: Camera) -> int:
    feat = _get_exposure_feature(cam)
    val = feat.get()
    try:
        return int(round(val))
    except Exception:
        return int(val)

def _nearest_with_increment(value: float, min_v: float, max_v: float, inc: float | None) -> float:
    v = max(min_v, min(max_v, value))
    if inc and inc > 0:
        steps = round((v - min_v) / inc)
        v = min_v + steps * inc
        v = max(min_v, min(max_v, v))
    return v

def set_exposure_us(cam: Camera, requested_us: int) -> int:
    # Ensure manual exposure
    try:
        cam.ExposureAuto.set("Off")
    except (AttributeError, VmbFeatureError):
        pass

    feat = _get_exposure_feature(cam)

    # Camera's valid range/increment
    try:
        min_v, max_v = feat.get_range()
    except (VmbFeatureError, TypeError, ValueError):
        min_v, max_v = 10.0, 1_000_000.0

    inc = None
    try:
        inc = feat.get_increment()
    except (VmbFeatureError, AttributeError):
        inc = None

    # UI clamp and step first
    ui_min, ui_max, ui_step = 1000, 50000, 1000
    target = max(ui_min, min(ui_max, int(requested_us)))
    target = (target // ui_step) * ui_step

    # Respect camera increment
    target = _nearest_with_increment(float(target), float(min_v), float(max_v), float(inc) if inc else None)

    feat.set(target)
    current = feat.get()
    try:
        return int(round(current))
    except Exception:
        return int(current)

# ----- Colormap handling -----

def _cubiczero_lut():
    # Build a simple cubic-bezier style LUT (uint8) for a custom palette.
    # This creates smooth cubic curves for each channel over intensity t in [0,1].
    t = np.linspace(0.0, 1.0, 256, dtype=np.float64)
    def cubic_smooth(x):  # 3x^2 - 2x^3
        return (3.0 * x * x) - (2.0 * x * x * x)
    r = np.clip((cubic_smooth(t))**1.0, 0.0, 1.0)
    g = np.clip((cubic_smooth(1.0 - np.abs(t - 0.5) * 2.0)), 0.0, 1.0)
    b = np.clip((cubic_smooth(1.0 - t))**1.0, 0.0, 1.0)
    lut_r = (r * 255.0).astype(np.uint8)
    lut_g = (g * 255.0).astype(np.uint8)
    lut_b = (b * 255.0).astype(np.uint8)
    return lut_b, lut_g, lut_r

_LUT_B, _LUT_G, _LUT_R = _cubiczero_lut()

def apply_colormap(frame_bgr: np.ndarray, mode: str) -> np.ndarray:
    # Convert input BGR to grayscale intensity first (for consistent mapping)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    if mode == "grey":
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if mode == "jet":
        return cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    if mode == "cubiczero":
        b = cv2.LUT(gray, _LUT_B)
        g = cv2.LUT(gray, _LUT_G)
        r = cv2.LUT(gray, _LUT_R)
        return cv2.merge([b, g, r])
    # Fallback to grey
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def gen_mjpeg():
    # Generator that yields multipart JPEG frames
    while True:
        frame_bgr = handler.get_image()
        # Snapshot current colormap without holding the lock across encode
        with _colormap_lock:
            cm = _colormap
        frame_cm = apply_colormap(frame_bgr, cm)
        ok, jpg = cv2.imencode(".jpg", frame_cm)
        if not ok:
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
        )

INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Allied Vision Stream + Exposure & Colormap</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 16px; }
    .row { display: flex; gap: 24px; align-items: flex-start; }
    .panel { padding: 12px; border: 1px solid #ddd; border-radius: 6px; }
    .slider { width: 420px; }
    label { display: block; margin-bottom: 8px; font-weight: bold; }
    .value { font-family: monospace; }
    select { width: 200px; padding: 4px; }
  </style>
</head>
<body>
  <h3>Allied Vision Camera Stream</h3>
  <div class="row">
    <div class="panel">
      <img id="stream" src="/video_feed" alt="Stream" />
    </div>
    <div class="panel">
      <label for="exp">Exposure (Integration Time) [µs]</label>
      <input id="exp" class="slider" type="range" min="1000" max="50000" step="1000" />
      <div>Current: <span id="val" class="value">-</span> µs</div>
      <hr />
      <label for="cm">Colormap</label>
      <select id="cm">
        <option value="grey">grey</option>
        <option value="jet">jet</option>
        <option value="cubiczero">cubiczero</option>
      </select>
    </div>
  </div>

  <script>
    const slider = document.getElementById('exp');
    const valueEl = document.getElementById('val');
    const cmSelect = document.getElementById('cm');

    function setLabel(v) { valueEl.textContent = v; }

    async function fetchCurrent() {
      try {
        const [expRes, cmRes] = await Promise.all([
          fetch('/exposure', { cache: 'no-store' }),
          fetch('/colormap', { cache: 'no-store' })
        ]);
        if (expRes.ok) {
          const data = await expRes.json();
          const v = Math.min(50000, Math.max(1000, Math.round(data.value || 0)));
          slider.value = v; setLabel(v);
        }
        if (cmRes.ok) {
          const data = await cmRes.json();
          if (data && data.value) cmSelect.value = data.value;
        }
      } catch (e) { /* ignore */ }
    }

    // Update label live while sliding
    slider.addEventListener('input', () => setLabel(slider.value));

    // Apply exposure on change (after releasing)
    slider.addEventListener('change', async () => {
      const v = parseInt(slider.value, 10);
      try {
        const res = await fetch('/exposure', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ value: v })
        });
        const data = await res.json();
        if (typeof data.value === 'number') {
          slider.value = data.value;
          setLabel(data.value);
        }
      } catch (e) { /* ignore */ }
    });

    // Apply colormap when selection changes
    cmSelect.addEventListener('change', async () => {
      const value = cmSelect.value;
      try {
        const res = await fetch('/colormap', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ value })
        });
        const data = await res.json();
        if (data && data.value) cmSelect.value = data.value;
      } catch (e) { /* ignore */ }
    });

    fetchCurrent();
  </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(INDEX_HTML)

@app.route("/video_feed")
def video_feed():
    return Response(gen_mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/exposure", methods=["GET", "POST"])
def exposure():
    # Read or set exposure time in microseconds
    if request.method == "GET":
        try:
            with _exposure_lock:
                val = get_exposure_us(cam)
            return jsonify({"value": val})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # POST
    try:
        data = request.get_json(silent=True) or {}
        req_val = int(data.get("value"))
    except Exception:
        return jsonify({"error": "Invalid value"}), 400

    try:
        with _exposure_lock:
            applied = set_exposure_us(cam, req_val)
        return jsonify({"value": applied})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/colormap", methods=["GET", "POST"])
def colormap():
    global _colormap
    if request.method == "GET":
        with _colormap_lock:
            return jsonify({"value": _colormap})
    # POST
    data = request.get_json(silent=True) or {}
    value = str(data.get("value", "")).lower()
    if value not in ("grey", "jet", "cubiczero"):
        return jsonify({"error": "Invalid colormap"}), 400
    with _colormap_lock:
        _colormap = value
        return jsonify({"value": _colormap})

if __name__ == "__main__":
    # Keep VmbSystem and camera streaming alive while the Flask app is running
    with VmbSystem.get_instance() as vmb:
        try:
            cam = vmb.get_camera_by_id(CAMERA_ID)
        except VmbCameraError:
            raise SystemExit(f"Failed to access camera '{CAMERA_ID}'. Abort.")

        with cam:
            setup_camera(cam)
            setup_pixel_format(cam)
            cam.start_streaming(handler=handler, buffer_count=10)
            try:
                # Bind on all interfaces for remote access
                app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
            finally:
                cam.stop_streaming()
