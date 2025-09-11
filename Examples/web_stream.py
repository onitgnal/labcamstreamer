# Simple MJPEG web streamer for Allied Vision camera using VmbPy + Flask.
# Serves http://localhost:5000 with an <img> that streams from /video_feed.

from queue import Queue
import cv2
from flask import Flask, Response, render_template_string
from vmbpy import *

CAMERA_ID = "DEV_000F31F42C02"
opencv_display_format = PixelFormat.Bgr8

app = Flask(__name__)

class Handler:
    def __init__(self):
        # Small queue to keep latency low
        self.display_queue = Queue(maxsize=2)

    def get_image(self):
        # Blocks until a frame is available
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
                # If queue is full or interrupted, drop frame silently
                pass

        # Re-queue the frame for next acquisition
        cam.queue_frame(frame)

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

handler = Handler()

def gen_mjpeg():
    # Generator that yields multipart JPEG frames
    while True:
        frame_bgr = handler.get_image()
        ok, jpg = cv2.imencode(".jpg", frame_bgr)
        if not ok:
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
        )

@app.route("/")
def index():
    # Minimal page showing the stream
    return render_template_string(
        "<html><body><h3>Allied Vision Camera Stream</h3>"
        "<img src='/video_feed' /></body></html>"
    )

@app.route("/video_feed")
def video_feed():
    return Response(gen_mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")

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
                # Disable debug reloader to avoid double initialization
                app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
            finally:
                cam.stop_streaming()
