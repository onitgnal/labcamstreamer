import os
import threading
import time
from queue import Queue
from typing import Optional, Tuple

import cv2
import numpy as np
from vmbpy import (VmbSystem, VmbFeatureError, VmbCameraError, PixelFormat, COLOR_PIXEL_FORMATS,
                   MONO_PIXEL_FORMATS, FrameStatus, Camera, Stream, Frame, intersect_pixel_formats)

# Camera display pixel format for OpenCV/JPEG
OPENCV_DISPLAY_FORMAT = PixelFormat.Bgr8


def _cubiczero_lut() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 1.0, 256, dtype=np.float64)

    def cubic_smooth(x: np.ndarray) -> np.ndarray:
        return (3.0 * x * x) - (2.0 * x * x * x)

    r = np.clip((cubic_smooth(t)) ** 1.0, 0.0, 1.0)
    g = np.clip((cubic_smooth(1.0 - np.abs(t - 0.5) * 2.0)), 0.0, 1.0)
    b = np.clip((cubic_smooth(1.0 - t)) ** 1.0, 0.0, 1.0)
    lut_r = (r * 255.0).astype(np.uint8)
    lut_g = (g * 255.0).astype(np.uint8)
    lut_b = (b * 255.0).astype(np.uint8)
    return lut_b, lut_g, lut_r


_LUT_B, _LUT_G, _LUT_R = _cubiczero_lut()


def apply_colormap_to_gray(gray: np.ndarray, mode: str) -> np.ndarray:
    if mode == "grey":
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if mode == "jet":
        return cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    if mode == "cubiczero":
        b = cv2.LUT(gray, _LUT_B)
        g = cv2.LUT(gray, _LUT_G)
        r = cv2.LUT(gray, _LUT_R)
        return cv2.merge([b, g, r])
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def apply_colormap_to_bgr(bgr: np.ndarray, mode: str) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return apply_colormap_to_gray(gray, mode)


def _placeholder_image(text: str, size: Tuple[int, int] = (640, 480)) -> np.ndarray:
    w, h = size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(img, text, (20, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def _nearest_with_increment(value: float, min_v: float, max_v: float, inc: Optional[float]) -> float:
    v = max(min_v, min(max_v, value))
    if inc and inc > 0:
        steps = round((v - min_v) / inc)
        v = min_v + steps * inc
        v = max(min_v, min(max_v, v))
    return v


class CameraService:
    """
    Owns VmbSystem/Camera, streaming handler, queues, and locks.
    Keeps latest raw BGR and latest GRAY frames for metrics/ROI/Save.
    Thread-safe exposure and colormap accessors.
    """

    class _Handler:
        def __init__(self, service: "CameraService"):
            self.service = service

        def __call__(self, cam: Camera, stream: Stream, frame: Frame):
            try:
                if frame.get_status() == FrameStatus.Complete:
                    # Convert to BGR8 if needed for OpenCV/JPEG
                    if frame.get_pixel_format() == OPENCV_DISPLAY_FORMAT:
                        display = frame
                    else:
                        display = frame.convert_pixel_format(OPENCV_DISPLAY_FORMAT)

                    # NumPy BGR image
                    bgr = display.as_opencv_image()

                    # Update latest frames (copy to decouple from buffer reuse)
                    with self.service._frame_lock:
                        self.service._latest_bgr = bgr.copy()
                        self.service._latest_gray = cv2.cvtColor(self.service._latest_bgr, cv2.COLOR_BGR2GRAY)

                    # Push to display queue (drop if full to keep latency low)
                    try:
                        self.service._display_queue.put(self.service._latest_bgr, block=False)
                    except Exception:
                        pass

                    # Notify metrics loop that a new frame is available (non-blocking)
                    try:
                        self.service._metrics_signal.put_nowait(1)
                    except Exception:
                        pass
            finally:
                # Re-queue the frame for next acquisition
                cam.queue_frame(frame)

    def __init__(self, camera_id: Optional[str] = None):
        self._camera_id = camera_id or os.environ.get("CAMERA_ID")
        self._vmb: Optional[VmbSystem] = None
        self._cam: Optional[Camera] = None
        self._handler = CameraService._Handler(self)

        self._running = False
        self._display_queue: "Queue[np.ndarray]" = Queue(maxsize=2)
        self._metrics_signal: "Queue[int]" = Queue(maxsize=1)

        # Shared state
        self._frame_lock = threading.Lock()
        self._exposure_lock = threading.Lock()
        self._colormap_lock = threading.Lock()

        self._latest_bgr: Optional[np.ndarray] = None
        self._latest_gray: Optional[np.ndarray] = None
        self._colormap: str = "grey"

    # ---------- Lifecycle ----------

    def start(self) -> None:
        if self._running:
            return

        # Enter VmbSystem context
        self._vmb = VmbSystem.get_instance()
        self._vmb.__enter__()  # keep context open

        # Pick camera (try preferred id, fall back to first available)
        cam: Optional[Camera] = None
        try:
            if self._camera_id:
                try:
                    cam = self._vmb.get_camera_by_id(self._camera_id)
                except VmbCameraError:
                    cams = self._vmb.get_all_cameras()
                    if not cams:
                        raise
                    cam = cams[0]
            else:
                cams = self._vmb.get_all_cameras()
                if not cams:
                    raise VmbCameraError("No cameras found.")
                cam = cams[0]
        except VmbCameraError as e:
            # Properly exit VmbSystem before raising
            self._vmb.__exit__(None, None, None)
            self._vmb = None
            raise e

        self._cam = cam
        # Enter camera context
        self._cam.__enter__()

        # Configure camera
        self._setup_camera(self._cam)
        self._setup_pixel_format(self._cam)

        # Prime capture by queuing all frames
        # Start streaming with handler
        self._cam.start_streaming(handler=self._handler, buffer_count=10)
        self._running = True

    def stop(self) -> None:
        if not self._running:
            return
        try:
            if self._cam:
                try:
                    self._cam.stop_streaming()
                except Exception:
                    pass
        finally:
            # Leave camera and Vmb contexts
            if self._cam:
                try:
                    self._cam.__exit__(None, None, None)
                except Exception:
                    pass
                self._cam = None

            if self._vmb:
                try:
                    self._vmb.__exit__(None, None, None)
                except Exception:
                    pass
                self._vmb = None

            # Clear queues
            try:
                while not self._display_queue.empty():
                    self._display_queue.get_nowait()
            except Exception:
                pass
            try:
                while not self._metrics_signal.empty():
                    self._metrics_signal.get_nowait()
            except Exception:
                pass

            self._running = False

    def is_running(self) -> bool:
        return self._running

    # ---------- Camera setup helpers (ported/extended from baseline) ----------

    def _setup_camera(self, cam: Camera) -> None:
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
        except (AttributeError, VmbFeatureError, IndexError):
            pass

    def _setup_pixel_format(self, cam: Camera) -> None:
        cam_formats = cam.get_pixel_formats()

        if OPENCV_DISPLAY_FORMAT in cam_formats:
            cam.set_pixel_format(OPENCV_DISPLAY_FORMAT)
            return

        cam_color_formats = intersect_pixel_formats(cam_formats, COLOR_PIXEL_FORMATS)
        convertible_color = [f for f in cam_color_formats if OPENCV_DISPLAY_FORMAT in f.get_convertible_formats()]
        if convertible_color:
            cam.set_pixel_format(convertible_color[0])
            return

        cam_mono_formats = intersect_pixel_formats(cam_formats, MONO_PIXEL_FORMATS)
        convertible_mono = [f for f in cam_mono_formats if OPENCV_DISPLAY_FORMAT in f.get_convertible_formats()]
        if convertible_mono:
            cam.set_pixel_format(convertible_mono[0])
            return

        raise RuntimeError("Camera does not support an OpenCV compatible format (BGR8).")

    # ---------- Exposure helpers ----------

    def _get_exposure_feature(self):
        assert self._cam is not None
        for name in ("ExposureTime", "ExposureTimeAbs"):
            try:
                feat = self._cam.get_feature_by_name(name)
                _ = feat.get()
                return feat
            except (VmbFeatureError, AttributeError):
                continue
        raise VmbFeatureError("No exposure time feature found on this camera.")

    def get_exposure_us(self) -> int:
        if not self._cam:
            return 0
        with self._exposure_lock:
            try:
                feat = self._get_exposure_feature()
                val = feat.get()
            except Exception:
                return 0
            try:
                return int(round(val))
            except Exception:
                try:
                    return int(val)
                except Exception:
                    return 0

    def set_exposure_us(self, requested_us: int) -> int:
        if not self._cam:
            return 0

        with self._exposure_lock:
            # Ensure manual exposure
            try:
                self._cam.ExposureAuto.set("Off")
            except (AttributeError, VmbFeatureError):
                pass

            try:
                feat = self._get_exposure_feature()
            except Exception:
                return 0

            # Camera range/increment
            try:
                min_v, max_v = feat.get_range()
            except (VmbFeatureError, TypeError, ValueError):
                min_v, max_v = 10.0, 1_000_000.0

            inc = None
            try:
                inc = feat.get_increment()
            except (VmbFeatureError, AttributeError):
                inc = None

            # UI clamp and step
            ui_min, ui_max, ui_step = 1000, 50000, 1000
            target = max(ui_min, min(ui_max, int(requested_us)))
            target = (target // ui_step) * ui_step

            target = _nearest_with_increment(float(target), float(min_v), float(max_v), float(inc) if inc else None)

            feat.set(target)
            current = feat.get()
            try:
                return int(round(current))
            except Exception:
                try:
                    return int(current)
                except Exception:
                    return 0

    # ---------- Colormap ----------

    def get_colormap(self) -> str:
        with self._colormap_lock:
            return self._colormap

    def set_colormap(self, mode: str) -> str:
        mode = (mode or "").lower()
        if mode not in ("grey", "jet", "cubiczero"):
            raise ValueError("Invalid colormap")
        with self._colormap_lock:
            self._colormap = mode
            return self._colormap

    # ---------- Frames and MJPEG ----------

    def get_latest_bgr(self) -> Optional[np.ndarray]:
        with self._frame_lock:
            if self._latest_bgr is None:
                return None
            return self._latest_bgr.copy()

    def get_latest_gray(self) -> Optional[np.ndarray]:
        with self._frame_lock:
            if self._latest_gray is None:
                return None
            return self._latest_gray.copy()

    def get_frame_size(self) -> Optional[Tuple[int, int]]:
        with self._frame_lock:
            if self._latest_bgr is None:
                return None
            h, w = self._latest_bgr.shape[:2]
            return (w, h)

    def get_camera_id(self) -> Optional[str]:
        # Return active camera id if available, else the configured id
        if self._cam:
            try:
                return self._cam.get_id()
            except Exception:
                pass
        return self._camera_id

    def gen_overview_mjpeg(self):
        # Yields MJPEG frames; applies current colormap on server side
        boundary = b"--frame\r\n"
        while True:
            if not self._running:
                # Yield placeholder at a slow rate
                img = _placeholder_image("Camera Off")
                with self._colormap_lock:
                    cm = self._colormap
                frame_cm = apply_colormap_to_bgr(img, cm)
                ok, jpg = cv2.imencode(".jpg", frame_cm)
                if ok:
                    yield boundary + b"Content-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
                time.sleep(0.5)
                continue

            try:
                bgr = self._display_queue.get(timeout=1.0)
            except Exception:
                # If queue empty, emit placeholder
                img = _placeholder_image("Waiting for frames")
                with self._colormap_lock:
                    cm = self._colormap
                frame_cm = apply_colormap_to_bgr(img, cm)
                ok, jpg = cv2.imencode(".jpg", frame_cm)
                if ok:
                    yield boundary + b"Content-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
                continue

            with self._colormap_lock:
                cm = self._colormap

            frame_cm = apply_colormap_to_bgr(bgr, cm)
            ok, jpg = cv2.imencode(".jpg", frame_cm)
            if not ok:
                continue
            yield boundary + b"Content-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"

    # ---------- Metrics signaling ----------

    def wait_for_frame_signal(self, timeout: float = 1.0) -> bool:
        try:
            self._metrics_signal.get(timeout=timeout)
            return True
        except Exception:
            return False
