import logging
import os
import threading
import time
from queue import Queue
from typing import Optional, Tuple, Union, List, Dict

import cv2
import numpy as np
try:
    from vmbpy import (VmbSystem, VmbFeatureError, VmbCameraError, PixelFormat, COLOR_PIXEL_FORMATS,
                       MONO_PIXEL_FORMATS, FrameStatus, Camera, Stream, Frame, intersect_pixel_formats)
except ImportError:
    # Create dummy classes if vmbpy is not available, so fake camera can run
    class VmbSystem:
        @staticmethod
        def get_instance():
            return VmbSystem()
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
        def get_all_cameras(self):
            return []
    class VmbFeatureError(Exception): pass
    class VmbCameraError(Exception): pass
    class PixelFormat:
        Bgr8 = "Bgr8"
        Mono16 = "Mono16"
        Mono8 = "Mono8"
    COLOR_PIXEL_FORMATS = []
    MONO_PIXEL_FORMATS = []
    class FrameStatus:
        Complete = "Complete"
    class Camera: pass
    class Stream: pass
    class Frame: pass
    def intersect_pixel_formats(a, b):
        return []

from fake_camera import FakeCamera, FakeFrame

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

        def __call__(self, cam: Union[Camera, FakeCamera], stream: Optional[Stream], frame: Union[Frame, FakeFrame]):
            try:
                if frame.get_status() == FrameStatus.Complete:
                    # Convert to BGR8 if needed for OpenCV/JPEG
                    if frame.get_pixel_format() == OPENCV_DISPLAY_FORMAT:
                        display = frame
                    else:
                        display = frame.convert_pixel_format(OPENCV_DISPLAY_FORMAT)

                    # NumPy BGR image and raw grayscale
                    bgr_input = display.as_opencv_image()
                    gray_raw = cv2.cvtColor(bgr_input, cv2.COLOR_BGR2GRAY)

                    with self.service._frame_lock:
                        bg_sub_enabled = self.service._background_subtraction_enabled
                        bg_frame = self.service._background_frame

                    if bg_sub_enabled and bg_frame is not None:
                        gray_float = gray_raw.astype(np.float32)
                        subtracted_gray_float = gray_float - bg_frame
                        final_gray = np.clip(subtracted_gray_float, 0, 255).astype(np.uint8)
                        final_bgr = cv2.cvtColor(final_gray, cv2.COLOR_GRAY2BGR)
                    else:
                        final_gray = gray_raw
                        final_bgr = bgr_input

                    # Update latest frames (copy to decouple from buffer reuse)
                    with self.service._frame_lock:
                        self.service._latest_gray_raw = gray_raw.copy()
                        self.service._latest_bgr = final_bgr.copy()
                        self.service._latest_gray = final_gray.copy()

                    # Push to display queue (drop if full to keep latency low)
                    try:
                        self.service._display_queue.put(final_bgr, block=False)
                    except Exception:
                        pass

                    # Notify metrics loop that a new frame is available (non-blocking)
                    try:
                        self.service._metrics_signal.put_nowait(1)
                    except Exception:
                        pass
            finally:
                # Re-queue the frame for next acquisition (only for real cameras)
                if isinstance(cam, Camera) and isinstance(frame, Frame):
                    cam.queue_frame(frame)

    def __init__(self, logger: logging.Logger, default_camera_id: Optional[str] = None):
        self._logger = logger
        self._default_camera_id = default_camera_id or os.environ.get("CAMERA_ID")
        self._active_camera_id: Optional[str] = None

        self._vmb: Optional[VmbSystem] = None
        self._cam: Optional[Union[Camera, FakeCamera]] = None
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
        self._latest_gray_raw: Optional[np.ndarray] = None
        self._colormap: str = "jet"
        self._background_frame: Optional[np.ndarray] = None
        self._background_subtraction_enabled = False

    def set_default_camera_id(self, camera_id: str):
        self._logger.info(f"Default camera ID set to: {camera_id}")
        self._default_camera_id = camera_id

    # ---------- Lifecycle ----------

    def list_available_cameras(self) -> List[Dict[str, str]]:
        """Lists all available physical cameras and adds the FakeCamera option."""
        cameras = [{"id": "fake", "name": "Fake Camera"}]
        try:
            with VmbSystem.get_instance() as vmb:
                real_cams = vmb.get_all_cameras()
                for cam in real_cams:
                    cameras.append({"id": cam.get_id(), "name": cam.get_name()})
        except Exception as e:
            self._logger.error(f"Could not list VmbPy cameras: {e}", exc_info=True)
        return cameras

    def start(self, camera_id: Optional[str] = None) -> None:
        if self._running:
            return

        target_camera_id = camera_id or self._default_camera_id
        self._logger.info(f"Attempting to start camera: {target_camera_id}")

        if not target_camera_id:
            # If no camera is specified at all (not in UI, not in args), try to find one.
            try:
                with VmbSystem.get_instance() as vmb:
                    available_cams = vmb.get_all_cameras()
                    if available_cams:
                        target_camera_id = available_cams[0].get_id()
                        self._logger.info(f"No camera specified, defaulting to first found: {target_camera_id}")
                    else:
                        # If no real cameras, default to fake one
                        target_camera_id = "fake"
                        self._logger.info("No real cameras found, defaulting to Fake Camera.")
            except Exception:
                target_camera_id = "fake"
                self._logger.info("VmbPy not available, defaulting to Fake Camera.")


        if target_camera_id == "fake":
            self._logger.info("Initializing FakeCamera...")
            cam = FakeCamera(camera_id="fake", logger=self._logger)
            self._cam = cam
            self._cam.__enter__()
            self._cam.start_streaming(handler=self._handler)
            self._active_camera_id = "fake"
            self._running = True
            self._logger.info("FakeCamera started.")
            return

        # --- Real VmbPy Camera Logic ---
        self._vmb = VmbSystem.get_instance()
        self._vmb.__enter__()

        cam: Optional[Camera] = None
        try:
            cam = self._vmb.get_camera_by_id(target_camera_id)
        except VmbCameraError as e:
            self._logger.error(f"Failed to get camera by ID '{target_camera_id}': {e}", exc_info=True)
            self._vmb.__exit__(None, None, None)
            self._vmb = None
            raise e

        self._cam = cam
        self._active_camera_id = self._cam.get_id()
        self._cam.__enter__()

        self._setup_camera(self._cam)
        self._setup_pixel_format(self._cam)

        self._cam.start_streaming(handler=self._handler, buffer_count=10)
        self._running = True
        self._logger.info(f"Real camera {self._active_camera_id} started.")

    def stop(self) -> None:
        if not self._running:
            return
        self._logger.info(f"Stopping camera: {self.get_camera_id()}")
        with self._frame_lock:
            background_active = self._background_subtraction_enabled or self._background_frame is not None
        if background_active:
            self.stop_background_subtraction()
        try:
            if self._cam:
                self._cam.stop_streaming()
        finally:
            if self._cam:
                self._cam.__exit__(None, None, None)
                self._cam = None
            if self._vmb:
                self._vmb.__exit__(None, None, None)
                self._vmb = None

            while not self._display_queue.empty():
                try: self._display_queue.get_nowait()
                except Exception: break
            while not self._metrics_signal.empty():
                try: self._metrics_signal.get_nowait()
                except Exception: break

            self._running = False
            self._active_camera_id = None
            self._logger.info("Camera stopped.")

    def is_running(self) -> bool:
        return self._running

    # ---------- Camera setup helpers (ported/extended from baseline) ----------

    def _setup_camera(self, cam: Camera) -> None:
        try:
            self.set_exposure_us(1000)
        except (AttributeError, VmbFeatureError):
            pass

        try:
            cam.BalanceWhiteAuto.set("Continuous")
        except (AttributeError, VmbFeatureError):
            pass

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
            try:
                self._cam.ExposureAuto.set("Off")
            except (AttributeError, VmbFeatureError):
                pass

            try:
                feat = self._get_exposure_feature()
            except Exception:
                return 0

            try:
                min_v, max_v = feat.get_range()
            except (VmbFeatureError, TypeError, ValueError):
                min_v, max_v = 10.0, 1_000_000.0

            inc = None
            try:
                inc = feat.get_increment()
            except (VmbFeatureError, AttributeError):
                inc = None

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

    def start_background_subtraction(self, num_frames: int):
        if not self._cam:
            raise RuntimeError("Camera not running")

        frames: List[np.ndarray] = []
        while len(frames) < num_frames:
            if not self._running:
                raise RuntimeError("Camera not running")
            got_signal = self.wait_for_frame_signal(timeout=0.2)
            if not got_signal:
                continue
            latest_gray = self.get_latest_gray(raw=True)
            if latest_gray is None:
                continue
            frames.append(latest_gray)

        if not frames:
            raise RuntimeError("Could not capture any background frames.")

        # Average the collected frames. Convert to float for calculation.
        avg_frame_float = np.mean([f.astype(np.float32) for f in frames], axis=0)

        with self._frame_lock:
            self._background_frame = avg_frame_float
            self._background_subtraction_enabled = True

        self._logger.info(f"Background subtraction enabled, averaged {len(frames)} frames.")

    def stop_background_subtraction(self):
        with self._frame_lock:
            self._background_frame = None
            self._background_subtraction_enabled = False
        self._logger.info("Background subtraction disabled.")

    # ---------- Frames and MJPEG ----------

    def get_latest_bgr(self) -> Optional[np.ndarray]:
        with self._frame_lock:
            if self._latest_bgr is None:
                return None
            return self._latest_bgr.copy()

    def get_latest_gray(self, *, raw: bool = False) -> Optional[np.ndarray]:
        with self._frame_lock:
            src = self._latest_gray_raw if raw else self._latest_gray
            if src is None:
                return None
            return src.copy()

    def get_background_frame(self) -> Optional[np.ndarray]:
        with self._frame_lock:
            if self._background_frame is None:
                return None
            return self._background_frame.copy()

    def is_background_subtraction_enabled(self) -> bool:
        with self._frame_lock:
            return self._background_subtraction_enabled

    def get_frame_size(self) -> Optional[Tuple[int, int]]:
        with self._frame_lock:
            if self._latest_bgr is None:
                return None
            h, w = self._latest_bgr.shape[:2]
            return (w, h)

    def get_camera_id(self) -> Optional[str]:
        return self._active_camera_id

    def gen_overview_mjpeg(self):
        boundary = b"--frame\r\n"
        while True:
            if not self._running:
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
