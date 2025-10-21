import logging
import math
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
                    sensor_gray = None
                    if self.service._sensor_capture_enabled_state():
                        try:
                            sensor_data = frame.as_numpy_ndarray()
                        except AttributeError:
                            sensor_data = None
                        except Exception:
                            sensor_data = None
                        if sensor_data is not None:
                            sensor_arr = np.asarray(sensor_data)
                            if sensor_arr.ndim == 3 and sensor_arr.shape[-1] == 1:
                                sensor_arr = sensor_arr[..., 0]
                            elif sensor_arr.ndim == 3 and sensor_arr.shape[-1] >= 3:
                                sensor_arr = sensor_arr.max(axis=2)
                            sensor_gray = np.ascontiguousarray(sensor_arr)

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
                        if sensor_gray is not None and sensor_gray.size > 0:
                            self.service._latest_sensor_raw = sensor_gray.copy()

                    if sensor_gray is not None and sensor_gray.size > 0:
                        self.service._notify_sensor_frame_captured(sensor_gray)

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
        self._sensor_lock = threading.Lock()

        self._latest_bgr: Optional[np.ndarray] = None
        self._latest_gray: Optional[np.ndarray] = None
        self._latest_gray_raw: Optional[np.ndarray] = None
        self._latest_sensor_raw: Optional[np.ndarray] = None
        self._colormap: str = "jet"
        self._background_frame: Optional[np.ndarray] = None
        self._background_subtraction_enabled = False
        self._sensor_max_value: Optional[float] = None
        self._sensor_max_dtype: Optional[np.dtype] = None
        self._sensor_last_observed_peak: float = 0.0
        self._sensor_capture_enabled: bool = False
        self._sensor_capture_event = threading.Event()
        self._sensor_capture_generation: int = 0

    def set_default_camera_id(self, camera_id: str):
        self._logger.info(f"Default camera ID set to: {camera_id}")
        self._default_camera_id = camera_id

    def _reset_sensor_calibration(self) -> None:
        with self._sensor_lock:
            self._sensor_max_value = None
            self._sensor_max_dtype = None
            self._sensor_last_observed_peak = 0.0
            self._sensor_capture_enabled = False
            self._sensor_capture_generation = 0
            self._sensor_capture_event.clear()

    def _sensor_capture_enabled_state(self) -> bool:
        with self._sensor_lock:
            return self._sensor_capture_enabled

    def _notify_sensor_frame_captured(self, sensor_frame: np.ndarray) -> None:
        self._update_sensor_observation(sensor_frame)
        with self._sensor_lock:
            self._sensor_capture_generation += 1
            self._sensor_capture_event.set()

    def _fetch_sensor_frame(self, timeout: float = 0.4) -> Optional[np.ndarray]:
        timeout = max(0.0, float(timeout))
        deadline = time.monotonic() + timeout
        with self._sensor_lock:
            if not self._sensor_capture_enabled:
                self._sensor_capture_enabled = True
                self._sensor_capture_event.clear()
            start_generation = self._sensor_capture_generation

        captured: Optional[np.ndarray] = None
        while True:
            with self._frame_lock:
                if self._latest_sensor_raw is not None:
                    captured = self._latest_sensor_raw.copy()
            if captured is not None:
                with self._sensor_lock:
                    if self._sensor_capture_generation != start_generation:
                        return captured
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            self._sensor_capture_event.wait(min(remaining, 0.05))

        if captured is None:
            with self._frame_lock:
                if self._latest_sensor_raw is not None:
                    captured = self._latest_sensor_raw.copy()
        return captured

    def _maybe_disable_sensor_capture(self, *, force: bool = False) -> None:
        with self._sensor_lock:
            if force or self._sensor_max_value is not None:
                self._sensor_capture_enabled = False
                self._sensor_capture_event.clear()

    def _update_sensor_observation(self, sensor_frame: np.ndarray) -> None:
        if sensor_frame is None or sensor_frame.size == 0:
            return
        peak = float(np.max(sensor_frame))
        if not np.isfinite(peak):
            return
        dtype = sensor_frame.dtype
        with self._sensor_lock:
            if self._sensor_max_dtype is not None and self._sensor_max_dtype != dtype:
                # Sensor output type changed; discard previous calibration.
                self._sensor_max_value = None
                self._sensor_last_observed_peak = 0.0
            self._sensor_max_dtype = dtype
            if peak > self._sensor_last_observed_peak:
                self._sensor_last_observed_peak = peak
            if (
                self._sensor_max_value is not None
                and self._sensor_max_dtype == dtype
                and peak > self._sensor_max_value
            ):
                bits = self._estimate_integer_bit_depth(peak)
                candidate = self._integer_level_from_bits(bits)
                dtype_max = self._dtype_max_from_dtype(dtype)
                candidate = min(dtype_max, max(candidate, peak))
                self._sensor_max_value = candidate

    @staticmethod
    def _dtype_max_from_dtype(dtype: np.dtype) -> float:
        try:
            if np.issubdtype(dtype, np.integer):
                info = np.iinfo(dtype)
                return float(info.max)
            if np.issubdtype(dtype, np.floating):
                finfo = np.finfo(dtype)
                return float(finfo.max)
        except (ValueError, TypeError):
            pass
        # Fallback: assume full-range unsigned integer.
        try:
            bits = np.dtype(dtype).itemsize * 8
            return float((1 << bits) - 1)
        except Exception:
            return 1.0

    @staticmethod
    def _integer_level_from_bits(bits: int) -> float:
        if bits <= 0:
            return 0.0
        try:
            return float((1 << bits) - 1)
        except OverflowError:
            return float((1 << 31) - 1)

    @staticmethod
    def _estimate_integer_bit_depth(value: float) -> int:
        if not np.isfinite(value) or value <= 0.0:
            return 0
        return int(math.ceil(math.log2(value + 1.0)))

    def _collect_sensor_peak(self, *, samples: int = 6, timeout: float = 0.3) -> float:
        peak = 0.0
        for _ in range(max(1, samples)):
            frame = self._fetch_sensor_frame(timeout=timeout)
            if frame is None or frame.size == 0:
                continue
            sample_peak = float(np.max(frame))
            if not np.isfinite(sample_peak):
                continue
            peak = max(peak, sample_peak)
        return peak

    def _measure_peak_at_extreme_exposure(self) -> float:
        try:
            feat = self._get_exposure_feature()
        except Exception:
            return 0.0
        try:
            min_exp, max_exp = feat.get_range()
        except (VmbFeatureError, TypeError, ValueError):
            return 0.0

        current_exp = max(1, self.get_exposure_us())
        try:
            target_exp = int(round(float(max_exp)))
        except Exception:
            target_exp = current_exp

        if target_exp <= 0:
            return 0.0

        if target_exp == current_exp:
            peak = self._collect_sensor_peak(samples=8, timeout=0.3)
            self._maybe_disable_sensor_capture()
            return peak

        applied = self.set_exposure_us(target_exp)
        if applied != target_exp:
            target_exp = applied
        try:
            peak = self._collect_sensor_peak(samples=8, timeout=0.35)
        finally:
            if target_exp != current_exp:
                self.set_exposure_us(current_exp)
                # Wait for a couple of frames to flush the high-exposure measurement.
                self._collect_sensor_peak(samples=3, timeout=0.2)
            self._maybe_disable_sensor_capture()
        return peak

    def _measure_sensor_max_value(self, sample: Optional[np.ndarray]) -> float:
        if sample is not None and sample.size > 0:
            dtype = sample.dtype
        else:
            latest = self._fetch_sensor_frame(timeout=0.3)
            if latest is not None and latest.size > 0:
                dtype = latest.dtype
                sample = latest
            else:
                raw = self.get_latest_gray(raw=True)
                if raw is None or raw.size == 0:
                    return 255.0
                dtype = raw.dtype
                sample = raw

        dtype_max = self._dtype_max_from_dtype(dtype)
        if dtype_max <= 0.0 or not np.isfinite(dtype_max):
            dtype_max = 1.0

        observed_peak = 0.0
        if sample is not None and sample.size > 0:
            observed_peak = max(observed_peak, float(np.max(sample)))

        observed_peak = max(observed_peak, self._collect_sensor_peak(samples=6, timeout=0.25))

        bits = self._estimate_integer_bit_depth(observed_peak)
        candidate_max = self._integer_level_from_bits(bits)
        if candidate_max <= 0.0 or not np.isfinite(candidate_max):
            candidate_max = dtype_max
        candidate_max = min(dtype_max, max(candidate_max, observed_peak))

        need_extreme_check = (
            candidate_max >= dtype_max * 0.95
            or observed_peak < 0.1 * dtype_max
        )
        if need_extreme_check:
            extreme_peak = self._measure_peak_at_extreme_exposure()
            if extreme_peak > observed_peak:
                observed_peak = extreme_peak
                bits = self._estimate_integer_bit_depth(observed_peak)
                candidate_max = self._integer_level_from_bits(bits)
                if candidate_max <= 0.0 or not np.isfinite(candidate_max):
                    candidate_max = dtype_max
                candidate_max = min(dtype_max, max(candidate_max, observed_peak))

        if candidate_max < dtype_max:
            return max(1.0, candidate_max)

        return max(1.0, dtype_max)

    def _ensure_sensor_max_value(self, sample: Optional[np.ndarray] = None) -> float:
        dtype = None
        if sample is not None and sample.size > 0:
            dtype = sample.dtype

        with self._sensor_lock:
            if (
                self._sensor_max_value is not None
                and self._sensor_max_dtype is not None
                and (dtype is None or self._sensor_max_dtype == dtype)
            ):
                return self._sensor_max_value

        measured = self._measure_sensor_max_value(sample)

        with self._sensor_lock:
            if dtype is None and sample is not None and sample.size > 0:
                dtype = sample.dtype
            if dtype is None and self._sensor_max_dtype is not None:
                dtype = self._sensor_max_dtype
            self._sensor_max_value = measured
            self._sensor_max_dtype = dtype
        self._maybe_disable_sensor_capture()
        return measured

    def _signal_peak_metrics(
        self,
        arr: Optional[np.ndarray],
        *,
        inspect_count: int = 4096,
        hot_gap_ratio: float = 8.0,
        hot_cluster_min: int = 4,
        max_cluster: int = 256,
    ) -> Dict[str, float]:
        """
        Analyse the brightest pixels in ``arr`` and return robust peak statistics.

        The routine inspects the brightest ``inspect_count`` samples (default 4096),
        looks for the first large gap in the descending sorted values, and treats
        tiny clusters above that gap as hot pixels to be ignored before computing
        a robust peak estimate.  The resulting dictionary includes:

            - ``max``: absolute max value seen in the inspected samples.
            - ``robust``: representative peak after excluding hot outliers.
            - ``cluster_size``: number of pixels supporting the robust peak.
            - ``dropped``: number of suspected hot-pixel samples ignored.
            - ``hot_gap``: ratio between the cluster boundary and the next value.
            - ``inspect_count``: number of samples included in the inspection.
        """
        stats: Dict[str, float] = {
            "max": 0.0,
            "robust": 0.0,
            "cluster_size": 0,
            "dropped": 0,
            "hot_gap": float("inf"),
            "inspect_count": 0,
        }

        if arr is None:
            return stats
        arr_np = np.asarray(arr)
        if arr_np.size == 0:
            return stats

        flat = np.ravel(arr_np).astype(np.float64, copy=False)
        flat = flat[np.isfinite(flat)]
        count = flat.size
        if count == 0:
            return stats

        stats["inspect_count"] = int(min(count, inspect_count))
        stats["max"] = float(np.max(flat))

        inspect = stats["inspect_count"]
        if inspect <= 0:
            stats["robust"] = stats["max"]
            stats["cluster_size"] = 1 if stats["max"] > 0 else 0
            return stats

        top = np.partition(flat, count - inspect)[count - inspect:]
        top.sort()
        descending = top[::-1]

        candidate_len = min(max_cluster, descending.size)
        if candidate_len == 0:
            stats["robust"] = stats["max"]
            stats["cluster_size"] = 1 if stats["max"] > 0 else 0
            stats["hot_gap"] = float("inf")
            return stats

        candidates = descending[:candidate_len]
        stats["max"] = float(candidates[0])

        ref_idx = min(hot_cluster_min, candidates.size) - 1
        reference = candidates[ref_idx]
        if not np.isfinite(reference) or reference <= 0.0:
            reference = float(np.median(candidates))
            if reference <= 0.0:
                reference = float(np.mean(candidates))

        hot_threshold = reference * hot_gap_ratio if reference > 0.0 else float("inf")
        drop_count = 0
        if np.isfinite(hot_threshold):
            while drop_count < candidates.size and candidates[drop_count] > hot_threshold:
                drop_count += 1
        if drop_count > 0:
            max_hot_drop = max(2, hot_cluster_min - 1)
            if drop_count > max_hot_drop:
                drop_count = max_hot_drop

        trimmed = candidates[drop_count:]
        if drop_count > 0:
            suspect = candidates[:drop_count]
            if trimmed.size < hot_cluster_min or suspect.size <= 4:
                keep = max(drop_count, min(hot_cluster_min, candidates.size))
                trimmed = candidates[:keep]
                drop_count = 0

        if trimmed.size == 0:
            tail_count = min(hot_cluster_min, candidates.size)
            trimmed = candidates[-tail_count:]
            drop_count = max(0, candidates.size - trimmed.size)

        if trimmed.size >= 5:
            robust_value = float(np.median(trimmed))
        elif trimmed.size > 1:
            robust_value = float(np.mean(trimmed))
        else:
            robust_value = float(trimmed[0])

        stats["robust"] = robust_value
        stats["cluster_size"] = int(trimmed.size)
        stats["dropped"] = int(drop_count)
        if drop_count > 0 and drop_count < candidates.size:
            stats["hot_gap"] = float(candidates[drop_count - 1] / max(candidates[drop_count], 1e-6))
        else:
            stats["hot_gap"] = 1.0 if drop_count == 0 else float("inf")
        if stats["cluster_size"] <= 3:
            stats["robust"] = float(stats["max"])
        return stats

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

        with self._frame_lock:
            self._latest_bgr = None
            self._latest_gray = None
            self._latest_gray_raw = None
            self._latest_sensor_raw = None
        self._reset_sensor_calibration()

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
            with self._frame_lock:
                self._latest_sensor_raw = None
            self._reset_sensor_calibration()
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
            min_v = float(min_v)
            max_v = float(max_v)

            inc = None
            try:
                inc = float(feat.get_increment())
            except (VmbFeatureError, AttributeError, TypeError, ValueError):
                inc = None

            ui_min, ui_max, ui_step = 1, 1e6, 1000
            requested = float(requested_us)

            target = max(min_v, min(max_v, requested))
            target = _nearest_with_increment(target, min_v, max_v, inc)

            if requested >= ui_min:
                snapped = max(float(ui_min), min(float(ui_max), requested))
                snapped = (int(snapped) // ui_step) * ui_step
                snapped = max(min_v, min(max_v, float(snapped)))
                target = _nearest_with_increment(snapped, min_v, max_v, inc)

            feat.set(target)
            current = feat.get()
            try:
                return int(round(current))
            except Exception:
                try:
                    return int(current)
                except Exception:
                    return 0

    def _get_gain_auto_feature(self):
        if not self._cam:
            return None
        try:
            feat = self._cam.get_feature_by_name("GainAuto")
        except (AttributeError, VmbFeatureError):
            return None
        try:
            feat.get()
        except Exception:
            return None
        return feat

    def _trigger_gain_auto_once(self) -> bool:
        feat = self._get_gain_auto_feature()
        if feat is None:
            return False
        try:
            candidates = []
            try:
                candidates = feat.get_values()
            except Exception:
                candidates = []
            target = None
            lowered = {str(v).lower(): v for v in candidates}
            if "once" in lowered:
                target = lowered["once"]
            elif not candidates:
                target = "Once"
            if target is None:
                return False
            feat.set(target)
            return True
        except Exception:
            return False

    def auto_adjust_exposure(self, target_fraction: float = 0.9, tolerance: float = 0.05) -> Dict[str, object]:
        """
        Adjust exposure to target a peak intensity fraction of the sensor range.
        Optionally falls back to triggering camera gain auto if exposure limits are hit.
        """
        if not self.is_running():
            raise RuntimeError("Camera must be running to auto-adjust exposure.")

        try:
            sensor_frame = self._fetch_sensor_frame(timeout=0.35)
            raw_frame = self.get_latest_gray(raw=True)
            processed_frame = self.get_latest_gray()

            frame = processed_frame if processed_frame is not None and getattr(processed_frame, "size", 0) > 0 else raw_frame
            if frame is None or getattr(frame, "size", 0) == 0:
                raise RuntimeError("No frame available for auto exposure.")

            frame_arr = np.asarray(frame)
            frame_peak = float(np.max(frame_arr))
            if frame_peak <= 0.0 or not np.isfinite(frame_peak):
                raise RuntimeError("Frame signal too low for auto exposure.")

            if np.issubdtype(frame_arr.dtype, np.integer):
                frame_scale = float(np.iinfo(frame_arr.dtype).max)
            else:
                frame_scale = float(np.max(np.abs(frame_arr.astype(np.float32))))
                if not np.isfinite(frame_scale) or frame_scale <= 0.0:
                    frame_scale = frame_peak
            frame_fraction = frame_peak / max(frame_scale, 1e-6)
            raw_arr = None
            raw_metrics = None
            raw_scale = frame_scale
            raw_peak_max = frame_peak
            if raw_frame is not None and getattr(raw_frame, "size", 0) > 0:
                raw_arr = np.asarray(raw_frame)
                if np.issubdtype(raw_arr.dtype, np.integer):
                    raw_scale = float(np.iinfo(raw_arr.dtype).max)
                else:
                    raw_scale = float(np.max(np.abs(raw_arr.astype(np.float32))))
                    if not np.isfinite(raw_scale) or raw_scale <= 0.0:
                        raw_scale = float(np.max(np.abs(raw_arr)))
                    if not np.isfinite(raw_scale) or raw_scale <= 0.0:
                        raw_scale = frame_scale
                raw_metrics = self._signal_peak_metrics(raw_arr)
                raw_peak_max = max(raw_peak_max, raw_metrics.get("max", frame_peak))

            sensor_arr = None
            sensor_scale = None
            sensor_metrics: Optional[Dict[str, float]] = None
            if sensor_frame is not None and getattr(sensor_frame, "size", 0) > 0:
                sensor_arr = np.asarray(sensor_frame)
                try:
                    sensor_scale = self._ensure_sensor_max_value(sensor_arr)
                except Exception as exc:
                    self._logger.debug(f"Sensor max calibration failed: {exc}")
                    sensor_scale = None
                if sensor_scale is not None and (not np.isfinite(sensor_scale) or sensor_scale <= 0.0):
                    sensor_scale = None
                if sensor_scale is not None:
                    sensor_metrics = self._signal_peak_metrics(sensor_arr)

            using_sensor = sensor_scale is not None and sensor_scale > 0.0 and sensor_metrics is not None

            if using_sensor:
                control_metrics = sensor_metrics
                control_scale = float(sensor_scale)
                raw_peak_max = sensor_metrics.get("max", raw_peak_max)
            elif raw_metrics is not None:
                control_metrics = raw_metrics
                control_scale = float(raw_scale)
            else:
                control_metrics = None
                control_scale = float(raw_scale)

            control_peak = float(control_metrics["robust"]) if control_metrics and control_metrics.get("robust", 0.0) > 0.0 else frame_peak
            if control_peak <= 0.0:
                control_peak = frame_peak
                control_scale = frame_scale

            control_scale = max(control_scale, 1e-6)
            control_fraction = control_peak / control_scale
            if control_fraction <= 0.0:
                control_fraction = frame_fraction
                control_peak = frame_peak
                control_scale = frame_scale

            sensor_peak = sensor_metrics.get("max") if using_sensor and sensor_metrics else 0.0
            sensor_peak_robust = sensor_metrics.get("robust") if using_sensor and sensor_metrics else 0.0
            sensor_fraction = (
                (sensor_peak_robust or 0.0) / max(float(sensor_scale or 1.0), 1e-6)
                if using_sensor and sensor_metrics
                else None
            )
            raw_peak_robust = control_peak
            raw_fraction = control_fraction
            if not using_sensor and raw_metrics is not None:
                sensor_peak = 0.0
                raw_peak_max = raw_metrics.get("max", raw_peak_max)
            elif not using_sensor:
                sensor_peak = 0.0
                raw_peak_max = raw_peak_max
            raw_cluster_size = (
                int(sensor_metrics.get("cluster_size", 0)) if using_sensor and sensor_metrics else int(raw_metrics.get("cluster_size", 0) if raw_metrics else 0)
            )
            raw_dropped = (
                int(sensor_metrics.get("dropped", 0)) if using_sensor and sensor_metrics else int(raw_metrics.get("dropped", 0) if raw_metrics else 0)
            )

            current_exp = max(1, self.get_exposure_us())
            min_exp = 10.0
            max_exp = 1_000_000.0
            inc = None
            try:
                feat = self._get_exposure_feature()
                try:
                    min_exp, max_exp = feat.get_range()
                except (VmbFeatureError, TypeError, ValueError):
                    min_exp, max_exp = 10.0, 1_000_000.0
                try:
                    inc = feat.get_increment()
                except (VmbFeatureError, AttributeError):
                    inc = None
            except Exception:
                feat = None

            min_exp = float(min_exp)
            max_exp = float(max_exp)

            factor = target_fraction / max(raw_fraction, 1e-6)
            unclamped_exp = float(current_exp) * factor
            desired_exp = unclamped_exp
            desired_exp = max(min_exp, min(max_exp, desired_exp))
            if inc:
                desired_exp = _nearest_with_increment(desired_exp, min_exp, max_exp, inc)
            desired_int = int(round(desired_exp))

            applied = self.set_exposure_us(desired_int)
            changed = applied != current_exp
            tol = inc if inc else 1.0
            reduction_requested = factor < 0.999
            increase_requested = factor > 1.001
            clamped_lower = unclamped_exp <= min_exp + tol
            clamped_upper = unclamped_exp >= max_exp - tol
            hit_lower = clamped_lower or (
                reduction_requested and (float(applied) <= min_exp + tol or not changed)
            )
            hit_upper = clamped_upper or (
                increase_requested and (float(applied) >= max_exp - tol or not changed)
            )

            expected_fraction = raw_fraction * (float(applied) / max(1.0, float(current_exp)))
            expected_fraction = min(1.0, expected_fraction)
            delta = abs(expected_fraction - target_fraction)

            gain_auto_feat = self._get_gain_auto_feature()
            gain_auto_available = gain_auto_feat is not None
            gain_triggered = False

            if delta > tolerance and gain_auto_available and (hit_lower or hit_upper):
                gain_triggered = self._trigger_gain_auto_once()

            peak_label = "sensor robust peak" if using_sensor else "raw robust peak"
            display_scale = control_scale

            if delta <= tolerance:
                message = "Exposure already within target range."
            elif changed:
                message = (
                    f"Exposure adjusted from {current_exp}us to {applied}us "
                    f"({peak_label} {raw_peak_robust:.1f}/{display_scale:.0f})."
                )
            else:
                limit = "minimum" if hit_lower else "maximum"
                message = (
                    f"Exposure change limited by camera {limit} ({peak_label} {raw_peak_robust:.1f}/{display_scale:.0f})."
                )
            if raw_dropped > 0:
                suffix = "s" if raw_dropped != 1 else ""
                message += f" Ignored {raw_dropped} hot pixel{suffix}."
            if gain_triggered:
                message += " Gain auto was triggered."
            elif gain_auto_available and delta > tolerance and (hit_lower or hit_upper):
                message += " Gain auto unavailable or failed."

            result = {
                "value": int(applied),
                "previous_value": int(current_exp),
                "target_fraction": float(target_fraction),
                "frame_peak": float(frame_peak),
                "frame_peak_fraction": float(frame_fraction),
                "frame_peak_raw": float(raw_peak_max),
                "frame_peak_raw_robust": float(raw_peak_robust),
                "frame_peak_raw_cluster_size": int(raw_cluster_size),
                "frame_peak_raw_hot_pixels_ignored": int(raw_dropped),
                "frame_peak_raw_fraction": float(raw_fraction),
                "sensor_peak": float(sensor_peak) if using_sensor else None,
                "sensor_peak_robust": float(sensor_peak_robust) if using_sensor else None,
                "sensor_peak_cluster_size": int(sensor_metrics.get("cluster_size", 0)) if using_sensor and sensor_metrics else None,
                "sensor_hot_pixels_ignored": int(sensor_metrics.get("dropped", 0)) if using_sensor and sensor_metrics else None,
                "sensor_peak_fraction": float(sensor_fraction) if sensor_fraction is not None else None,
                "sensor_max_value": float(sensor_scale) if sensor_scale is not None else None,
                "expected_fraction": float(expected_fraction),
                "changed": bool(changed),
                "gain_auto_available": bool(gain_auto_available),
                "gain_auto_triggered": bool(gain_triggered),
                "message": message,
                "min_exposure": float(min_exp),
                "max_exposure": float(max_exp),
            }
            return result
        finally:
            self._maybe_disable_sensor_capture(force=True)

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

    def capture_background(self, num_frames: int) -> np.ndarray:
        if num_frames <= 0:
            raise ValueError("num_frames must be positive")

        acc_frame: Optional[np.ndarray] = None
        captured = 0
        try:
            while captured < num_frames:
                if not self._running:
                    raise RuntimeError("Camera not running")
                got_signal = self.wait_for_frame_signal(timeout=0.2)
                if not got_signal:
                    continue

                sensor_sample = self._fetch_sensor_frame(timeout=0.25)
                frame_f32: Optional[np.ndarray] = None

                if sensor_sample is not None and getattr(sensor_sample, "size", 0) > 0:
                    sensor_arr = np.asarray(sensor_sample)
                    if sensor_arr.ndim == 3 and sensor_arr.shape[-1] == 1:
                        sensor_arr = sensor_arr[..., 0]
                    elif sensor_arr.ndim == 3 and sensor_arr.shape[-1] >= 3:
                        sensor_arr = sensor_arr.max(axis=2)
                    sensor_arr = np.asarray(sensor_arr)
                    if sensor_arr.size > 0:
                        if np.issubdtype(sensor_arr.dtype, np.integer):
                            scale = float(np.iinfo(sensor_arr.dtype).max)
                        else:
                            scale = float(np.max(np.abs(sensor_arr.astype(np.float32, copy=False))))
                        if not np.isfinite(scale) or scale <= 0.0:
                            scale = float(np.max(sensor_arr))
                        if not np.isfinite(scale) or scale <= 0.0:
                            scale = 1.0
                        frame_f32 = (sensor_arr.astype(np.float32, copy=False) / scale) * 255.0
                        frame_f32 = np.clip(frame_f32, 0.0, 255.0)

                if frame_f32 is None:
                    latest_gray = self.get_latest_gray(raw=True)
                    if latest_gray is None or getattr(latest_gray, "size", 0) == 0:
                        continue
                    frame_f32 = latest_gray.astype(np.float32, copy=False)

                if acc_frame is None:
                    acc_frame = frame_f32.copy()
                else:
                    acc_frame += frame_f32
                captured += 1
        finally:
            self._maybe_disable_sensor_capture(force=True)

        if acc_frame is None or captured == 0:
            raise RuntimeError("Could not capture any background frames.")

        avg_frame_float = acc_frame / float(captured)

        with self._frame_lock:
            self._background_frame = avg_frame_float

        self._logger.info(f"Background frame captured from {captured} frames.")
        return np.clip(avg_frame_float, 0.0, 255.0).astype(np.uint8)

    def start_background_subtraction(self, num_frames: int) -> np.ndarray:
        preview = self.capture_background(num_frames)
        self.enable_background_subtraction()
        return preview

    def enable_background_subtraction(self) -> None:
        with self._frame_lock:
            if self._background_frame is None:
                raise RuntimeError("No background frame captured.")
            self._background_subtraction_enabled = True
        self._logger.info("Background subtraction enabled.")

    def disable_background_subtraction(self, *, clear: bool = False) -> None:
        with self._frame_lock:
            self._background_subtraction_enabled = False
            had_frame = self._background_frame is not None
            if clear:
                self._background_frame = None
        if clear and had_frame:
            self._logger.info("Background subtraction disabled and frame cleared.")
        else:
            self._logger.info("Background subtraction disabled.")

    def stop_background_subtraction(self, *, clear: bool = True):
        self.disable_background_subtraction(clear=clear)

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

    def get_latest_sensor_raw(self) -> Optional[np.ndarray]:
        with self._frame_lock:
            if self._latest_sensor_raw is None:
                return None
            return self._latest_sensor_raw.copy()

    def get_background_frame(self) -> Optional[np.ndarray]:
        with self._frame_lock:
            if self._background_frame is None:
                return None
            return self._background_frame.copy()

    def is_background_subtraction_enabled(self) -> bool:
        with self._frame_lock:
            return self._background_subtraction_enabled

    def has_background_frame(self) -> bool:
        with self._frame_lock:
            return self._background_frame is not None

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
