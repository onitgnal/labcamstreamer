import logging
import threading
import time
from typing import Callable, Dict, Iterable, Optional, Tuple, Union

import cv2
import numpy as np
try:
    from vmbpy import Frame, FrameStatus, PixelFormat
except ImportError:
    # Create dummy classes if vmbpy is not available
    class Frame: pass
    class FrameStatus:
        Complete = "Complete"
    class PixelFormat:
        Bgr8 = "Bgr8"
        Mono16 = "Mono16"
        Mono8 = "Mono8"

# ----- Mock VmbPy objects for API compatibility -----

class FakeFeature:
    """A mock VmbPy Feature for controlling camera settings."""
    def __init__(
        self,
        name: str,
        initial_value: float,
        min_value: float,
        max_value: float,
        increment: float = 1.0,
        on_change: Optional[Callable[[float], None]] = None,
    ):
        self._name = name
        self._value = initial_value
        self._min = min_value
        self._max = max_value
        self._inc = increment
        self._on_change = on_change

    def get(self) -> float:
        return self._value

    def set(self, value: float) -> None:
        clamped = max(self._min, min(self._max, value))
        if self._inc > 0:
            steps = round((clamped - self._min) / self._inc)
            clamped = self._min + steps * self._inc
            clamped = max(self._min, min(self._max, clamped))
        self._value = clamped
        if self._on_change:
            self._on_change(self._value)

    def get_range(self) -> Tuple[float, float]:
        return self._min, self._max

    def get_increment(self) -> float:
        return self._inc


class FakeEnumFeature:
    """Simple enum-like feature for GainAuto behaviour."""
    def __init__(
        self,
        name: str,
        values: Iterable[Union[str, int]],
        initial_value: Union[str, int],
        on_change: Optional[Callable[[Union[str, int]], None]] = None,
    ):
        self._name = name
        self._values = tuple(values)
        if initial_value not in self._values:
            raise ValueError(f"{initial_value} not valid for {name}")
        self._value = initial_value
        self._on_change = on_change

    def get(self) -> Union[str, int]:
        return self._value

    def set(self, value: Union[str, int]) -> None:
        if value not in self._values:
            raise ValueError(f"{value} not valid for {self._name}")
        self._value = value
        if self._on_change:
            self._on_change(value)

    def get_values(self) -> Tuple[Union[str, int], ...]:
        return self._values

    def set_internal(self, value: Union[str, int]) -> None:
        if value not in self._values:
            raise ValueError(f"{value} not valid for {self._name}")
        self._value = value

class FakeFrame(Frame):
    """
    A mock VmbPy Frame that wraps a NumPy array.
    This allows the FakeCamera to be used by the existing CameraService handler.
    """
    def __init__(self, frame_data: np.ndarray, pixel_format: PixelFormat = PixelFormat.Mono16):
        # This is a mock object, so we don't call super().__init__()
        self._data = frame_data
        self._format = pixel_format
        self._status = FrameStatus.Complete

    def get_status(self) -> FrameStatus:
        return self._status

    def get_pixel_format(self) -> PixelFormat:
        return self._format

    def as_numpy_ndarray(self) -> np.ndarray:
        """Return a copy of the underlying frame data as a NumPy array."""
        return self._data.copy()

    def as_opencv_image(self) -> np.ndarray:
        # Note: This is simplified. A real implementation would need to handle
        # various pixel format conversions more robustly.
        if self._format == PixelFormat.Bgr8:
            return self._data
        elif self._format == PixelFormat.Mono16:
            # Fixed 16-bit to 8-bit scaling to avoid frame-to-frame flicker
            gray8 = (self._data >> 8).astype(np.uint8)
            return cv2.cvtColor(gray8, cv2.COLOR_GRAY2BGR)
        elif self._format == PixelFormat.Mono8:
            return cv2.cvtColor(self._data, cv2.COLOR_GRAY2BGR)
        else:
            raise TypeError(f"Unsupported pixel format for OpenCV conversion: {self._format}")

    def convert_pixel_format(self, target_format: PixelFormat) -> 'FakeFrame':
        if self._format == target_format:
            return self
        if target_format == PixelFormat.Bgr8:
            # This is the primary conversion needed by the app
            img = self.as_opencv_image()
            return FakeFrame(img, PixelFormat.Bgr8)
        else:
            # For simplicity, we don't support other conversions
            raise TypeError(f"Unsupported pixel format conversion to {target_format}")

# ----- The main FakeCamera class -----

class FakeCamera:
    """
    A synthetic image generator that mimics the VmbPy Camera API.
    It produces a stream of 16-bit monochrome frames with simulated
    laser beam profiles, noise, and other artifacts.
    """
    def __init__(self, camera_id: str, logger: logging.Logger):
        self._id = camera_id
        self._logger = logger
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._handler: Optional[Callable] = None
        self._control_lock = threading.RLock()

        # --- Configurable Parameters ---
        self.width = 1280
        self.height = 1024
        self.fps = 20
        self.random_seed = int(time.time())
        self._rng = np.random.default_rng(self.random_seed)

        # Beam characteristics (FWHM in pixels for UI friendliness)
        self._beam_center = (self.width / 2.0, self.height / 2.0)
        self._beam_fwhm = 80.0  # pixels (full width at half maximum)
        self._beam_peak = 18000.0  # intensity before exposure/gain scaling
        self._beam_fluctuation = 0.05  # relative standard deviation per frame

        # Background controls
        self._background_level = 600.0
        self._background_gradient_strength = 500.0
        self._background_inhomogeneity_strength = 350.0

        # Noise controls
        self._shot_noise_scale = 3.0
        self._read_noise_std = 12.0

        # Gain / exposure simulation
        self._exposure_reference = 20000.0  # microseconds for unit gain reference
        self._gain_db = 0.0
        self._gain_limits = (0.0, 24.0)
        self._gain_auto_pending = False

        # Pre-compute static maps for backgrounds
        self._gradient_map = self._create_gradient_map()
        self._inhomogeneity_map = self._create_inhomogeneity_map()

        # --- Feature map ---
        self._features = {
            "ExposureTime": FakeFeature(
                "ExposureTime",
                10000.0,
                1000.0,
                100000.0,
                increment=100.0,
                on_change=self._on_exposure_change,
            ),
            "Gain": FakeFeature(
                "Gain",
                self._gain_db,
                self._gain_limits[0],
                self._gain_limits[1],
                increment=0.1,
                on_change=self._on_gain_change,
            ),
            "GainAuto": FakeEnumFeature(
                "GainAuto",
                ("Off", "Once"),
                "Off",
                on_change=self._on_gain_auto_change,
            ),
        }

        # Derived noise calibration
        self._recompute_noise_model()

    # ----- Feature callbacks -----

    def _on_exposure_change(self, _value: float) -> None:
        self._recompute_noise_model()

    def _on_gain_change(self, value: float) -> None:
        with self._control_lock:
            self._gain_db = float(value)

    def _on_gain_auto_change(self, value: Union[str, int]) -> None:
        if isinstance(value, str) and value.lower() == "once":
            with self._control_lock:
                self._gain_auto_pending = True

    # ----- Public configuration helpers -----

    def get_control_snapshot(self) -> Dict[str, float]:
        with self._control_lock:
            return {
                "beam_fwhm": float(self._beam_fwhm),
                "beam_peak": float(self._beam_peak),
                "beam_fluctuation": float(self._beam_fluctuation),
                "background_level": float(self._background_level),
                "background_gradient": float(self._background_gradient_strength),
                "background_inhomogeneity": float(self._background_inhomogeneity_strength),
                "gain_db": float(self._gain_db),
                "exposure_us": float(self._features["ExposureTime"].get()),
            }

    def update_controls(
        self,
        *,
        beam_fwhm: Optional[float] = None,
        beam_peak: Optional[float] = None,
        beam_fluctuation: Optional[float] = None,
        background_level: Optional[float] = None,
        background_gradient: Optional[float] = None,
        background_inhomogeneity: Optional[float] = None,
    ) -> Dict[str, float]:
        with self._control_lock:
            if beam_fwhm is not None:
                self._beam_fwhm = float(np.clip(beam_fwhm, 6.0, min(self.width, self.height)))
            if beam_peak is not None:
                self._beam_peak = float(np.clip(beam_peak, 100.0, 60000.0))
            if beam_fluctuation is not None:
                self._beam_fluctuation = float(np.clip(beam_fluctuation, 0.0, 0.5))
            if background_level is not None:
                self._background_level = float(np.clip(background_level, 0.0, 5000.0))
            if background_gradient is not None:
                self._background_gradient_strength = float(np.clip(background_gradient, 0.0, 5000.0))
            if background_inhomogeneity is not None:
                self._background_inhomogeneity_strength = float(np.clip(background_inhomogeneity, 0.0, 5000.0))

            self._recompute_noise_model()
            return self.get_control_snapshot()

    def get_id(self) -> str:
        return self._id

    def get_pixel_formats(self):
        return [PixelFormat.Mono16, PixelFormat.Bgr8]

    def set_pixel_format(self, fmt: PixelFormat):
        self._logger.info(f"FakeCamera: Pixel format set to {fmt.name}")

    def get_feature_by_name(self, name: str):
        if name in self._features:
            return self._features[name]
        if "ExposureTime" in name:
            return self._features["ExposureTime"]
        if name == "GainAuto":
            return self._features["GainAuto"]
        if name == "Gain":
            return self._features["Gain"]
        raise AttributeError(f"Feature '{name}' not found in FakeCamera")

    def start_streaming(self, handler: Callable, buffer_count: int = 5):
        if self._running:
            self._logger.warning("FakeCamera: Start streaming called but already running.")
            return
        self._logger.info("FakeCamera: Starting stream...")
        self._handler = handler
        self._running = True
        with self._control_lock:
            self._gain_auto_pending = False
        self._thread = threading.Thread(target=self._generate_frames, daemon=True)
        self._thread.start()

    def stop_streaming(self):
        if not self._running:
            return
        self._logger.info("FakeCamera: Stopping stream...")
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        self._thread = None
        self._handler = None
        self._logger.info("FakeCamera: Stream stopped.")

    @staticmethod
    def _fwhm_to_sigma(fwhm_px: float) -> float:
        return max(1.0, float(fwhm_px) / (2.0 * np.sqrt(2.0 * np.log(2.0))))

    def _create_gradient_map(self) -> np.ndarray:
        y_coords, x_coords = np.mgrid[0:self.height, 0:self.width]
        gradient = (x_coords / max(1, self.width - 1) + y_coords / max(1, self.height - 1)) * 0.5
        return gradient.astype(np.float32)

    def _create_inhomogeneity_map(self) -> np.ndarray:
        base_noise = self._rng.normal(0.0, 1.0, (self.height, self.width)).astype(np.float32)
        smooth = cv2.GaussianBlur(base_noise, (0, 0), sigmaX=25.0, sigmaY=25.0, borderType=cv2.BORDER_REFLECT)
        smooth -= smooth.min()
        denom = smooth.max() - smooth.min()
        if denom <= 0:
            denom = 1.0
        smooth /= denom
        return smooth

    def _recompute_noise_model(self) -> None:
        with self._control_lock:
            exp_us = float(self._features["ExposureTime"].get())
            exposure_factor = max(exp_us / self._exposure_reference, 1e-4)
            expected_peak = self._beam_peak * exposure_factor
            baseline = self._background_level * exposure_factor
            combined = max(1.0, expected_peak + baseline)
            self._shot_noise_scale = np.sqrt(combined) * 0.15
            self._read_noise_std = max(3.0, combined * 0.002)

    def _generate_frames(self):
        frame_duration = 1.0 / self.fps

        while self._running:
            start_time = time.monotonic()

            with self._control_lock:
                exposure_us = float(self._features["ExposureTime"].get())
                gain_db = float(self._gain_db)
                beam_fwhm = float(self._beam_fwhm)
                base_peak = float(self._beam_peak)
                fluctuation = float(self._beam_fluctuation)
                background_level = float(self._background_level)
                gradient_strength = float(self._background_gradient_strength)
                inhom_strength = float(self._background_inhomogeneity_strength)
                perform_gain_auto = self._gain_auto_pending
                if perform_gain_auto:
                    self._gain_auto_pending = False

            if perform_gain_auto:
                self._auto_tune_gain(exposure_us)
                with self._control_lock:
                    self._features["GainAuto"].set_internal("Off")

            exposure_factor = max(exposure_us / self._exposure_reference, 1e-6)
            gain_factor = 10.0 ** (gain_db / 20.0)

            # Base background
            background = (
                background_level
                + gradient_strength * self._gradient_map
                + inhom_strength * self._inhomogeneity_map
            ).astype(np.float32)

            # Gaussian beam rendering (isotropic)
            sigma = self._fwhm_to_sigma(beam_fwhm)
            patch_radius = int(max(6.0, sigma * 3.5))
            cx, cy = self._beam_center
            x_start = max(0, int(cx - patch_radius))
            x_end = min(self.width, int(cx + patch_radius + 1))
            y_start = max(0, int(cy - patch_radius))
            y_end = min(self.height, int(cy + patch_radius + 1))

            if x_start >= x_end or y_start >= y_end:
                signal = np.zeros_like(background)
            else:
                y_patch, x_patch = np.mgrid[y_start:y_end, x_start:x_end]
                dx = x_patch - cx
                dy = y_patch - cy
                exponent = -0.5 * ((dx * dx + dy * dy) / (sigma * sigma))
                frame_peak = base_peak * max(0.0, 1.0 + self._rng.normal(0.0, fluctuation))
                # enforce positive intensity
                frame_peak = max(0.0, frame_peak)
                gaussian_patch = frame_peak * np.exp(exponent)
                signal = np.zeros_like(background)
                signal[y_start:y_end, x_start:x_end] += gaussian_patch.astype(np.float32)

            ideal_frame = (background + signal) * exposure_factor * gain_factor

            # Noise simulation (shot + read)
            shot_noise = self._rng.normal(0.0, self._shot_noise_scale, ideal_frame.shape).astype(np.float32)
            read_noise = self._rng.normal(0.0, self._read_noise_std, ideal_frame.shape).astype(np.float32)
            noisy_frame = ideal_frame + shot_noise * np.sqrt(np.clip(ideal_frame, 0.0, None)) + read_noise

            final_frame = np.clip(noisy_frame, 0, 65535).astype(np.uint16)

            # --- Frame delivery ---
            if self._handler:
                try:
                    mock_frame = FakeFrame(final_frame, PixelFormat.Mono16)
                    self._handler(self, None, mock_frame)
                except Exception as e:
                    self._logger.error(f"FakeCamera: Error in handler: {e}", exc_info=True)
                    self.stop_streaming()
                    break

            # --- Maintain FPS ---
            elapsed = time.monotonic() - start_time
            sleep_duration = max(0, frame_duration - elapsed)
            time.sleep(sleep_duration)

    def _auto_tune_gain(self, exposure_us: float) -> None:
        with self._control_lock:
            current_gain = self._gain_db
            current_peak = self._beam_peak
        exposure_factor = max(exposure_us / self._exposure_reference, 1e-6)
        linear_signal = current_peak * exposure_factor
        if linear_signal <= 0.0:
            target_gain = self._gain_limits[0]
        else:
            target_fraction = 0.65  # aim for 65% of full scale
            desired_linear = target_fraction * 65535.0
            gain_linear = desired_linear / linear_signal
            gain_linear = max(10 ** (self._gain_limits[0] / 20.0), min(10 ** (self._gain_limits[1] / 20.0), gain_linear))
            target_gain = 20.0 * np.log10(gain_linear)
        self._features["Gain"].set(target_gain)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_streaming()

    def generate_background_frame(self) -> np.ndarray:
        """Generates a noise-only frame as a NumPy array."""
        with self._control_lock:
            exposure_us = float(self._features["ExposureTime"].get())
            gain_db = float(self._gain_db)
            background_level = float(self._background_level)
            gradient_strength = float(self._background_gradient_strength)
            inhom_strength = float(self._background_inhomogeneity_strength)

        exposure_factor = max(exposure_us / self._exposure_reference, 1e-6)
        gain_factor = 10.0 ** (gain_db / 20.0)
        background = (
            background_level
            + gradient_strength * self._gradient_map
            + inhom_strength * self._inhomogeneity_map
        ).astype(np.float32)

        ideal_frame = background * exposure_factor * gain_factor
        shot_noise = self._rng.normal(0.0, self._shot_noise_scale, ideal_frame.shape).astype(np.float32)
        read_noise = self._rng.normal(0.0, self._read_noise_std, ideal_frame.shape).astype(np.float32)
        noisy_frame = ideal_frame + shot_noise * np.sqrt(np.clip(ideal_frame, 0.0, None)) + read_noise
        return np.clip(noisy_frame, 0, 65535).astype(np.uint16)

# Example usage (for testing)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("FakeCamTest")

    # The handler would be CameraService._Handler in the real app
    def test_handler(cam, stream, frame):
        logger.info(f"Received frame with status: {frame.get_status()}")
        img = frame.as_opencv_image()
        logger.info(f"Image shape: {img.shape}, dtype: {img.dtype}")
        cv2.imshow("Fake Camera", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cam.stop_streaming()

    fake_cam = FakeCamera("fake_test_cam", logger)
    try:
        fake_cam.start_streaming(handler=test_handler)
        # In a real app, the camera runs in the background. Here we just wait.
        while fake_cam._running:
            time.sleep(0.1)
    except KeyboardInterrupt:
        logger.info("Interrupted, shutting down.")
    finally:
        fake_cam.stop_streaming()
        cv2.destroyAllWindows()
        logger.info("Test finished.")
