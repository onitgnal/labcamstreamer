import logging
import threading
import time
from typing import Callable, Optional, Tuple

import cv2
import numpy as np
from vmbpy import Frame, FrameStatus, PixelFormat

# ----- Mock VmbPy objects for API compatibility -----

class FakeFeature:
    """A mock VmbPy Feature for controlling camera settings."""
    def __init__(self, name: str, initial_value: float, min_value: float, max_value: float, increment: float = 1.0):
        self._name = name
        self._value = initial_value
        self._min = min_value
        self._max = max_value
        self._inc = increment

    def get(self) -> float:
        return self._value

    def set(self, value: float) -> None:
        self._value = max(self._min, min(self._max, value))

    def get_range(self) -> Tuple[float, float]:
        return self._min, self._max

    def get_increment(self) -> float:
        return self._inc

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

    def as_opencv_image(self) -> np.ndarray:
        # Note: This is simplified. A real implementation would need to handle
        # various pixel format conversions more robustly.
        if self._format == PixelFormat.Bgr8:
            return self._data
        elif self._format == PixelFormat.Mono16:
            # Normalize 16-bit to 8-bit for display
            normalized = cv2.normalize(self._data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            return cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
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

        # --- Configurable Parameters ---
        self.width = 1280
        self.height = 1024
        self.fps = 20
        self.random_seed = int(time.time())
        self.num_beams_range = (2, 5)
        self.amplitude_range = (8000, 20000)
        self.sigma_range = (20.0, 50.0)
        self.noise_level = 500.0
        self.background_level = 2000
        self.background_gradient_strength = 4000.0
        self.hot_pixel_prob = 1e-5
        self.drift_position_std_dev = 1.5
        self.drift_amplitude_std_dev = 300.0

        # --- Internal State ---
        self._rng = np.random.default_rng(self.random_seed)
        self._beams = []
        self._static_gradient: Optional[np.ndarray] = None
        self._features = {
            "ExposureTime": FakeFeature("ExposureTime", 10000.0, 1000.0, 50000.0),
            "Gain": FakeFeature("Gain", 0.0, 0.0, 20.0)
        }

    def get_id(self) -> str:
        return self._id

    def get_pixel_formats(self):
        return [PixelFormat.Mono16, PixelFormat.Bgr8]

    def set_pixel_format(self, fmt: PixelFormat):
        # The fake camera always generates Mono16 internally, but we acknowledge the setting.
        self._logger.info(f"FakeCamera: Pixel format set to {fmt.name}")

    def get_feature_by_name(self, name: str) -> FakeFeature:
        if name in self._features:
            return self._features[name]
        # Allow fallback for features like ExposureTimeAbs
        if "ExposureTime" in name:
            return self._features["ExposureTime"]
        raise AttributeError(f"Feature '{name}' not found in FakeCamera")

    def start_streaming(self, handler: Callable, buffer_count: int = 5):
        if self._running:
            self._logger.warning("FakeCamera: Start streaming called but already running.")
            return
        self._logger.info("FakeCamera: Starting stream...")
        self._handler = handler
        self._running = True
        self._initialize_beams()
        self._initialize_static_gradient()
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

    def _initialize_beams(self):
        self._beams = []
        num_beams = self._rng.integers(self.num_beams_range[0], self.num_beams_range[1] + 1)
        for _ in range(num_beams):
            angle = self._rng.uniform(0, np.pi)
            sx = self._rng.uniform(self.sigma_range[0], self.sigma_range[1])
            sy = self._rng.uniform(self.sigma_range[0], self.sigma_range[1]) * 0.7
            self._beams.append({
                'center': (self._rng.uniform(0, self.width), self._rng.uniform(0, self.height)),
                'amplitude': self._rng.uniform(self.amplitude_range[0], self.amplitude_range[1]),
                'cov_matrix': self._create_covariance_matrix(sx, sy, angle)
            })

    def _initialize_static_gradient(self):
        """Generates a low-frequency gradient pattern that remains static for the session."""
        y_coords, x_coords = np.mgrid[0:self.height, 0:self.width]
        # Lower frequency for a smoother, larger-scale gradient.
        # The frequency is proportional to 1/dimension, so the wavelength is on the order of the image size.
        grad_y_freq = self._rng.uniform(0.2, 0.7) / self.height
        grad_x_freq = self._rng.uniform(0.2, 0.7) / self.width
        gradient = (np.sin(y_coords * grad_y_freq * 2 * np.pi) *
                    np.cos(x_coords * grad_x_freq * 2 * np.pi))
        self._static_gradient = gradient * self.background_gradient_strength

    def _create_covariance_matrix(self, sx, sy, angle):
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        S = np.array([[sx**2, 0], [0, sy**2]])
        return R @ S @ R.T

    def _update_beams(self):
        for beam in self._beams:
            # Drift position
            dx = self._rng.normal(0, self.drift_position_std_dev)
            dy = self._rng.normal(0, self.drift_position_std_dev)
            beam['center'] = (beam['center'][0] + dx, beam['center'][1] + dy)
            # Drift amplitude
            da = self._rng.normal(0, self.drift_amplitude_std_dev)
            beam['amplitude'] = max(0, beam['amplitude'] + da)

    def _generate_2d_gaussian(self, x, y, beam):
        center = beam['center']
        amplitude = beam['amplitude']
        cov_inv = np.linalg.inv(beam['cov_matrix'])
        pos = np.dstack((x, y))
        diff = pos - center
        exponent = -0.5 * np.einsum('...k,kl,...l->...', diff, cov_inv, diff)
        return amplitude * np.exp(exponent)

    def _generate_frames(self):
        frame_duration = 1.0 / self.fps
        y_coords, x_coords = np.mgrid[0:self.height, 0:self.width]

        while self._running:
            start_time = time.monotonic()

            # 1. Base Canvas (as float32 for calculations)
            frame = np.full((self.height, self.width), float(self.background_level), dtype=np.float32)

            # 2. Background Noise
            frame += self._rng.normal(0, self.noise_level, frame.shape)

            # 3. Background Gradient
            if self._static_gradient is not None:
                frame += self._static_gradient

            # 4. Synthetic Beams
            self._update_beams()
            for beam in self._beams:
                frame += self._generate_2d_gaussian(x_coords, y_coords, beam)

            # 5. Exposure and Gain Simulation (Non-linear)
            # This section models the non-linear effects of exposure and gain,
            # as requested in the prompt. The simulation makes two assumptions:
            # - Brighter parts of the image (the "signal") are affected more
            #   by changes in exposure and gain than the darker background.
            # - The relationship is non-linear (logarithmic) to better
            #   simulate how real sensors might behave.
            exposure_ms = self._features["ExposureTime"].get() / 1000.0
            gain_db = self._features["Gain"].get()

            # A simple mask to separate "signal" (beams) from "background".
            signal_mask = frame > (self.background_level + self.noise_level * 3)

            # Apply gain factor, scaled to have a larger effect on the signal.
            gain_factor = 1.0 + (10**(gain_db / 20.0) - 1.0) * 0.8
            frame[signal_mask] *= gain_factor

            # Apply exposure factor, with a much smaller effect on the background
            # and a logarithmic response for more realistic saturation behavior.
            exposure_factor_bg = 1.0 + np.log1p(exposure_ms / 10.0) * 0.2
            exposure_factor_signal = 1.0 + np.log1p(exposure_ms / 10.0)
            frame[~signal_mask] *= exposure_factor_bg
            frame[signal_mask] *= exposure_factor_signal

            # 6. Convert to 16-bit and add read noise/hot pixels
            frame = np.clip(frame, 0, 65535).astype(np.uint16)

            # Add a touch of read noise after conversion
            frame += self._rng.normal(0, 5, frame.shape).astype(np.uint16)

            # 7. Hot Pixels
            if self.hot_pixel_prob > 0:
                hot_pixels = self._rng.choice([True, False], size=frame.shape,
                                              p=[self.hot_pixel_prob, 1 - self.hot_pixel_prob])
                frame[hot_pixels] = 65535

            # --- Frame delivery ---
            if self._handler:
                try:
                    # Wrap in a mock Frame object for compatibility
                    mock_frame = FakeFrame(frame, PixelFormat.Mono16)
                    # Pass the frame to the handler (CameraService._Handler)
                    self._handler(self, None, mock_frame)
                except Exception as e:
                    self._logger.error(f"FakeCamera: Error in handler: {e}", exc_info=True)
                    self.stop_streaming()
                    break

            # --- Maintain FPS ---
            elapsed = time.monotonic() - start_time
            sleep_duration = max(0, frame_duration - elapsed)
            time.sleep(sleep_duration)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_streaming()

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
