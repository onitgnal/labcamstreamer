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
        self.amplitude_range = (15000, 30000)
        self.sigma_range = (25.0, 60.0)
        self.background_level = 1000
        self.background_gradient_strength = 500.0
        self.hot_pixel_prob = 1e-6
        self.drift_position_std_dev = 0.5
        self.drift_amplitude_std_dev = 150.0
        # Noise parameters for a more realistic model
        self.shot_noise_factor = 0.8  # Proportional to signal
        self.read_noise_level = 150.0 # Constant

        # --- Internal State ---
        self._rng = np.random.default_rng(self.random_seed)
        self._beams = []
        self._static_gradient: Optional[np.ndarray] = None
        self._features = {
            "ExposureTime": FakeFeature("ExposureTime", 10000.0, 1000.0, 50000.0),
        }

    def get_id(self) -> str:
        return self._id

    def get_pixel_formats(self):
        return [PixelFormat.Mono16, PixelFormat.Bgr8]

    def set_pixel_format(self, fmt: PixelFormat):
        self._logger.info(f"FakeCamera: Pixel format set to {fmt.name}")

    def get_feature_by_name(self, name: str) -> FakeFeature:
        if name in self._features:
            return self._features[name]
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
        """Initializes exactly three beams in predictable locations."""
        self._beams = []
        positions = [
            (self.width * 0.25, self.height * 0.5),
            (self.width * 0.5, self.height * 0.3),
            (self.width * 0.75, self.height * 0.7),
        ]
        for i, pos in enumerate(positions):
            angle = self._rng.uniform(0, np.pi)
            sx = self._rng.uniform(self.sigma_range[0], self.sigma_range[1]) * (1.0 + i * 0.1)
            sy = sx * self._rng.uniform(0.6, 0.9)
            self._beams.append({
                'center': pos,
                'amplitude': self._rng.uniform(self.amplitude_range[0], self.amplitude_range[1]),
                'cov_matrix': self._create_covariance_matrix(sx, sy, angle)
            })

    def _initialize_static_gradient(self):
        """Generates a simpler, static, non-uniform background."""
        y_coords, x_coords = np.mgrid[0:self.height, 0:self.width]
        gradient = (x_coords / self.width + y_coords / self.height) * 0.5
        self._static_gradient = gradient * self.background_gradient_strength

    def _create_covariance_matrix(self, sx, sy, angle):
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        S = np.array([[sx**2, 0], [0, sy**2]])
        return R @ S @ R.T

    def _update_beams(self):
        for beam in self._beams:
            dx = self._rng.normal(0, self.drift_position_std_dev)
            dy = self._rng.normal(0, self.drift_position_std_dev)
            beam['center'] = (beam['center'][0] + dx, beam['center'][1] + dy)
            da = self._rng.normal(0, self.drift_amplitude_std_dev)
            beam['amplitude'] = max(0, beam['amplitude'] + da)

    def _generate_gaussian_patch(self, beam: dict) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Generates a Gaussian beam on a smaller patch for performance."""
        center_x, center_y = beam['center']
        # Estimate sigma from the covariance matrix for patch size calculation
        sx = np.sqrt(abs(beam['cov_matrix'][0, 0]))
        sy = np.sqrt(abs(beam['cov_matrix'][1, 1]))
        patch_half_size = int(3 * max(sx, sy)) # Use a generous patch size

        x_start = max(0, int(center_x - patch_half_size))
        x_end = min(self.width, int(center_x + patch_half_size))
        y_start = max(0, int(center_y - patch_half_size))
        y_end = min(self.height, int(center_y + patch_half_size))

        if x_start >= x_end or y_start >= y_end:
            return np.array([[]]), (0, 0)

        y_patch, x_patch = np.mgrid[y_start:y_end, x_start:x_end]

        cov_inv = np.linalg.inv(beam['cov_matrix'])
        pos = np.dstack((x_patch, y_patch))
        diff = pos - beam['center']
        exponent = -0.5 * np.einsum('...k,kl,...l->...', diff, cov_inv, diff)

        patch = beam['amplitude'] * np.exp(exponent)
        return patch, (y_start, x_start)

    def _generate_frames(self):
        frame_duration = 1.0 / self.fps

        while self._running:
            start_time = time.monotonic()

            # 1. Base Canvas (ideal signal, float32 for calculations)
            ideal_frame = np.full((self.height, self.width), float(self.background_level), dtype=np.float32)

            # 2. Add static background gradient
            if self._static_gradient is not None:
                ideal_frame += self._static_gradient

            # 3. Add synthetic beams (optimized)
            self._update_beams()
            for beam in self._beams:
                patch, (y_start, x_start) = self._generate_gaussian_patch(beam)
                if patch.size > 0:
                    y_end, x_end = y_start + patch.shape[0], x_start + patch.shape[1]
                    ideal_frame[y_start:y_end, x_start:x_end] += patch

            # 4. Exposure Simulation (linear scaling of ideal signal)
            exposure_us = self._features["ExposureTime"].get()
            exposure_factor = exposure_us / 20000.0
            exposed_frame = ideal_frame * exposure_factor

            # 5. Noise Simulation
            shot_noise_variance = np.maximum(0, exposed_frame) * self.shot_noise_factor
            shot_noise = self._rng.normal(0, np.sqrt(shot_noise_variance)).astype(np.float32)
            read_noise = self._rng.normal(0, self.read_noise_level, ideal_frame.shape).astype(np.float32)
            noisy_frame = exposed_frame + shot_noise + read_noise

            # 6. Hot Pixels (applied before final conversion)
            if self.hot_pixel_prob > 0:
                hot_pixels = self._rng.choice([True, False], size=noisy_frame.shape,
                                              p=[self.hot_pixel_prob, 1 - self.hot_pixel_prob])
                noisy_frame[hot_pixels] = 65535

            # 7. Convert to final 16-bit format
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
