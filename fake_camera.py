import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

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

# ----- MPC parameter modeling -----

@dataclass(frozen=True)
class MPCParameters:
    """Design parameters for the synthetic Herriott-type multi-pass cell."""
    k: int = 1
    N: int = 24
    R: float = 0.5               # meters
    wavelength: float = 1030e-9  # meters
    n: float = 1.0               # refractive index

    def validate(self) -> "MPCParameters":
        if self.N < 2:
            raise ValueError("N must be at least 2.")
        if self.N > 400:
            raise ValueError("N must be <= 400 to keep the MPC simulation responsive.")
        if self.k < 1 or self.k >= self.N:
            raise ValueError("k must be between 1 and N-1.")
        if self.R <= 0:
            raise ValueError("R must be positive.")
        if self.wavelength <= 0:
            raise ValueError("Wavelength must be positive.")
        if self.n <= 0:
            raise ValueError("Refractive index n must be positive.")
        C = 1.0 - math.cos(math.pi * self.k / self.N)
        if not (0.0 < C < 2.0):
            raise ValueError("The selected k and N combination would make the cell unstable (C outside (0, 2)).")
        return self

    def with_updates(self, **kwargs: object) -> "MPCParameters":
        data = {
            "k": self.k,
            "N": self.N,
            "R": self.R,
            "wavelength": self.wavelength,
            "n": self.n,
        }

        if "k" in kwargs and kwargs["k"] is not None:
            try:
                data["k"] = int(kwargs["k"])
            except (TypeError, ValueError):
                raise ValueError("k must be an integer.")

        if "N" in kwargs and kwargs["N"] is not None:
            try:
                data["N"] = int(kwargs["N"])
            except (TypeError, ValueError):
                raise ValueError("N must be an integer.")

        if "R" in kwargs and kwargs["R"] is not None:
            try:
                data["R"] = float(kwargs["R"])
            except (TypeError, ValueError):
                raise ValueError("R must be a number.")

        wavelength_keys = ("wavelength_nm", "lambda_nm", "wavelength", "lambda")
        for key in wavelength_keys:
            if key in kwargs and kwargs[key] is not None:
                try:
                    if key.endswith("_nm"):
                        data["wavelength"] = float(kwargs[key]) * 1e-9
                    else:
                        data["wavelength"] = float(kwargs[key])
                except (TypeError, ValueError):
                    raise ValueError("Wavelength must be a number.")
                break

        if "n" in kwargs and kwargs["n"] is not None:
            try:
                data["n"] = float(kwargs["n"])
            except (TypeError, ValueError):
                raise ValueError("n must be a number.")

        return MPCParameters(**data).validate()

    def derived(self) -> Dict[str, float]:
        params = self.validate()
        xi = 2.0 * math.pi * params.k / params.N
        C = 1.0 - math.cos(math.pi * params.k / params.N)
        lambda_medium = params.wavelength / params.n
        if lambda_medium <= 0:
            raise ValueError("The effective wavelength λ/n must be positive.")

        stability_term = max(C * (2.0 - C), 0.0)
        sqrt_term = math.sqrt(stability_term)

        zr = 0.5 * params.R * sqrt_term
        w0_sq = (params.R * lambda_medium / (2.0 * math.pi)) * sqrt_term
        w0 = math.sqrt(max(w0_sq, 0.0))

        denom = 2.0 - C
        if denom <= 0.0:
            raise ValueError("Derived denominator became non-positive; check k/N.")
        sqrt_ratio = math.sqrt(max(C / denom, 0.0))
        wm_sq = (params.R * lambda_medium / math.pi) * sqrt_ratio
        wm = math.sqrt(max(wm_sq, 0.0))

        return {
            "xi_rad": float(xi),
            "C": float(C),
            "L_m": float(params.R * C),
            "lambda_medium_m": float(lambda_medium),
            "zr_m": float(zr),
            "w0_m": float(w0),
            "wm_m": float(wm),
        }

    def to_payload(self) -> Dict[str, object]:
        derived = self.derived()
        return {
            "k": self.k,
            "N": self.N,
            "R": float(self.R),
            "wavelength_m": float(self.wavelength),
            "wavelength_nm": float(self.wavelength * 1e9),
            "n": float(self.n),
            "derived": derived,
        }

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
        self.drift_position_std_dev = 0.5
        self.max_amplitude_variation = 0.05  # +/-5 % envelope

        # Noise parameters
        self.background_noise_fraction = 0.5  # relative to short-exposure peak
        self.read_noise_fraction = 0.2

        # --- Internal State ---
        self._rng = np.random.default_rng(self.random_seed)
        self._beams = []
        self._static_gradient: Optional[np.ndarray] = None
        self._features = {
            "ExposureTime": FakeFeature("ExposureTime", 10000.0, 1000.0, 50000.0),
        }

        min_exp, _ = self._features["ExposureTime"].get_range()
        self._min_exposure_us = float(min_exp)
        short_exposure_factor = self._min_exposure_us / 20000.0
        estimated_peak = self.amplitude_range[1] * short_exposure_factor
        self.background_noise_std = max(1.0, estimated_peak * self.background_noise_fraction)
        self.read_noise_level = max(1.0, estimated_peak * self.read_noise_fraction)

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
        """Initializes beams with random positions and shape parameters."""
        self._beams = []
        num_beams = 3
        margin_x = self.width * 0.1
        margin_y = self.height * 0.1
        for _ in range(num_beams):
            cx = self._rng.uniform(margin_x, self.width - margin_x)
            cy = self._rng.uniform(margin_y, self.height - margin_y)
            angle = self._rng.uniform(0.0, np.pi)
            sigma_major = self._rng.uniform(self.sigma_range[0], self.sigma_range[1])
            aspect_ratio = self._rng.uniform(0.3, 1.0)
            sigma_minor = sigma_major * aspect_ratio
            amp = self._rng.uniform(self.amplitude_range[0], self.amplitude_range[1])
            self._beams.append({
                'center': (cx, cy),
                'amplitude': amp,
                'base_amplitude': amp,
                'cov_matrix': self._create_covariance_matrix(sigma_major, sigma_minor, angle)
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

            base_amp = beam.get('base_amplitude', beam['amplitude'])
            variation = self._rng.uniform(-self.max_amplitude_variation, self.max_amplitude_variation)
            target_amp = base_amp * (1.0 + variation)
            updated = 0.9 * beam['amplitude'] + 0.1 * target_amp
            lower = base_amp * (1.0 - self.max_amplitude_variation)
            upper = base_amp * (1.0 + self.max_amplitude_variation)
            beam['amplitude'] = float(np.clip(updated, lower, upper))

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

            # 5. Noise Simulation (exposure-independent)
            background_noise = self._rng.normal(0.0, self.background_noise_std, ideal_frame.shape).astype(np.float32)
            read_noise = self._rng.normal(0.0, self.read_noise_level, ideal_frame.shape).astype(np.float32)
            noisy_frame = exposed_frame + background_noise + read_noise

            # 6. Convert to final 16-bit format
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

    def generate_background_frame(self) -> np.ndarray:
        """Generates a noise-only frame as a NumPy array."""
        ideal_frame = np.full((self.height, self.width), float(self.background_level), dtype=np.float32)

        if self._static_gradient is not None:
            ideal_frame += self._static_gradient

        exposure_us = self._features["ExposureTime"].get()
        exposure_factor = exposure_us / 20000.0
        exposed_frame = ideal_frame * exposure_factor

        background_noise = self._rng.normal(0.0, self.background_noise_std, ideal_frame.shape).astype(np.float32)
        read_noise = self._rng.normal(0.0, self.read_noise_level, ideal_frame.shape).astype(np.float32)
        noisy_frame = exposed_frame + background_noise + read_noise

        final_frame = np.clip(noisy_frame, 0, 65535).astype(np.uint16)
        return final_frame


class MPCCamera(FakeCamera):
    """
    A specialized FakeCamera that renders a Herriott multi-pass cell spot pattern.
    The user-configurable MPCParameters drive both the simulated beam geometry and
    the derived design metrics that are reported through the API/UI.
    """

    def __init__(self, camera_id: str, logger: logging.Logger, parameters: Optional[MPCParameters] = None):
        super().__init__(camera_id=camera_id, logger=logger)
        self._param_lock = threading.Lock()
        self._parameters = (parameters or MPCParameters()).validate()
        self._derived = self._parameters.derived()
        self._phase = 0.0
        self._phase_increment = self._compute_phase_increment(self._derived["xi_rad"], self._parameters.N)
        self._base_amplitude = (self.amplitude_range[0] + self.amplitude_range[1]) * 0.5
        self.drift_position_std_dev = 0.0
        self.max_amplitude_variation = 0.02
        self.background_gradient_strength = 200.0
        self.background_level = 800
        self.background_noise_fraction = min(self.background_noise_fraction, 0.2)
        self.read_noise_fraction = min(self.read_noise_fraction, 0.1)
        self.background_noise_std = max(1.0, self.background_noise_std * 0.5)
        self.read_noise_level = max(1.0, self.read_noise_level * 0.5)
        self._needs_rebuild = True
        self._recompute_geometry_locked()

    @staticmethod
    def _compute_phase_increment(xi: float, passes: int) -> float:
        if passes <= 0 or abs(xi) < 1e-9:
            return 0.0
        return xi / max(passes, 1) * 0.25

    def _recompute_geometry_locked(self) -> None:
        """Recalculate pixel-space geometry derived from the analytic solution."""
        derived = self._derived
        min_dim = float(min(self.width, self.height))
        w0 = max(derived["w0_m"], 1e-12)
        wm = max(derived["wm_m"], w0)
        ratio = max(1.0, min(wm / w0, 50.0))

        density_scale = min(1.0, 64.0 / max(self._parameters.N, 1))
        radius = min_dim * (0.2 + 0.015 * ratio)
        if density_scale < 1.0:
            radius *= density_scale ** 0.5
        radius = float(max(min_dim * 0.18, min(radius, min_dim * 0.45)))

        sigma = min_dim * 0.01 * math.sqrt(ratio)
        sigma = float(max(4.0, min(sigma, min_dim * 0.09)))

        self._circle_radius_px = radius
        self._sigma_base_px = sigma

    def set_parameters(self, params: MPCParameters) -> None:
        params = params.validate()
        with self._param_lock:
            self._parameters = params
            self._derived = params.derived()
            self._phase = 0.0
            self._phase_increment = self._compute_phase_increment(self._derived["xi_rad"], params.N)
            self._base_amplitude = (self.amplitude_range[0] + self.amplitude_range[1]) * 0.5
            self._recompute_geometry_locked()
            self._needs_rebuild = True

    def get_parameters(self) -> MPCParameters:
        with self._param_lock:
            return self._parameters

    def get_payload(self) -> Dict[str, object]:
        with self._param_lock:
            return self._parameters.to_payload()

    def _initialize_beams(self):
        with self._param_lock:
            params = self._parameters
            derived = self._derived
            radius = self._circle_radius_px
            sigma = self._sigma_base_px
        self._beams = []
        cx, cy = self.width / 2.0, self.height / 2.0
        xi = derived["xi_rad"]
        jitter_scale = sigma * 0.15

        for idx in range(params.N):
            theta = self._phase + idx * xi
            radial_offset = float(self._rng.normal(0.0, jitter_scale * 0.3))
            sigma_major = max(2.0, sigma * (1.05 + 0.1 * math.sin(theta)))
            sigma_minor = max(2.0, sigma * (0.95 - 0.1 * math.sin(theta)))
            angle = 0.5 * theta
            cov = self._create_covariance_matrix(sigma_major, sigma_minor, angle)
            amplitude = self._base_amplitude * (0.85 + 0.15 * math.cos(theta))
            beam = {
                "index": idx,
                "center": (
                    cx + (radius + radial_offset) * math.cos(theta),
                    cy + (radius + radial_offset) * math.sin(theta),
                ),
                "amplitude": float(np.clip(amplitude, 0.0, 60000.0)),
                "base_amplitude": float(np.clip(amplitude, 0.0, 60000.0)),
                "cov_matrix": cov,
                "radial_offset": radial_offset,
            }
            self._beams.append(beam)

        self._needs_rebuild = False

    def _update_beams(self):
        with self._param_lock:
            params = self._parameters
            derived = self._derived
            radius = self._circle_radius_px
            sigma = self._sigma_base_px

        if self._needs_rebuild or len(self._beams) != params.N:
            self._phase = 0.0
            self._initialize_beams()
            return

        xi = derived["xi_rad"]
        self._phase = (self._phase + self._phase_increment) % (2.0 * math.pi)
        cx, cy = self.width / 2.0, self.height / 2.0

        for beam in self._beams:
            idx = beam.get("index", 0)
            theta = self._phase + idx * xi

            radial_offset = beam.get("radial_offset", 0.0)
            new_offset = 0.98 * radial_offset + self._rng.normal(0.0, sigma * 0.01)
            beam["radial_offset"] = float(np.clip(new_offset, -sigma, sigma))

            effective_radius = radius + beam["radial_offset"]
            beam["center"] = (
                cx + effective_radius * math.cos(theta),
                cy + effective_radius * math.sin(theta),
            )

            modulation = 0.1 * math.sin(theta * 2.0)
            beam["amplitude"] = float(np.clip(beam["base_amplitude"] * (1.0 + modulation), 0.0, 60000.0))

            sigma_major = max(2.0, sigma * (1.05 + 0.08 * math.cos(theta + 0.3)))
            sigma_minor = max(2.0, sigma * (0.95 - 0.08 * math.cos(theta + 0.3)))
            angle = 0.5 * theta
            beam["cov_matrix"] = self._create_covariance_matrix(sigma_major, sigma_minor, angle)

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
