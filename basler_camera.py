import logging
import threading
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

try:
    from pypylon import pylon
except ImportError:  # pragma: no cover - optional dependency
    pylon = None  # type: ignore[assignment]

FrameHandler = Callable[[object, Optional[object], object], None]

_FRAME_STATUS_COMPLETE = "Complete"
_PIXEL_MONO8 = "Mono8"
_PIXEL_BGR8 = "Bgr8"


def _normalize_pixel_name(fmt: Union[str, object]) -> str:
    if isinstance(fmt, str):
        return fmt
    name = getattr(fmt, "name", None)
    if isinstance(name, str):
        return name
    return str(fmt)


class BaslerFrame:
    """Minimal frame wrapper so CameraService can treat Basler like VmbPy."""

    def __init__(self, frame_data: np.ndarray, pixel_format: str = _PIXEL_MONO8):
        self._data = np.ascontiguousarray(frame_data)
        self._format = pixel_format
        self._status = _FRAME_STATUS_COMPLETE

    def get_status(self) -> str:
        return self._status

    def get_pixel_format(self) -> str:
        return self._format

    def as_numpy_ndarray(self) -> np.ndarray:
        return self._data.copy()

    def as_opencv_image(self) -> np.ndarray:
        fmt = self._format.lower()
        if fmt == _PIXEL_BGR8.lower():
            return self._data
        if fmt == "rgb8":
            return cv2.cvtColor(self._data, cv2.COLOR_RGB2BGR)
        if fmt in ("mono8", "mono8signed"):
            return cv2.cvtColor(self._data, cv2.COLOR_GRAY2BGR)
        raise TypeError(f"Unsupported pixel format for OpenCV conversion: {self._format}")

    def convert_pixel_format(self, target_format: Union[str, object]) -> "BaslerFrame":
        target = _normalize_pixel_name(target_format).lower()
        if target == _PIXEL_BGR8.lower():
            if self._format.lower() == _PIXEL_BGR8.lower():
                return self
            return BaslerFrame(self.as_opencv_image(), _PIXEL_BGR8)
        raise TypeError(f"Unsupported pixel format conversion to {target_format}")


class _BaslerFloatFeature:
    def __init__(self, parameter):
        self._param = parameter

    def get(self) -> float:
        return float(self._param.GetValue())

    def set(self, value: float) -> None:
        val = float(value)
        try:
            self._param.SetValue(val)
        except TypeError:
            self._param.SetValue(int(round(val)))

    def get_range(self) -> Tuple[float, float]:
        try:
            return float(self._param.GetMin()), float(self._param.GetMax())
        except Exception:
            val = float(self._param.GetValue())
            return val, val

    def get_increment(self) -> float:
        try:
            inc = float(self._param.GetInc())
        except Exception:
            return 0.0
        return inc


class _BaslerEnumFeature:
    def __init__(self, parameter):
        self._param = parameter

    def get(self) -> str:
        return str(self._param.GetValue())

    def set(self, value: Union[str, object]) -> None:
        self._param.SetValue(str(value))

    def get_values(self) -> Tuple[str, ...]:
        try:
            values = self._param.GetSymbolics()
        except Exception:
            values = []
        return tuple(str(v) for v in values)


@dataclass(frozen=True)
class BaslerDeviceInfo:
    serial: str
    model: str
    friendly: str

    @property
    def id(self) -> str:
        return f"basler:{self.serial}" if self.serial else "basler"

    @property
    def label(self) -> str:
        base = self.model or self.friendly or "Basler Camera"
        serial = self.serial or "unknown"
        return f"{base} ({serial})"


def _enumerate_pylon_devices() -> Sequence["pylon.CDeviceInfo"]:
    if pylon is None:
        return []
    factory = pylon.TlFactory.GetInstance()
    try:
        return factory.EnumerateDevices()
    except Exception:
        return []


def list_basler_devices() -> List[BaslerDeviceInfo]:
    devices: List[BaslerDeviceInfo] = []
    for info in _enumerate_pylon_devices():
        serial = ""
        model = ""
        friendly = ""
        try:
            serial = info.GetSerialNumber()
        except Exception:
            pass
        try:
            friendly = info.GetFriendlyName()
        except Exception:
            friendly = ""
        try:
            model = info.GetModelName()
        except Exception:
            model = friendly
        devices.append(
            BaslerDeviceInfo(
                serial=serial or "",
                model=model or friendly or "Basler",
                friendly=friendly or model or "Basler",
            )
        )
    return devices


class BaslerCamera:
    """
    Thin wrapper around pypylon.InstantCamera that mimics the subset of the VmbPy API
    consumed by CameraService.
    """

    def __init__(
        self,
        logger: logging.Logger,
        device_serial: Optional[str] = None,
        pixel_format: str = _PIXEL_MONO8,
        grab_timeout_ms: int = 5000,
    ):
        if pylon is None:
            raise ImportError("pypylon is not available on this platform.")
        self._logger = logger
        self._factory = pylon.TlFactory.GetInstance()
        self._device_info = self._select_device(device_serial)
        self._camera = pylon.InstantCamera(self._factory.CreateDevice(self._device_info))
        self._pixel_format = pixel_format
        self._grab_timeout_ms = max(100, int(grab_timeout_ms))
        self._handler: Optional[FrameHandler] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._feature_cache: Dict[str, Union[_BaslerFloatFeature, _BaslerEnumFeature]] = {}

    def __enter__(self) -> "BaslerCamera":
        if not self._camera.IsOpen():
            self._camera.Open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop_streaming()
        if self._camera.IsOpen():
            self._camera.Close()

    def get_id(self) -> str:
        serial = ""
        try:
            serial = self._device_info.GetSerialNumber()
        except Exception:
            pass
        return f"basler:{serial}" if serial else "basler"

    def start_streaming(self, handler: FrameHandler, buffer_count: int = 6) -> None:
        if self._running:
            return
        self.__enter__()
        self._handler = handler
        try:
            self._camera.MaxNumBuffer = max(self._camera.MaxNumBuffer, max(4, buffer_count))
        except Exception:
            pass
        self._configure_pixel_format()
        self._camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly, pylon.GrabLoop_ProvidedByUser)
        self._running = True
        self._thread = threading.Thread(target=self._acquisition_loop, name="BaslerCameraStream", daemon=True)
        self._thread.start()
        self._logger.info("BaslerCamera: Grabbing started for %s", self.get_id())

    def stop_streaming(self) -> None:
        if not self._running:
            return
        self._running = False
        try:
            if self._camera.IsGrabbing():
                self._camera.StopGrabbing()
        except Exception:
            pass
        if self._thread:
            self._thread.join(timeout=1.5)
        self._thread = None
        self._handler = None
        self._logger.info("BaslerCamera: Grabbing stopped.")

    def _configure_pixel_format(self) -> None:
        try:
            pixel_param = self._camera.PixelFormat
            if pixel_param.IsWritable():
                pixel_param.SetValue(self._pixel_format)
        except Exception as exc:
            self._logger.debug(f"BaslerCamera: Could not set pixel format to {self._pixel_format}: {exc}")

    def _acquisition_loop(self) -> None:
        while self._running and self._camera.IsGrabbing():
            try:
                grab = self._camera.RetrieveResult(self._grab_timeout_ms, pylon.TimeoutHandling_ThrowException)
            except Exception as exc:
                if self._running:
                    self._logger.error(f"BaslerCamera: RetrieveResult failed: {exc}")
                continue

            with grab:
                try:
                    if not grab.GrabSucceeded():
                        self._logger.warning(f"BaslerCamera: Frame grab failed: {grab.ErrorDescription}")
                        continue
                except Exception as exc:
                    if self._running:
                        self._logger.error(f"BaslerCamera: Grab result invalid: {exc}")
                    continue
                arr = grab.Array
                if arr is None:
                    continue
                frame = BaslerFrame(np.array(arr, copy=True), _PIXEL_MONO8)
                if not self._handler:
                    continue
                try:
                    self._handler(self, None, frame)
                except Exception as exc:
                    self._logger.error(f"BaslerCamera: Handler error: {exc}", exc_info=True)
                    self._running = False
                    break
        try:
            if self._camera.IsGrabbing():
                self._camera.StopGrabbing()
        except Exception:
            pass
        self._running = False

    def _select_device(self, target_serial: Optional[str]):
        devices = list(_enumerate_pylon_devices())
        if not devices:
            raise RuntimeError("No Basler cameras detected.")
        if target_serial:
            for dev in devices:
                try:
                    if dev.GetSerialNumber() == target_serial:
                        return dev
                except Exception:
                    continue
            raise RuntimeError(f"Basler camera with serial '{target_serial}' not found.")
        return devices[0]

    def __getattr__(self, item: str):
        try:
            return self.get_feature_by_name(item)
        except AttributeError as exc:
            raise AttributeError(item) from exc

    def get_feature_by_name(self, name: str):
        key = (name or "").lower()
        if key in self._feature_cache:
            return self._feature_cache[key]

        mapping = {
            "exposuretime": (("ExposureTime", "ExposureTimeAbs", "ExposureTimeRaw"), _BaslerFloatFeature),
            "exposuretimeabs": (("ExposureTimeAbs", "ExposureTime", "ExposureTimeRaw"), _BaslerFloatFeature),
            "exposureauto": (("ExposureAuto",), _BaslerEnumFeature),
            "gain": (("Gain", "GainRaw", "GainAbs"), _BaslerFloatFeature),
            "gainauto": (("GainAuto",), _BaslerEnumFeature),
        }
        if key not in mapping:
            raise AttributeError(f"Feature '{name}' not supported by Basler camera.")

        attr_names, wrapper_cls = mapping[key]
        if isinstance(attr_names, str):
            attr_names = (attr_names,)
        parameter = None
        for attr_name in attr_names:
            try:
                parameter = getattr(self._camera, attr_name)
            except Exception:
                parameter = None
            if parameter is not None:
                break
        if parameter is None:
            raise AttributeError(f"Basler camera missing feature '{name}'")

        wrapper = wrapper_cls(parameter)
        self._feature_cache[key] = wrapper
        return wrapper
