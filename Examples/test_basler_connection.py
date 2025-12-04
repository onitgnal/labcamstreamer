"""
Quick probe script to verify that the Pixi environment can talk to a Basler
camera through the pypylon SDK.

Run with:

    pixi run python Examples/test_basler_connection.py
"""
from __future__ import annotations

import logging
import sys
from typing import Optional

import numpy as np
from pypylon import pylon


def _configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(levelname)s] %(message)s",
        stream=sys.stdout,
    )


def _get_first_device() -> Optional[pylon.CDeviceInfo]:
    tl_factory = pylon.TlFactory.GetInstance()
    devices = tl_factory.EnumerateDevices()
    if not devices:
        return None
    return devices[0]


def probe_camera(timeout_ms: int = 5000) -> int:
    dev = _get_first_device()
    if dev is None:
        logging.error("No Basler cameras detected. Is the camera connected and powered?")
        return 1

    tl_factory = pylon.TlFactory.GetInstance()
    camera = pylon.InstantCamera(tl_factory.CreateDevice(dev))
    logging.info(
        "Connecting to Basler camera %s (S/N: %s)",
        dev.GetModelName(),
        dev.GetSerialNumber(),
    )

    try:
        camera.Open()
        logging.info("Camera opened. Starting single-frame acquisition...")
        camera.StartGrabbingMax(1)
        with camera.RetrieveResult(timeout_ms, pylon.TimeoutHandling_ThrowException) as result:
            if not result.GrabSucceeded():
                logging.error("Frame grab failed: %s", result.ErrorDescription)
                return 2

            array = result.Array
            if isinstance(array, np.ndarray):
                min_val = float(array.min())
                max_val = float(array.max())
                shape = array.shape
                dtype = array.dtype
            else:
                min_val = max_val = 0.0
                shape = "unknown"
                dtype = type(array)
            logging.info(
                "Grab succeeded: shape=%s dtype=%s min=%.1f max=%.1f",
                shape,
                dtype,
                min_val,
                max_val,
            )
    except pylon.RuntimeException as exc:
        logging.exception("Pylon runtime error: %s", exc)
        return 3
    except Exception as exc:  # Catch-all so we always close the device.
        logging.exception("Unexpected error while grabbing: %s", exc)
        return 4
    finally:
        if camera.IsGrabbing():
            camera.StopGrabbing()
        if camera.IsOpen():
            camera.Close()
        logging.info("Camera closed.")

    return 0


def main(argv: list[str]) -> int:
    verbose = "-v" in argv or "--verbose" in argv
    timeout_arg = next((arg for arg in argv if arg.startswith("--timeout=")), None)
    timeout_ms = 5000
    if timeout_arg:
        try:
            timeout_ms = int(timeout_arg.split("=", 1)[1])
        except ValueError:
            logging.warning("Invalid timeout override %s, using default %d ms", timeout_arg, timeout_ms)
    _configure_logging(verbose=verbose)
    return probe_camera(timeout_ms=timeout_ms)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
