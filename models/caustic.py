from __future__ import annotations

import json
import math
import re
import shutil
import threading
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from datetime import datetime

import numpy as np
from scipy.optimize import curve_fit

__all__ = [
    "CausticPoint",
    "CausticRadiiSource",
    "CausticManager",
    "convert_length_to_meters",
    "convert_meters_to_length",
    "format_caustic_raw_filename",
]


class CausticRadiiSource(str, Enum):
    GAUSS_1E2 = "gauss_1e2"
    MOMENT_2SIGMA = "moment_2sigma"

    @classmethod
    def from_label(cls, label: str) -> "CausticRadiiSource":
        norm = (label or "").strip().lower()
        if norm in {"gauss", "gaussian", "1e2", "1/e2"}:
            return cls.GAUSS_1E2
        if norm in {"2sigma", "moment", "second_moment", "second-moment"}:
            return cls.MOMENT_2SIGMA
        for member in cls:
            if member.value == norm:
                return member
        raise ValueError(f"Unknown caustic radii source: {label!r}")


_UNIT_FACTORS: Dict[str, float] = {
    "m": 1.0,
    "meter": 1.0,
    "meters": 1.0,
    "mm": 1e-3,
    "millimeter": 1e-3,
    "millimeters": 1e-3,
    "cm": 1e-2,
    "centimeter": 1e-2,
    "centimeters": 1e-2,
    "um": 1e-6,
    "micrometer": 1e-6,
    "micrometers": 1e-6,
    "in": 0.0254,
    "inch": 0.0254,
    "inches": 0.0254,
    "ft": 0.3048,
    "foot": 0.3048,
    "feet": 0.3048,
}


_FILENAME_SANITIZER = re.compile(r"[^A-Za-z0-9._-]+")


def _sanitize_filename_part(value: str, fallback: str) -> str:
    trimmed = str(value or "").strip()
    cleaned = _FILENAME_SANITIZER.sub("_", trimmed)
    cleaned = cleaned.strip("_")
    return cleaned or fallback


def format_caustic_raw_filename(
    timestamp_iso: Optional[str],
    z_value: float,
    unit: str,
) -> str:
    """
    Build a filename like YYYYMMDD_pos_4.123_mm.bmp, preserving sign information.
    """
    if timestamp_iso:
        date_part = _sanitize_filename_part(timestamp_iso.split("T", 1)[0].replace("-", ""), "date")
    else:
        date_part = datetime.utcnow().strftime("%Y%m%d")

    unit_part = _sanitize_filename_part(unit, "unit")

    value = float(z_value)
    magnitude = abs(value)
    value_part = f"{magnitude:.6f}".rstrip("0").rstrip(".")
    if "." not in value_part:
        value_part += ".0"
    sign_part = "neg_" if value < 0 else ""

    return f"{date_part}_pos_{sign_part}{value_part}_{unit_part}.bmp"


def convert_length_to_meters(value: float, unit: str) -> float:
    factor = _UNIT_FACTORS.get((unit or "").strip().lower())
    if factor is None:
        raise ValueError(f"Unsupported length unit: {unit!r}")
    return float(value) * factor


def convert_meters_to_length(value_m: float, unit: str) -> float:
    factor = _UNIT_FACTORS.get((unit or "").strip().lower())
    if factor is None or factor == 0.0:
        raise ValueError(f"Unsupported length unit: {unit!r}")
    return float(value_m) / factor


@dataclass
class CausticPoint:
    point_id: str
    roi_id: str
    timestamp_iso: str
    z_m: float
    position_unit_at_capture: str
    wavelength_nm: float
    pixel_size_m: Optional[float]
    wx_1e2: float
    wy_1e2: float
    wx_2sigma: float
    wy_2sigma: float
    profile_img_path: str
    cut_x_img_path: str
    cut_y_img_path: str
    raw_roi_img_path: Optional[str] = None

    def cleanup_files(self) -> None:
        for path_str in (
            self.profile_img_path,
            self.cut_x_img_path,
            self.cut_y_img_path,
            self.raw_roi_img_path,
        ):
            if not path_str:
                continue
            try:
                path = Path(path_str)
                if path.exists():
                    path.unlink()
                    parent = path.parent
                    if parent.exists() and not any(parent.iterdir()):
                        parent.rmdir()
            except OSError:
                # Non-fatal cleanup failure.
                continue


def _fit_profile_model(z: np.ndarray, w0: float, z0: float, B: float) -> np.ndarray:
    inside = np.clip(w0**2 + B * (z - z0) ** 2, a_min=0.0, a_max=None)
    return np.sqrt(inside, dtype=np.float64)


def _fit_axis(
    z_m: Iterable[float],
    w_m: Iterable[float],
    wavelength_m: float,
) -> Tuple[Dict[str, float], np.ndarray]:
    z_arr = np.asarray(list(z_m), dtype=np.float64)
    w_arr = np.asarray(list(w_m), dtype=np.float64)
    if z_arr.size != w_arr.size or z_arr.size < 3:
        raise ValueError("Need at least three samples per axis to fit M^2.")
    if np.any(w_arr <= 0):
        raise ValueError("All beam radii must be positive for fitting.")

    idx_min = int(np.argmin(w_arr))
    w0_guess = max(1e-9, float(w_arr[idx_min]))
    z0_guess = float(z_arr[idx_min])
    span = float(np.max(np.abs(z_arr - z0_guess))) or 1.0
    w_span = float(np.max(w_arr) - np.min(w_arr))
    if w_span <= 0.0:
        B_guess = 1e-12
    else:
        B_guess = max(1e-18, (w_span**2) / (span**2))

    bounds = ((1e-9, -np.inf, 0.0), (np.inf, np.inf, np.inf))
    popt, pcov = curve_fit(
        _fit_profile_model,
        z_arr,
        w_arr,
        p0=(w0_guess, z0_guess, B_guess),
        bounds=bounds,
        maxfev=10000,
    )
    w0, z0, B = (float(p) for p in popt)
    if B < 0.0:
        B = 0.0

    zr_prime = math.inf
    if B > 0.0:
        zr_prime = w0 / math.sqrt(B)
    m2 = math.inf
    if zr_prime != math.inf and zr_prime > 0.0:
        m2 = (math.pi * w0 * math.sqrt(B)) / max(wavelength_m, 1e-18)

    sigma_w0 = math.sqrt(abs(pcov[0, 0])) if pcov.size else float("nan")
    sigma_z0 = math.sqrt(abs(pcov[1, 1])) if pcov.size > 1 else float("nan")
    sigma_B = math.sqrt(abs(pcov[2, 2])) if pcov.size > 4 else float("nan")
    sigma_zr = float("nan")
    sigma_m2 = float("nan")
    if B > 0.0 and np.isfinite(zr_prime):
        d_zr_dw0 = 1.0 / math.sqrt(B)
        d_zr_dB = -0.5 * w0 * (B ** -1.5)
        var_zr = (
            d_zr_dw0**2 * sigma_w0**2
            + d_zr_dB**2 * sigma_B**2
            + 2.0 * d_zr_dw0 * d_zr_dB * pcov[0, 2]
        )
        sigma_zr = math.sqrt(abs(var_zr))

        d_m_dw0 = (math.pi * math.sqrt(B)) / max(wavelength_m, 1e-18)
        d_m_dB = (math.pi * w0) / (2.0 * max(wavelength_m, 1e-18) * math.sqrt(B))
        var_m2 = (
            d_m_dw0**2 * sigma_w0**2
            + d_m_dB**2 * sigma_B**2
            + 2.0 * d_m_dw0 * d_m_dB * pcov[0, 2]
        )
        sigma_m2 = math.sqrt(abs(var_m2))

    return {
        "w0_m": w0,
        "z0_m": z0,
        "zR_prime_m": zr_prime,
        "M2": m2,
        "sigma": {
            "w0_m": sigma_w0,
            "z0_m": sigma_z0,
            "B": sigma_B,
            "zR_prime_m": sigma_zr,
            "M2": sigma_m2,
        },
    }, pcov


class CausticManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._points: List[CausticPoint] = []
        self._radii_source: CausticRadiiSource = CausticRadiiSource.GAUSS_1E2
        self._wavelength_nm: float = 1030.0
        self._position_unit: str = "mm"
        self._fit_results: Dict[str, Dict[str, float]] = {}
        self._autosave_dir: Optional[Path] = None
        self._default_pixel_size_m: Optional[float] = None

    def set_autosave_dir(self, path: Path) -> None:
        with self._lock:
            self._autosave_dir = path
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    def set_radii_source(self, source: CausticRadiiSource) -> None:
        with self._lock:
            self._radii_source = source
        self._autosave()

    def set_wavelength_nm(self, value: float) -> None:
        with self._lock:
            self._wavelength_nm = float(value)
        self._autosave()

    def set_position_unit(self, unit: str) -> None:
        convert_length_to_meters(1.0, unit)
        with self._lock:
            self._position_unit = unit
        self._autosave()

    def config_snapshot(self) -> Dict[str, object]:
        with self._lock:
            return {
                "wavelength_nm": self._wavelength_nm,
                "position_unit": self._position_unit,
                "radii_source": self._radii_source.value,
            }

    def generate_point_id(self) -> str:
        return uuid.uuid4().hex

    def add_point(self, point: CausticPoint) -> CausticPoint:
        with self._lock:
            if (point.pixel_size_m is None or point.pixel_size_m <= 0.0) and self._default_pixel_size_m:
                point.pixel_size_m = self._default_pixel_size_m
            if point.pixel_size_m is not None and point.pixel_size_m > 0.0:
                self._default_pixel_size_m = float(point.pixel_size_m)
            self._points.append(point)
        self._autosave()
        return point

    def remove_point(self, point_id: str) -> bool:
        removed = False
        with self._lock:
            for idx, pt in enumerate(self._points):
                if pt.point_id == point_id:
                    popped = self._points.pop(idx)
                    popped.cleanup_files()
                    removed = True
                    break
        if removed:
            self._autosave()
        return removed

    def list_points(self) -> List[CausticPoint]:
        with self._lock:
            return list(self._points)

    def get_point(self, point_id: str) -> Optional[CausticPoint]:
        with self._lock:
            for pt in self._points:
                if pt.point_id == point_id:
                    return pt
        return None

    def get_plot_series(self) -> Dict[str, List[float]]:
        with self._lock:
            unit = self._position_unit
            source = self._radii_source
            points = list(self._points)

        z_vals: List[float] = []
        wx_vals: List[float] = []
        wy_vals: List[float] = []

        for pt in sorted(points, key=lambda p: p.z_m):
            z_vals.append(convert_meters_to_length(pt.z_m, unit))
            if source is CausticRadiiSource.GAUSS_1E2:
                wx_vals.append(float(pt.wx_1e2))
                wy_vals.append(float(pt.wy_1e2))
            else:
                wx_vals.append(float(pt.wx_2sigma))
                wy_vals.append(float(pt.wy_2sigma))

        return {"z": z_vals, "wx": wx_vals, "wy": wy_vals}

    def compute_m2_fit(self) -> Dict[str, Dict[str, float]]:
        with self._lock:
            if not self._points:
                raise ValueError("No caustic points collected.")
            points = list(self._points)
            wavelength_nm = self._wavelength_nm
            source = self._radii_source
            default_pixel_size = self._default_pixel_size_m

        wavelength_m = max(1e-12, float(wavelength_nm) * 1e-9)
        z_data: List[float] = []
        w_x: List[float] = []
        w_y: List[float] = []

        ordered_points = sorted(points, key=lambda p: p.z_m)

        fallback_pixel_size = None
        for pt in ordered_points:
            if pt.pixel_size_m is not None and pt.pixel_size_m > 0.0:
                fallback_pixel_size = float(pt.pixel_size_m)
                break
        if fallback_pixel_size is None and default_pixel_size is not None and default_pixel_size > 0.0:
            fallback_pixel_size = float(default_pixel_size)

        for pt in ordered_points:
            if pt.pixel_size_m is None or pt.pixel_size_m <= 0.0:
                if fallback_pixel_size is not None:
                    pt.pixel_size_m = fallback_pixel_size
                else:
                    raise ValueError(
                        f"Pixel size must be configured for all points to compute M^2 (missing for ROI {pt.roi_id})."
                    )
            scale = float(pt.pixel_size_m)
            z_data.append(pt.z_m)
            if source is CausticRadiiSource.GAUSS_1E2:
                w_x.append(pt.wx_1e2 * scale)
                w_y.append(pt.wy_1e2 * scale)
            else:
                w_x.append(pt.wx_2sigma * scale)
                w_y.append(pt.wy_2sigma * scale)

        fit_x, _ = _fit_axis(z_data, w_x, wavelength_m)
        fit_y, _ = _fit_axis(z_data, w_y, wavelength_m)

        with self._lock:
            self._fit_results = {"x": fit_x, "y": fit_y}
        self._autosave()
        return {"x": dict(fit_x), "y": dict(fit_y)}

    def fit_results(self) -> Dict[str, Dict[str, float]]:
        with self._lock:
            return dict(self._fit_results)

    def export_dataset(self, target_dir: Path, *, clean: bool = False) -> Dict[str, object]:
        points = self.list_points()
        fits = self.fit_results()

        if clean and target_dir.exists():
            shutil.rmtree(target_dir, ignore_errors=True)

        target_dir.mkdir(parents=True, exist_ok=True)
        images_dir = target_dir / "images"
        images_dir.mkdir(exist_ok=True)

        config = self.config_snapshot()
        meta = {
            "wavelength_nm": config["wavelength_nm"],
            "position_unit": config["position_unit"],
            "radii_source": config["radii_source"],
            "created_at": points[0].timestamp_iso if points else None,
            "point_count": len(points),
            "fit_results": fits,
        }
        (target_dir / "caustic_meta.json").write_text(json.dumps(meta, indent=2))

        csv_lines = [
            "timestamp_iso,roi_id,z,wavelength_nm,wx_1e2,wy_1e2,wx_2sigma,wy_2sigma,profile_img,cut_x_img,cut_y_img,raw_roi_img"
        ]

        for idx, pt in enumerate(points, start=1):
            prefix = f"{idx:03d}_{pt.point_id}"
            profile_dst = images_dir / f"{prefix}_profile.png"
            cut_x_dst = images_dir / f"{prefix}_cut_x.png"
            cut_y_dst = images_dir / f"{prefix}_cut_y.png"

            has_profile = False
            has_cut_x = False
            has_cut_y = False
            has_raw = False

            if pt.profile_img_path:
                src = Path(pt.profile_img_path)
                if src.exists():
                    shutil.copy2(src, profile_dst)
                    has_profile = True
            if pt.cut_x_img_path:
                src = Path(pt.cut_x_img_path)
                if src.exists():
                    shutil.copy2(src, cut_x_dst)
                    has_cut_x = True
            if pt.cut_y_img_path:
                src = Path(pt.cut_y_img_path)
                if src.exists():
                    shutil.copy2(src, cut_y_dst)
                    has_cut_y = True
            try:
                z_display = convert_meters_to_length(pt.z_m, config["position_unit"])
            except ValueError:
                z_display = pt.z_m

            raw_filename = format_caustic_raw_filename(
                pt.timestamp_iso,
                z_display,
                config["position_unit"],
            )
            raw_dst = images_dir / f"{prefix}_{raw_filename}"
            if pt.raw_roi_img_path:
                src = Path(pt.raw_roi_img_path)
                if src.exists():
                    shutil.copy2(src, raw_dst)
                    has_raw = True

            csv_lines.append(
                ",".join(
                    [
                        pt.timestamp_iso,
                        pt.roi_id,
                        f"{z_display:.9g}",
                        f"{pt.wavelength_nm:.6g}",
                        f"{pt.wx_1e2:.6g}",
                        f"{pt.wy_1e2:.6g}",
                        f"{pt.wx_2sigma:.6g}",
                        f"{pt.wy_2sigma:.6g}",
                        str(profile_dst.relative_to(target_dir)) if has_profile else "",
                        str(cut_x_dst.relative_to(target_dir)) if has_cut_x else "",
                        str(cut_y_dst.relative_to(target_dir)) if has_cut_y else "",
                        str(raw_dst.relative_to(target_dir)) if has_raw else "",
                    ]
                )
            )

        (target_dir / "caustic_points.csv").write_text("\n".join(csv_lines))
        (target_dir / "fits.json").write_text(json.dumps(fits or {}, indent=2))

        return {
            "path": str(target_dir),
            "images_dir": str(images_dir),
            "points": len(points),
            "files": [str(p) for p in target_dir.rglob("*") if p.is_file()],
        }

    def _autosave(self) -> None:
        with self._lock:
            autosave_dir = self._autosave_dir
        if not autosave_dir:
            return
        try:
            target = autosave_dir / "latest"
            self.export_dataset(target, clean=True)
        except Exception:
            pass

