"""
beam_plot_example.py
---------------------

This example script demonstrates how to use the :mod:`beam_analysis`
library to analyse a beam profile stored as a BMP image and plot the
integrated spectra along the principal axes together with the Gaussian
fits returned by :func:`beam_analysis.analyze_beam`.

The script expects the path to a BMP image file as its first command line
argument.  It then loads the image, performs the analysis and creates
two plots: one for the spectrum along the x‑axis (Ix) and one for the
y‑axis (Iy).  Each plot shows both the measured spectrum and the best
fit Gaussian curve.  A legend indicates the fitted 1/e² radius for
convenience.

Example usage::

    python beam_plot_example.py path/to/beam_profile.bmp

An optional ``--pixel-size`` argument lets you provide the physical size
of a pixel (together with ``--pixel-unit``).  When supplied, the script
reports ISO and Gaussian radii in both pixels and physical units and
adds secondary axes with the converted scales.

Requirements:

* numpy
* matplotlib
* Pillow (for BMP loading)
* SciPy (beam_analysis dependencies)
* beam_analysis.py located in the same directory or installed on the
  Python path

"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from PIL import Image

from typing import Optional, Tuple

# import the analyze_beam function from the beam_analysis module
from beam_analysis import analyze_beam


def main(image_path: str, pixel_size: Optional[float] = None, pixel_unit: str = "µm") -> None:
    """Analyse a beam image and plot the Ix and Iy spectra with fits.

    Parameters
    ----------
    image_path : str
        Path to the BMP image containing the beam profile.
    """
    # Load the BMP image and convert to grayscale
    with Image.open(image_path) as img:
        data = np.asarray(img.convert("L"), dtype=float)

    # Build analysis options from CLI if provided
    _opts = globals().get('_beam_example_opts')
    if _opts is None:
        _compute = "both"
        _clip_negatives = "none"
        _angle_clip_mode = "otsu"
        _background_subtraction = True
        _rotation_mode = "auto"
        _fixed_angle_deg = None
    else:
        _compute = _opts.get("compute", "both")
        _clip_negatives = _opts.get("clip_negatives", "none")
        _angle_clip_mode = _opts.get("angle_clip_mode", "otsu")
        _background_subtraction = _opts.get("background_subtraction", True)
        _rotation_mode = _opts.get("rotation_mode", "auto")
        _fixed_angle_deg = _opts.get("fixed_angle_deg", None)

    # Determine rotation angle per CLI:
    # - auto: ignore any fixed angle and auto-detect principal axis
    # - fixed: use provided fixed angle in degrees, or 0 if omitted
    rotation_angle = None
    if _rotation_mode == "auto":
        rotation_angle = None
    elif _rotation_mode == "fixed":
        if _fixed_angle_deg is None:
            rotation_angle = 0.0
        else:
            rotation_angle = np.deg2rad(float(_fixed_angle_deg))

    # Map angle clip mode 'same' to None to reuse clip_negatives
    angle_mode_param = None if (_angle_clip_mode is None or _angle_clip_mode == "same") else _angle_clip_mode

    # Run the analysis unless disabled
    if _compute == "none":
        result = None
    else:
        result = analyze_beam(
            data,
            clip_negatives=_clip_negatives,
            angle_clip_mode=angle_mode_param,
            background_subtraction=_background_subtraction,
            rotation_angle=rotation_angle,
            compute_gaussian=(_compute in ("both", "gauss")),
        )

    # If no analysis requested, show the raw image and return
    if result is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(data, cmap="gray")
        ax.set_title("Beam image (no analysis)")
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")
        fig.tight_layout()
        plt.show()
        return

    # Unpack the spectra and fit parameters
    x_positions, Ix = result["Ix_spectrum"]
    y_positions, Iy = result["Iy_spectrum"]
    fit_x = result["gauss_fit_x"]
    fit_y = result["gauss_fit_y"]

    cx = result["cx"]
    cy = result["cy"]
    rx_iso = result["rx_iso"]
    ry_iso = result["ry_iso"]
    is_major_x = rx_iso >= ry_iso

    if pixel_size is not None and fit_x is not None and fit_y is not None:
        print(
            "Pixel size set to "
            f"{pixel_size:.6g} {pixel_unit}\n"
            f"ISO radii: rₓ = {rx_iso:.3f} px ({rx_iso * pixel_size:.3f} {pixel_unit}), "
            f"rᵧ = {ry_iso:.3f} px ({ry_iso * pixel_size:.3f} {pixel_unit})\n"
            f"Gaussian radii: wₓ = {fit_x['radius']:.3f} px ({fit_x['radius'] * pixel_size:.3f} {pixel_unit}), "
            f"wᵧ = {fit_y['radius']:.3f} px ({fit_y['radius'] * pixel_size:.3f} {pixel_unit})"
        )

    # Build fitted curves for plotting (only when Gaussian fits are available)
    def gauss_curve(x: np.ndarray, params: dict) -> np.ndarray:
        return params["amplitude"] * np.exp(-2.0 * ((x - params["centre"]) / params["radius"]) ** 2)

    _compute = globals().get('_beam_example_opts', {}).get('compute', 'both')
    Ix_fit = gauss_curve(x_positions, fit_x) if (_compute in ("both", "gauss") and fit_x is not None) else None
    Iy_fit = gauss_curve(y_positions, fit_y) if (_compute in ("both", "gauss") and fit_y is not None) else None

    # Create plots
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=False)

    # Plot Ix
    axes[0].plot(x_positions, Ix, label="Ix (integrated)", color="C0")
    _compute = globals().get('_beam_example_opts', {}).get('compute', 'both')
    gauss_x_label = "Gaussian fit"
    if _compute in ("both", "gauss") and fit_x is not None:
        if pixel_size is not None:
            gauss_x_label = (
                f"Gaussian fit (w={fit_x['radius']:.2f} px / "
                f"{fit_x['radius'] * pixel_size:.2f} {pixel_unit})"
            )
        else:
            gauss_x_label = f"Gaussian fit (w={fit_x['radius']:.2f} px)"
    _compute = globals().get('_beam_example_opts', {}).get('compute', 'both')
    if _compute in ("both", "gauss") and Ix_fit is not None:
        axes[0].plot(x_positions, Ix_fit, label=gauss_x_label, color="C1")
    # Indicate second moment radius (2σ) on the x spectrum
    x_label = "ISO rₓ (major)" if is_major_x else "ISO rₓ (minor)"
    if pixel_size is not None:
        iso_rx_label = f"{x_label} = {rx_iso:.2f} px ({rx_iso * pixel_size:.2f} {pixel_unit})"
    else:
        iso_rx_label = f"{x_label} = {rx_iso:.2f} px"
    _compute = globals().get('_beam_example_opts', {}).get('compute', 'both')
    if _compute in ("both", "second"):
        axes[0].axvline(cx - rx_iso, linestyle="--", color="C2", alpha=0.7, label=iso_rx_label)
        axes[0].axvline(cx + rx_iso, linestyle="--", color="C2", alpha=0.7)
    axes[0].set_xlabel("X position (pixels)")
    axes[0].set_ylabel("Integrated intensity")
    axes[0].set_title("Spectrum along principal x‑axis")
    axes[0].legend()
    if pixel_size is not None:
        def px_to_phys_x(x_value: float) -> float:
            return (x_value - cx) * pixel_size

        def phys_to_px_x(x_value: float) -> float:
            return (x_value / pixel_size) + cx

        secax_x = axes[0].secondary_xaxis('top', functions=(px_to_phys_x, phys_to_px_x))
        secax_x.set_xlabel(f"X position relative to centre ({pixel_unit})")

    # Plot Iy
    axes[1].plot(y_positions, Iy, label="Iy (integrated)", color="C0")
    _compute = globals().get('_beam_example_opts', {}).get('compute', 'both')
    gauss_y_label = "Gaussian fit"
    if _compute in ("both", "gauss") and fit_y is not None:
        if pixel_size is not None:
            gauss_y_label = (
                f"Gaussian fit (w={fit_y['radius']:.2f} px / "
                f"{fit_y['radius'] * pixel_size:.2f} {pixel_unit})"
            )
        else:
            gauss_y_label = f"Gaussian fit (w={fit_y['radius']:.2f} px)"
    _compute = globals().get('_beam_example_opts', {}).get('compute', 'both')
    if _compute in ("both", "gauss") and Iy_fit is not None:
        axes[1].plot(y_positions, Iy_fit, label=gauss_y_label, color="C1")
    # Indicate second moment radius (2σ) on the y spectrum
    cy = result["cy"]
    y_label = "ISO rᵧ (major)" if not is_major_x else "ISO rᵧ (minor)"
    if pixel_size is not None:
        iso_ry_label = f"{y_label} = {ry_iso:.2f} px ({ry_iso * pixel_size:.2f} {pixel_unit})"
    else:
        iso_ry_label = f"{y_label} = {ry_iso:.2f} px"
    _compute = globals().get('_beam_example_opts', {}).get('compute', 'both')
    if _compute in ("both", "second"):
        axes[1].axvline(cy - ry_iso, linestyle="--", color="C2", alpha=0.7, label=iso_ry_label)
        axes[1].axvline(cy + ry_iso, linestyle="--", color="C2", alpha=0.7)
    axes[1].set_xlabel("Y position (pixels)")
    axes[1].set_ylabel("Integrated intensity")
    axes[1].set_title("Spectrum along principal y‑axis")
    axes[1].legend()
    if pixel_size is not None:
        def px_to_phys_y(x_value: float) -> float:
            return (x_value - cy) * pixel_size

        def phys_to_px_y(x_value: float) -> float:
            return (x_value / pixel_size) + cy

        secax_y = axes[1].secondary_xaxis('top', functions=(px_to_phys_y, phys_to_px_y))
        secax_y.set_xlabel(f"Y position relative to centre ({pixel_unit})")

    fig.tight_layout()
    plt.show()

    # ------------------------------------------------------------------
    # Plot the input image with ellipses and principal axes
    # ------------------------------------------------------------------
    # Display the (non-rotated) processed image in a new figure
    fig2, ax_img = plt.subplots(figsize=(6, 6))
    processed_display = result["img_for_spec"]
    img_origin = result.get("img_for_spec_origin", (0, 0))
    ymin_img, xmin_img = img_origin
    if processed_display.size == 0:
        raise ValueError("Beam analysis returned an empty processed image; cannot build overlay plot.")
    height_img, width_img = processed_display.shape
    x_display = xmin_img + np.arange(width_img)
    y_display = ymin_img + np.arange(height_img)
    base_cmap = plt.cm.get_cmap("jet", 256)
    cmap_colors = base_cmap(np.linspace(0, 1, 256))
    min_val = float(processed_display.min())
    max_val = float(processed_display.max())
    limit = abs(min_val) if min_val < 0 else 0.0
    if limit > 0 and max_val > 0:
        threshold = min(limit, max_val)
        if threshold > 0:
            # map data range [0, max_val] onto colormap indices and set the
            # low-positive band up to ``threshold`` to grey to match the spec.
            threshold_idx = int(np.clip(np.floor((threshold / max_val) * 255.0), 0, 255))
            cmap_colors[: threshold_idx + 1] = np.array([0.5, 0.5, 0.5, 1.0])
    display_cmap = mcolors.ListedColormap(cmap_colors)
    display_cmap.set_under("white")
    vmin = 0.0 if min_val < 0 else None
    vmax = max_val if max_val > 0 else None
    x_min_display = x_display[0]
    x_max_display = x_display[-1]
    y_min_display = y_display[-1]
    y_max_display = y_display[0]
    ax_img.imshow(
        processed_display,
        cmap=display_cmap,
        origin="upper",
        extent=(x_min_display, x_max_display, y_min_display, y_max_display),
        vmin=vmin,
        vmax=vmax,
    )
    ax_img.set_title("Beam profile with ISO and Gaussian ellipses")
    ax_img.set_xlabel("X (pixels)")
    ax_img.set_ylabel("Y (pixels)")

    # Principal-axis orientation returned by the analysis is referenced to the
    # original, non-rotated image.  Convert this angle into unit vectors that
    # we can use directly in the display coordinate system.
    theta = result["theta"]
    major_radius = rx_iso
    minor_radius = ry_iso
    angle_image = theta
    if ry_iso > rx_iso:
        major_radius, minor_radius = minor_radius, major_radius
        angle_image += 0.5 * np.pi

    major_vec = np.array([np.cos(angle_image), np.sin(angle_image)], dtype=float)
    minor_vec = np.array([-np.sin(angle_image), np.cos(angle_image)], dtype=float)
    major_norm = np.linalg.norm(major_vec)
    minor_norm = np.linalg.norm(minor_vec)
    if major_norm == 0.0 or minor_norm == 0.0:
        raise ValueError("Failed to derive principal-axis directions; zero-length vector encountered.")
    major_vec /= major_norm
    minor_vec /= minor_norm

    gauss_rx = fit_x["radius"] if fit_x is not None else None
    gauss_ry = fit_y["radius"] if fit_y is not None else None
    if gauss_rx is not None and gauss_ry is not None:
        if ry_iso > rx_iso:
            gauss_major_radius = gauss_ry
            gauss_minor_radius = gauss_rx
        else:
            gauss_major_radius = gauss_rx
            gauss_minor_radius = gauss_ry
    else:
        gauss_major_radius = None
        gauss_minor_radius = None

    def ellipse_coords(a_radius: float, b_radius: float) -> Tuple[np.ndarray, np.ndarray]:
        t = np.linspace(0.0, 2.0 * np.pi, 361)
        cos_t = np.cos(t)
        sin_t = np.sin(t)
        x_vals = cx + a_radius * cos_t * major_vec[0] + b_radius * sin_t * minor_vec[0]
        y_vals = cy + a_radius * cos_t * major_vec[1] + b_radius * sin_t * minor_vec[1]
        return x_vals, y_vals

    iso_x, iso_y = ellipse_coords(major_radius, minor_radius)
    _compute = globals().get('_beam_example_opts', {}).get('compute', 'both')
    if _compute in ("both", "second"):
        ax_img.plot(iso_x, iso_y, color="white", linewidth=2.0, label="ISO ellipse")
    if _compute in ("both", "gauss") and gauss_major_radius is not None and gauss_minor_radius is not None:
        gauss_x, gauss_y = ellipse_coords(gauss_major_radius, gauss_minor_radius)
        ax_img.plot(gauss_x, gauss_y, color="yellow", linewidth=2.0, linestyle="--", label="Gaussian ellipse")

    # Principal axis lines drawn using the eigenvectors
    major_line_x = [cx - major_radius * major_vec[0], cx + major_radius * major_vec[0]]
    major_line_y = [cy - major_radius * major_vec[1], cy + major_radius * major_vec[1]]
    minor_line_x = [cx - minor_radius * minor_vec[0], cx + minor_radius * minor_vec[0]]
    minor_line_y = [cy - minor_radius * minor_vec[1], cy + minor_radius * minor_vec[1]]

    if _compute in ("both", "second"):
        ax_img.plot(major_line_x, major_line_y, color="cyan", linewidth=1.5, label="Major axis")
        ax_img.plot(minor_line_x, minor_line_y, color="magenta", linewidth=1.5, label="Minor axis")

    ax_img.legend(loc="upper right")
    ax_img.set_xlim(x_min_display, x_max_display)
    ax_img.set_ylim(y_min_display, y_max_display)
    # origin='upper' already places the minimum y at the top through extent
    ax_img.set_aspect('equal')
    if pixel_size is not None:
        def px_to_phys_img_x(x_value: float) -> float:
            return (x_value - cx) * pixel_size

        def phys_to_px_img_x(x_value: float) -> float:
            return (x_value / pixel_size) + cx

        def px_to_phys_img_y(y_value: float) -> float:
            return (y_value - cy) * pixel_size

        def phys_to_px_img_y(y_value: float) -> float:
            return (y_value / pixel_size) + cy

        secax_img_x = ax_img.secondary_xaxis('top', functions=(px_to_phys_img_x, phys_to_px_img_x))
        secax_img_x.set_xlabel(f"X relative to centre ({pixel_unit})")
        secax_img_y = ax_img.secondary_yaxis('right', functions=(px_to_phys_img_y, phys_to_px_img_y))
        secax_img_y.set_ylabel(f"Y relative to centre ({pixel_unit})")
    fig2.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Analyse a beam-profile BMP image and plot the ISO/gaussian cuts."
        )
    )
    parser.add_argument(
        "image_path",
        help="Path to the BMP image containing the beam profile.",
    )
    parser.add_argument(
        "--pixel-size",
        type=float,
        default=None,
        help="Physical size of one pixel (same units as --pixel-unit).",
    )
    parser.add_argument(
        "--pixel-unit",
        default="µm",
        help="Unit string used when displaying physical radii (default: µm).",
    )
    # Additional analysis options
    parser.add_argument(
        "--compute",
        choices=["both", "second", "gauss", "none"],
        default="both",
        help="Choose what to calculate/display: both, ISO second-moment, Gaussian fit, or none.",
    )
    parser.add_argument(
        "--clip-negatives",
        choices=["none", "zero", "otsu"],
        default="none",
        help="Negative-value handling after background subtraction (none, zero, otsu).",
    )
    parser.add_argument(
        "--angle-clip-mode",
        choices=["same", "none", "zero", "otsu"],
        default="otsu",
        help="Background handling for angle estimation; 'same' reuses --clip-negatives.",
    )
    parser.add_argument(
        "--no-background-subtraction",
        action="store_true",
        help="Disable background subtraction in the analysis.",
    )
    parser.add_argument(
        "--rotation",
        choices=["auto", "fixed"],
        default="auto",
        help="Principal-axis rotation control: automatic (default) or fixed angle.",
    )
    parser.add_argument(
        "--fixed-angle",
        type=float,
        default=None,
        help="Fixed rotation angle in degrees (requires --rotation fixed).",
    )
    args = parser.parse_args()
    image_file = Path(args.image_path)
    if not image_file.is_file():
        print(f"Error: {image_file} does not exist or is not a file.")
        sys.exit(1)
    if args.pixel_size is not None and args.pixel_size <= 0:
        print("Error: --pixel-size must be positive if provided.")
        sys.exit(1)
    _beam_example_opts = {
        "compute": args.compute,
        "clip_negatives": args.clip_negatives,
        "angle_clip_mode": args.angle_clip_mode,
        "background_subtraction": (not args.no_background_subtraction),
        "rotation_mode": args.rotation,
        "fixed_angle_deg": args.fixed_angle,
    }
    globals()["_beam_example_opts"] = _beam_example_opts
    main(str(image_file), pixel_size=args.pixel_size, pixel_unit=args.pixel_unit)

