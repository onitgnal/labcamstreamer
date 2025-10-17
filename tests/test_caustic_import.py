import time
from pathlib import Path

import cv2
import numpy as np
import pytest

import app as app_module
from models.caustic import CausticFilenameError, parse_caustic_filename


@pytest.fixture
def app_client(tmp_path):
    orig_manager = app_module.caustic_manager
    orig_cache_dir = app_module.CAUSTIC_CACHE_DIR
    orig_autosave_dir = app_module.CAUSTIC_AUTOSAVE_DIR
    orig_service = getattr(app_module, '_caustic_import_service', None)
    orig_beam_opts = dict(app_module._beam_opts)

    cache_dir = tmp_path / "cache"
    autosave_dir = tmp_path / "autosave"
    cache_dir.mkdir()
    autosave_dir.mkdir()

    manager = app_module.CausticManager()
    manager.set_autosave_dir(autosave_dir)
    app_module.caustic_manager = manager

    app_module.CAUSTIC_CACHE_DIR = cache_dir
    app_module.CAUSTIC_AUTOSAVE_DIR = autosave_dir

    if orig_service is not None:
        try:
            orig_service._shutdown()
        except Exception:
            pass
    app_module._caustic_import_service = app_module.CausticImportService()

    app_module._beam_opts.update({
        "compute": "both",
        "background_subtraction": False,
    })

    app_module.app.config['TESTING'] = True
    client = app_module.app.test_client()

    yield client

    app_module._caustic_import_service._shutdown()
    app_module._caustic_import_service = app_module.CausticImportService()

    app_module.caustic_manager = orig_manager
    app_module.CAUSTIC_CACHE_DIR = orig_cache_dir
    app_module.CAUSTIC_AUTOSAVE_DIR = orig_autosave_dir
    if orig_manager is not None and orig_autosave_dir is not None:
        try:
            orig_manager.set_autosave_dir(orig_autosave_dir)
        except Exception:
            pass

    app_module._beam_opts.clear()
    app_module._beam_opts.update(orig_beam_opts)
def _write_gaussian_bmp(path: Path, *, dtype: str) -> None:
    size = 64
    y, x = np.indices((size, size))
    cx = cy = size / 2.0
    sigma = size / 6.0
    gaussian = np.exp(-(((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * sigma ** 2)))
    if dtype == "uint8":
        arr = np.clip(gaussian * 255.0, 0, 255).astype(np.uint8)
    else:
        arr = np.clip(gaussian * 55000.0, 0, 65535).astype(np.uint16)
    cv2.imwrite(str(path), arr)


def test_parse_caustic_filename_examples():
    pixel_size, z_mm = parse_caustic_filename("sample_pixelsize_16e-6_m_pos_-2.1_mm.bmp")
    assert pixel_size == pytest.approx(1.6e-5)
    assert z_mm == pytest.approx(-2.1)

    pixel_size2, z_mm2 = parse_caustic_filename("foo_pixelsize_1.00E-5_m_pos_0_mm.BMP")
    assert pixel_size2 == pytest.approx(1.0e-5)
    assert z_mm2 == pytest.approx(0.0)


@pytest.mark.parametrize(
    "filename",
    [
        "bad_name.bmp",
        "img_pixelsize_-1e-6_m_pos_0_mm.bmp",
        "img_pixelsize_1e-5_m_pos_0_mm.png",
    ],
)
def test_parse_caustic_filename_invalid(filename):
    with pytest.raises(CausticFilenameError):
        parse_caustic_filename(filename)


def test_caustic_import_pipeline(app_client, tmp_path):
    folder = tmp_path / "frames"
    folder.mkdir()

    file1 = folder / "001_sample_pixelsize_16e-6_m_pos_-2.1_mm.bmp"
    file2 = folder / "002_sample_pixelsize_1.00E-5_m_pos_0_mm.bmp"
    _write_gaussian_bmp(file1, dtype="uint16")
    _write_gaussian_bmp(file2, dtype="uint8")

    response = app_client.post(
        "/api/caustic/import",
        json={"folder": str(folder), "recursive": False},
    )
    assert response.status_code == 202
    payload = response.get_json()
    task_id = payload.get("task_id")
    assert task_id

    status = payload
    for _ in range(60):
        status = app_client.get(f"/api/caustic/import/{task_id}").get_json()
        if status["status"] in ("completed", "failed"):
            break
        time.sleep(0.1)

    assert status["status"] == "completed"
    counts = status["counts"]
    assert counts["imported"] == 2
    assert counts["duplicates"] == 0
    assert counts["malformed"] == 0
    assert counts["io_errors"] == 0
    assert status.get("total_files") == 2
    assert status.get("skipped") == []

    state = status.get("caustic_state")
    assert state
    assert len(state.get("points", [])) == 2

    points = app_module.caustic_manager.list_points()
    assert len(points) == 2
    pixel_sizes = sorted(pt.pixel_size_m for pt in points)
    assert pixel_sizes == pytest.approx([1.0e-5, 1.6e-5])
    z_values = sorted(pt.z_m for pt in points)
    assert z_values == pytest.approx([-2.1e-3, 0.0])
