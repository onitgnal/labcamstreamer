# Allied Vision ROI Web App (Flask + OpenCV + NumPy + Matplotlib)

Python-only, cross-platform Flask app to stream an Allied Vision camera (vmbpy), draw/manage multiple ROIs from the browser, compute exposure-corrected integrations, render a live bar plot, show intensity-normalized ROI mini-streams, and save snapshots (JSON + PNG) as a ZIP.

## Features
- Live MJPEG overview: `/video_feed` (server-side colormap)
- Controls in left sidebar: Camera On/Off, Exposure (µs), Colormap
- Multi-ROI workflow: add/move/resize with mouse on overlay canvas
- Metrics: per-frame ROI sum, exposure-corrected value/ms, rolling FPS (`/metrics`)
- Bar Plot MJPEG at ~5 Hz: `/bar_feed`
- ROI mini-streams at ~10 Hz each: `/roi_feed/<id>` (intensity-normalized per frame)
- Save Data: downloads `<base>.zip` containing `<base>.json` + `<base>.png`

## Requirements
- Python 3.10+
- vmbpy (Allied Vision SDK) installed and working (do not pip install)
- pip packages in `requirements.txt`:
  - Flask, numpy, opencv-python, matplotlib

## Quick Start (Windows, similar on macOS/Linux)
1) Create/activate a venv
   - PowerShell:
     - python -m venv .venv
     - .venv\Scripts\Activate.ps1
2) Install deps
   - pip install -r requirements.txt
3) (Optional) Choose camera
   - set CAMERA_ID=DEV_XXXX (or leave empty to use first camera)
4) Run
   - python app.py
   - Open http://localhost:5000

Firewall note (Windows): allow Python inbound for local network access.

## File Structure
- app.py                      Flask app & routes
- camera_service.py           Camera lifecycle, streaming, exposure/colormap, latest frames, MJPEG
- roi.py                      ROI dataclass + thread-safe registry (CRUD, bounds clamp)
- metrics.py                  Per-frame ROI integration, exposure-corrected metrics, FPS
- templates/index.html        UI layout (sidebar + workspace)
- static/app.js               ROI interactions, REST, polling, save workflow
- static/styles.css           Responsive styles
- requirements.txt            Python deps
- README.md                   This file

## Endpoints (Server)
- GET /                       UI
- GET /video_feed             MJPEG overview (keeps baseline endpoint)
- GET|POST /exposure          JSON {value} in µs (keeps baseline endpoint)
- GET|POST /colormap          JSON {value: "grey|jet|cubiczero"} (keeps baseline endpoint)
- POST /camera                {enabled: bool} → start/stop streaming
- GET|POST /rois              List / create
- GET|PUT|DELETE /rois/<id>   ROI CRUD
- GET /metrics                Latest snapshot JSON:
  {
    "timestamp": float,
    "fps": float,
    "exposure_us": int,
    "rois": [{"id": str, "sum_gray": int, "value_per_ms": float}, ...]
  }
- GET /bar_feed               MJPEG bar plot (~5 Hz)
- GET /roi_feed/<id>         Per-ROI intensity-normalized MJPEG (~10 Hz)
- GET /save_bundle?base=NAME  Returns NAME.zip with JSON + PNG

## ROI Interactions (Client)
- Add: click-drag on live view
- Move: drag inside ROI
- Resize: drag handles (corners/edges)
- Coordinates are kept in natural image pixel space; client converts display↔stream via scale factor.

## Save Data ZIP
- NAME.json:
  - timestamp_iso, camera_id (None), frame_size (w,h), exposure_us, colormap, fps, rois (id,x,y,w,h)
- NAME.png:
  - Current raw overview frame (BGR before colormap)

## Notes
- Gray computed once per frame; reused for metrics and ROI streams
- Sums use uint64; per-ms normalization: sum_gray / max(1, exposure_us) * 1000
- Thread safety via locks; queues for low-latency MJPEG
- Bar plot ticks at ~5 Hz; ROI mini-streams ~10 Hz per ROI (budget dependent)

## Troubleshooting
- vmbpy not found: ensure Allied Vision SDK is installed and Python bindings are available
- Camera not found: set CAMERA_ID to the exact device ID (see Examples/list_cameras.py)
- Port in use: change port in app.py run()
- Black/placeholder video: ensure camera is enabled from the sidebar; verify exposure
- Slow FPS: reduce ROI feed count, lower resolution, or increase system resources

## License
Internal/Project use. Verify Allied Vision SDK license for deployment constraints.
