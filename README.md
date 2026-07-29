# 20M DASH SPEED
![image alt](https://github.com/DhananjaySingh0/20M_DASH_SPEED/blob/f9568bbc1bee4cf9ce5c4f65df1c0fc2b59b351c/Screenshot1.png)
![image alt](https://github.com/DhananjaySingh0/20M_DASH_SPEED/blob/21dfaea2e0c54e172e8937c7457d1011deda6750/Screenshot2.png)
AI-powered sprint analysis tool that processes a 20-metre dash video, tracks the
runner with YOLOv8 + ByteTrack, and returns speed/form metrics along with an
annotated playback video and an HTML/JSON report.

---

## Folder Structure

```
20M_DASH_SPEED/
├── env/                         <- Python virtual environment
├── inputs/                      <- Uploaded videos land here (auto-created)
├── modules/
│   ├── __pycache__/
│   ├── app.py                   <- Flask backend / routes
│   ├── module_speed_20m.py      <- Core analysis logic (YOLO tracking + speed calc)
│   ├── utils.py                 <- Shared helpers (video I/O, HUD overlay, logo)
│   ├── __init__.py
│   └── PCL_Logo.png             <- Watermark used on annotated output video
├── outputs/                     <- Annotated video + HTML/JSON reports (auto-created)
│   ├── <timestamp>_<uid>_annotated.mp4
│   ├── <timestamp>_<uid>_report.html
│   └── <timestamp>_<uid>_report.json
├── static/
│   └── styles.css               <- Not linked anywhere; safe to remove (page has its own inline <style>)
├── templates/
│   └── 20m_dash_blog.html       <- Landing page + upload UI (served at "/")
├── Video/                       <- Demo/sample videos served at /video/<file> (auto-created)
├── yolov8n.pt                   <- YOLOv8 nano weights (project root)
├── requirements.txt
└── README.md
```

> `inputs/`, `outputs/`, and `Video/` are created automatically on startup via
> `os.makedirs(..., exist_ok=True)` in `app.py` — you don't need to create
> them by hand.
>
> **`yolov8n.pt` must sit at the project root** (one level above `modules/`),
> because `app.py` is normally launched as `python modules/app.py` from the
> project root, so the working directory is the root and the relative path
> `"yolov8n.pt"` resolves there. If it's missing, `ultralytics` will
> auto-download it on first run.
>
> `static/styles.css` currently isn't referenced by `20m_dash_blog.html` —
> the page uses its own embedded `<style>` block. You can delete it or wire
> it up with a `<link>` tag if you plan to move the CSS out of the HTML file.

---

## Requirements

- Python 3.9+
- [FFmpeg](https://ffmpeg.org/) on PATH (optional but recommended — used to
  re-encode video to browser-compatible H.264 if OpenCV's built-in codec
  isn't available)

Python packages (pinned in `requirements.txt`):

```
certifi==2026.7.22
charset-normalizer==3.4.9
click==8.4.2
filelock==3.32.0
Flask==3.1.3
fonttools==4.63.0
fsspec==2026.6.0
idna==3.18
numpy
nvidia-ml-py==13.610.43
opencv-python==5.0.0.93
pillow==12.3.0
polars==1.43.0
polars-runtime-32==1.43.0
torch==2.13.0
torchvision==0.28.0
typing_extensions==4.16.0
ultralytics==8.4.104
ultralytics-thop==2.0.20
```

Install with:

```bash
pip install -r requirements.txt
```

> `nvidia-ml-py` is only useful on machines with an NVIDIA GPU. If you're
> deploying to a CPU-only environment (e.g. Render's standard instances),
> it's harmless but unused — safe to leave in or remove.

You also need the YOLOv8 nano weights file `yolov8n.pt` placed at the
**project root** (one level above `modules/` — see Folder Structure above).
It will be downloaded automatically by `ultralytics` on first run if it's
not already present.

---

## Running the App

Run from the **project root** (not from inside `modules/`), so the
relative `yolov8n.pt` path resolves correctly:

```bash
cd 20M_DASH_SPEED
python modules/app.py
```

Then open **http://localhost:5000** in your browser.

---

## How It Works (Async Job Flow)

The upload → analysis flow is fully asynchronous so the UI can show live
progress while a video is processed in a background thread:

1. **`POST /analyse`** — saves the uploaded file to `inputs/`, kicks off a
   background thread, and immediately returns `{"uid": "..."}` (HTTP 202).
2. **`GET /progress/<uid>`** — Server-Sent Events (SSE) stream that pushes
   live `{pct, label, done}` updates while the background thread runs.
3. **`GET /result/<uid>`** — returns HTTP 202 while still processing, and the
   full result JSON (or the error payload) once the job is done.

The analysis itself (`module_speed_20m.py`):

- Runs YOLOv8 person detection + ByteTrack tracking on every frame.
- Picks the largest detected person on the first frame as the "primary"
  sprinter and follows that track ID for the rest of the video.
- Converts frame-to-frame pixel displacement into km/h using a configurable
  `METERS_PER_PIXEL` scale factor, smoothed with a rolling mean.
- Detects the sprint "block" (continuous run above a minimum speed threshold)
  and computes duration, average speed, peak speed, a form-adjusted speed,
  and a 4–10 form score.
- Draws a live speed bar, stat HUD, and the PCL logo watermark onto the
  output video via `utils.py`, and writes it as browser-ready H.264 MP4.
- Generates an HTML report and a JSON report, saved to `outputs/`.

---

## API Endpoints

| Method | Route                 | Description                                      |
|--------|------------------------|---------------------------------------------------|
| GET    | `/`                    | Renders the landing page / upload UI              |
| GET    | `/health`               | Health check — `{"status": "ok"}`                 |
| POST   | `/analyse`               | Upload a video, starts analysis, returns `{uid}`  |
| GET    | `/progress/<uid>`       | SSE stream of live progress                       |
| GET    | `/result/<uid>`         | Final result JSON (202 while processing)          |
| GET    | `/outputs/<filename>`   | Serves annotated video / HTML / JSON reports      |
| GET    | `/video/<filename>`     | Serves demo/sample videos from `Video/`           |

---

## Configuration

Tunable constants live at the top of `module_speed_20m.py`:

| Constant              | Default | Purpose                                                        |
|------------------------|---------|------------------------------------------------------------------|
| `SPRINT_DISTANCE_M`     | `20.0`  | Nominal sprint distance (metres)                                 |
| `METERS_PER_PIXEL`      | `0.025` | **Camera-specific.** Converts pixel displacement to metres — recalibrate for your camera distance/angle |
| `SPEED_SMOOTH_WINDOW`   | `7`     | Rolling-mean window for speed smoothing                          |
| `MIN_SPRINT_SPEED`      | `2.0`   | Minimum km/h to count as "sprinting"                              |
| `MIN_SPRINT_SEC`        | `0.5`   | Minimum valid sprint duration (seconds)                          |
| `SPRINT_ENTRY_FRAMES`   | `2`     | Frames above threshold needed to open a sprint block              |
| `SPRINT_EXIT_FRAMES`    | `15`    | Frames below threshold needed to close a sprint block              |

> **`METERS_PER_PIXEL` is the most important value to calibrate.** The rough
> guide in the code: if the person covers ~200px over 1 metre at mid-frame,
> set `METERS_PER_PIXEL = 1/200 = 0.005`. The default (`0.025`) is tuned for
> a standard 1080p track camera ~15m away.

Other settings:

- `app.config["MAX_CONTENT_LENGTH"]` — max upload size, currently 200 MB
  (in `app.py`).
- Accepted file types: `.mp4 .avi .mov .webm .mkv` for video, `.jpg .jpeg
  .png .bmp` for images (note: this analyser currently **requires a video**
  — image uploads return a 422 error).

---

## Notes / Known Limitations

- The 20m-dash analyser requires a **video** file — a still image will be
  rejected with a 422 error.
- Speed is estimated from raw pixel displacement assuming a rear-facing
  camera (runner moving away from the camera); it is **not** perspective-
  corrected, so accuracy depends heavily on a correctly calibrated
  `METERS_PER_PIXEL`.
- If `ultralytics` isn't installed, analysis requests will fail with a clear
  `RuntimeError` message telling you to `pip install ultralytics`.
- If FFmpeg isn't available and OpenCV's H.264 codecs aren't either, output
  video falls back to `mp4v`, which may not play in all browsers.
