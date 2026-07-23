"""
utils.py — PCL Body Analyser (Speed 20m project)
Shared utilities for the speed_20m analysis module.

Provides:
    frame_to_b64(frame)                     — JPEG → base64 string
    RollingMean(window)                     — smoothing helper
    process_video_or_image(...)             — unified video/image runner
    draw_footer_hud(frame, items)           — bottom stat bar overlay
    draw_pcl_logo(frame)                    — PCL logo watermark
"""

import os
import cv2
import base64
import threading
import subprocess
import numpy as np
from collections import deque
from typing import Callable, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_LOGO_PATH = os.path.join(_HERE, "PCL_Logo.png")


# ════════════════════════════════════════════════════════════════
# Global: last processed video frames for browser playback
# ════════════════════════════════════════════════════════════════
_last_video_frames: dict = {"frames": [], "fps": 6.0}

# ════════════════════════════════════════════════════════════════
# Global: live progress store, keyed by uid (progress_uid)
#   { uid: {"pct": int, "label": str, "done": bool, "jump_event": dict|None} }
# Read by the /progress/<uid> SSE route in app.py, written from
# inside process_video_or_image() while a video is being processed.
# ════════════════════════════════════════════════════════════════
_progress_store: dict = {}
_progress_lock = threading.Lock()


def set_progress(uid, pct, label, done=False, jump_event=None):
    """Update the live progress for a given job uid. No-op if uid is None."""
    if not uid:
        return
    with _progress_lock:
        entry = {"pct": int(pct), "label": label, "done": bool(done)}
        if jump_event is not None:
            entry["jump_event"] = jump_event
        _progress_store[uid] = entry


def get_progress(uid):
    with _progress_lock:
        return _progress_store.get(uid)


# ════════════════════════════════════════════════════════════════
# Frame → Base64  (JPEG quality 65 — faster transfer)
# ════════════════════════════════════════════════════════════════
def frame_to_b64(frame) -> str:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
    return base64.b64encode(buf).decode("utf-8")


# ════════════════════════════════════════════════════════════════
# Rolling mean smoother
# ════════════════════════════════════════════════════════════════
class RollingMean:
    def __init__(self, window: int = 5):
        self._buf = deque(maxlen=window)

    def update(self, value: float) -> float:
        self._buf.append(value)
        return float(np.mean(self._buf))

    def reset(self):
        self._buf.clear()


# ════════════════════════════════════════════════════════════════
# Browser-compatible video writer
# Priority: avc1 (H.264) -> H264 -> mp4v + FFmpeg re-encode
# ════════════════════════════════════════════════════════════════
def _ensure_mp4_path(output_path: str) -> str:
    base, ext = os.path.splitext(output_path)
    return base + ".mp4" if ext.lower() != ".mp4" else output_path


def _make_writer(output_path: str, fps: float, width: int, height: int):
    output_path = _ensure_mp4_path(output_path)

    # 1st: avc1 — direct H.264, browser-ready, no re-encode needed
    for codec in ("avc1", "H264"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if writer.isOpened():
            print(f"  [CODEC]  {codec} (H.264 direct) — browser-ready: {output_path}")
            return writer, codec
        writer.release()

    # Fallback: mp4v — FFmpeg re-encode needed after writing
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    print(f"  [CODEC]  mp4v fallback — FFmpeg re-encode will run after: {output_path}")
    return writer, "mp4v"


def _ffmpeg_available() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=5, check=True)
        return True
    except Exception:
        return False


def _reencode_h264(src: str, dst: str) -> bool:
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", src,
                "-vcodec", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-an",
                dst,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=300,
        )
        return (result.returncode == 0
                and os.path.exists(dst)
                and os.path.getsize(dst) > 0)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# ════════════════════════════════════════════════════════════════
# Unified video / image processor
# ════════════════════════════════════════════════════════════════
def process_video_or_image(
    path: str,
    is_video: bool,
    frame_processor: Callable,
    output_path: Optional[str] = None,
    snap_pcts: Optional[List[float]] = None,
    analysis_skip: int = 2,
    progress_uid: Optional[str] = None,   # accepted for API compatibility
) -> List[str]:
    if snap_pcts is None:
        snap_pcts = [0.1, 0.3, 0.5, 0.7, 0.9]

    snapshots: List[str] = []

    # ── IMAGE ────────────────────────────────────────────────────
    if not is_video:
        frame = cv2.imread(path)
        if frame is None:
            raise ValueError(f"Could not read image: {path}")
        h, w = frame.shape[:2]
        if max(h, w) > 1280:
            scale = 1280 / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        processed = frame_processor(frame, 0, 1)
        if processed is not None:
            frame = processed

        if output_path:
            cv2.imwrite(output_path, frame)

        snapshots.append(frame_to_b64(frame))
        return snapshots

    # ── VIDEO ────────────────────────────────────────────────────
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")

    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    scale = 1.0
    if max(width, height) > 1280:
        scale  = 1280 / max(width, height)
        width  = int(width  * scale)
        height = int(height * scale)

    # ── Output video setup ──────────────────────────────────────
    writer     = None
    used_codec = None
    temp_path  = None

    if output_path:
        output_path = _ensure_mp4_path(output_path)
        writer, used_codec = _make_writer(output_path, fps, width, height)

        # mp4v fallback: write to temp, re-encode to H.264 via FFmpeg after
        if used_codec == "mp4v" and _ffmpeg_available():
            base, _ = os.path.splitext(output_path)
            temp_path = base + "_tmp.mp4"
            writer.release()
            writer, _ = _make_writer(temp_path, fps, width, height)
            print("  [FFMPEG] Available — will re-encode after writing")

    snap_indices = set(int(p * max(total - 1, 1)) for p in snap_pcts)

    # ── 6fps canvas + analysis skip for speed ───────────────────
    video_fps_target = 6.0
    frame_skip       = max(1, int(fps / video_fps_target))
    MAX_VIDEO_FRAMES = 200
    ANALYSIS_SKIP    = analysis_skip  # MediaPipe har Nth frame pe (1=har frame, 2=har 2nd frame)
    video_frames_b64 = []
    last_frame       = None     # skipped frames ke liye last annotated frame reuse

    # Progress reporting: reserve 5–90% for the frame loop below.
    # (0–5% is "uploading / initialising", handled client-side + app.py.)
    PROGRESS_UPDATE_EVERY = max(1, total // 40)  # ~40 updates over the whole video

    set_progress(progress_uid, 5, "Loading video…")

    fc = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if scale < 1.0:
            frame = cv2.resize(frame, (width, height))

        if fc % ANALYSIS_SKIP == 0:
            processed = frame_processor(frame, fc, total)
            if processed is not None:
                frame = processed
            last_frame = frame
        else:
            if last_frame is not None:
                frame = last_frame

        if writer:
            writer.write(frame)

        if fc in snap_indices:
            snapshots.append(frame_to_b64(frame))

        # Browser video frames — skip + max cap
        if fc % frame_skip == 0 and len(video_frames_b64) < MAX_VIDEO_FRAMES:
            video_frames_b64.append(frame_to_b64(frame))

        if fc % PROGRESS_UPDATE_EVERY == 0 and total > 0:
            pct = 5 + int(85 * (fc / total))
            set_progress(progress_uid, min(pct, 90),
                         f"Analysing frame {fc}/{total}…")

        fc += 1

    cap.release()
    if writer:
        writer.release()

    set_progress(progress_uid, 92, "Encoding annotated video…")

    # ── Re-encode mp4v → H.264 (only when FFmpeg was available) ──
    if output_path and temp_path and os.path.exists(temp_path):
        print("  [FFMPEG] Re-encoding mp4v → H.264 for browser playback …")
        ok = _reencode_h264(temp_path, output_path)
        try:
            os.remove(temp_path)
        except OSError:
            pass
        if ok:
            print("  [FFMPEG] H.264 re-encode successful ✓")
        else:
            print("  [FFMPEG] Re-encode failed — copying mp4v as fallback")
            # temp file ko output path pe rename karo agar output missing/empty
            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                try:
                    os.rename(temp_path, output_path)
                except OSError:
                    pass

    while len(snapshots) < len(snap_pcts):
        snapshots.append(snapshots[-1] if snapshots else "")

    _last_video_frames["frames"] = video_frames_b64
    _last_video_frames["fps"]    = video_fps_target

    set_progress(progress_uid, 95, "Computing metrics…")

    return snapshots


# ════════════════════════════════════════════════════════════════
# Logo watermark  (expects PCL_Logo.png next to this file)
# ════════════════════════════════════════════════════════════════
_logo_cache = {"img": None, "tried": False}


def _load_logo():
    if not _logo_cache["tried"]:
        _logo_cache["tried"] = True
        if os.path.exists(_LOGO_PATH):
            _logo_cache["img"] = cv2.imread(_LOGO_PATH, cv2.IMREAD_UNCHANGED)
        else:
            print(f"  [HUD]    Logo not found at {_LOGO_PATH} — skipping watermark")
    return _logo_cache["img"]


def draw_pcl_logo(frame, position="top-right", scale=0.14, margin=14, opacity=0.85):
    """Draws PCL_Logo.png onto `frame` as a small semi-transparent watermark."""
    logo = _load_logo()
    if logo is None:
        return frame

    h, w = frame.shape[:2]
    logo_w = max(1, int(w * scale))
    aspect = logo.shape[0] / logo.shape[1]
    logo_h = max(1, int(logo_w * aspect))

    resized = cv2.resize(logo, (logo_w, logo_h), interpolation=cv2.INTER_AREA)

    if position == "top-left":
        x, y = margin, margin
    elif position == "bottom-right":
        x, y = w - logo_w - margin, h - logo_h - margin
    elif position == "bottom-left":
        x, y = margin, h - logo_h - margin
    else:  # "top-right" default
        x, y = w - logo_w - margin, margin

    x = max(0, min(x, w - logo_w))
    y = max(0, min(y, h - logo_h))

    roi = frame[y:y + logo_h, x:x + logo_w]

    if resized.ndim == 3 and resized.shape[2] == 4:
        # PNG with alpha channel — blend per-pixel
        alpha = (resized[:, :, 3].astype(float) / 255.0) * opacity
        for c in range(3):
            roi[:, :, c] = (alpha * resized[:, :, c] +
                             (1 - alpha) * roi[:, :, c]).astype(roi.dtype)
    else:
        cv2.addWeighted(resized[:, :, :3], opacity, roi, 1 - opacity, 0, roi)

    frame[y:y + logo_h, x:x + logo_w] = roi
    return frame


# ════════════════════════════════════════════════════════════════
# Footer stat bar
# ════════════════════════════════════════════════════════════════
def draw_footer_hud(frame, items, height=54, bg_color=(20, 20, 20), alpha=0.65):
    """
    Draws a semi-transparent stat bar across the bottom of `frame`.
    items: list of (label, value) tuples, e.g. [("SPEED","12.3"), ...]
    """
    h, w = frame.shape[:2]
    bar_top = max(0, h - height)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, bar_top), (w, h), bg_color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    n = max(1, len(items))
    col_w = w / n

    for i, (label, value) in enumerate(items):
        cx = int(col_w * i + col_w / 2)

        (val_w, _), _ = cv2.getTextSize(str(value), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.putText(frame, str(value), (cx - val_w // 2, bar_top + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        (lbl_w, _), _ = cv2.getTextSize(str(label), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.putText(frame, str(label), (cx - lbl_w // 2, bar_top + 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 200, 255), 1, cv2.LINE_AA)

        if i > 0:
            div_x = int(col_w * i)
            cv2.line(frame, (div_x, bar_top + 8), (div_x, h - 8), (80, 80, 80), 1)

    return frame