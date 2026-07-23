import os
import math
import cv2
import numpy as np

# ── Graceful YOLO import ──────────────────────────────────────────
try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False

# ── PCL shared utilities ──────────────────────────────────────────
from utils import (
    RollingMean,
    process_video_or_image,   # <-- this writes the output video
    draw_footer_hud,
    draw_pcl_logo,
)

# ════════════════════════════════════════════════════════════════
# CONFIG  —  tune these for your camera / lane
# ════════════════════════════════════════════════════════════════

YOLO_MODEL_PATH   = "yolov8n.pt"   # same file already uploaded

SPRINT_DISTANCE_M = 20.0

# Camera is rear-facing (person runs away from camera).
# Speed uses direct pixel distance × METERS_PER_PIXEL scale.
# ── TUNE THIS for your camera ──────────────────────────────────
# Rough guide: if person covers ~200px over 1 metre at mid-frame,
# set METERS_PER_PIXEL = 1/200 = 0.005.
# Default 0.025 suits a standard 1080p track camera ~15m away.
METERS_PER_PIXEL    = 0.025

SPEED_SMOOTH_WINDOW = 7          # RollingMean window (more = smoother)
MIN_SPRINT_SPEED    = 2.0        # km/h — below this = not sprinting
MIN_SPRINT_SEC      = 0.5        # minimum valid sprint duration
SPRINT_ENTRY_FRAMES = 2          # frames above threshold → open block
SPRINT_EXIT_FRAMES  = 15         # frames below threshold → close block


# ════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════

def _speed_kmph(prev_xy, cur_xy, fps):
    """Direct pixel distance → km/h. No perspective warp needed for
    rear-facing camera — the runner moves mostly in the Y axis
    (away from camera) so raw pixel displacement is proportional."""
    dx = cur_xy[0] - prev_xy[0]
    dy = cur_xy[1] - prev_xy[1]
    return math.sqrt(dx*dx + dy*dy) * METERS_PER_PIXEL * fps * 3.6


def _form_score(peak_kph, avg_kph, duration_s):
    score = 4
    if   peak_kph >= 28: score += 3
    elif peak_kph >= 24: score += 2
    elif peak_kph >= 18: score += 1
    if peak_kph > 0:
        ratio = avg_kph / peak_kph
        if   ratio >= 0.85: score += 2
        elif ratio >= 0.70: score += 1
    if duration_s > 0:
        ideal = SPRINT_DISTANCE_M / max(avg_kph / 3.6, 0.1)
        if duration_s >= ideal * 0.85:
            score += 1
    return max(4, min(10, score))


def _feedback(peak_kph, avg_kph, form_kph, duration_s, detected):
    issues = []
    strengths = []

    if not detected:
        issues.append(
            "Person not reliably detected — ensure full body is visible "
            "and lighting is adequate")
        return issues, strengths

    if peak_kph < 10:
        issues.append(
            f"Very low peak speed ({peak_kph:.1f} km/h) — "
            "check camera angle and ensure a true sprint effort")
    elif peak_kph < 18:
        issues.append(
            f"Below-average sprint speed ({peak_kph:.1f} km/h) — "
            "focus on drive phase and arm mechanics")
    elif peak_kph < 24:
        strengths.append(f"Decent top speed ({peak_kph:.1f} km/h)")
    else:
        strengths.append(f"Strong peak velocity ({peak_kph:.1f} km/h)")

    if peak_kph > 0:
        consistency = avg_kph / peak_kph
        if consistency < 0.65:
            issues.append(
                f"Speed consistency low ({consistency:.0%}) — "
                "work on maintaining max velocity across full 20 m")
        elif consistency >= 0.80:
            strengths.append(
                f"Excellent speed maintenance ({consistency:.0%} of peak avg)")

    if 0 < duration_s < MIN_SPRINT_SEC * 2:
        issues.append(
            "Sprint appears very short — ensure the full 20 m is captured")
    elif duration_s > 0:
        strengths.append(
            f"Full sprint detected ({duration_s:.2f}s over "
            f"{SPRINT_DISTANCE_M:.0f} m)")

    if not issues:
        issues = ["No major issues detected"]
    return issues, strengths


# ════════════════════════════════════════════════════════════════
# MAIN ANALYSER
# ════════════════════════════════════════════════════════════════

def analyse_speed_20m(
    path,
    is_video,
    output_path=None,
    progress_uid=None,
):
    # ── Guards ─────────────────────────────────────────────────────
    if not _YOLO_AVAILABLE:
        raise RuntimeError(
            "ultralytics not installed. Run: pip install ultralytics")

    if not is_video:
        raise ValueError(
            "Speed 20m analysis requires a video file, not an image.")

    # ── Resolve YOLO model ─────────────────────────────────────────
    model_path = YOLO_MODEL_PATH
    if not os.path.exists(model_path):
        here = os.path.dirname(os.path.abspath(__file__))
        alt  = os.path.join(here, YOLO_MODEL_PATH)
        if os.path.exists(alt):
            model_path = alt
        else:
            raise FileNotFoundError(
                f"YOLO model not found at '{YOLO_MODEL_PATH}'. "
                "Place yolov8n.pt next to app.py.")

    model = YOLO(model_path)

    # ── Read FPS once (needed for speed calc inside callback) ──────
    _cap = cv2.VideoCapture(path)
    fps  = _cap.get(cv2.CAP_PROP_FPS) or 30.0
    _cap.release()

    # ════════════════════════════════════════════════════════════════
    # STATE  —  all mutable state is in lists/dicts so the closure
    #           inside pf() can write to them (Python nonlocal trick)
    # ════════════════════════════════════════════════════════════════
    smoother        = RollingMean(SPEED_SMOOTH_WINDOW)
    prev_pos        = {}       # track_id → (wx, wy) in warped plane
    primary_id      = [None]   # the sprinter we follow
    detected_ok     = [False]

    all_speeds      = []       # every smoothed speed value

    # Sprint block detection
    entry_ctr       = [0]
    exit_ctr        = [0]
    in_sprint       = [False]
    sprint_start_fc = [None]
    sprint_blocks   = []       # [(start_fc, end_fc, [speeds])]
    block_speeds    = []

    # Live HUD
    live_speed  = [0.0]
    live_peak   = [0.0]
    live_time   = [0.0]

    total_frames = [0]

    # ── Per-frame callback — same pattern as every PCL module ──────
    def pf(frame, fc, total):
        nonlocal block_speeds
        total_frames[0] = fc

        # ── YOLO tracking ────────────────────────────────────────
        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],
            verbose=False,
        )

        current_speed = 0.0

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids   = results[0].boxes.id.cpu().numpy().astype(int)

            # Choose primary track on first detection (largest box)
            if primary_id[0] is None and len(ids) > 0:
                areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
                primary_id[0] = int(ids[np.argmax(areas)])

            for box, tid in zip(boxes, ids):
                tid = int(tid)
                x1, y1, x2, y2 = box
                foot_x = (x1 + x2) / 2
                foot_y = y2

                # Use raw pixel coords directly (rear-facing camera)
                cur_pos = (foot_x, foot_y)

                if tid in prev_pos:
                    raw_spd    = _speed_kmph(prev_pos[tid], cur_pos, fps)
                    smooth_spd = smoother.update(raw_spd)
                else:
                    smooth_spd = 0.0

                prev_pos[tid] = cur_pos

                # Track only primary sprinter for analysis
                if tid == primary_id[0]:
                    detected_ok[0]  = True
                    current_speed   = smooth_spd
                    all_speeds.append(smooth_spd)
                    if smooth_spd > live_peak[0]:
                        live_peak[0] = smooth_spd

                # ── Draw box ─────────────────────────────────────
                clr = (255, 255, 255) if tid == primary_id[0] else (160, 160, 160)
                cv2.rectangle(frame,
                              (int(x1), int(y1)), (int(x2), int(y2)),
                              clr, 2)
                cv2.putText(frame,
                            f"ID {tid}  {smooth_spd:.1f} km/h",
                            (int(x1), max(0, int(y1) - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                            (128, 0, 0), 2, cv2.LINE_AA)

        live_speed[0] = current_speed

        # ── Sprint block state machine ────────────────────────────
        if current_speed >= MIN_SPRINT_SPEED:
            entry_ctr[0] += 1
            exit_ctr[0]   = 0
            if not in_sprint[0] and entry_ctr[0] >= SPRINT_ENTRY_FRAMES:
                in_sprint[0]      = True
                sprint_start_fc[0]= fc - SPRINT_ENTRY_FRAMES + 1
                block_speeds      = []
        else:
            exit_ctr[0]  += 1
            entry_ctr[0]  = 0
            if in_sprint[0] and exit_ctr[0] >= SPRINT_EXIT_FRAMES:
                end_fc      = fc - SPRINT_EXIT_FRAMES
                duration_fc = max(1, end_fc - (sprint_start_fc[0] or fc))
                if duration_fc / fps >= MIN_SPRINT_SEC:
                    sprint_blocks.append((
                        sprint_start_fc[0], end_fc, block_speeds.copy()))
                in_sprint[0]  = False
                block_speeds  = []

        if in_sprint[0]:
            block_speeds.append(current_speed)
            live_time[0] = (fc - (sprint_start_fc[0] or fc)) / fps

        # ── Speed bar ─────────────────────────────────────────────
        bar_w = int(min(1.0, current_speed / 30.0) * 200)
        bar_c = (0, 200, 100) if current_speed < 20 else (0, 80, 255)
        cv2.rectangle(frame, (10, 10), (10 + bar_w, 30), bar_c, -1)
        cv2.rectangle(frame, (10, 10), (210, 30), (180, 180, 180), 1)
        cv2.putText(frame, "0           30 km/h",
                    (10, 45), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (200, 200, 200), 1, cv2.LINE_AA)

        # ── HUD footer ────────────────────────────────────────────
        t_disp = f"{live_time[0]:.2f}s" if live_time[0] > 0 else "---"
        draw_footer_hud(frame, [
            ("SPEED",  f"{current_speed:.1f}"),
            ("PEAK",   f"{live_peak[0]:.1f}"),
            ("TIME",   t_disp),
        ])
        draw_pcl_logo(frame)

        return frame   # ← process_video_or_image writes this to disk

    # ── Run — process_video_or_image handles ALL video I/O ─────────
    snaps = process_video_or_image(
        path, is_video, pf,
        output_path=output_path,      # ← H.264 output video here
        snap_pcts=[0.1, 0.3, 0.5, 0.7, 0.9],
        analysis_skip=1,
        progress_uid=progress_uid,
    )

    # ── Close any open block at end of video ───────────────────────
    if in_sprint[0] and block_speeds:
        dur_fc = max(1, total_frames[0] - (sprint_start_fc[0] or 0))
        if dur_fc / fps >= MIN_SPRINT_SEC:
            sprint_blocks.append((
                sprint_start_fc[0], total_frames[0], block_speeds.copy()))

    # ── Guard: no person detected ──────────────────────────────────
    if not detected_ok[0]:
        raise ValueError(
            "No person detected in video. "
            "Ensure the full body is visible and lighting is adequate.")

    # ── Pick best sprint block (longest continuous run) ────────────
    if sprint_blocks:
        best = max(sprint_blocks, key=lambda b: b[1] - b[0])
        b_start, b_end, b_speeds = best
        duration_s   = max(b_end - b_start, 1) / fps
        valid        = [s for s in b_speeds if s > 0]
    else:
        # Fallback: use everything collected
        duration_s   = total_frames[0] / fps
        valid        = [s for s in all_speeds if s > 0]

    if valid:
        peak_kph  = round(max(valid), 1)
        avg_kph   = round(sum(valid) / len(valid), 1)
        top_half  = sorted(valid, reverse=True)[:max(1, len(valid)//2)]
        form_kph  = round(sum(top_half) / len(top_half), 1)
    else:
        peak_kph  = avg_kph = form_kph = 0.0

    form_score = _form_score(peak_kph, avg_kph, duration_s)
    issues, strengths = _feedback(
        peak_kph, avg_kph, form_kph, duration_s, detected_ok[0])

    # Pad snapshots to 5
    while len(snaps) < 5:
        snaps.append(snaps[-1] if snaps else "")

    dur_str = f"{duration_s:.2f}s" if duration_s > 0 else "N/A"

    return {
        "exercise"        : "20m Sprint",
        "duration_sec"    : round(duration_s, 2),
        "avg_speed_kph"   : avg_kph,
        "peak_speed_kph"  : peak_kph,
        "form_speed_kph"  : form_kph,
        "form_score"      : form_score,
        "sprint_blocks"   : len(sprint_blocks),
        "issues"          : issues,
        "strengths"       : strengths,
        "snapshots"       : snaps,
        "metrics": [
            {"label": "Duration",      "value": dur_str},
            {"label": "Avg Speed",     "value": f"{avg_kph:.1f} km/h"},
            {"label": "Peak Speed",    "value": f"{peak_kph:.1f} km/h"},
            {"label": "Max-Vel Phase", "value": f"{form_kph:.1f} km/h"},
            {"label": "Form Score",    "value": f"{form_score}/10"},
        ],
    }