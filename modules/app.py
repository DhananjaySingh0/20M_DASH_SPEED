r"""
Speed 20m Analyser — Flask Backend

Folder structure:
    New folder/
    ├── modules/
    │   ├── app.py              <- yeh file
    │   ├── module_speed_20m.py
    │   └── utils.py
    ├── inputs/
    ├── outputs/
    ├── templates/
    │   └── index.html
    ├── static/
    │   └── styles.css
    └── Video/

Run: python -u "...New folder\modules\app.py"

── ASYNC JOB FLOW (matches templates/index.html's JS) ──────────
    1. POST /analyse          → saves the upload, starts a background
                                 thread, returns {"uid": ...} immediately (202).
    2. GET  /progress/<uid>   → Server-Sent Events stream of live
                                 {pct, label, done} updates while the
                                 background thread runs.
    3. GET  /result/<uid>     → 202 while still processing, then the
                                 full result JSON once done (or the
                                 error payload if analysis failed).
"""

import os
import re
import uuid
import json
import time
import threading
import datetime
import sys

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.dirname(BASE_DIR)
MODULES_DIR = BASE_DIR

sys.path.insert(0, MODULES_DIR)

from flask import Flask, request, jsonify, send_from_directory, Response, render_template
from module_speed_20m import analyse_speed_20m
from utils import get_progress, set_progress, _last_video_frames

app = Flask(
    __name__,
    template_folder=os.path.join(ROOT_DIR, "templates"),
    static_folder=os.path.join(ROOT_DIR, "static"),
)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024

INPUTS_DIR  = os.path.join(ROOT_DIR, "inputs")
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")
VIDEO_DIR   = os.path.join(ROOT_DIR, "Video")

os.makedirs(INPUTS_DIR,  exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR,   exist_ok=True)

VIDEO_EXTS   = {".mp4", ".avi", ".mov", ".webm", ".mkv"}
IMAGE_EXTS   = {".jpg", ".jpeg", ".png", ".bmp"}
ALLOWED_EXTS = VIDEO_EXTS | IMAGE_EXTS

def allowed_file(filename):
    return os.path.splitext(filename.lower())[1] in ALLOWED_EXTS

def is_video_file(filename):
    return os.path.splitext(filename.lower())[1] in VIDEO_EXTS


# ════════════════════════════════════════════════════════════════
# In-memory job store — { uid: {"status": "processing"|"done"|"error", ...} }
# ════════════════════════════════════════════════════════════════
_jobs: dict = {}
_jobs_lock = threading.Lock()


def _set_job(uid, **fields):
    with _jobs_lock:
        job = _jobs.setdefault(uid, {})
        job.update(fields)


def _get_job(uid):
    with _jobs_lock:
        return _jobs.get(uid)


# ── HTML REPORT GENERATOR ────────────────────────────────────────
def generate_html_report(result, safe_name):
    ts  = datetime.datetime.now().strftime("%d %b %Y, %H:%M")
    sid = result.get("session_id", "—")
    src = result.get("input_file", "—")

    metrics_html = ""
    for m in result.get("metrics", []):
        metrics_html += f"""
        <div class="metric">
          <div class="metric-val">{m['value']}</div>
          <div class="metric-lbl">{m['label']}</div>
        </div>"""

    issues    = result.get("issues", [])
    strengths = result.get("strengths", [])
    corrections = [i for i in issues if not re.search(r'no major|good|full|excellent|perfect|stable|strong|controlled|symmetr', i, re.I)]
    if not strengths:
        strengths = [i for i in issues if re.search(r'good|full|excellent|perfect|stable|strong|controlled|symmetr', i, re.I)]

    str_html = "".join(f"<li>{s}</li>" for s in (strengths or ["Analysis complete"]))
    cor_html = "".join(f"<li>{c}</li>" for c in (corrections or ["No major issues detected — keep up the great work!"]))

    raw_data = {k: v for k, v in result.items()
                if k not in ("snapshots", "video_frames", "video_fps")}

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>Speed 20m Report — {sid}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
  *,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:#eaf1ff;color:#001f5b;font-family:'DM Sans',sans-serif;font-size:15px;line-height:1.6;-webkit-font-smoothing:antialiased}}
  .wrap{{max-width:900px;margin:0 auto;padding:40px 24px 80px}}
  /* HEADER */
  .header{{background:linear-gradient(135deg,#001f5b 0%,#003b9a 100%);border-radius:16px;padding:32px 36px;margin-bottom:28px;display:flex;justify-content:space-between;align-items:flex-end;flex-wrap:wrap;gap:16px}}
  .header-title{{font-family:'Bebas Neue',sans-serif;font-size:2.6rem;letter-spacing:0.05em;color:#fff;line-height:1}}
  .header-sub{{font-family:'JetBrains Mono',monospace;font-size:0.7rem;letter-spacing:0.1em;text-transform:uppercase;color:rgba(255,255,255,0.55);margin-top:6px}}
  .header-meta{{text-align:right;font-family:'JetBrains Mono',monospace;font-size:0.68rem;color:rgba(255,255,255,0.5);line-height:1.8}}
  .header-meta strong{{color:rgba(255,255,255,0.85)}}
  /* METRICS */
  .metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:14px;margin-bottom:24px}}
  .metric{{background:#fff;border:1px solid #c8d0e8;border-radius:12px;padding:20px 16px;text-align:center;box-shadow:0 2px 10px rgba(0,31,91,0.07)}}
  .metric-val{{font-family:'Bebas Neue',sans-serif;font-size:2rem;letter-spacing:0.03em;color:#001f5b;line-height:1}}
  .metric-lbl{{font-family:'JetBrains Mono',monospace;font-size:0.6rem;letter-spacing:0.1em;text-transform:uppercase;color:#7a8bb0;margin-top:4px}}
  /* FINDINGS */
  .findings{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:24px}}
  @media(max-width:600px){{.findings{{grid-template-columns:1fr}}}}
  .finding{{border-radius:12px;padding:18px 20px}}
  .finding.pos{{background:rgba(0,160,112,0.07);border:1px solid rgba(0,160,112,0.25)}}
  .finding.neg{{background:rgba(255,61,0,0.06);border:1px solid rgba(255,61,0,0.18)}}
  .finding-lbl{{font-family:'JetBrains Mono',monospace;font-size:0.65rem;letter-spacing:0.1em;text-transform:uppercase;font-weight:700;margin-bottom:10px}}
  .finding.pos .finding-lbl{{color:#00A070}}
  .finding.neg .finding-lbl{{color:#cc3300}}
  .finding ul{{list-style:none;padding:0}}
  .finding ul li{{font-size:0.88rem;color:#3a4a6b;padding:3px 0;border-bottom:1px solid rgba(0,31,91,0.06)}}
  .finding ul li:last-child{{border-bottom:none}}
  .finding ul li::before{{content:'— ';color:#7a8bb0}}
  /* RAW DATA */
  .section-title{{font-family:'Bebas Neue',sans-serif;font-size:1.2rem;letter-spacing:0.08em;color:#001f5b;text-transform:uppercase;margin-bottom:12px;padding-bottom:6px;border-bottom:2px solid #c8d0e8}}
  .json-box{{background:#fff;border:1px solid #c8d0e8;border-radius:12px;padding:20px;overflow-x:auto;box-shadow:0 2px 10px rgba(0,31,91,0.06)}}
  .json-box pre{{font-family:'JetBrains Mono',monospace;font-size:0.75rem;color:#3a4a6b;white-space:pre-wrap;word-break:break-all}}
  /* FOOTER */
  .footer{{margin-top:40px;text-align:center;font-family:'JetBrains Mono',monospace;font-size:0.65rem;letter-spacing:0.1em;text-transform:uppercase;color:#7a8bb0}}
</style>
</head>
<body>
<div class="wrap">

  <div class="header">
    <div>
      <div class="header-title">Speed 20m Dash Report</div>
      <div class="header-sub">PCL Body Analyser · AI Movement Analysis</div>
    </div>
    <div class="header-meta">
      <div>Generated &nbsp;<strong>{ts}</strong></div>
      <div>Session &nbsp;<strong>{sid}</strong></div>
      <div>Source &nbsp;<strong>{src}</strong></div>
    </div>
  </div>

  <div class="metrics">{metrics_html}</div>

  <div class="findings">
    <div class="finding pos">
      <div class="finding-lbl">✓ Strengths</div>
      <ul>{str_html}</ul>
    </div>
    <div class="finding neg">
      <div class="finding-lbl">✗ Corrections</div>
      <ul>{cor_html}</ul>
    </div>
  </div>

  <div class="section-title">Full Data</div>
  <div class="json-box">
    <pre>{json.dumps(raw_data, indent=2, ensure_ascii=False)}</pre>
  </div>

  <div class="footer">PCL Body Analyser &nbsp;·&nbsp; Speed 20m &nbsp;·&nbsp; {ts}</div>
</div>
</body>
</html>"""


# ════════════════════════════════════════════════════════════════
# BACKGROUND WORKER — runs the actual analysis off the request thread
# ════════════════════════════════════════════════════════════════
def _run_analysis_job(uid, input_path, is_video, output_path,
                       input_filename, output_filename, safe_name):
    try:
        set_progress(uid, 5, "Loading video…")

        result = analyse_speed_20m(
            path=input_path,
            is_video=is_video,
            output_path=output_path,
            progress_uid=uid,
        )

        if "metrics" not in result or not result["metrics"]:
            result["metrics"] = [
                {"label": "Duration",   "value": f"{result.get('duration_sec', 0):.1f}s"},
                {"label": "Avg Speed",  "value": f"{result.get('avg_speed_kph', 0):.1f} km/h"},
                {"label": "Peak Speed", "value": f"{result.get('peak_speed_kph', 0):.1f} km/h"},
                {"label": "Form Speed", "value": f"{result.get('form_speed_kph', 0):.1f} km/h"},
                {"label": "Form Score", "value": f"{result.get('form_score', 0)}/10"},
            ]

        result["session_id"]  = uid
        result["input_file"]  = input_filename
        result["output_file"] = output_filename
        result["output_url"]  = f"/outputs/{output_filename}"
        result["media_type"]  = "video" if is_video else "image"

        if is_video:
            result["video_frames"] = _last_video_frames.get("frames", [])
            result["video_fps"]    = _last_video_frames.get("fps", 6.0)

        set_progress(uid, 97, "Generating report…")

        # ── Save HTML report to outputs/ ──────────────────────
        html_filename = f"{safe_name}_report.html"
        html_path     = os.path.join(OUTPUTS_DIR, html_filename)
        html_content  = generate_html_report(result, safe_name)
        with open(html_path, "w", encoding="utf-8") as hf:
            hf.write(html_content)
        result["report_url"]  = f"/outputs/{html_filename}"
        result["report_file"] = html_filename
        print(f"  [REPORT] saved → outputs/{html_filename}")

        # Save JSON report to outputs/
        json_filename = f"{safe_name}_report.json"
        json_path     = os.path.join(OUTPUTS_DIR, json_filename)
        json_data = {k: v for k, v in result.items()
                     if k not in ("snapshots", "video_frames", "video_fps")}
        json_data["generated_at"] = datetime.datetime.now().isoformat()
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(json_data, jf, indent=2, ensure_ascii=False)
        result["json_url"]  = f"/outputs/{json_filename}"
        result["json_file"] = json_filename
        print(f"  [JSON]   saved → outputs/{json_filename}")

        print(f"  [OUTPUT] annotated → outputs/{output_filename}")

        _set_job(uid, status="done", result=result)
        set_progress(uid, 100, "Complete", done=True)

    except ValueError as ve:
        msg = str(ve)
        _set_job(uid, status="error", code=422, error=msg,
                  remark="Ensure the full sprint is clearly visible in the video.")
        set_progress(uid, 100, "Error", done=True)

    except Exception as e:
        app.logger.exception("Speed 20m analysis failed")
        _set_job(uid, status="error", code=500, error=f"Analysis failed: {str(e)}",
                  remark="Please try again.")
        set_progress(uid, 100, "Error", done=True)


# ════════════════════════════════════════════════════════════════
# ROUTES
# ════════════════════════════════════════════════════════════════

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "exercise": "speed_20m"}), 200


@app.route("/analyse", methods=["POST"])
def analyse():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "Empty filename."}), 400
    if not allowed_file(f.filename):
        return jsonify({"error": "Unsupported file type.", "remark": "Upload MP4, MOV, AVI, WEBM, JPG, or PNG."}), 400

    ext       = os.path.splitext(f.filename)[1].lower()
    uid       = str(uuid.uuid4())[:8]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = f"{timestamp}_{uid}"

    input_filename = f"{safe_name}_{f.filename}"
    input_path     = os.path.join(INPUTS_DIR, input_filename)

    is_video        = is_video_file(f.filename)
    out_ext         = ".mp4" if is_video else ext
    output_filename = f"{safe_name}_annotated{out_ext}"
    output_path     = os.path.join(OUTPUTS_DIR, output_filename)

    try:
        f.save(input_path)
    except Exception as e:
        return jsonify({"error": f"Could not save upload: {e}"}), 500

    if not is_video:
        return jsonify({"error": "Speed 20m analysis requires a video file, not an image.",
                         "remark": "Upload MP4, MOV, AVI, or WEBM."}), 422

    print(f"  [INPUT]  saved → inputs/{input_filename}")

    _set_job(uid, status="processing")
    set_progress(uid, 2, "Queued…")

    thread = threading.Thread(
        target=_run_analysis_job,
        args=(uid, input_path, is_video, output_path,
              input_filename, output_filename, safe_name),
        daemon=True,
    )
    thread.start()

    return jsonify({"uid": uid}), 202


@app.route("/progress/<uid>")
def progress_stream(uid):
    def generate():
        last_sent = None
        start = time.time()
        while True:
            p = get_progress(uid)
            if p and p != last_sent:
                yield f"data: {json.dumps(p)}\n\n"
                last_sent = p
                if p.get("done"):
                    break
            if time.time() - start > 480:   # matches client-side 480000ms timeout
                yield f"data: {json.dumps({'pct': 100, 'label': 'Timeout', 'done': True})}\n\n"
                break
            time.sleep(0.3)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.route("/result/<uid>")
def get_result(uid):
    job = _get_job(uid)
    if job is None:
        return jsonify({"error": "Unknown job id."}), 404

    status = job.get("status")
    if status == "processing":
        return jsonify({}), 202
    if status == "error":
        return jsonify({"error": job.get("error", "Analysis failed."),
                         "remark": job.get("remark", "")}), job.get("code", 500)

    # status == "done"
    return jsonify(job.get("result", {})), 200


@app.route("/outputs/<path:filename>")
def serve_output(filename):
    file_path = os.path.join(OUTPUTS_DIR, filename)
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404

    file_size = os.path.getsize(file_path)

    if filename.lower().endswith(".html"):
        return send_from_directory(OUTPUTS_DIR, filename, mimetype="text/html")

    mime = "video/mp4" if filename.lower().endswith(".mp4") else "application/octet-stream"

    range_header = request.headers.get("Range")
    if range_header:
        byte_start, byte_end = 0, file_size - 1
        m = re.search(r"bytes=(\d+)-(\d*)", range_header)
        if m:
            byte_start = int(m.group(1))
            if m.group(2):
                byte_end = int(m.group(2))
        length = byte_end - byte_start + 1

        def generate():
            with open(file_path, "rb") as fh:
                fh.seek(byte_start)
                remaining = length
                while remaining > 0:
                    chunk = fh.read(min(65536, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk

        resp = Response(generate(), status=206, mimetype=mime, content_type=mime, direct_passthrough=True)
        resp.headers["Content-Range"]  = f"bytes {byte_start}-{byte_end}/{file_size}"
        resp.headers["Accept-Ranges"]  = "bytes"
        resp.headers["Content-Length"] = str(length)
        return resp

    return send_from_directory(OUTPUTS_DIR, filename, mimetype=mime)


@app.route("/video/<path:filename>")
def video(filename):
    file_path = os.path.join(VIDEO_DIR, filename)
    if not os.path.exists(file_path):
        return jsonify({"error": f"Demo video '{filename}' not found."}), 404
    mime = "video/mp4" if filename.lower().endswith(".mp4") else "video/webm"
    return send_from_directory(VIDEO_DIR, filename, mimetype=mime)


if __name__ == "__main__":
    print("=" * 50)
    print("  Speed 20m Analyser — Flask Server")
    print("=" * 50)
    print(f"  ROOT_DIR  : {ROOT_DIR}")
    print(f"  inputs/   : {INPUTS_DIR}")
    print(f"  outputs/  : {OUTPUTS_DIR}")
    print(f"  templates/: {app.template_folder}")
    print(f"  static/   : {app.static_folder}")
    print()
    print("  Open : http://localhost:5000")
    print("=" * 50)
    # threaded=True so /progress SSE streams don't block other requests
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False, threaded=True)