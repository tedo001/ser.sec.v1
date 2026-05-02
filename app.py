"""
AI Surveillance Admin Application
Run with:  python app.py
"""
from __future__ import annotations

import json
import os
import time

import cv2
import yaml
from flask import (
    Flask,
    Response,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    abort,
    url_for,
)

from src.alerts import AlertManager
from src.pipeline import SurveillancePipeline
from src.utils import load_config

# ---------------------------------------------------------------------------
CONFIG_PATH = "config.yaml"
cfg = load_config(CONFIG_PATH)
alerts = AlertManager(cfg)
pipeline = SurveillancePipeline(cfg, alerts)

app = Flask(__name__)
app.secret_key = os.urandom(24)

# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

@app.before_request
def _auto_start():
    """Start the pipeline on first request if source is set."""
    if not pipeline.state.snapshot()["running"]:
        if cfg["source"]["input"] != "":
            pipeline.start()


# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------

@app.get("/")
def dashboard():
    return render_template("dashboard.html", page="dashboard", cfg=cfg)


@app.get("/events")
def events_page():
    evs = alerts.recent(limit=200)
    # Attach just the filename for display, strip the save_dir prefix
    for ev in evs:
        ev["thumb"] = os.path.basename(ev.get("image_path", ""))
    return render_template("events.html", page="events", events=evs[::-1])


@app.get("/settings")
def settings_page():
    raw = open(CONFIG_PATH).read()
    return render_template("settings.html", page="settings",
                           config_yaml=raw, cfg=cfg)


# ---------------------------------------------------------------------------
# Video stream  (MJPEG)
# ---------------------------------------------------------------------------

def _gen_frames():
    jpeg_quality = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
    while True:
        frame = pipeline.state.get_frame()
        if frame is None:
            # Emit a blank 640×480 frame so the stream never stalls
            frame = _blank_frame()
        ok, buf = cv2.imencode(".jpg", frame, jpeg_quality)
        if ok:
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                + buf.tobytes()
                + b"\r\n"
            )
        time.sleep(0.033)  # ~30 fps cap


def _blank_frame():
    import numpy as np
    f = __import__("numpy").zeros((480, 640, 3), dtype="uint8")
    cv2.putText(f, "No signal — pipeline stopped", (80, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 80, 80), 2)
    return f


@app.get("/stream/video")
def stream_video():
    return Response(
        _gen_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


# ---------------------------------------------------------------------------
# Server-Sent Events  (real-time status ticker)
# ---------------------------------------------------------------------------

@app.get("/stream/status")
def stream_status():
    def _gen():
        while True:
            data = json.dumps(pipeline.state.snapshot())
            yield f"data: {data}\n\n"
            time.sleep(1)

    return Response(_gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


# ---------------------------------------------------------------------------
# REST API
# ---------------------------------------------------------------------------

@app.get("/api/status")
def api_status():
    return jsonify(pipeline.state.snapshot())


@app.get("/api/events")
def api_events():
    limit = int(request.args.get("limit", 100))
    return jsonify(alerts.recent(limit=limit))


@app.delete("/api/events")
def api_clear_events():
    import shutil
    # Wipe in-memory queue (replace deque with empty one)
    from collections import deque
    alerts._events = deque(maxlen=200)
    return jsonify({"cleared": True})


@app.get("/api/config")
def api_config():
    return jsonify(cfg)


@app.post("/api/config")
def api_save_config():
    """Accept YAML text body and hot-reload pipeline."""
    raw = request.get_data(as_text=True)
    try:
        new_cfg = yaml.safe_load(raw)
    except yaml.YAMLError as e:
        return jsonify({"error": str(e)}), 400

    with open(CONFIG_PATH, "w") as f:
        f.write(raw)
    cfg.clear()
    cfg.update(new_cfg)
    pipeline.restart(cfg)
    return jsonify({"ok": True})


@app.post("/api/pipeline/start")
def api_start():
    pipeline.start()
    return jsonify({"running": True})


@app.post("/api/pipeline/stop")
def api_stop():
    pipeline.stop()
    return jsonify({"running": False})


@app.post("/api/pipeline/restart")
def api_restart():
    pipeline.restart(cfg)
    return jsonify({"restarted": True})


# ---------------------------------------------------------------------------
# Image serving
# ---------------------------------------------------------------------------

@app.get("/events/image/<filename>")
def event_image(filename: str):
    save_dir = os.path.abspath(alerts.save_dir)
    path = os.path.abspath(os.path.join(save_dir, filename))
    if not path.startswith(save_dir + os.sep) or not os.path.isfile(path):
        abort(404)
    return send_file(path, mimetype="image/jpeg")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    srv = cfg.get("server", {})
    host = srv.get("host", "0.0.0.0")
    port = int(srv.get("port", 5000))
    print(f"[admin] http://{host}:{port}")
    # threaded=True is required for MJPEG + SSE to work concurrently
    app.run(host=host, port=port, debug=False, threaded=True, use_reloader=False)
