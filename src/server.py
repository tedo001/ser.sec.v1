"""Tiny Flask API exposing recent events and the latest snapshot."""
from __future__ import annotations

import os
import threading

from flask import Flask, jsonify, send_file, abort


def build_app(alert_manager) -> Flask:
    app = Flask(__name__)

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.get("/events")
    def events():
        return jsonify(alert_manager.recent(limit=100))

    @app.get("/events/latest")
    def latest():
        items = alert_manager.recent(limit=1)
        if not items:
            return jsonify({}), 204
        return jsonify(items[-1])

    @app.get("/image/<path:name>")
    def image(name: str):
        # Only serve files inside the configured save directory.
        save_dir = os.path.abspath(alert_manager.save_dir)
        full = os.path.abspath(os.path.join(save_dir, name))
        if not full.startswith(save_dir + os.sep) or not os.path.exists(full):
            abort(404)
        return send_file(full, mimetype="image/jpeg")

    return app


def run_in_background(alert_manager, host: str, port: int) -> threading.Thread:
    app = build_app(alert_manager)
    t = threading.Thread(
        target=lambda: app.run(host=host, port=port, debug=False, use_reloader=False),
        daemon=True,
    )
    t.start()
    print(f"[server] Flask API on http://{host}:{port}")
    return t
