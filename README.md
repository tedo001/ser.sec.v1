# AI Surveillance System

End-to-end home-surveillance prototype: YOLO-based detection, behaviour
tracking, dynamic risk scoring, and alerts. Runs on a laptop for testing
and is light enough for a Raspberry Pi for deployment to a CCTV stream.

## Features

- **Detection** — person, face mask, weapons/tools (knife, rod, crowbar, ...).
  Uses **YOLO26** if the weights are available locally, otherwise falls back
  to **YOLOv8n** (auto-downloaded by Ultralytics).
- **Night enhancement** — CLAHE on the L channel of LAB when low light is
  detected.
- **Tracking** — fast IoU + centroid tracker (no torch dependency); enough
  to detect loitering reliably.
- **Risk scoring** (configurable in `config.yaml`):

  | Signal | Weight |
  |---|---|
  | Person | +0.30 |
  | Night | +0.25 |
  | Mask | +0.30 |
  | Weapon/tool | +0.50 |
  | Loitering (>8s) | +0.20 |

  Threshold: **0.6 at night**, **0.8 by day**.
- **Alerts** — frame snapshot (`events/event_*.jpg`) + cross-platform beep
  + JSONL event log. Cooldown prevents spam.
- **Flask API** — `GET /health`, `GET /events`, `GET /events/latest`,
  `GET /image/<file>`.

## Project layout

```
ser.sec.v1/
├── main.py              # entrypoint
├── config.yaml          # all thresholds / paths
├── requirements.txt
└── src/
    ├── detector.py      # YOLO wrapper (YOLO26 → YOLOv8 fallback)
    ├── tracker.py       # IoU + centroid tracker
    ├── risk.py          # risk scoring engine
    ├── preprocess.py    # CLAHE night enhancement
    ├── alerts.py        # snapshot + beep + JSONL log
    ├── server.py        # Flask API
    └── utils.py         # config loader + HUD overlay
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# laptop webcam
python main.py --source 0

# pre-recorded clip
python main.py --source path/to/video.mp4

# CCTV / IP camera
python main.py --source rtsp://user:pass@192.168.1.42:554/stream

# headless (Raspberry Pi)
python main.py --no-display
```

Press `q` to quit the display window. Snapshots land in `events/`, and the
Flask API serves them at `http://<host>:5000/`.

## Configuration

All knobs live in `config.yaml`: model weights, device (`cpu`/`cuda`/`mps`),
input source, risk weights, thresholds, loitering window, alert cooldown,
night-enhance settings, and Flask host/port.

For custom-trained weights (recommended for masks and weapons), drop the
`.pt` file in `weights/` and point `model.custom_weapons` /
`model.custom_mask` at it. They run alongside the primary model and their
detections are merged.

## Datasets

Use these to fine-tune the mask / weapon heads:

- **Masks** — Roboflow *Face Mask Detection*, Kaggle *Face Mask Detection*
  (Larxel), MAFA dataset.
- **Weapons / tools** — Roboflow *Weapon Detection*, *Knife Detection*,
  the **Sohas Weapon dataset** (UGR), **OD-WeaponDetection**.
- **Low-light persons** — ExDark, NightOwls.

Train with Ultralytics:

```bash
yolo detect train data=weapons.yaml model=yolov8n.pt imgsz=640 epochs=50
```

## Optimisation tips for edge devices (Raspberry Pi)

1. Use `yolov8n.pt` (or YOLO26 nano variant) and `imgsz: 320`.
2. Export to **NCNN** or **ONNX** and run with quantisation:
   `yolo export model=yolov8n.pt format=ncnn int8=True`.
3. Lower input FPS — process every 2nd or 3rd frame.
4. Add a **Coral USB TPU** and run a TFLite-quantised model (~30 FPS on Pi 4).
5. Disable the Flask server (`server.enabled: false`) if not needed.
6. Run with `--no-display` to skip OpenCV's GUI overhead.

## Workflow (PyCharm ↔ GitHub ↔ Claude Code)

1. Clone the repo in PyCharm: `git clone <repo-url>`.
2. Create / switch to the working branch: `git checkout claude/ai-surveillance-system-9qyAJ`.
3. Open Claude Code in the repo root; it shares the working tree with
   PyCharm so edits made by Claude show up immediately in the IDE.
4. After changes: `git add -A && git commit -m "..." && git push`.

## Roadmap

- Telegram / email push notifications.
- Replace the IoU tracker with DeepSORT once accuracy is bottlenecked.
- Web UI on top of the Flask API.
- Multi-camera support.
