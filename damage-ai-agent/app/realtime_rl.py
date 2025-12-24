# app/realtime_rl.py
"""
Realtime RL loop:
- Connect to RTSP
- Capture frames every N seconds
- Run detect_damage() -> autonomous_agent()
- Log RL step with reward=0 (human feedback expected later)
- Save captured image into data/realtime/

Usage:
    python -m app.realtime_rl --rtsp "rtsp://..." --interval 5
"""
import time
import argparse
from pathlib import Path
from datetime import datetime
import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.append(str(ROOT))

from app.model import detect_damage
from app.agent_core import autonomous_agent
from app.rl_memory import log_rl_step

OUT_DIR = Path("data/realtime")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def capture_loop(rtsp_url, interval=5, max_iters=None):
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        raise ConnectionError("Cannot open RTSP stream.")
    it = 0
    try:
        while True:
            it += 1
            ret, frame = cap.read()
            if not ret:
                time.sleep(1.0)
                continue
            # Convert BGR -> RGB
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except Exception:
                frame_rgb = frame
            img = Image.fromarray(frame_rgb)
            ts = int(datetime.utcnow().timestamp())
            filename = OUT_DIR / f"realtime_{ts}_{it}.jpg"
            img.save(filename)

            # run detection + agent
            try:
                yolo_preds, yolo_boxes = detect_damage(str(filename))
            except Exception as e:
                print("YOLO error:", e)
                yolo_preds, yolo_boxes = {}, []

            try:
                agent_out = autonomous_agent(
                    image_path=str(filename),
                    yolo_preds=yolo_preds,
                    yolo_boxes=yolo_boxes
                )
            except Exception as e:
                agent_out = {"action": "ASK_HUMAN", "confidence": "low", "damage_type":"unknown", "reason": str(e)}

            # Compose state (keep it small)
            state = {
                "yolo": {
                    "label": yolo_preds.get("label") if isinstance(yolo_preds, dict) else None,
                    "confidence": float(yolo_preds.get("confidence", 0.0)) if isinstance(yolo_preds, dict) else 0.0
                },
                "agent": agent_out,
                "image": str(filename)
            }

            # log with reward 0 for now
            log_rl_step(state=state, action=agent_out.get("action", "ASK_HUMAN"), reward=0.0, info={"realtime": True})

            print(f"[{it}] Logged frame {filename.name} action={agent_out.get('action')}")

            if max_iters and it >= max_iters:
                break

            time.sleep(interval)

    except KeyboardInterrupt:
        print("Interrupted by user, exiting.")
    finally:
        cap.release()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rtsp", required=True, help="rtsp url")
    p.add_argument("--interval", type=int, default=5)
    p.add_argument("--iters", type=int, default=0)
    args = p.parse_args()
    capture_loop(args.rtsp, interval=args.interval, max_iters=(args.iters or None))
