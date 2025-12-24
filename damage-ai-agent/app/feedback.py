from pathlib import Path
import json, shutil
from PIL import Image
from datetime import datetime

CLASS_MAP = {"dent": 0, "hole": 1, "rust": 2}

FB = Path(r"D:\Rushikesh\project\AI Agent\damage-ai-agent\data\feedback")
META = Path(r"D:\Rushikesh\project\AI Agent\damage-ai-agent\data\feedback_meta")
YOLO = Path(r"D:\Rushikesh\project\AI Agent\damage-ai-agent\data\yolo_labels")
ERR = Path(r"D:\Rushikesh\project\AI Agent\damage-ai-agent\data\errors")

def save_class_feedback(image_path, user_label, model_label, confidence):
    (FB / user_label).mkdir(parents=True, exist_ok=True)
    META.mkdir(exist_ok=True)

    name = Path(image_path).name
    shutil.copy(image_path, FB / user_label / name)

    meta = {
        "image": name,
        "user_label": user_label,
        "model_label": model_label,
        "confidence": confidence,
        "time": datetime.utcnow().isoformat()
    }
    with open(META / f"{name}.json", "w") as f:
        json.dump(meta, f, indent=2)

def save_yolo_boxes(image_path, boxes, label):
    if label not in CLASS_MAP:
        return

    YOLO.mkdir(exist_ok=True)
    img = Image.open(image_path)
    w, h = img.size
    out = YOLO / f"{Path(image_path).stem}.txt"

    with open(out, "w") as f:
        for b in boxes:
            x, y, bw, bh = b["left"], b["top"], b["width"], b["height"]
            xc = (x + bw/2) / w
            yc = (y + bh/2) / h
            f.write(f"{CLASS_MAP[label]} {xc} {yc} {bw/w} {bh/h}\n")

def log_error(image_path, model_label, user_label):
    if model_label == user_label:
        return
    (ERR / "mismatch").mkdir(parents=True, exist_ok=True)
    shutil.copy(image_path, ERR / "mismatch" / Path(image_path).name)
