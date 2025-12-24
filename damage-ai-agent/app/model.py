from ultralytics import YOLO

model = YOLO(r"D:\Rushikesh\project\AI Agent\damage-ai-agent\models\best.pt")

def detect_damage(image_path):
    results = model(image_path)[0]
    preds = {}
    boxes = []

    if results.boxes is None:
        return preds, boxes

    for b in results.boxes:
        cls = int(b.cls[0])
        conf = float(b.conf[0])
        label = model.names[cls]
        x1, y1, x2, y2 = map(int, b.xyxy[0])

        preds[label] = max(preds.get(label, 0), conf)
        boxes.append({
            "label": label,
            "confidence": conf,
            "bbox": (x1, y1, x2, y2)
        })

    return preds, boxes
