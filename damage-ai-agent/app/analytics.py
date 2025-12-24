from pathlib import Path
import json
from collections import Counter

META = Path(r"D:\Rushikesh\project\AI Agent\damage-ai-agent\data\feedback_meta")

def stats():
    total, correct = 0, 0
    classes = Counter()

    for f in META.glob("*.json"):
        d = json.loads(f.read_text())
        total += 1
        classes[d["user_label"]] += 1
        if d["user_label"] == d["model_label"]:
            correct += 1

    return {
        "total": total,
        "accuracy": round(correct/total, 3) if total else 0,
        "class_distribution": dict(classes)
    }
