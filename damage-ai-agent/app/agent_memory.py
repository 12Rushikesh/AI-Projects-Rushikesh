# app/agent_memory.py
# ============================================================
# AGENT MEMORY + BIAS PENALTY
# ============================================================

import json
import time
from pathlib import Path

MEMORY_DIR = Path(r"D:\Rushikesh\project\AI Agent\damage-ai-agent\data/agent_memory")
MEMORY_DIR.mkdir(parents=True, exist_ok=True)

CONFIRM_FILE = MEMORY_DIR / "confirmations.jsonl"
CORRECT_FILE = MEMORY_DIR / "corrections.jsonl"


# ------------------------------------------------------------
# Internal helper
# ------------------------------------------------------------
def _append_jsonl(path: Path, data: dict):
    data = dict(data)
    data["timestamp"] = time.time()
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data) + "\n")


# ------------------------------------------------------------
# Public APIs
# ------------------------------------------------------------
def record_confirmation(damage_type: str, image: str = None):
    _append_jsonl(CONFIRM_FILE, {
        "type": "confirm",
        "damage_type": damage_type,
        "image": image
    })


def record_correction(damage_type: str, image: str = None):
    _append_jsonl(CORRECT_FILE, {
        "type": "correction",
        "damage_type": damage_type,
        "image": image
    })


def read_memory(limit=200):
    records = []
    for p in [CONFIRM_FILE, CORRECT_FILE]:
        if not p.exists():
            continue
        with open(p, "r", encoding="utf-8") as f:
            for line in f.readlines()[-limit:]:
                records.append(json.loads(line))
    return records


# ------------------------------------------------------------
# 🔥 BIAS PENALTY (THIS FIXES YOUR ERROR)
# ------------------------------------------------------------
def bias_penalty(damage_type: str) -> float:
    """
    Returns a penalty factor [0.0 – 0.5] based on past mistakes.
    Used to reduce confidence for frequently misclassified classes.
    """
    if not CORRECT_FILE.exists():
        return 0.0

    with open(CORRECT_FILE, "r", encoding="utf-8") as f:
        corrections = [
            json.loads(l)
            for l in f
            if json.loads(l).get("damage_type") == damage_type
        ]

    mistake_count = len(corrections)

    # Simple rule-based penalty
    if mistake_count >= 20:
        return 0.5
    elif mistake_count >= 10:
        return 0.3
    elif mistake_count >= 5:
        return 0.15
    else:
        return 0.0
