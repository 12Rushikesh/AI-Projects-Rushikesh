

# app/rl_memory.py
import json
import time
from pathlib import Path

RL_DIR = Path(r"D:/Rushikesh/project/AI Agent/damage-ai-agent/data/rl_experience.jsonl")
RL_DIR.mkdir(parents=True, exist_ok=True)

RL_LOG = RL_DIR / "rl_steps.jsonl"


def log_rl_step(state, action, reward, info=None):
    """
    Append one RL experience record to data/rl/rl_steps.jsonl
    state, action, reward, info should be JSON-serializable.
    """
    rec = {
        "timestamp": time.time(),
        "state": state,
        "action": action,
        "reward": float(reward),
        "info": info or {}
    }
    with open(RL_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def read_all(limit=None):
    """
    Read all RL records. If limit provided, return last `limit` records.
    """
    if not RL_LOG.exists():
        return []
    with open(RL_LOG, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if limit is None:
        records = [json.loads(l) for l in lines]
    else:
        records = [json.loads(l) for l in lines[-limit:]]
    return records

