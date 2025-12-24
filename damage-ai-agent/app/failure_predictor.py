# app/failure_predictor.py
from typing import Dict
import math

def estimate_failure_risk(history_summary: Dict) -> float:
    """
    Compute a simple failure risk score [0.0, 1.0] from a history summary.
    history_summary can contain keys like:
      - rust_count
      - dent_count
      - dent_growth_rate (0..1)
      - last_inspection_days
      - age_years
    This is intentionally simple and explainable.
    """
    risk = 0.0
    rust = float(history_summary.get("rust_count", 0))
    dent = float(history_summary.get("dent_count", 0))
    dent_growth = float(history_summary.get("dent_growth_rate", 0))
    last_days = float(history_summary.get("last_inspection_days", 30))
    age = float(history_summary.get("age_years", 0))

    # rust contributes strongly
    risk += min(rust * 0.12, 0.5)

    # dent count and growth contribute
    risk += min(dent * 0.05, 0.25)
    risk += min(dent_growth * 0.4, 0.2)

    # older equipment slightly more risky
    if age > 5:
        risk += min((age - 5) * 0.02, 0.1)

    # long time since last inspection increases risk
    if last_days > 30:
        risk += min(((last_days - 30) / 30) * 0.05, 0.1)

    # clamp 0..1
    risk = max(0.0, min(1.0, risk))
    return risk


def make_history_summary(past_detections: list) -> dict:
    """
    Convert a list of past detection records (e.g. from dataset or agent_memory)
    into a compact summary for the predictor.
    Each detection record expected to include 'label' and 'timestamp'.
    """
    rust_count = sum(1 for r in past_detections if r.get("label") == "rust")
    dent_count = sum(1 for r in past_detections if r.get("label") == "dent")
    # compute a naive growth rate (delta count over time)
    # this is placeholder logic — replace with more robust time-series later
    dent_growth_rate = 0.0
    if len(past_detections) >= 2:
        # take last half vs first half
        mid = len(past_detections) // 2
        first = sum(1 for r in past_detections[:mid] if r.get("label") == "dent")
        second = sum(1 for r in past_detections[mid:] if r.get("label") == "dent")
        denom = max(1, first)
        dent_growth_rate = max(0.0, (second - first) / denom)

    last_ts = max((r.get("timestamp", 0) for r in past_detections), default=0)
    from time import time
    last_days = (time() - last_ts) / (24 * 3600) if last_ts else 999

    # age_years not derivable here; user/system must provide it via metadata
    return {
        "rust_count": rust_count,
        "dent_count": dent_count,
        "dent_growth_rate": dent_growth_rate,
        "last_inspection_days": last_days,
        "age_years": past_detections[0].get("age_years", 0) if past_detections else 0
    }
