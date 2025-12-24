import json
from app.vl_reasoner import vision_reason
from app.thinker import think
from app.agent_memory import bias_penalty
from app.utils import confidence_to_float


def autonomous_agent(image_path, yolo_preds, yolo_boxes):
    # -----------------------------
    # Vision reasoning (VL model)
    # -----------------------------
    try:
        vl = json.loads(vision_reason(image_path, yolo_boxes))
    except Exception:
        return {
            "action": "ASK_HUMAN",
            "reason": "Vision model failed",
            "confidence": "low",
            "damage_type": "unknown"
        }

    # -----------------------------
    # Extract confidence SAFELY
    # -----------------------------
    confidence_raw = vl.get("confidence", "low")
    confidence_score = confidence_to_float(confidence_raw)

    # -----------------------------
    # Bias / memory penalty
    # -----------------------------
    penalty = bias_penalty(vl.get("damage_type", "unknown"))
    confidence_score = max(0.0, confidence_score - penalty)

    # -----------------------------
    # Decision logic
    # -----------------------------
    if confidence_score >= 0.8:
        action = "AUTO_ACCEPT"
    elif confidence_score >= 0.4:
        action = "ASK_HUMAN"
    else:
        action = "REJECT"

    return {
        "action": action,
        "confidence": confidence_raw,   # keep human-readable
        "confidence_score": confidence_score,  # numeric
        "damage_type": vl.get("damage_type", "unknown"),
        "reason": vl.get("reason", "")
    }
