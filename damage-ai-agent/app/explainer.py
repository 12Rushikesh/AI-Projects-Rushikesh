# app/explainer.py
from typing import Dict, List

def explain_simple(agent_out: Dict, yolo_boxes: List[Dict]) -> str:
    """
    Produces a short human-readable explanation for the decision.
    agent_out is expected to include:
      - action
      - confidence (optional)
      - damage_types (optional) or damage_type
    yolo_boxes is list of boxes with keys 'label' and 'confidence'
    """
    action = agent_out.get("action", "ASK_HUMAN")
    conf = agent_out.get("confidence", None)
    damage_types = agent_out.get("damage_types") or agent_out.get("damage_type")
    if isinstance(damage_types, list):
        dtype_summary = ", ".join(damage_types[:3])
    else:
        dtype_summary = str(damage_types)

    parts = []
    if action == "AUTO_ACCEPT":
        parts.append("Agent accepted the item automatically.")
        if dtype_summary and dtype_summary != "unknown":
            parts.append(f"Detected: {dtype_summary}.")
        if conf is not None:
            parts.append(f"Confidence ~ {float(conf):.2f}.")
        parts.append("No immediate action required.")
    elif action == "ASK_HUMAN":
        parts.append("Agent requests human review.")
        parts.append(f"Reason: low confidence or ambiguous pattern ({dtype_summary}).")
        if yolo_boxes:
            top = sorted(yolo_boxes, key=lambda b: -b.get("confidence",0))[:3]
            box_info = "; ".join([f"{b.get('label')}({b.get('confidence',0):.2f})" for b in top])
            parts.append(f"Top model hints: {box_info}.")
    elif action == "PREVENTIVE_MAINTENANCE":
        parts.append("High failure risk detected; schedule preventive maintenance.")
        parts.append(f"Key signals: {dtype_summary}.")
    else:
        parts.append(f"Agent decision: {action}.")
    return " ".join(parts)
