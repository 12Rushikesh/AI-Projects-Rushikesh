HIGH_CONF = 0.75
MID_CONF = 0.45

def agent_decision(preds):
    if not preds:
        return {"action": "HUMAN_ONLY", "label": None, "confidence": 0.0}

    label = max(preds, key=preds.get)
    conf = preds[label]

    if conf >= HIGH_CONF:
        action = "ACCEPT"
    elif conf >= MID_CONF:
        action = "ASK_HUMAN"
    else:
        action = "HUMAN_ONLY"

    return {
        "action": action,
        "label": label,
        "confidence": round(conf, 3)
    }
