# service/agent_service.py
import time
from pathlib import Path
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))


from app.model import detect_damage
from app.agent import agent_decision
from app.agent_core import autonomous_agent
from app.auto_accept import auto_accept_save
from app.feedback import save_class_feedback, log_error
from app.rl_memory import log_rl_step
from app.agent_memory import record_confirmation, record_correction
from app.failure_predictor import estimate_failure_risk, make_history_summary
from app.explainer import explain_simple

INCOMING = Path(r"D:\Rushikesh\project\AI Agent\damage-ai-agent\data\incoming")
PROCESSED = Path(r"D:\Rushikesh\project\AI Agent\damage-ai-agent\data\processed")
INCOMING.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)

# configure classes/thresholds
CLASSES = ["dent", "hole", "rust", "not_damaged"]
AUTO_ACCEPT_CONF = 0.85  # tune later


def vectorize_preds(yolo_preds):
    """
    Convert yolo_preds (dict-like label->conf) into fixed vector summary.
    This enables logging and compact RL state features.
    """
    vec = {}
    for c in CLASSES:
        vec[f"conf_{c}"] = float(yolo_preds.get(c, 0) or 0)
    vec["mean_conf"] = sum(vec[f"conf_{c}"] for c in CLASSES) / max(1, len(CLASSES))
    return vec


def decide_and_act(image_path: Path):
    # 1) detect
    yolo_preds, yolo_boxes = detect_damage(str(image_path))

    # 2) compact decision
    yolo_decision = agent_decision(yolo_preds)

    # 3) fuller reasoning via agent_core
    agent_out = autonomous_agent(image_path=str(image_path), yolo_preds=yolo_preds, yolo_boxes=yolo_boxes)

    # 4) failure prediction (simple history lookup)
    # If you have historic detection logs for this container/item, load and summarize here.
    # For demo, we create an empty history -> risk low
    history = []  # TODO: load real history keyed by container id if you have it
    hist_summary = make_history_summary(history)
    failure_risk = estimate_failure_risk(hist_summary)
    agent_out["failure_risk"] = failure_risk

    # 5) produce explanation (human-readable)
    explanation = explain_simple(agent_out, yolo_boxes)

    # 6) decide action and automatic behavior
    action = agent_out.get("action", "ASK_HUMAN")
    # ensure it's AUTO_ACCEPT if confident (safety rule)
    conf = float(agent_out.get("confidence", 0.0) or 0.0)
    if conf >= AUTO_ACCEPT_CONF and action == "AUTO_ACCEPT":
        taken_action = "AUTO_ACCEPT"
    elif action == "PREVENTIVE_MAINTENANCE" or failure_risk > 0.7:
        taken_action = "PREVENTIVE_MAINTENANCE"
    else:
        taken_action = action

    # 7) execute action: auto_accept => save to dataset automatically
    if taken_action == "AUTO_ACCEPT":
        try:
            auto_accept_save(
                image_path=str(image_path),
                yolo_boxes=yolo_boxes,
                classes=CLASSES,
                W=agent_out.get("image_w", None) or 0,
                H=agent_out.get("image_h", None) or 0,
                dataset_img=Path("data/dataset/images/train"),
                dataset_lbl=Path("data/dataset/labels/train"),
            )
        except Exception as e:
            # if save fails, log error but continue
            log_error(str(image_path), yolo_decision.get("label", "unknown"), f"auto_accept_save_error: {e}")

    # 8) RL reward heuristic (smarter)
    # Default reward: small negative for asking human (inefficient).
    reward = -0.2 if taken_action == "ASK_HUMAN" else 0.0
    # If auto-accepted with high conf -> small positive
    if taken_action == "AUTO_ACCEPT" and conf >= AUTO_ACCEPT_CONF:
        reward += 1.2
    # Penalize missed damage (false negative) heavily: if agent label is not 'not_damaged' but yolo detects damage, small penalty not applicable here
    # If failure_risk high and not preventive -> penalty
    if failure_risk > 0.7 and taken_action != "PREVENTIVE_MAINTENANCE":
        reward -= 2.0

    # 9) log RL step
    log_rl_step(
        state={"yolo_summary": vectorize_preds(yolo_preds), "num_boxes": len(yolo_boxes)},
        action=taken_action,
        reward=reward,
        info={"image": image_path.name, "explanation": explanation, "failure_risk": failure_risk}
    )

    # 10) create a human-readable audit record (also used as dataset metadata)
    audit = {
        "image": image_path.name,
        "action": taken_action,
        "yolo_decision": yolo_decision,
        "agent_out": agent_out,
        "failure_risk": failure_risk,
        "explanation": explanation,
        "time": time.time()
    }

    # 11) save feedback logs and move file to processed
    try:
        save_class_feedback(str(image_path), user_label=taken_action, model_label=yolo_decision.get("label", "unknown"), confidence=conf)
    except Exception:
        pass

    # save audit json
    audit_dir = Path("data/audit")
    audit_dir.mkdir(parents=True, exist_ok=True)
    with open(audit_dir / f"{image_path.stem}.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    # move processed file
    dest = PROCESSED / image_path.name
    shutil.move(str(image_path), str(dest))

    return audit


def run_loop(poll_interval=1.0):
    print("Agent service started — monitoring data/incoming/")
    while True:
        files = sorted(INCOMING.glob("*.jpg")) + sorted(INCOMING.glob("*.png"))
        if not files:
            time.sleep(poll_interval)
            continue
        for f in files:
            try:
                audit = decide_and_act(f)
                print(f"Processed {f.name} -> action: {audit['action']}; risk: {audit['failure_risk']:.2f}")
            except Exception as exc:
                print(f"Error processing {f}: {exc}")
        time.sleep(poll_interval)
MODE = "SHADOW"  # SHADOW | AUTO

def act_on_decision(agent_output, image_path):
    if MODE == "SHADOW":
        print("[SHADOW MODE]")
        print("Would take action:", agent_output)
        return

    if MODE == "AUTO":
        if agent_output["action"] == "AUTO_ACCEPT":
            print("Executing AUTO_ACCEPT on", image_path)


if __name__ == "__main__":
    run_loop()
