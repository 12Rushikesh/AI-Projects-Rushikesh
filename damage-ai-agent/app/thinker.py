from app.llm_clients import call_thinker

def think(yolo_preds, vision_result):
    prompt = f"""
You are an AI decision agent.

YOLO predictions:
{yolo_preds}

Vision reasoning:
{vision_result}

Choose ONE action:
AUTO_ACCEPT
ASK_HUMAN
REJECT

Respond ONLY in JSON:
{{
  "action": "...",
  "reason": "short explanation"
}}
"""
    return call_thinker(prompt)
