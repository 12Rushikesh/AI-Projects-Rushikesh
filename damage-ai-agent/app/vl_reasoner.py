import json
from app.llm_clients import call_vl

def vision_reason(image_path, yolo_boxes):
    prompt = f"""
You are a container damage inspection expert.

YOLO detections:
{json.dumps(yolo_boxes, indent=2)}

Look at the image and detections.
Respond ONLY in JSON:

{{
  "damage_present": true/false,
  "damage_types": ["dent","hole","rust"],
  "confidence": "low/medium/high",
  "notes": "short explanation"
}}
"""
    return call_vl(prompt, image_path)
