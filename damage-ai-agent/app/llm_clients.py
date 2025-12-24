import requests
import base64
import json
from PIL import Image
import io

# =====================================================
# Vision LLM (Qwen-VL) → PORT 8080
# =====================================================
VL_URL = "http://127.0.0.1:8080/v1/chat/completions"
VL_MODEL = "Qwen3VL-2B-Instruct"

def call_vl(prompt, image_path, temperature=0.2):
    try:
        img = Image.open(image_path).convert("RGB")
        img.thumbnail((768, 768))

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        payload = {
            "model": VL_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "images": [img_b64],   # 🔥 THIS IS THE KEY
            "temperature": temperature
        }

        r = requests.post(VL_URL, json=payload, timeout=120)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    except Exception as e:
        return json.dumps({
            "damage_present": False,
            "damage_types": [],
            "confidence": "low",
            "notes": f"Vision model error: {str(e)}"
        })

    try:
        img = Image.open(image_path).convert("RGB")
        img.thumbnail((768, 768))

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()  # BASE64 ONLY

        payload = {
            "model": VL_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image",
                            "image": img_b64
                        }
                    ]
                }
            ],
            "temperature": temperature
        }

        r = requests.post(VL_URL, json=payload, timeout=120)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    except Exception as e:
        return json.dumps({
            "damage_present": False,
            "confidence": "low",
            "notes": f"Vision model error: {str(e)}"
        })


# =====================================================
# Thinking LLM (Qwen3-4B) → PORT 8081
# =====================================================
THINK_URL = "http://127.0.0.1:8081/v1/chat/completions"
THINK_MODEL = "Qwen3-4B-Thinking"

def call_thinker(prompt, temperature=0.2):
    try:
        payload = {
            "model": THINK_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": temperature
        }

        r = requests.post(THINK_URL, json=payload, timeout=120)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    except Exception as e:
        # SAFE FALLBACK
        return json.dumps({
            "action": "ASK_HUMAN",
            "confidence": "low",
            "reason": f"Thinking model error: {str(e)}"
        })
