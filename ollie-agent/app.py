# ============================================
# Ollie AI Agent Backend (Flask + Persistent Memory)
# ============================================

from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import json
import os

# ============================================
# CONFIG
# ============================================

LLAMA_URL = "http://127.0.0.1:8081/v1/chat/completions"
MODEL_NAME = "llama"
MEMORY_FILE = "memory.json"

# ============================================
# LOAD SYSTEM PROMPT
# ============================================

with open("system_prompt.txt", "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()

# ============================================
# APP INIT
# ============================================

app = Flask(__name__)
CORS(app)

# ============================================
# LOAD / SAVE MEMORY
# ============================================

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"name": None, "goal": None}

def save_memory(mem):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(mem, f, indent=2)

memory = load_memory()

# ============================================
# CONVERSATION STATE
# ============================================

messages = [
    {"role": "system", "content": SYSTEM_PROMPT}
]

# ============================================
# HELPERS
# ============================================

def extract_memory(user_text: str):
    text = user_text.lower()

    if "my name is" in text:
        try:
            memory["name"] = user_text.split("my name is")[-1].strip().split()[0]
        except:
            pass

    if "my goal is" in text:
        try:
            memory["goal"] = user_text.split("my goal is")[-1].strip()
        except:
            pass

    save_memory(memory)

def memory_context():
    ctx = []
    if memory.get("name"):
        ctx.append(f"The user's name is {memory['name']}.")
    if memory.get("goal"):
        ctx.append(f"The user's goal is {memory['goal']}.")
    return " ".join(ctx)

# ============================================
# ROUTES
# ============================================

@app.route("/chat", methods=["POST"])
def chat():
    global messages

    user_message = request.json.get("message", "").strip()
    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    # Extract & persist memory
    extract_memory(user_message)

    # Inject memory
    mem_ctx = memory_context()
    if mem_ctx:
        messages.append({"role": "system", "content": mem_ctx})

    # User message
    messages.append({"role": "user", "content": user_message})

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.6,
        "max_tokens": 512
    }

    try:
        r = requests.post(
            LLAMA_URL,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=120
        )
        r.raise_for_status()
        data = r.json()
        reply = data["choices"][0]["message"]["content"]
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    messages.append({"role": "assistant", "content": reply})

    return jsonify({
        "reply": reply,
        "memory": memory
    })

# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    print("🤖 Ollie backend with PERSISTENT MEMORY running")
    print("➡ http://127.0.0.1:5000/chat")
    app.run(host="127.0.0.1", port=5000, debug=True)
