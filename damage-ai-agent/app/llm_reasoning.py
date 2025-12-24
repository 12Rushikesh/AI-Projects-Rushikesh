import requests
import json

LLM_URL = "http://127.0.0.1:8081/v1/chat/completions"

def call_llm(prompt):
    payload = {
        "model": "local",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.2
    }

    r = requests.post(LLM_URL, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]
