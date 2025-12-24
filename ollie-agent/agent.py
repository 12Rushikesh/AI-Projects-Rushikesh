import requests
import json

LLAMA_URL = "http://127.0.0.1:8081/v1/chat/completions"
MODEL_NAME = "llama"

# Load system prompt
with open("system_prompt.txt", "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()

messages = [
    {"role": "system", "content": SYSTEM_PROMPT}
]

print("\n🤖 Ollie is ready. Type 'exit' to stop.\n")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ["exit", "quit"]:
        print("\nOllie: Take care. Keep learning 😊")
        break

    messages.append({"role": "user", "content": user_input})

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.6,
        "max_tokens": 512
    }

    response = requests.post(
        LLAMA_URL,
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload)
    )

    result = response.json()
    reply = result["choices"][0]["message"]["content"]

    messages.append({"role": "assistant", "content": reply})

    print(f"\nOllie: {reply}\n")
