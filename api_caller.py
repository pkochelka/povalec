#!/usr/bin/env python3
"""
Demo: call the UFAL AI model
https://ai.ufal.mff.cuni.cz/c/76b3d87c-1299-4ea4-90e3-76bd2e919ccf
"""

import json
import sys
from pathlib import Path

import requests

# ----------------------------------------------------------------------
# 1️⃣  Configuration
# ----------------------------------------------------------------------
BASE_URL = "https://ai.ufal.mff.cuni.cz/api/v1"
#MODEL_ID = "LLM3-AMD-MI210.llama3.3:latest" # LLAMA
MODEL_ID = "LLM3-AMD-MI210.gpt-oss:120b" # Gpt

# If the service ever requires a bearer token you can set it here.
# For the public endpoint no token is needed, so AUTH_TOKEN stays None.
AUTH_TOKEN: str | None = "sk-413c5bcb19af4e81a8a94af9a995a938"   # e.g. "my‑jwt‑or‑api‑key"

HEADERS = {
    "Content-Type": "application/json",
}
if AUTH_TOKEN:
    HEADERS["Authorization"] = f"Bearer {AUTH_TOKEN}"

def find_ids(obj):
    """Recursively find all 'id' values in nested dicts/lists."""
    ids = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "id":
                ids.append(v)
            else:
                ids.extend(find_ids(v))
    elif isinstance(obj, list):
        for item in obj:
            ids.extend(find_ids(item))
    return ids

# ----------------------------------------------------------------------
# 2️⃣  Helper: build the request payload
# ----------------------------------------------------------------------
def build_payload(user_message: str, model_id: str, max_tokens: int = 512, temperature: float = 0) -> dict:
    return {
        "model": model_id,
        "messages": [
            {"role": "user", "content": user_message}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        # optional flags – uncomment if you need them
        #"stream": True,
        # "top_p": 0.95,
    }

# ----------------------------------------------------------------------
# 3️⃣  Call the API
# ----------------------------------------------------------------------
def call_api(user_message: str, model_id: str = MODEL_ID) -> dict():
    payload = build_payload(user_message, model_id)

    try:
        resp = requests.post(
            f"{BASE_URL}/chat/completions",
            headers=HEADERS,
            json=payload,
            timeout=300,               # seconds
        )
        resp.raise_for_status()   # raise on HTTP 4xx/5xx
    except requests.HTTPError as http_err:
        # Try to decode a JSON error body – most UFAL errors are JSON.
        try:
            err_body = resp.json()
            pretty_err = json.dumps(err_body, indent=2, ensure_ascii=False)
        except Exception:
            # Fallback to raw text if it isn’t JSON.
            pretty_err = resp.text[:500] + ("…" if len(resp.text) > 500 else "")
        raise SystemExit(
            f"\n❌ HTTP {resp.status_code} error while calling UFAL API:\n{pretty_err}"
        ) from http_err
    except requests.RequestException as exc:
        raise SystemExit(f"❌ Request failed: {exc}") from exc

    return resp.json()

def create_chat(system_prompt: str, model_id: str = MODEL_ID, title: str = "Persistent chat"):
    payload = {
        "chat": {
            "model": model_id,
            "title": title,
            "system": system_prompt
        }
    }

    resp = requests.post(
        f"{BASE_URL}/chats/new",
        headers=HEADERS,
        json=payload,
        timeout=300,
    )

    print(resp.json())

# ----------------------------------------------------------------------
# 4️⃣  Simple CLI interface (optional)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Take the whole command line after the script name as the query
        query = " ".join(sys.argv[1:])
    else:
        # Interactive fallback
        query = input("💬 Your question: ").strip()

    if not query:
        sys.exit("⚠️ No input – exiting.")

    print("\n⏳ Generating answer …")
    answer = call_api(query)
    print("\n🤖 Answer:\n")
    print(answer)