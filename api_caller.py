#!/usr/bin/env python3
import json
import requests

import os
from dotenv import load_dotenv

load_dotenv(".env.local")

AUTH_TOKEN: str | None = os.getenv("AUTH_TOKEN")
BASE_URL = "https://ai.ufal.mff.cuni.cz/api/v1"

#MODEL_ID = "LLM3-AMD-MI210.llama3.3:latest" # LLAMA
MODEL_ID = "LLM3-AMD-MI210.gpt-oss:120b" # Gpt


HEADERS = {
    "Content-Type": "application/json",
}
if AUTH_TOKEN:
    HEADERS["Authorization"] = f"Bearer {AUTH_TOKEN}"

# ----------------------------------------------------------------------
# Helper: build the request payload
# ----------------------------------------------------------------------
def build_payload(user_message: str, model_id: str, max_tokens: int = 512, temperature: float = 0) -> dict:
    return {
        "model": model_id,
        "messages": [
            {"role": "user", "content": user_message}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

# ----------------------------------------------------------------------
# Call the API
# ----------------------------------------------------------------------
def call_api(user_message: str, model_id: str = MODEL_ID) -> dict():
    payload = build_payload(user_message, model_id)

    try:
        resp = requests.post(
            f"{BASE_URL}/chat/completions",
            headers=HEADERS,
            json=payload,
            timeout=300, # seconds
        )
        resp.raise_for_status()   # raise on HTTP 4xx/5xx
    except requests.HTTPError as http_err:
        # Try to decode a JSON error body – most errors are JSON.
        try:
            err_body = resp.json()
            pretty_err = json.dumps(err_body, indent=2, ensure_ascii=False)
        except Exception:
            # Fallback to raw text if it isn’t JSON.
            pretty_err = resp.text[:500] + ("…" if len(resp.text) > 500 else "")
        raise SystemExit(
            f"\n❌ HTTP {resp.status_code} error while calling API:\n{pretty_err}"
        ) from http_err
    except requests.RequestException as exc:
        raise SystemExit(f"❌ Request failed: {exc}") from exc

    return resp.json()
