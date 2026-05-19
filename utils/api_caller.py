#!/usr/bin/env python3
"""Minimal client for an OpenAI-compatible chat completions endpoint."""
import os
import requests
from dotenv import load_dotenv

load_dotenv(".env.local")

BASE_URL = os.environ["BASE_URL"].rstrip("/")
AUTH_TOKEN = os.getenv("AUTH_TOKEN")

_BASE_URL2 = os.getenv("BASE_URL2", "").rstrip("/")
_AUTH_TOKEN2 = os.getenv("AUTH_TOKEN2")

REQUEST_TIMEOUT = 540

def _make_headers(auth_token: str | None) -> dict:
    headers = {"Content-Type": "application/json"}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
    return headers

_HEADERS = _make_headers(AUTH_TOKEN)

def call_api(
    user_message: str,
    model_id: str,
    *,
    max_tokens: int = 512,
    temperature: float = 0.0,
    second_provider: bool = False,
) -> dict:
    """Send a single user message to the chat completions endpoint."""
    if second_provider:
        if not _BASE_URL2:
            raise RuntimeError("BASE_URL2 is not set in .env.local")
        base_url = _BASE_URL2
        headers = _make_headers(_AUTH_TOKEN2)
    else:
        base_url = BASE_URL
        headers = _HEADERS

    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": user_message}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    try:
        resp = requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
    except requests.HTTPError as err:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}") from err
    except requests.RequestException as err:
        raise RuntimeError(f"Request failed: {err}") from err

    return resp.json()