#!/usr/bin/env python3
"""Minimal client for an OpenAI-compatible chat completions endpoint."""
import os

import requests
from requests.adapters import HTTPAdapter
from dotenv import load_dotenv

load_dotenv(".env.local")

# Resolved on first call, not at import. `from utils import <anything>` runs this
# module via the package __init__, so reading os.environ["BASE_URL"] here made every
# script in the repo -- including ones that never touch the API -- fail at import
# time when .env.local was absent. Missing config is now an error where it matters.
def _base_url() -> str:
    url = os.getenv("BASE_URL")
    if not url:
        raise RuntimeError("BASE_URL is not set (put it in .env.local)")
    return url.rstrip("/")


AUTH_TOKEN = os.getenv("AUTH_TOKEN")

REQUEST_TIMEOUT = 1080

_POOL_SIZE = int(os.getenv("HTTP_POOL_SIZE", "200"))


def _make_headers(auth_token: str | None) -> dict:
    headers = {"Content-Type": "application/json"}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
    return headers

_HEADERS = _make_headers(AUTH_TOKEN)


def _make_session() -> requests.Session:
    session = requests.Session()
    adapter = HTTPAdapter(pool_connections=_POOL_SIZE, pool_maxsize=_POOL_SIZE)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

_SESSION = _make_session()

def call_api(
    user_message: str,
    model_id: str,
    *,
    max_tokens: int = 8096,
    temperature: float = 0.0,
    enable_thinking: bool | None = None,
) -> dict:
    """Send a single user message to the chat completions endpoint.

    enable_thinking toggles the reasoning phase on reasoning models. vLLM forwards
    chat_template_kwargs to the model's chat template; Qwen3 and similar templates
    read enable_thinking to switch the <think> phase on or off. Leave it None to use
    the server/model default; set False to suppress reasoning (so the token budget is
    spent on the answer, not an internal deliberation that can exhaust max_tokens).
    """
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": user_message}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if enable_thinking is not None:
        payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}

    try:
        resp = _SESSION.post(
            f"{_base_url()}/chat/completions",
            headers=_HEADERS,
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
    except requests.HTTPError as err:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}") from err
    except requests.RequestException as err:
        raise RuntimeError(f"Request failed: {err}") from err

    return resp.json()