#!/usr/bin/env python3
"""Minimal client for an OpenAI-compatible chat completions endpoint."""
import os
import itertools
import threading
from concurrent.futures import ThreadPoolExecutor

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

_BASE_URL2 = os.getenv("BASE_URL2", "").rstrip("/")

REQUEST_TIMEOUT = 1080

_POOL_SIZE = int(os.getenv("HTTP_POOL_SIZE", "200"))

# The second provider limits concurrency per API key, so we spread load across
# as many keys as are configured (KEY1, KEY2, ...). Each worker thread is pinned
# to one key, and keys are handed out round-robin, so no key ever sees more than
# SECOND_PROVIDER_WORKERS_PER_KEY concurrent requests.
SECOND_PROVIDER_WORKERS_PER_KEY = 20


def _discover_second_provider_keys() -> list[str]:
    keys = []
    i = 1
    while True:
        token = os.getenv(f"KEY{i}")
        if not token:
            break
        keys.append(token)
        i += 1
    if not keys:
        legacy = os.getenv("AUTH_TOKEN2")
        if legacy:
            keys = [legacy]
    return keys

_SECOND_PROVIDER_KEYS = _discover_second_provider_keys()

_key_cycle = itertools.cycle(range(len(_SECOND_PROVIDER_KEYS))) if _SECOND_PROVIDER_KEYS else None
_key_lock = threading.Lock()
_thread_local = threading.local()


def second_provider_num_keys() -> int:
    return len(_SECOND_PROVIDER_KEYS)


def _next_key_index() -> int:
    with _key_lock:
        return next(_key_cycle)


def assign_thread_key() -> None:
    """ThreadPoolExecutor initializer: pin a second-provider key to this worker thread."""
    if _key_cycle is not None:
        _thread_local.key_index = _next_key_index()


def _current_second_provider_key() -> str:
    if not _SECOND_PROVIDER_KEYS:
        raise RuntimeError("No second-provider keys configured (set KEY1, KEY2, ... in .env.local)")
    idx = getattr(_thread_local, "key_index", None)
    if idx is None:
        # Thread was not initialized via assign_thread_key (e.g. a bare call); fall back to round-robin.
        idx = _next_key_index()
        _thread_local.key_index = idx
    return _SECOND_PROVIDER_KEYS[idx]


def make_pool(per_key_workers: int, second_provider: bool) -> ThreadPoolExecutor:
    """Build a ThreadPoolExecutor sized for the target provider.

    For the second provider the pool spans every configured key, so total
    concurrency is per_key_workers * num_keys, with each key pinned to its own
    workers. For the primary provider per_key_workers is simply the pool size.
    """
    if second_provider:
        n = max(1, second_provider_num_keys())
        return ThreadPoolExecutor(max_workers=per_key_workers * n, initializer=assign_thread_key)
    return ThreadPoolExecutor(max_workers=per_key_workers)


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
    second_provider: bool = False,
    enable_thinking: bool | None = None,
) -> dict:
    """Send a single user message to the chat completions endpoint.

    enable_thinking toggles the reasoning phase on reasoning models. vLLM forwards
    chat_template_kwargs to the model's chat template; Qwen3 and similar templates
    read enable_thinking to switch the <think> phase on or off. Leave it None to use
    the server/model default; set False to suppress reasoning (so the token budget is
    spent on the answer, not an internal deliberation that can exhaust max_tokens).
    """
    if second_provider:
        if not _BASE_URL2:
            raise RuntimeError("BASE_URL2 is not set in .env.local")
        base_url = _BASE_URL2
        headers = _make_headers(_current_second_provider_key())
    else:
        base_url = _base_url()
        headers = _HEADERS

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