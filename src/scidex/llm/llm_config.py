"""
Shared LLM configuration for all Nautilex hackathon challenges.
Uses GitHub Copilot Models via the OpenAI-compatible API.

Usage:
    from shared.llm_config import get_llm_client, get_s2_session

    # For LLM calls
    client = get_llm_client()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello"}]
    )

    # For Semantic Scholar API calls (rate-limited to 1 req/sec)
    session = get_s2_session()
    resp = session.get("https://api.semanticscholar.org/graph/v1/paper/search", params={"query": "test"})
"""

import os
import time
import threading
from pathlib import Path

from dotenv import load_dotenv

# Load .env from shared directory
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)


def get_llm_client(model: str = "gpt-4o"):
    """Get an OpenAI-compatible client pointing at GitHub Copilot Models."""
    from openai import OpenAI

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError(
            "GITHUB_TOKEN not set. Export your GitHub token:\n"
            "  export GITHUB_TOKEN=$(gh auth token)"
        )

    return OpenAI(
        base_url="https://models.inference.ai.azure.com",
        api_key=token,
    )


class _RateLimitedSession:
    """requests.Session wrapper that enforces 1 req/sec for Semantic Scholar."""

    def __init__(self):
        import requests

        self._session = requests.Session()
        api_key = os.environ.get("S2_API_KEY", "")
        if api_key:
            self._session.headers["x-api-key"] = api_key
        self._lock = threading.Lock()
        self._last_request = 0.0

    def _throttle(self):
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_request
            if elapsed < 1.0:
                time.sleep(1.0 - elapsed)
            self._last_request = time.monotonic()

    def get(self, url, **kwargs):
        self._throttle()
        return self._session.get(url, **kwargs)

    def post(self, url, **kwargs):
        self._throttle()
        return self._session.post(url, **kwargs)


_s2_session = None


def get_s2_session():
    """Get a rate-limited session for Semantic Scholar API (1 req/sec)."""
    global _s2_session
    if _s2_session is None:
        _s2_session = _RateLimitedSession()
    return _s2_session
