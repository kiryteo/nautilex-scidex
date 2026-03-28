"""LLM client wrapper using GitHub Copilot Models via OpenAI-compatible API."""

from __future__ import annotations

import os
import time
import logging

from openai import OpenAI, APIError, RateLimitError, APITimeoutError

logger = logging.getLogger(__name__)

_client: OpenAI | None = None


def get_client() -> OpenAI:
    """Get a singleton OpenAI client pointing at GitHub Copilot Models."""
    global _client
    if _client is None:
        token = os.environ.get("GITHUB_TOKEN")
        if not token:
            raise RuntimeError(
                "GITHUB_TOKEN not set. Export your GitHub token:\n"
                "  export GITHUB_TOKEN=$(gh auth token)"
            )
        _client = OpenAI(
            base_url="https://models.inference.ai.azure.com",
            api_key=token,
        )
    return _client


def chat(
    messages: list[dict],
    model: str = "gpt-4o",
    temperature: float = 0.7,
    max_tokens: int = 2048,
    retries: int = 3,
    backoff: float = 2.0,
) -> str:
    """Send a chat completion request with retry logic.

    Args:
        messages: List of message dicts (role, content).
        model: Model name (gpt-4o or gpt-4o-mini).
        temperature: Sampling temperature.
        max_tokens: Max tokens in response.
        retries: Number of retries on transient failures.
        backoff: Backoff multiplier between retries.

    Returns:
        The assistant's response text.
    """
    client = get_client()
    last_error = None

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except (RateLimitError, APITimeoutError) as e:
            last_error = e
            wait = backoff**attempt
            logger.warning(
                f"LLM request failed (attempt {attempt + 1}/{retries}): {e}. "
                f"Retrying in {wait:.1f}s..."
            )
            time.sleep(wait)
        except APIError as e:
            last_error = e
            if e.status_code and e.status_code >= 500:
                wait = backoff**attempt
                logger.warning(
                    f"LLM server error (attempt {attempt + 1}/{retries}): {e}. "
                    f"Retrying in {wait:.1f}s..."
                )
                time.sleep(wait)
            else:
                raise

    raise RuntimeError(f"LLM request failed after {retries} retries: {last_error}")
