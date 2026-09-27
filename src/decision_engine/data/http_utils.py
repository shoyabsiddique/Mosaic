"""Shared HTTP-with-retry helper for every data loader.

Live public APIs/mirrors are occasionally slow or return a transient error -- a
government API timeout killed a customer_support.py run once, then an unrelated 502
from Hugging Face's datasets-server killed a trust_safety.py run the same way. Every
loader making a live call should retry transient failures the same way, not just
whichever one happened to fail first and get a bespoke fix.
"""
from __future__ import annotations

import time
from typing import Optional

import requests


def get_with_retry(url: str, params: Optional[dict] = None, headers: Optional[dict] = None,
                    retries: int = 2, timeout: int = 60, backoff_s: float = 2.0) -> requests.Response:
    """GETs `url`, retrying transient network/HTTP errors (timeouts, 5xx, connection
    resets) with linear backoff. Raises the last error if every attempt fails."""
    last_error: Exception = RuntimeError("unreachable")
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp
        except requests.exceptions.RequestException as e:
            last_error = e
            if attempt < retries:
                time.sleep(backoff_s * (attempt + 1))
    raise last_error
