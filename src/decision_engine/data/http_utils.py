"""Shared HTTP-with-retry helper for every data loader.

Live public APIs/mirrors are occasionally slow or return a transient error -- a
government API timeout killed a customer_support.py run once, then an unrelated 502
from Hugging Face's datasets-server killed a trust_safety.py run the same way. Every
loader making a live call should retry transient failures the same way, not just
whichever one happened to fail first and get a bespoke fix.

Two things this deliberately does NOT treat as retryable:
- A 403 (a real CFPB response hit from a Colab notebook, not this local machine --
  cloud-provider IP ranges get blocked by some `.gov`/WAF-protected sites more often
  than residential ones) or other 4xx: retrying an identical request won't change a
  deliberate block or a bad request, it just delays the real failure. Only 429 and
  5xx are retried.
- A missing/default User-Agent contributing to that block in the first place --
  `requests`' bare `python-requests/x.y` string is a common, cheap bot-signal for
  simple WAFs. A realistic default is sent unless the caller overrides it.
"""
from __future__ import annotations

import time
from typing import Optional

import requests

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def get_with_retry(url: str, params: Optional[dict] = None, headers: Optional[dict] = None,
                    retries: int = 2, timeout: int = 60, backoff_s: float = 2.0) -> requests.Response:
    """GETs `url`, retrying transient failures (timeouts, connection resets, 429/5xx)
    with linear backoff. A non-retryable HTTP error (403, 404, other 4xx) raises
    immediately without wasting time on identical retries. Raises the last error if
    every retryable attempt fails."""
    merged_headers = {"User-Agent": DEFAULT_USER_AGENT, **(headers or {})}

    last_error: Exception = RuntimeError("unreachable")
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, params=params, headers=merged_headers, timeout=timeout)
            resp.raise_for_status()
            return resp
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            if status is not None and status not in RETRYABLE_STATUS_CODES:
                raise
            last_error = e
        except requests.exceptions.RequestException as e:
            last_error = e
        if attempt < retries:
            time.sleep(backoff_s * (attempt + 1))
    raise last_error
