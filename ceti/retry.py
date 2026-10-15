"""Retry helper with exponential backoff. Sync, stdlib-only.

Used for cloud API calls, where a single transient failure
(timeout, 5xx, dropped connection) shouldn't kill the task.
"""

from __future__ import annotations

import time


def retry(fn, attempts: int = 3, delay: float = 1.0,
          backoff: float = 2.0,
          exceptions: tuple[type[BaseException], ...] = (Exception,),
          on_retry=None):
    """Call fn(); retry on `exceptions` up to `attempts` times.

    Waits delay * backoff**n between attempts. Returns fn()'s result.
    Raises the last exception if every attempt fails.
    """
    if attempts < 1:
        raise ValueError("attempts must be >= 1")
    last: BaseException | None = None
    for n in range(attempts):
        try:
            return fn()
        except exceptions as e:  # noqa: BLE001
            last = e
            if not getattr(e, "retryable", True):
                raise  # futile (e.g. HTTP 4xx) — don't burn time
            if n < attempts - 1:
                if on_retry is not None:
                    on_retry(n + 1, e)
                time.sleep(delay * (backoff ** n))
    assert last is not None
    raise last
