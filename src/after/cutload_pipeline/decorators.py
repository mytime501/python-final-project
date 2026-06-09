from __future__ import annotations

import functools
import logging
import time
import tracemalloc
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])
LOGGER = logging.getLogger("cutload_pipeline")


def timed(func: F) -> F:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            LOGGER.info("%s elapsed_sec=%.6f", func.__name__, elapsed)

    return wrapper  # type: ignore[return-value]


def memory_traced(func: F) -> F:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        already_tracing = tracemalloc.is_tracing()
        if not already_tracing:
            tracemalloc.start()
        try:
            return func(*args, **kwargs)
        finally:
            current, peak = tracemalloc.get_traced_memory()
            LOGGER.info("%s peak_memory_mb=%.3f current_memory_mb=%.3f", func.__name__, peak / 1e6, current / 1e6)
            if not already_tracing:
                tracemalloc.stop()

    return wrapper  # type: ignore[return-value]


def log_call(func: F) -> F:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        LOGGER.info("call %s", func.__name__)
        return func(*args, **kwargs)

    return wrapper  # type: ignore[return-value]
