"""Shared utilities for jpy-ml Python modules."""
import os
import threading
import contextlib

_stderr_lock = threading.Lock()


@contextlib.contextmanager
def suppress_stderr_fd():
    """Context manager to suppress stderr at the file descriptor level.

    Thread-safe: acquires a lock so concurrent callers do not corrupt fd state.
    Suppresses the harmless 'Unrecognized option: -c' JVM error from safetensors
    Rust extension fork on macOS.
    """
    with _stderr_lock:
        old_stderr = os.dup(2)
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull_fd, 2)
        os.close(devnull_fd)
        try:
            yield
        finally:
            os.dup2(old_stderr, 2)
            os.close(old_stderr)
