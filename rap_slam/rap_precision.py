"""Scoped precision policy for RAP-SLAM inference (M2/F02).

The lock serializes cooperating backend calls. PyTorch's matmul policy is
process-global: unrelated torch callers must use this scope or a separate process.
FP16 neural inference remains enabled; float32 geometry uses IEEE matmul.
"""
from contextlib import contextmanager
from functools import wraps
from threading import RLock

_LOCK = RLock()


@contextmanager
def ieee_matmul():
    import torch

    with _LOCK:
        previous = torch.get_float32_matmul_precision()
        try:
            torch.set_float32_matmul_precision("highest")
            yield
        finally:
            torch.set_float32_matmul_precision(previous)


def precise_registration(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with ieee_matmul():
            return function(*args, **kwargs)
    return wrapped
