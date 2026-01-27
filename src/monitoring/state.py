'''
A simple tracking state module to the state of the Remaining Useful Life (RUL)
'''

from threading import Lock
from typing import Optional

_latest_rul: Optional[float] = None
_lock = Lock()

def set_latest_rul(value: float) -> None:
    global _latest_rul
    with _lock:
        _latest_rul = value

def get_latest_rul() -> Optional[float]:
    with _lock:
        return _latest_rul
