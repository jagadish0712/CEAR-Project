# dashboard/state_store.py
from typing import Any, Dict, Optional
from django.core.cache import cache

KEY_FMT = "mstate:{id}"

def _key(measurement_id: int) -> str:
    return KEY_FMT.format(id=measurement_id)

def get_state(measurement_id: int) -> Dict[str, Any]:
    return cache.get(_key(measurement_id), {}) or {}

def set_state(measurement_id: int, state: Dict[str, Any], ttl: Optional[int] = None) -> None:
    cache.set(_key(measurement_id), state, timeout=ttl)

def update_state(measurement_id: int, **patch: Any) -> Dict[str, Any]:
    state = get_state(measurement_id)
    state.update({k: v for k, v in patch.items() if v is not None})
    set_state(measurement_id, state)
    return state

def clear_state(measurement_id: int) -> None:
    cache.delete(_key(measurement_id))
