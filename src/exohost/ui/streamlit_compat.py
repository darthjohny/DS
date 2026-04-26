# Файл `streamlit_compat.py` слоя `ui`.
#
# Этот файл отвечает только за:
# - совместимость helper-слоя `ui` с окружением без установленного `streamlit`;
# - безопасные no-op обертки вокруг кэш-декораторов Streamlit для unit-тестов.
#
# Следующий слой:
# - loader- и helper-модули интерфейса;
# - unit-тесты, которые не должны зависеть от отдельного UI-окружения.

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any, ParamSpec, TypeVar

P = ParamSpec("P")
T = TypeVar("T")

_streamlit: Any | None
try:
    import streamlit as _streamlit
except ModuleNotFoundError:
    _streamlit = None


def cache_data(
    *,
    show_spinner: bool = False,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    # В production используем настоящий Streamlit cache, а в unit-тестах подменяем его no-op
    # декоратором, чтобы helper-модули импортировались без отдельного UI-окружения.
    if _streamlit is not None:
        return _streamlit.cache_data(show_spinner=show_spinner)

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
            return func(*args, **kwargs)

        return wrapped

    return decorator


def clear_cached_call(func: Any) -> None:
    # После создания нового `run_dir` аккуратно сбрасываем только те кэши,
    # которые реально поддерживают `.clear()`.
    clear_method = getattr(func, "clear", None)
    if callable(clear_method):
        clear_method()


__all__ = [
    "cache_data",
    "clear_cached_call",
]
