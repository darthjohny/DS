"""Нормализация и label-contract для Gaussian router."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import pandas as pd

SPEC_CLASSES: list[str] = ["A", "B", "F", "G", "K", "M", "O"]
EVOLUTION_STAGES: list[str] = ["dwarf", "evolved"]


def is_missing_scalar(value: Any) -> bool:
    """Вернуть `True`, если скаляр считается пропущенным значением."""
    if value is None or value is pd.NA:
        return True
    try:
        return bool(math.isnan(float(value)))
    except (TypeError, ValueError):
        return False


def has_missing_values(values: Iterable[Any]) -> bool:
    """Проверить, что среди значений нет пропусков в ключевых признаках."""
    return any(is_missing_scalar(value) for value in values)


def normalize_spec_class(spec_class: Any) -> str:
    """Нормализовать спектральный класс к контракту router."""
    value = str(spec_class).strip().upper()
    if value not in SPEC_CLASSES:
        raise ValueError(f"Unsupported spec_class: {spec_class}")
    return value


def normalize_evolution_stage(evolution_stage: Any) -> str:
    """Нормализовать эволюционную стадию к контракту router."""
    value = str(evolution_stage).strip().lower()
    if value not in EVOLUTION_STAGES:
        raise ValueError(f"Unsupported evolution_stage: {evolution_stage}")
    return value


def make_router_label(spec_class: Any, evolution_stage: Any) -> str:
    """Собрать составной router label из класса и стадии."""
    return (
        f"{normalize_spec_class(spec_class)}_"
        f"{normalize_evolution_stage(evolution_stage)}"
    )


def split_router_label(router_label: str) -> tuple[str, str]:
    """Разбить составной router label обратно на класс и стадию."""
    try:
        spec_class, evolution_stage = router_label.split("_", 1)
    except ValueError as exc:
        raise ValueError(f"Invalid router label: {router_label}") from exc
    return spec_class, evolution_stage


__all__ = [
    "EVOLUTION_STAGES",
    "SPEC_CLASSES",
    "has_missing_values",
    "is_missing_scalar",
    "make_router_label",
    "normalize_evolution_stage",
    "normalize_spec_class",
    "split_router_label",
]
