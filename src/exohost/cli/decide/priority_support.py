# Файл `priority_support.py` слоя `cli`.
#
# Этот файл отвечает только за:
# - CLI-команды и orchestration entrypoints;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - CLI-команды или support-модули этого же домена;
# - пользовательский запуск через `python -m exohost.cli.main`.

from __future__ import annotations

import argparse

from exohost.posthoc.priority_integration import PriorityIntegrationConfig
from exohost.ranking.priority_score import (
    DEFAULT_HOST_SCORE_COLUMN,
    DEFAULT_PRIORITY_THRESHOLDS,
    PriorityThresholds,
)


def build_priority_integration_config_from_namespace(
    namespace: argparse.Namespace,
) -> PriorityIntegrationConfig:
    # Собираем explicit priority integration config из CLI namespace.
    return PriorityIntegrationConfig(
        host_score_column=_coerce_string(
            namespace.host_score_column,
            field_name="host_score_column",
            default=DEFAULT_HOST_SCORE_COLUMN,
        ),
        thresholds=PriorityThresholds(
            high_min=_coerce_optional_float(
                namespace.priority_high_min,
                default=DEFAULT_PRIORITY_THRESHOLDS.high_min,
            ),
            medium_min=_coerce_optional_float(
                namespace.priority_medium_min,
                default=DEFAULT_PRIORITY_THRESHOLDS.medium_min,
            ),
            low_priority_class_cap=DEFAULT_PRIORITY_THRESHOLDS.low_priority_class_cap,
            evolved_stage_penalty=DEFAULT_PRIORITY_THRESHOLDS.evolved_stage_penalty,
        ),
    )


def _coerce_optional_float(value: object, *, default: float) -> float:
    if value is None:
        return float(default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("Expected optional numeric CLI value.")
    return float(value)


def _coerce_string(value: object, *, field_name: str, default: str) -> str:
    if value is None:
        return default
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    normalized = value.strip()
    if not normalized:
        return default
    return normalized
