# Файл `quality_gate_support.py` слоя `cli`.
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

from exohost.posthoc.quality_gate_tuning import (
    DEFAULT_QUALITY_GATE_TUNING_CONFIG,
    QualityGateTuningConfig,
)


def build_quality_gate_tuning_config_from_namespace(
    namespace: argparse.Namespace,
) -> QualityGateTuningConfig:
    # Собираем explicit quality-gate policy override из CLI namespace.
    ruwe_threshold = _coerce_optional_float(
        namespace.quality_ruwe_unknown_threshold,
        default=DEFAULT_QUALITY_GATE_TUNING_CONFIG.ruwe_unknown_threshold,
    )
    parallax_threshold = _coerce_optional_float(
        namespace.quality_parallax_snr_unknown_threshold,
        default=DEFAULT_QUALITY_GATE_TUNING_CONFIG.parallax_snr_unknown_threshold,
    )
    require_flame_for_pass = _coerce_optional_bool(
        namespace.quality_require_flame_for_pass,
        default=DEFAULT_QUALITY_GATE_TUNING_CONFIG.require_flame_for_pass,
    )
    is_baseline = (
        ruwe_threshold == DEFAULT_QUALITY_GATE_TUNING_CONFIG.ruwe_unknown_threshold
        and parallax_threshold
        == DEFAULT_QUALITY_GATE_TUNING_CONFIG.parallax_snr_unknown_threshold
        and require_flame_for_pass
        == DEFAULT_QUALITY_GATE_TUNING_CONFIG.require_flame_for_pass
    )
    policy_name = "baseline" if is_baseline else "decide_tuned_quality_gate"
    return QualityGateTuningConfig(
        policy_name=policy_name,
        ruwe_unknown_threshold=ruwe_threshold,
        parallax_snr_unknown_threshold=parallax_threshold,
        require_flame_for_pass=require_flame_for_pass,
    )


def _coerce_optional_float(value: object, *, default: float | None) -> float | None:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("Expected optional numeric CLI value.")
    return float(value)


def _coerce_optional_bool(value: object, *, default: bool) -> bool:
    if value is None:
        return bool(default)
    if not isinstance(value, bool):
        raise TypeError("Expected optional boolean CLI value.")
    return value
