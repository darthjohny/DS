# Тестовый файл `test_cli_decide_quality_gate_support.py` домена `cli`.
#
# Этот файл проверяет только:
# - проверку логики домена: CLI-команды и их orchestration-сценарии;
# - регрессионные сценарии и ожидаемые контракты целевого слоя.
#
# Следующий слой:
# - реализация домена `cli` в `src/exohost`;
# - соседние тесты и testkit этого же пакета.

from __future__ import annotations

import argparse

from exohost.cli.decide.quality_gate_support import (
    build_quality_gate_tuning_config_from_namespace,
)


def test_build_quality_gate_tuning_config_from_namespace_uses_defaults() -> None:
    namespace = argparse.Namespace(
        quality_ruwe_unknown_threshold=None,
        quality_parallax_snr_unknown_threshold=None,
        quality_require_flame_for_pass=None,
    )

    config = build_quality_gate_tuning_config_from_namespace(namespace)

    assert config.policy_name == "baseline"
    assert config.ruwe_unknown_threshold == 1.4
    assert config.parallax_snr_unknown_threshold == 5.0
    assert config.require_flame_for_pass is True


def test_build_quality_gate_tuning_config_from_namespace_applies_overrides() -> None:
    namespace = argparse.Namespace(
        quality_ruwe_unknown_threshold=1.4,
        quality_parallax_snr_unknown_threshold=5.0,
        quality_require_flame_for_pass=False,
    )

    config = build_quality_gate_tuning_config_from_namespace(namespace)

    assert config.policy_name == "decide_tuned_quality_gate"
    assert config.ruwe_unknown_threshold == 1.4
    assert config.parallax_snr_unknown_threshold == 5.0
    assert config.require_flame_for_pass is False
