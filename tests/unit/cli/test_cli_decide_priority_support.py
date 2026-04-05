# Тестовый файл `test_cli_decide_priority_support.py` домена `cli`.
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

from exohost.cli.decide.priority_support import (
    build_priority_integration_config_from_namespace,
)


def test_build_priority_integration_config_from_namespace_uses_defaults() -> None:
    namespace = argparse.Namespace(
        host_score_column="host_similarity_score",
        priority_high_min=None,
        priority_medium_min=None,
    )

    config = build_priority_integration_config_from_namespace(namespace)

    assert config.host_score_column == "host_similarity_score"
    assert config.thresholds.high_min == 0.75
    assert config.thresholds.medium_min == 0.45


def test_build_priority_integration_config_from_namespace_applies_overrides() -> None:
    namespace = argparse.Namespace(
        host_score_column="custom_host_score",
        priority_high_min=0.85,
        priority_medium_min=0.55,
    )

    config = build_priority_integration_config_from_namespace(namespace)

    assert config.host_score_column == "custom_host_score"
    assert config.thresholds.high_min == 0.85
    assert config.thresholds.medium_min == 0.55
