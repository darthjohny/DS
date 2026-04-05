# Тестовый файл `test_posthoc_quality_gate_tuning.py` домена `posthoc`.
#
# Этот файл проверяет только:
# - проверку логики домена: post-hoc routing, gate и final decision policy;
# - регрессионные сценарии и ожидаемые контракты целевого слоя.
#
# Следующий слой:
# - реализация домена `posthoc` в `src/exohost`;
# - соседние тесты и testkit этого же пакета.

from __future__ import annotations

import pandas as pd

from exohost.posthoc.quality_gate_tuning import (
    QualityGateTuningConfig,
    apply_quality_gate_tuning,
)


def _build_quality_gate_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_id": ["1", "2", "3", "4"],
            "quality_state": ["pass", "unknown", "unknown", "reject"],
            "quality_reason": [
                "pass",
                "review_missing_radius_flame",
                "review_high_ruwe",
                "reject_missing_core_features",
            ],
            "review_bucket": [
                "pass",
                "review_missing_radius_flame",
                "review_high_ruwe",
                "reject_missing_core_features",
            ],
            "ruwe": [1.1, 1.0, 1.55, 1.0],
            "parallax_over_error": [10.0, 7.0, 10.0, 9.0],
            "radius_flame": [1.0, pd.NA, 1.2, 1.1],
            "has_missing_flame_features": [False, True, False, False],
        }
    )


def test_apply_quality_gate_tuning_keeps_baseline_policy() -> None:
    result = apply_quality_gate_tuning(_build_quality_gate_frame())

    assert result["quality_state"].tolist() == ["pass", "unknown", "unknown", "reject"]
    assert result["quality_reason"].tolist() == [
        "pass",
        "review_missing_radius_flame",
        "review_high_ruwe",
        "reject_missing_core_features",
    ]


def test_apply_quality_gate_tuning_can_relax_flame_requirement_only() -> None:
    result = apply_quality_gate_tuning(
        _build_quality_gate_frame(),
        config=QualityGateTuningConfig(
            policy_name="no_flame",
            ruwe_unknown_threshold=1.4,
            parallax_snr_unknown_threshold=5.0,
            require_flame_for_pass=False,
        ),
    )

    assert result["quality_state"].tolist() == ["pass", "pass", "unknown", "reject"]
    assert result["quality_reason"].tolist() == [
        "pass",
        "pass",
        "review_high_ruwe",
        "reject_missing_core_features",
    ]
    assert result["review_bucket"].tolist() == [
        "pass",
        "pass",
        "review_high_ruwe",
        "reject_missing_core_features",
    ]
