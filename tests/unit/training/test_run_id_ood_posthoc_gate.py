# Тестовый файл `test_run_id_ood_posthoc_gate.py` домена `training`.
#
# Этот файл проверяет только:
# - проверку логики домена: обучающие orchestration-сценарии и benchmark-runner;
# - регрессионные сценарии и ожидаемые контракты целевого слоя.
#
# Следующий слой:
# - реализация домена `training` в `src/exohost`;
# - соседние тесты и testkit этого же пакета.

from __future__ import annotations

from typing import cast

import pandas as pd
from sqlalchemy.engine import Engine

from exohost.training.run_id_ood_posthoc_gate import (
    run_id_ood_posthoc_gate_with_engine,
)


def build_id_ood_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index in range(80):
        rows.append(
            {
                "source_id": index + 1,
                "domain_target": "id" if index < 40 else "ood",
                "teff_gspphot": 5500.0 + index,
                "logg_gspphot": 4.2,
                "mh_gspphot": 0.0,
                "bp_rp": 0.2 if index < 40 else 0.9,
                "parallax": 10.0,
                "parallax_over_error": 12.0,
                "ruwe": 1.0,
                "phot_g_mean_mag": 12.0,
            }
        )
    return pd.DataFrame(rows)


def test_run_id_ood_posthoc_gate_with_engine_returns_gate_result(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "exohost.training.run_id_ood_posthoc_gate.load_hierarchical_prepared_training_frame",
        lambda engine, task_name, limit=None: build_id_ood_frame(),
    )

    result = run_id_ood_posthoc_gate_with_engine(
        engine=cast(Engine, object()),
        model_name="hist_gradient_boosting",
        limit=80,
    )

    assert result.task_name == "gaia_id_ood_classification"
    assert result.model_name == "hist_gradient_boosting"
    assert "ood_probability" in result.test_scored_df.columns
    assert "ood_threshold_policy_version" in result.test_scored_df.columns
    assert result.threshold_policy.threshold_metric == "balanced_accuracy"
    assert tuple(result.metrics_df["split_name"].tolist()) == ("train", "test")
