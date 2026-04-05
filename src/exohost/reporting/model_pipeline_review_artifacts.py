# Файл `model_pipeline_review_artifacts.py` слоя `reporting`.
#
# Этот файл отвечает только за:
# - review-слой, notebook helper и отчетные dataframe;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - active notebook, docs и review-слой поверх этих helper;
# - unit-тесты reporting-пакета.

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from exohost.reporting.id_ood_threshold_artifacts import load_id_ood_threshold_artifact
from exohost.reporting.model_artifacts import load_model_artifact


def build_model_artifact_summary_frame(
    model_run_dirs: Mapping[str, str | Path],
) -> pd.DataFrame:
    # Короткая сводка по сохраненным model artifacts для notebook observability.
    rows: list[dict[str, object]] = []
    for stage_name, run_dir in model_run_dirs.items():
        artifact = load_model_artifact(run_dir)
        rows.append(
            {
                "stage_name": stage_name,
                "run_dir": Path(run_dir).name,
                "task_name": artifact.task_name,
                "model_name": artifact.model_name,
                "target_column": artifact.target_column,
                "n_features": int(len(artifact.feature_columns)),
                "feature_columns_preview": ", ".join(artifact.feature_columns[:5]),
                "created_at_utc": artifact.metadata.get("created_at_utc", "unknown"),
            }
        )
    return pd.DataFrame.from_records(rows)


def build_threshold_artifact_summary_frame(run_dir: str | Path) -> pd.DataFrame:
    # Короткая сводка по tuned threshold artifact для ID/OOD gate.
    artifact = load_id_ood_threshold_artifact(run_dir)
    return pd.DataFrame(
        [
            {
                "run_dir": Path(run_dir).name,
                "task_name": artifact.task_name,
                "model_name": artifact.model_name,
                "threshold_name": artifact.policy.threshold_name,
                "threshold_value": float(artifact.policy.threshold_value),
                "candidate_ood_threshold": (
                    pd.NA
                    if artifact.policy.candidate_ood_threshold is None
                    else float(artifact.policy.candidate_ood_threshold)
                ),
                "threshold_metric": artifact.policy.threshold_metric,
                "threshold_fit_scope": artifact.policy.threshold_fit_scope,
                "threshold_policy_version": artifact.policy.threshold_policy_version,
                "created_at_utc": artifact.metadata.get("created_at_utc", "unknown"),
            }
        ]
    )


__all__ = [
    "build_model_artifact_summary_frame",
    "build_threshold_artifact_summary_frame",
]
