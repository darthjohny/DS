# Файл `run_id_ood_posthoc_gate.py` слоя `training`.
#
# Этот файл отвечает только за:
# - оркестрацию обучения и benchmark-прогонов;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - соседние модули слоя `training` и orchestration поверх них;
# - unit-тесты и более верхние слои проекта.

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sqlalchemy.engine import Engine

from exohost.evaluation.metrics import build_classification_metrics
from exohost.evaluation.protocol import DEFAULT_BENCHMARK_PROTOCOL
from exohost.evaluation.split import DatasetSplit, split_dataset
from exohost.models.registry import build_router_model_specs, get_model_spec
from exohost.posthoc.calibration import (
    DEFAULT_CALIBRATION_CONFIG,
    CalibrationConfig,
    build_calibrated_classifier,
)
from exohost.posthoc.id_ood_gate import (
    IdOodThresholdPolicy,
    build_id_ood_gate_scored_frame,
    build_id_ood_metrics_probability_frame,
    build_id_ood_threshold_policy,
)
from exohost.posthoc.threshold_tuning import (
    DEFAULT_THRESHOLD_TUNING_CONFIG,
    ThresholdTuningConfig,
    build_tuned_threshold_classifier,
)
from exohost.training.benchmark_orchestration import build_task_ready_frame
from exohost.training.benchmark_runner import build_metrics_row
from exohost.training.hierarchical_source import load_hierarchical_prepared_training_frame
from exohost.training.run_hierarchical_benchmark import get_hierarchical_benchmark_task


@dataclass(slots=True)
class IdOodPosthocGateRunResult:
    # Полный результат calibrated and tuned ID/OOD gate run.
    task_name: str
    model_name: str
    split: DatasetSplit
    metrics_df: pd.DataFrame
    threshold_policy: IdOodThresholdPolicy
    train_scored_df: pd.DataFrame
    test_scored_df: pd.DataFrame


def run_id_ood_posthoc_gate_with_engine(
    engine: Engine,
    *,
    model_name: str = "hist_gradient_boosting",
    limit: int | None = None,
    calibration_config: CalibrationConfig = DEFAULT_CALIBRATION_CONFIG,
    threshold_config: ThresholdTuningConfig = DEFAULT_THRESHOLD_TUNING_CONFIG,
    threshold_policy_version: str = "id_ood_threshold_v1",
    candidate_ood_threshold: float | None = None,
) -> IdOodPosthocGateRunResult:
    # Выполняем separate post-hoc gate pipeline для binary ID/OOD stage.
    task = get_hierarchical_benchmark_task("gaia_id_ood_classification")
    prepared_frame = load_hierarchical_prepared_training_frame(
        engine,
        task_name=task.name,
        limit=limit,
    )
    task_frame = build_task_ready_frame(
        prepared_frame,
        target_column=task.target_column,
        frame_name="id/ood posthoc gate frame",
    )
    split = split_dataset(
        task_frame,
        split_config=DEFAULT_BENCHMARK_PROTOCOL.split,
        stratify_columns=task.stratify_columns,
    )

    model_spec = get_model_spec(
        build_router_model_specs(task.feature_columns),
        model_name=model_name,
    )
    calibrated_estimator = build_calibrated_classifier(
        model_spec.estimator,
        config=calibration_config,
    )
    tuned_threshold_estimator = build_tuned_threshold_classifier(
        calibrated_estimator,
        config=threshold_config,
    )

    train_X = split.train_df.loc[:, list(task.feature_columns)]
    train_y = split.train_df.loc[:, task.target_column].astype(str)
    tuned_threshold_estimator.fit(train_X, train_y)

    threshold_policy = build_id_ood_threshold_policy(
        tuned_threshold=float(tuned_threshold_estimator.best_threshold_),
        threshold_policy_version=threshold_policy_version,
        candidate_ood_threshold=candidate_ood_threshold,
        threshold_metric=threshold_config.scoring,
        threshold_fit_scope=f"cv_{threshold_config.cv}",
    )

    train_scored_df = build_id_ood_gate_scored_frame(
        split.train_df,
        estimator=tuned_threshold_estimator,
        feature_columns=task.feature_columns,
        policy=threshold_policy,
    )
    test_scored_df = build_id_ood_gate_scored_frame(
        split.test_df,
        estimator=tuned_threshold_estimator,
        feature_columns=task.feature_columns,
        policy=threshold_policy,
    )
    train_predicted_series = pd.Series(
        train_scored_df.loc[:, "predicted_domain_target"],
        index=train_scored_df.index,
        dtype="string",
    )
    test_predicted_series = pd.Series(
        test_scored_df.loc[:, "predicted_domain_target"],
        index=test_scored_df.index,
        dtype="string",
    )

    train_metrics = build_classification_metrics(
        train_y,
        train_predicted_series,
        split_name="train",
        y_proba=build_id_ood_metrics_probability_frame(train_scored_df),
    )
    test_metrics = build_classification_metrics(
        split.test_df.loc[:, task.target_column].astype(str),
        test_predicted_series,
        split_name="test",
        y_proba=build_id_ood_metrics_probability_frame(test_scored_df),
    )
    metrics_df = pd.DataFrame.from_records(
        [
            build_metrics_row(model_spec.model_name, train_metrics),
            build_metrics_row(model_spec.model_name, test_metrics),
        ]
    )

    return IdOodPosthocGateRunResult(
        task_name=task.name,
        model_name=model_spec.model_name,
        split=split,
        metrics_df=metrics_df,
        threshold_policy=threshold_policy,
        train_scored_df=train_scored_df,
        test_scored_df=test_scored_df,
    )
