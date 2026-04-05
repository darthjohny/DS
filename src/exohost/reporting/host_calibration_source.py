# Файл `host_calibration_source.py` слоя `reporting`.
#
# Этот файл отвечает только за:
# - review-слой, notebook helper и отчетные dataframe;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - active notebook, docs и review-слой поверх этих helper;
# - unit-тесты reporting-пакета.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pandas as pd
from sklearn.base import clone
from sqlalchemy.engine import Engine

from exohost.datasets.build_host_field_training_dataset import (
    build_host_field_training_dataset,
)
from exohost.datasets.load_host_training_dataset import load_host_training_dataset
from exohost.datasets.load_router_training_dataset import load_router_training_dataset
from exohost.db.engine import make_read_only_engine
from exohost.evaluation.protocol import (
    DEFAULT_BENCHMARK_PROTOCOL,
    BenchmarkProtocol,
    ClassificationTask,
)
from exohost.evaluation.split import DatasetSplit, split_dataset
from exohost.features.training_frame import (
    prepare_host_training_frame,
    prepare_router_training_frame,
)
from exohost.models.inference import score_with_model
from exohost.models.protocol import ClassifierModel
from exohost.models.registry import build_router_model_specs, get_model_spec
from exohost.reporting.model_artifacts import (
    load_model_artifact_metadata,
    require_metadata_string,
)
from exohost.training.benchmark_orchestration import build_task_ready_frame
from exohost.training.run_host_benchmark import get_host_benchmark_task

DEFAULT_HOST_SCORE_COLUMN = "host_similarity_score"
DEFAULT_HOST_POSITIVE_LABEL = "host"


@dataclass(frozen=True, slots=True)
class HostCalibrationSource:
    # Holdout-source для calibration review без leakage из full-train model artifact.
    task_name: str
    model_name: str
    target_column: str
    positive_label: str
    host_score_column: str
    feature_columns: tuple[str, ...]
    split: DatasetSplit
    train_scored_df: pd.DataFrame
    test_scored_df: pd.DataFrame


def build_host_calibration_source_from_frames(
    host_df: pd.DataFrame,
    router_df: pd.DataFrame,
    *,
    task_name: str,
    model_name: str,
    field_to_host_ratio: int = 1,
    protocol: BenchmarkProtocol = DEFAULT_BENCHMARK_PROTOCOL,
    host_score_column: str = DEFAULT_HOST_SCORE_COLUMN,
) -> HostCalibrationSource:
    # Пересобираем host benchmark split и считаем честные holdout probabilities.
    task = get_host_benchmark_task(task_name)
    task_frame = _build_host_task_frame(
        host_df,
        router_df,
        task=task,
        field_to_host_ratio=field_to_host_ratio,
    )
    split = split_dataset(
        task_frame,
        split_config=protocol.split,
        stratify_columns=task.stratify_columns,
    )
    estimator = _build_host_estimator(
        feature_columns=task.feature_columns,
        model_name=model_name,
    )

    train_X = split.train_df.loc[:, list(task.feature_columns)]
    train_y = split.train_df.loc[:, task.target_column].astype(str)
    fitted_estimator = estimator.fit(train_X, train_y)

    train_scored_df = score_with_model(
        split.train_df,
        estimator=fitted_estimator,
        task_name=task.name,
        target_column=task.target_column,
        feature_columns=task.feature_columns,
        model_name=model_name,
        host_score_column=host_score_column,
    ).scored_df
    test_scored_df = score_with_model(
        split.test_df,
        estimator=fitted_estimator,
        task_name=task.name,
        target_column=task.target_column,
        feature_columns=task.feature_columns,
        model_name=model_name,
        host_score_column=host_score_column,
    ).scored_df

    return HostCalibrationSource(
        task_name=task.name,
        model_name=model_name,
        target_column=task.target_column,
        positive_label=DEFAULT_HOST_POSITIVE_LABEL,
        host_score_column=host_score_column,
        feature_columns=task.feature_columns,
        split=split,
        train_scored_df=train_scored_df,
        test_scored_df=test_scored_df,
    )


def build_host_calibration_source_with_engine(
    engine: Engine,
    *,
    task_name: str,
    model_name: str,
    host_limit: int | None = None,
    router_limit: int | None = None,
    field_to_host_ratio: int = 1,
    protocol: BenchmarkProtocol = DEFAULT_BENCHMARK_PROTOCOL,
    host_score_column: str = DEFAULT_HOST_SCORE_COLUMN,
) -> HostCalibrationSource:
    # Загружаем live host/router source из БД и строим holdout calibration source.
    raw_host_frame = load_host_training_dataset(engine, limit=host_limit)
    raw_router_frame = load_router_training_dataset(engine, limit=router_limit)
    host_frame = prepare_host_training_frame(raw_host_frame)
    router_frame = prepare_router_training_frame(raw_router_frame)
    return build_host_calibration_source_from_frames(
        host_frame,
        router_frame,
        task_name=task_name,
        model_name=model_name,
        field_to_host_ratio=field_to_host_ratio,
        protocol=protocol,
        host_score_column=host_score_column,
    )


def build_host_calibration_source_from_env(
    *,
    task_name: str,
    model_name: str,
    host_limit: int | None = None,
    router_limit: int | None = None,
    field_to_host_ratio: int = 1,
    protocol: BenchmarkProtocol = DEFAULT_BENCHMARK_PROTOCOL,
    host_score_column: str = DEFAULT_HOST_SCORE_COLUMN,
    dotenv_path: str = ".env",
    connect_timeout: int | None = 10,
) -> HostCalibrationSource:
    # Создаем read-only engine и строим calibration source из live relation.
    engine = make_read_only_engine(
        dotenv_path=dotenv_path,
        connect_timeout=connect_timeout,
    )
    try:
        return build_host_calibration_source_with_engine(
            engine,
            task_name=task_name,
            model_name=model_name,
            host_limit=host_limit,
            router_limit=router_limit,
            field_to_host_ratio=field_to_host_ratio,
            protocol=protocol,
            host_score_column=host_score_column,
        )
    finally:
        engine.dispose()


def build_host_calibration_source_from_model_artifact(
    model_run_dir: str | Path,
    *,
    host_limit: int | None = None,
    router_limit: int | None = None,
    field_to_host_ratio: int = 1,
    protocol: BenchmarkProtocol = DEFAULT_BENCHMARK_PROTOCOL,
    host_score_column: str = DEFAULT_HOST_SCORE_COLUMN,
    dotenv_path: str = ".env",
    connect_timeout: int | None = 10,
) -> HostCalibrationSource:
    # Строим calibration source по metadata существующего host model artifact.
    metadata = load_model_artifact_metadata(model_run_dir)
    return build_host_calibration_source_from_env(
        task_name=require_metadata_string(metadata, field_name="task_name"),
        model_name=require_metadata_string(metadata, field_name="model_name"),
        host_limit=host_limit,
        router_limit=router_limit,
        field_to_host_ratio=field_to_host_ratio,
        protocol=protocol,
        host_score_column=host_score_column,
        dotenv_path=dotenv_path,
        connect_timeout=connect_timeout,
    )


def _build_host_task_frame(
    host_df: pd.DataFrame,
    router_df: pd.DataFrame,
    *,
    task: ClassificationTask,
    field_to_host_ratio: int,
) -> pd.DataFrame:
    host_field_frame = build_host_field_training_dataset(
        host_df,
        router_df,
        field_to_host_ratio=field_to_host_ratio,
    )
    return build_task_ready_frame(
        host_field_frame,
        target_column=task.target_column,
        frame_name="host calibration frame",
    )


def _build_host_estimator(
    *,
    feature_columns: tuple[str, ...],
    model_name: str,
) -> ClassifierModel:
    model_spec = get_model_spec(
        build_router_model_specs(feature_columns),
        model_name=model_name,
    )
    return cast(ClassifierModel, clone(model_spec.estimator))
