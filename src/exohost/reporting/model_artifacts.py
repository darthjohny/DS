# Файл `model_artifacts.py` слоя `reporting`.
#
# Этот файл отвечает только за:
# - review-слой, notebook helper и отчетные dataframe;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - active notebook, docs и review-слой поверх этих helper;
# - unit-тесты reporting-пакета.

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib

from exohost.reporting.benchmark_artifacts import build_run_stamp, sanitize_artifact_name
from exohost.training.train_runner import TrainRunResult

DEFAULT_MODEL_OUTPUT_DIR = Path("artifacts/models")


@dataclass(frozen=True, slots=True)
class ModelArtifactPaths:
    # Пути к артефактам одного train-прогона.
    run_dir: Path
    model_joblib_path: Path
    label_distribution_csv_path: Path
    metadata_json_path: Path


@dataclass(frozen=True, slots=True)
class LoadedModelArtifact:
    # Загруженный model artifact вместе с metadata.
    estimator: object
    metadata: dict[str, Any]
    task_name: str
    target_column: str
    feature_columns: tuple[str, ...]
    model_name: str


def require_metadata_string(
    metadata: Mapping[str, Any],
    *,
    field_name: str,
) -> str:
    # Достаем обязательное строковое поле из metadata и валидируем тип.
    value = metadata.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Model artifact metadata field '{field_name}' must be a non-empty string.")
    return value


def require_metadata_string_tuple(
    metadata: Mapping[str, Any],
    *,
    field_name: str,
) -> tuple[str, ...]:
    # Достаем обязательный список строк из metadata и приводим его к tuple.
    raw_value = metadata.get(field_name)
    if not isinstance(raw_value, list) or not raw_value:
        raise ValueError(f"Model artifact metadata field '{field_name}' must be a non-empty list.")

    items: list[str] = []
    for raw_item in raw_value:
        if not isinstance(raw_item, str) or not raw_item.strip():
            raise ValueError(
                f"Model artifact metadata field '{field_name}' must contain non-empty strings."
            )
        items.append(raw_item)
    return tuple(items)


def build_model_artifact_paths(
    *,
    output_dir: str | Path,
    task_name: str,
    model_name: str,
    now: datetime | None = None,
) -> ModelArtifactPaths:
    # Собираем стандартную файловую структуру train-прогона.
    base_dir = Path(output_dir)
    run_name = (
        f"{sanitize_artifact_name(task_name)}__"
        f"{sanitize_artifact_name(model_name)}__"
        f"{build_run_stamp(now)}"
    )
    run_dir = base_dir / run_name
    return ModelArtifactPaths(
        run_dir=run_dir,
        model_joblib_path=run_dir / "model.joblib",
        label_distribution_csv_path=run_dir / "label_distribution.csv",
        metadata_json_path=run_dir / "metadata.json",
    )


def build_model_metadata(
    result: TrainRunResult,
    *,
    created_at: datetime,
    extra_metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    # Собираем metadata train-прогона.
    payload: dict[str, object] = {
        "task_name": result.task_name,
        "model_name": result.model_name,
        "created_at_utc": created_at.astimezone(UTC).isoformat(),
        "target_column": result.target_column,
        "feature_columns": list(result.feature_columns),
        "n_rows": result.n_rows,
        "class_labels": list(result.class_labels),
        "estimator_class_name": result.estimator.__class__.__name__,
    }
    if extra_metadata:
        payload["context"] = dict(extra_metadata)
    return payload


def save_model_artifacts(
    result: TrainRunResult,
    *,
    output_dir: str | Path = DEFAULT_MODEL_OUTPUT_DIR,
    now: datetime | None = None,
    extra_metadata: Mapping[str, object] | None = None,
) -> ModelArtifactPaths:
    # Сохраняем модель, label distribution и metadata в отдельный run_dir.
    created_at = now or datetime.now(UTC)
    paths = build_model_artifact_paths(
        output_dir=output_dir,
        task_name=result.task_name,
        model_name=result.model_name,
        now=created_at,
    )
    paths.run_dir.mkdir(parents=True, exist_ok=False)
    joblib.dump(result.estimator, paths.model_joblib_path)
    result.label_distribution_df.to_csv(paths.label_distribution_csv_path, index=False)
    metadata = build_model_metadata(
        result,
        created_at=created_at,
        extra_metadata=extra_metadata,
    )
    paths.metadata_json_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return paths


def load_model_artifact_metadata(run_dir: str | Path) -> dict[str, Any]:
    # Загружаем только metadata без materialize estimator из model.joblib.
    artifact_dir = Path(run_dir)
    return json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))


def load_model_artifact(run_dir: str | Path) -> LoadedModelArtifact:
    # Загружаем model artifact и metadata из одного run_dir.
    artifact_dir = Path(run_dir)
    estimator = joblib.load(artifact_dir / "model.joblib")
    metadata = load_model_artifact_metadata(artifact_dir)
    return LoadedModelArtifact(
        estimator=estimator,
        metadata=metadata,
        task_name=require_metadata_string(metadata, field_name="task_name"),
        target_column=require_metadata_string(metadata, field_name="target_column"),
        feature_columns=require_metadata_string_tuple(metadata, field_name="feature_columns"),
        model_name=require_metadata_string(metadata, field_name="model_name"),
    )
