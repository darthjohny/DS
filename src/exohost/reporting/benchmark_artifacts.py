# Файл `benchmark_artifacts.py` слоя `reporting`.
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
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from exohost.training.benchmark_runner import BenchmarkRunResult

DEFAULT_BENCHMARK_OUTPUT_DIR = Path("artifacts/benchmarks")


@dataclass(frozen=True, slots=True)
class BenchmarkArtifactPaths:
    # Пути к артефактам одного benchmark-прогона.
    run_dir: Path
    metrics_csv_path: Path
    cv_summary_csv_path: Path
    target_distribution_csv_path: Path
    metadata_json_path: Path


def sanitize_artifact_name(value: str) -> str:
    # Приводим имя артефакта к простому и безопасному filesystem-формату.
    normalized_value = value.strip().lower()
    sanitized = re.sub(r"[^a-z0-9]+", "_", normalized_value)
    return sanitized.strip("_") or "run"


def build_run_stamp(now: datetime | None = None) -> str:
    # Строим UTC timestamp для имени benchmark-прогона.
    current_time = now or datetime.now(UTC)
    current_time_utc = current_time.astimezone(UTC)
    return current_time_utc.strftime("%Y_%m_%d_%H%M%S_%f")


def build_benchmark_artifact_paths(
    *,
    output_dir: str | Path,
    task_name: str,
    now: datetime | None = None,
) -> BenchmarkArtifactPaths:
    # Собираем стандартный набор путей для benchmark-артефактов.
    base_dir = Path(output_dir)
    run_name = f"{sanitize_artifact_name(task_name)}_{build_run_stamp(now)}"
    run_dir = base_dir / run_name
    return BenchmarkArtifactPaths(
        run_dir=run_dir,
        metrics_csv_path=run_dir / "metrics.csv",
        cv_summary_csv_path=run_dir / "cv_summary.csv",
        target_distribution_csv_path=run_dir / "target_distribution.csv",
        metadata_json_path=run_dir / "metadata.json",
    )


def build_benchmark_metadata(
    result: BenchmarkRunResult,
    *,
    created_at: datetime,
    extra_metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    # Собираем компактный metadata-пакет benchmark-прогона.
    payload: dict[str, object] = {
        "task_name": result.task_name,
        "created_at_utc": created_at.astimezone(UTC).isoformat(),
        "n_rows_full": int(result.split.full_df.shape[0]),
        "n_rows_train": int(result.split.train_df.shape[0]),
        "n_rows_test": int(result.split.test_df.shape[0]),
        "metrics_columns": result.metrics_df.columns.astype(str).tolist(),
        "cv_summary_columns": result.cv_summary_df.columns.astype(str).tolist(),
        "target_distribution_columns": result.target_distribution_df.columns.astype(str).tolist(),
        "model_names": result.cv_summary_df.loc[:, "model_name"].astype(str).tolist(),
    }
    if extra_metadata:
        payload["context"] = dict(extra_metadata)
    return payload


def save_benchmark_artifacts(
    result: BenchmarkRunResult,
    *,
    output_dir: str | Path = DEFAULT_BENCHMARK_OUTPUT_DIR,
    now: datetime | None = None,
    extra_metadata: Mapping[str, object] | None = None,
) -> BenchmarkArtifactPaths:
    # Сохраняем benchmark-таблицы и metadata в отдельный run_dir.
    created_at = now or datetime.now(UTC)
    paths = build_benchmark_artifact_paths(
        output_dir=output_dir,
        task_name=result.task_name,
        now=created_at,
    )
    paths.run_dir.mkdir(parents=True, exist_ok=False)
    result.metrics_df.to_csv(paths.metrics_csv_path, index=False)
    result.cv_summary_df.to_csv(paths.cv_summary_csv_path, index=False)
    result.target_distribution_df.to_csv(paths.target_distribution_csv_path, index=False)
    metadata = build_benchmark_metadata(
        result,
        created_at=created_at,
        extra_metadata=extra_metadata,
    )
    paths.metadata_json_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return paths
