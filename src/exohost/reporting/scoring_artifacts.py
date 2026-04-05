# Файл `scoring_artifacts.py` слоя `reporting`.
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

import pandas as pd

from exohost.models.inference import ModelScoringResult
from exohost.reporting.benchmark_artifacts import build_run_stamp, sanitize_artifact_name

DEFAULT_SCORING_OUTPUT_DIR = Path("artifacts/scoring")


@dataclass(frozen=True, slots=True)
class ScoringArtifactPaths:
    # Пути к артефактам одного scoring-прогона.
    run_dir: Path
    scored_csv_path: Path
    metadata_json_path: Path


def load_scoring_artifacts(run_dir: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    # Загружаем scored DataFrame и metadata из одного scoring run_dir.
    scoring_dir = Path(run_dir)
    scored_df = pd.read_csv(scoring_dir / "scored.csv")
    metadata = json.loads((scoring_dir / "metadata.json").read_text(encoding="utf-8"))
    return scored_df, metadata


def build_scoring_artifact_paths(
    *,
    output_dir: str | Path,
    task_name: str,
    model_name: str,
    now: datetime | None = None,
) -> ScoringArtifactPaths:
    # Собираем стандартную файловую структуру scoring-прогона.
    base_dir = Path(output_dir)
    run_name = (
        f"{sanitize_artifact_name(task_name)}__"
        f"{sanitize_artifact_name(model_name)}__"
        f"{build_run_stamp(now)}"
    )
    run_dir = base_dir / run_name
    return ScoringArtifactPaths(
        run_dir=run_dir,
        scored_csv_path=run_dir / "scored.csv",
        metadata_json_path=run_dir / "metadata.json",
    )


def build_scoring_metadata(
    result: ModelScoringResult,
    *,
    created_at: datetime,
    extra_metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    # Собираем metadata scoring-прогона.
    payload: dict[str, object] = {
        "task_name": result.task_name,
        "target_column": result.target_column,
        "model_name": result.model_name,
        "created_at_utc": created_at.astimezone(UTC).isoformat(),
        "n_rows": result.n_rows,
        "columns": result.scored_df.columns.astype(str).tolist(),
    }
    if extra_metadata:
        payload["context"] = dict(extra_metadata)
    return payload


def save_scoring_artifacts(
    result: ModelScoringResult,
    *,
    output_dir: str | Path = DEFAULT_SCORING_OUTPUT_DIR,
    now: datetime | None = None,
    extra_metadata: Mapping[str, object] | None = None,
) -> ScoringArtifactPaths:
    # Сохраняем scored DataFrame и metadata в отдельный run_dir.
    created_at = now or datetime.now(UTC)
    paths = build_scoring_artifact_paths(
        output_dir=output_dir,
        task_name=result.task_name,
        model_name=result.model_name,
        now=created_at,
    )
    paths.run_dir.mkdir(parents=True, exist_ok=False)
    result.scored_df.to_csv(paths.scored_csv_path, index=False)
    metadata = build_scoring_metadata(
        result,
        created_at=created_at,
        extra_metadata=extra_metadata,
    )
    paths.metadata_json_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return paths
