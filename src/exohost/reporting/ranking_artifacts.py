# Файл `ranking_artifacts.py` слоя `reporting`.
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

from exohost.reporting.benchmark_artifacts import build_run_stamp, sanitize_artifact_name

DEFAULT_RANKING_OUTPUT_DIR = Path("artifacts/ranking")


@dataclass(frozen=True, slots=True)
class RankingArtifactPaths:
    # Пути к артефактам одного ranking-прогона.
    run_dir: Path
    ranking_csv_path: Path
    metadata_json_path: Path


def load_ranking_artifacts(run_dir: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    # Загружаем ranking-таблицу и metadata из одного ranking run_dir.
    ranking_dir = Path(run_dir)
    ranking_df = pd.read_csv(ranking_dir / "ranking.csv")
    metadata = json.loads((ranking_dir / "metadata.json").read_text(encoding="utf-8"))
    return ranking_df, metadata


def build_ranking_artifact_paths(
    *,
    output_dir: str | Path,
    ranking_name: str,
    now: datetime | None = None,
) -> RankingArtifactPaths:
    # Собираем стандартную файловую структуру ranking-прогона.
    base_dir = Path(output_dir)
    run_name = f"{sanitize_artifact_name(ranking_name)}_{build_run_stamp(now)}"
    run_dir = base_dir / run_name
    return RankingArtifactPaths(
        run_dir=run_dir,
        ranking_csv_path=run_dir / "ranking.csv",
        metadata_json_path=run_dir / "metadata.json",
    )


def build_ranking_metadata(
    ranking_df: pd.DataFrame,
    *,
    ranking_name: str,
    created_at: datetime,
    extra_metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    # Сохраняем компактный metadata-пакет ranking-прогона.
    payload: dict[str, object] = {
        "ranking_name": ranking_name,
        "created_at_utc": created_at.astimezone(UTC).isoformat(),
        "n_rows": int(ranking_df.shape[0]),
        "columns": ranking_df.columns.astype(str).tolist(),
    }
    if "priority_label" in ranking_df.columns:
        payload["priority_label_distribution"] = (
            ranking_df.loc[:, "priority_label"]
            .astype(str)
            .value_counts(dropna=False)
            .sort_index()
            .to_dict()
        )
    if extra_metadata:
        payload["context"] = dict(extra_metadata)
    return payload


def save_ranking_artifacts(
    ranking_df: pd.DataFrame,
    *,
    ranking_name: str,
    output_dir: str | Path = DEFAULT_RANKING_OUTPUT_DIR,
    now: datetime | None = None,
    extra_metadata: Mapping[str, object] | None = None,
) -> RankingArtifactPaths:
    # Сохраняем ranking-таблицу и metadata в отдельный run_dir.
    created_at = now or datetime.now(UTC)
    paths = build_ranking_artifact_paths(
        output_dir=output_dir,
        ranking_name=ranking_name,
        now=created_at,
    )
    paths.run_dir.mkdir(parents=True, exist_ok=False)
    ranking_df.to_csv(paths.ranking_csv_path, index=False)
    metadata = build_ranking_metadata(
        ranking_df,
        ranking_name=ranking_name,
        created_at=created_at,
        extra_metadata=extra_metadata,
    )
    paths.metadata_json_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return paths
