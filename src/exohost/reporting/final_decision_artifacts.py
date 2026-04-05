# Файл `final_decision_artifacts.py` слоя `reporting`.
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

DEFAULT_FINAL_DECISION_OUTPUT_DIR = Path("artifacts/decisions")


@dataclass(frozen=True, slots=True)
class FinalDecisionArtifactPaths:
    # Пути к артефактам одного final-decision прогона.
    run_dir: Path
    decision_input_csv_path: Path
    final_decision_csv_path: Path
    priority_input_csv_path: Path
    priority_ranking_csv_path: Path
    metadata_json_path: Path


@dataclass(frozen=True, slots=True)
class LoadedFinalDecisionArtifacts:
    # Загруженный bundle final-decision artifacts.
    decision_input_df: pd.DataFrame
    final_decision_df: pd.DataFrame
    priority_input_df: pd.DataFrame
    priority_ranking_df: pd.DataFrame
    metadata: dict[str, Any]


def build_final_decision_artifact_paths(
    *,
    output_dir: str | Path,
    pipeline_name: str,
    now: datetime | None = None,
) -> FinalDecisionArtifactPaths:
    # Собираем стандартную файловую структуру final-decision artifact run.
    base_dir = Path(output_dir)
    run_name = f"{sanitize_artifact_name(pipeline_name)}_{build_run_stamp(now)}"
    run_dir = base_dir / run_name
    return FinalDecisionArtifactPaths(
        run_dir=run_dir,
        decision_input_csv_path=run_dir / "decision_input.csv",
        final_decision_csv_path=run_dir / "final_decision.csv",
        priority_input_csv_path=run_dir / "priority_input.csv",
        priority_ranking_csv_path=run_dir / "priority_ranking.csv",
        metadata_json_path=run_dir / "metadata.json",
    )


def build_final_decision_metadata(
    *,
    pipeline_name: str,
    decision_input_df: pd.DataFrame,
    final_decision_df: pd.DataFrame,
    priority_input_df: pd.DataFrame,
    priority_ranking_df: pd.DataFrame,
    created_at: datetime,
    extra_metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    # Собираем compact metadata final-decision run.
    payload: dict[str, object] = {
        "pipeline_name": pipeline_name,
        "created_at_utc": created_at.astimezone(UTC).isoformat(),
        "n_rows_input": int(decision_input_df.shape[0]),
        "n_rows_final_decision": int(final_decision_df.shape[0]),
        "n_rows_priority_input": int(priority_input_df.shape[0]),
        "n_rows_priority_ranking": int(priority_ranking_df.shape[0]),
        "decision_input_columns": decision_input_df.columns.astype(str).tolist(),
        "final_decision_columns": final_decision_df.columns.astype(str).tolist(),
        "priority_input_columns": priority_input_df.columns.astype(str).tolist(),
        "priority_ranking_columns": priority_ranking_df.columns.astype(str).tolist(),
    }
    if "final_domain_state" in final_decision_df.columns:
        payload["final_domain_distribution"] = (
            final_decision_df["final_domain_state"]
            .astype(str)
            .value_counts(dropna=False)
            .sort_index()
            .to_dict()
        )
    if "priority_label" in priority_ranking_df.columns:
        payload["priority_label_distribution"] = (
            priority_ranking_df["priority_label"]
            .astype(str)
            .value_counts(dropna=False)
            .sort_index()
            .to_dict()
        )
    if extra_metadata:
        payload["context"] = dict(extra_metadata)
    return payload


def save_final_decision_artifacts(
    *,
    pipeline_name: str,
    decision_input_df: pd.DataFrame,
    final_decision_df: pd.DataFrame,
    priority_input_df: pd.DataFrame,
    priority_ranking_df: pd.DataFrame,
    output_dir: str | Path = DEFAULT_FINAL_DECISION_OUTPUT_DIR,
    now: datetime | None = None,
    extra_metadata: Mapping[str, object] | None = None,
) -> FinalDecisionArtifactPaths:
    # Сохраняем final-decision tables и metadata в отдельный run_dir.
    created_at = now or datetime.now(UTC)
    paths = build_final_decision_artifact_paths(
        output_dir=output_dir,
        pipeline_name=pipeline_name,
        now=created_at,
    )
    paths.run_dir.mkdir(parents=True, exist_ok=False)
    decision_input_df.to_csv(paths.decision_input_csv_path, index=False)
    final_decision_df.to_csv(paths.final_decision_csv_path, index=False)
    priority_input_df.to_csv(paths.priority_input_csv_path, index=False)
    priority_ranking_df.to_csv(paths.priority_ranking_csv_path, index=False)
    metadata = build_final_decision_metadata(
        pipeline_name=pipeline_name,
        decision_input_df=decision_input_df,
        final_decision_df=final_decision_df,
        priority_input_df=priority_input_df,
        priority_ranking_df=priority_ranking_df,
        created_at=created_at,
        extra_metadata=extra_metadata,
    )
    paths.metadata_json_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return paths


def load_final_decision_artifacts(run_dir: str | Path) -> LoadedFinalDecisionArtifacts:
    # Загружаем final-decision artifact bundle из одного run_dir.
    artifact_dir = Path(run_dir)
    return LoadedFinalDecisionArtifacts(
        decision_input_df=pd.read_csv(artifact_dir / "decision_input.csv"),
        final_decision_df=pd.read_csv(artifact_dir / "final_decision.csv"),
        priority_input_df=pd.read_csv(artifact_dir / "priority_input.csv"),
        priority_ranking_df=pd.read_csv(artifact_dir / "priority_ranking.csv"),
        metadata=json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8")),
    )
