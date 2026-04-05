# Файл `support.py` слоя `cli`.
#
# Этот файл отвечает только за:
# - CLI-команды и orchestration entrypoints;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - CLI-команды или support-модули этого же домена;
# - пользовательский запуск через `python -m exohost.cli.main`.

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import pandas as pd

from exohost.contracts.feature_contract import unique_columns
from exohost.reporting.model_artifacts import LoadedModelArtifact
from exohost.reporting.ranking_artifacts import (
    DEFAULT_RANKING_OUTPUT_DIR,
    RankingArtifactPaths,
)
from exohost.reporting.scoring_artifacts import (
    DEFAULT_SCORING_OUTPUT_DIR,
    ScoringArtifactPaths,
)


class SupportsDispose(Protocol):
    # Минимальный контракт DB engine для безопасного закрытия ресурсов.

    def dispose(self) -> None:
        # Освобождаем ресурсы после чтения relation.
        ...


EngineFactory = Callable[..., SupportsDispose]
DatasetLoader = Callable[..., pd.DataFrame]


def print_prioritize_stage(message: str) -> None:
    # Печатаем короткий статус сквозной prioritize-команды.
    print(f"[prioritize] {message}")


def print_scoring_artifact_paths(paths: ScoringArtifactPaths) -> None:
    # Печатаем каталог сохраненных scoring-артефактов.
    print(f"[artifacts] scoring_saved_to={paths.run_dir}")


def print_ranking_artifact_paths(paths: RankingArtifactPaths) -> None:
    # Печатаем каталог сохраненных ranking-артефактов.
    print(f"[artifacts] ranking_saved_to={paths.run_dir}")


def format_frame_preview(frame: pd.DataFrame, *, preview_rows: int) -> str:
    # Собираем компактный preview табличного результата.
    return frame.head(preview_rows).to_string(index=False)


def require_target_column(
    artifact: LoadedModelArtifact,
    *,
    expected_target_column: str,
    flag_name: str,
) -> None:
    # Валидируем, что поданный model artifact соответствует ожидаемой задаче.
    if artifact.target_column != expected_target_column:
        raise ValueError(
            f"{flag_name} must point to a model with target_column="
            f"{expected_target_column}, got {artifact.target_column}."
        )


def build_feature_union(
    *,
    router_artifact: LoadedModelArtifact,
    host_artifact: LoadedModelArtifact,
    stage_artifact: LoadedModelArtifact | None,
) -> tuple[str, ...]:
    # Объединяем признаки всех моделей для одного candidate source.
    if stage_artifact is None:
        return unique_columns(
            router_artifact.feature_columns,
            host_artifact.feature_columns,
        )
    return unique_columns(
        router_artifact.feature_columns,
        host_artifact.feature_columns,
        stage_artifact.feature_columns,
    )


def resolve_scoring_output_dir(namespace: argparse.Namespace) -> Path:
    # Разрешаем output dir scoring-артефактов.
    if namespace.output_dir is not None:
        return Path(namespace.output_dir)
    return DEFAULT_SCORING_OUTPUT_DIR


def resolve_ranking_output_dir(namespace: argparse.Namespace) -> Path:
    # Разрешаем output dir ranking-артефактов.
    if namespace.ranking_output_dir is not None:
        return Path(namespace.ranking_output_dir)
    return DEFAULT_RANKING_OUTPUT_DIR


def build_source_name(namespace: argparse.Namespace) -> str:
    # Собираем стабильное имя source для artifact naming.
    if namespace.relation_name is not None:
        return str(namespace.relation_name).replace(".", "__")
    if namespace.input_csv is not None:
        return Path(namespace.input_csv).stem
    return "candidates"


def load_candidate_frame(
    namespace: argparse.Namespace,
    *,
    feature_columns: tuple[str, ...],
    engine_factory: EngineFactory,
    dataset_loader: DatasetLoader,
) -> pd.DataFrame:
    # Загружаем candidate source либо из CSV, либо из relation в БД.
    if namespace.input_csv is not None:
        print_prioritize_stage(f"load input={namespace.input_csv}")
        return pd.read_csv(namespace.input_csv)

    if namespace.relation_name is None:
        raise RuntimeError("Candidate input source is not configured.")

    print_prioritize_stage(f"load relation={namespace.relation_name}")
    engine = engine_factory(
        dotenv_path=namespace.dotenv_path,
        connect_timeout=namespace.connect_timeout,
    )
    try:
        return dataset_loader(
            engine,
            relation_name=namespace.relation_name,
            feature_columns=feature_columns,
            limit=namespace.limit,
        )
    finally:
        engine.dispose()


def apply_router_predictions_for_ranking(
    df: pd.DataFrame,
    *,
    router_target_column: str,
    stage_target_column: str | None,
) -> pd.DataFrame:
    # Переносим предсказания router-слоя в канонические колонки ranking-контура.
    result = df.copy()
    predicted_class_column = f"predicted_{router_target_column}"
    if predicted_class_column not in result.columns:
        raise ValueError(
            "Prioritize pipeline expected router prediction column: "
            f"{predicted_class_column}"
        )
    result["spec_class"] = result.loc[:, predicted_class_column].astype(str)

    if stage_target_column is not None:
        predicted_stage_column = f"predicted_{stage_target_column}"
        if predicted_stage_column not in result.columns:
            raise ValueError(
                "Prioritize pipeline expected stage prediction column: "
                f"{predicted_stage_column}"
            )
        result["evolution_stage"] = result.loc[:, predicted_stage_column].astype(str)

    return result


def build_prioritize_context(
    namespace: argparse.Namespace,
    *,
    scoring_output_dir: Path,
    ranking_output_dir: Path,
    router_artifact: LoadedModelArtifact,
    host_artifact: LoadedModelArtifact,
    stage_artifact: LoadedModelArtifact | None,
) -> dict[str, object]:
    # Собираем metadata-контекст для scoring/ranking артефактов сквозного контура.
    context: dict[str, object] = {
        "score_mode": "prioritize_pipeline",
        "input_csv": None if namespace.input_csv is None else str(namespace.input_csv),
        "relation_name": None if namespace.relation_name is None else str(namespace.relation_name),
        "host_score_column": str(namespace.host_score_column),
        "scoring_output_dir": str(scoring_output_dir),
        "ranking_output_dir": str(ranking_output_dir),
        "preview_rows": int(namespace.preview_rows),
        "dotenv_path": str(namespace.dotenv_path),
        "connect_timeout": int(namespace.connect_timeout),
        "router_model_run_dir": str(namespace.router_model_run_dir),
        "host_model_run_dir": str(namespace.host_model_run_dir),
        "router_model_name": router_artifact.model_name,
        "host_model_name": host_artifact.model_name,
    }
    if namespace.stage_model_run_dir is not None and stage_artifact is not None:
        context["stage_model_run_dir"] = str(namespace.stage_model_run_dir)
        context["stage_model_name"] = stage_artifact.model_name
    if namespace.limit is not None:
        context["limit"] = int(namespace.limit)
    return context
