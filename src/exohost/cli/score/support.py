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

from exohost.ranking.priority_score import DEFAULT_HOST_SCORE_COLUMN
from exohost.reporting.ranking_artifacts import (
    DEFAULT_RANKING_OUTPUT_DIR,
    RankingArtifactPaths,
)
from exohost.reporting.scoring_artifacts import (
    DEFAULT_SCORING_OUTPUT_DIR,
    ScoringArtifactPaths,
)


class SupportsDispose(Protocol):
    # Минимальный контракт engine для безопасного закрытия DB-backed загрузки.

    def dispose(self) -> None:
        # Освобождаем ресурсы подключения после чтения.
        ...


EngineFactory = Callable[..., SupportsDispose]
DatasetLoader = Callable[..., pd.DataFrame]


def print_score_stage(message: str) -> None:
    # Печатаем короткий статус score-команды.
    print(f"[score] {message}")


def print_ranking_artifact_paths(paths: RankingArtifactPaths) -> None:
    # Печатаем каталог сохраненных ranking-артефактов.
    print(f"[artifacts] saved_to={paths.run_dir}")


def print_scoring_artifact_paths(paths: ScoringArtifactPaths) -> None:
    # Печатаем каталог сохраненных scoring-артефактов.
    print(f"[artifacts] saved_to={paths.run_dir}")


def format_frame_preview(frame: pd.DataFrame, *, preview_rows: int) -> str:
    # Собираем компактный preview табличного результата.
    preview_frame = frame.head(preview_rows)
    return preview_frame.to_string(index=False)


def build_score_context(
    namespace: argparse.Namespace,
    *,
    score_mode: str,
    output_dir: Path,
) -> dict[str, object]:
    # Собираем metadata-контекст score-команды.
    context: dict[str, object] = {
        "score_mode": score_mode,
        "input_csv": None if namespace.input_csv is None else str(namespace.input_csv),
        "relation_name": None if namespace.relation_name is None else str(namespace.relation_name),
        "host_score_column": str(namespace.host_score_column),
        "output_dir": str(output_dir),
        "with_ranking": bool(namespace.with_ranking),
        "preview_rows": int(namespace.preview_rows),
        "dotenv_path": str(namespace.dotenv_path),
        "connect_timeout": int(namespace.connect_timeout),
    }
    if namespace.model_run_dir is not None:
        context["model_run_dir"] = str(namespace.model_run_dir)
    if namespace.ranking_output_dir is not None:
        context["ranking_output_dir"] = str(namespace.ranking_output_dir)
    if namespace.limit is not None:
        context["limit"] = int(namespace.limit)
    return context


def resolve_score_output_dir(
    namespace: argparse.Namespace,
    *,
    score_mode: str,
) -> Path:
    # Выбираем стандартный output dir по режиму score-команды.
    if namespace.output_dir is not None:
        return Path(namespace.output_dir)
    if score_mode == "model_scoring":
        return DEFAULT_SCORING_OUTPUT_DIR
    return DEFAULT_RANKING_OUTPUT_DIR


def resolve_ranking_output_dir(namespace: argparse.Namespace) -> Path:
    # Для комбинированного model-scoring + ranking разрешаем отдельный output dir.
    if namespace.ranking_output_dir is not None:
        return Path(namespace.ranking_output_dir)
    return DEFAULT_RANKING_OUTPUT_DIR


def build_ranking_name_from_scoring(namespace: argparse.Namespace) -> str:
    # Собираем стабильное имя ranking-прогона поверх scored output.
    if namespace.relation_name is not None:
        source_name = str(namespace.relation_name).replace(".", "__")
    elif namespace.input_csv is not None:
        source_name = Path(namespace.input_csv).stem
    else:
        source_name = "model_scoring"
    return f"{source_name}__ranking"


def load_score_input_frame(
    namespace: argparse.Namespace,
    *,
    feature_columns: tuple[str, ...] | None,
    engine_factory: EngineFactory,
    dataset_loader: DatasetLoader,
) -> pd.DataFrame:
    # Загружаем score-вход либо из CSV, либо из relation в БД.
    if namespace.input_csv is not None:
        print_score_stage(f"load input={namespace.input_csv}")
        return pd.read_csv(namespace.input_csv)

    if namespace.relation_name is None:
        raise RuntimeError("Score input source is not configured.")

    if feature_columns is None:
        raise ValueError("DB-backed score currently requires --model-run-dir.")

    print_score_stage(f"load relation={namespace.relation_name}")
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


def resolve_host_score_column(namespace: argparse.Namespace) -> str:
    # Возвращаем host-like score column с сохранением CLI-default контракта.
    if namespace.host_score_column is None:
        return DEFAULT_HOST_SCORE_COLUMN
    return str(namespace.host_score_column)
