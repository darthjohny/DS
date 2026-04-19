# Файл `home_summary.py` слоя `ui`.
#
# Этот файл отвечает только за:
# - подготовку данных для главной страницы интерфейса;
# - компактную сводку главного прикладного результата и поиск схемы системы.
#
# Следующий слой:
# - визуальные компоненты главной страницы;
# - unit-тесты helper-слоя home summary.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from exohost.ui.loaders import UiLoadedRunBundle
from exohost.ui.run_browser import build_ui_top_candidates_frame
from exohost.ui.run_overview import build_ui_run_overview

DEFAULT_UI_SYSTEM_OVERVIEW_CANDIDATE_PATHS: tuple[Path, ...] = (
    Path("docs/assets/diagrams/system_overview_ru.svg"),
    Path("assets/diagrams/system_overview_ru.svg"),
)

HOME_TOP_CANDIDATE_COLUMNS: tuple[str, ...] = (
    "source_id",
    "spec_class",
    "spec_subclass",
    "priority_label",
    "priority_score",
    "host_similarity_score",
)


@dataclass(frozen=True, slots=True)
class UiHomeMainResult:
    # Главный прикладной результат latest run для домашней страницы без ручного парсинга CSV.
    run_dir_name: str
    created_at_utc: str
    total_objects: int
    id_count: int
    unknown_count: int
    ood_count: int
    ranked_count: int
    high_priority_count: int
    medium_priority_count: int
    low_priority_count: int
    high_priority_share: float


def build_ui_home_main_result(bundle: UiLoadedRunBundle) -> UiHomeMainResult:
    # Главная страница должна показывать не только общие счетчики, но и масштаб итогового shortlist.
    overview = build_ui_run_overview(bundle)
    ranked_count = (
        overview.high_priority_count
        + overview.medium_priority_count
        + overview.low_priority_count
    )
    high_priority_share = 0.0
    if ranked_count > 0:
        high_priority_share = overview.high_priority_count / float(ranked_count)

    return UiHomeMainResult(
        run_dir_name=overview.run_dir_name,
        created_at_utc=overview.created_at_utc,
        total_objects=overview.n_rows_final_decision,
        id_count=overview.id_count,
        unknown_count=overview.unknown_count,
        ood_count=overview.ood_count,
        ranked_count=ranked_count,
        high_priority_count=overview.high_priority_count,
        medium_priority_count=overview.medium_priority_count,
        low_priority_count=overview.low_priority_count,
        high_priority_share=high_priority_share,
    )


def build_ui_home_top_candidates_preview(
    bundle: UiLoadedRunBundle,
    *,
    top_n: int = 5,
) -> pd.DataFrame:
    # На главной странице нужен короткий preview shortlist, а не вся ranking-таблица.
    preview_df = build_ui_top_candidates_frame(bundle, top_n=top_n)
    if preview_df.empty:
        return pd.DataFrame(columns=HOME_TOP_CANDIDATE_COLUMNS)

    for column_name in HOME_TOP_CANDIDATE_COLUMNS:
        if column_name not in preview_df.columns:
            preview_df[column_name] = pd.NA
    return preview_df.loc[:, list(HOME_TOP_CANDIDATE_COLUMNS)].copy()


def resolve_ui_system_overview_path(
    candidate_paths: tuple[Path, ...] = DEFAULT_UI_SYSTEM_OVERVIEW_CANDIDATE_PATHS,
) -> Path | None:
    # Схему ищем по фиксированному набору repo-path, чтобы домашняя страница не знала о layout проекта.
    for candidate_path in candidate_paths:
        if candidate_path.exists() and candidate_path.is_file():
            return candidate_path.resolve()
    return None


__all__ = [
    "DEFAULT_UI_SYSTEM_OVERVIEW_CANDIDATE_PATHS",
    "HOME_TOP_CANDIDATE_COLUMNS",
    "UiHomeMainResult",
    "build_ui_home_main_result",
    "build_ui_home_top_candidates_preview",
    "resolve_ui_system_overview_path",
]
