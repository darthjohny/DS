# Файл `run_overview.py` слоя `ui`.
#
# Этот файл отвечает только за:
# - компактную сводку по одному готовому `run_dir`;
# - подготовку чисел, которые потом показывает главная страница интерфейса.
#
# Следующий слой:
# - компоненты и страницы read-only режима;
# - unit-тесты helper-слоя сводок.

from __future__ import annotations

from dataclasses import dataclass

from exohost.ui.loaders import UiLoadedRunBundle


@dataclass(frozen=True, slots=True)
class UiRunOverview:
    # Компактная сводка по одному рабочему прогону для главной страницы интерфейса.
    run_dir_name: str
    created_at_utc: str
    pipeline_name: str
    n_rows_input: int
    n_rows_final_decision: int
    id_count: int
    unknown_count: int
    ood_count: int
    high_priority_count: int
    medium_priority_count: int
    low_priority_count: int


def build_ui_run_overview(bundle: UiLoadedRunBundle) -> UiRunOverview:
    # Главной странице нужны короткие устойчивые числа без ручного парсинга DataFrame по месту.
    metadata = bundle.loaded_artifacts.metadata
    final_decision_df = bundle.loaded_artifacts.final_decision_df
    priority_ranking_df = bundle.loaded_artifacts.priority_ranking_df

    return UiRunOverview(
        run_dir_name=bundle.run_dir.name,
        created_at_utc=str(metadata.get("created_at_utc", "unknown")),
        pipeline_name=str(metadata.get("pipeline_name", "unknown")),
        n_rows_input=int(metadata.get("n_rows_input", int(final_decision_df.shape[0]))),
        n_rows_final_decision=int(
            metadata.get("n_rows_final_decision", int(final_decision_df.shape[0]))
        ),
        id_count=_count_string_rows(final_decision_df["final_domain_state"], "id"),
        unknown_count=_count_string_rows(final_decision_df["final_domain_state"], "unknown"),
        ood_count=_count_string_rows(final_decision_df["final_domain_state"], "ood"),
        high_priority_count=_count_string_rows(priority_ranking_df["priority_label"], "high"),
        medium_priority_count=_count_string_rows(priority_ranking_df["priority_label"], "medium"),
        low_priority_count=_count_string_rows(priority_ranking_df["priority_label"], "low"),
    )


def _count_string_rows(series, expected_label: str) -> int:
    return int(series.astype(str).eq(expected_label).sum())


__all__ = [
    "UiRunOverview",
    "build_ui_run_overview",
]
