# Тестовый файл `test_ui_run_browser.py` домена `ui`.
#
# Этот файл проверяет только:
# - read-only helper-слой страницы просмотра запуска;
# - устойчивые распределения и верхний список кандидатов.
#
# Следующий слой:
# - компонент и страница просмотра `run_dir`;
# - сценарные проверки интерфейса Streamlit.

from __future__ import annotations

from numbers import Real

from exohost.ui.run_browser import (
    build_ui_domain_distribution_frame,
    build_ui_priority_distribution_frame,
    build_ui_top_candidates_frame,
)

from .ui_testkit import build_ui_loaded_run_bundle


def test_build_ui_domain_distribution_frame_keeps_final_states() -> None:
    distribution_df = build_ui_domain_distribution_frame(build_ui_loaded_run_bundle())

    assert set(distribution_df["final_domain_state"].astype(str)) == {"id", "unknown"}
    assert int(
        distribution_df.loc[
            distribution_df["final_domain_state"].astype(str) == "id",
            "n_rows",
        ].iloc[0]
    ) == 2


def test_build_ui_priority_distribution_frame_keeps_priority_labels() -> None:
    distribution_df = build_ui_priority_distribution_frame(build_ui_loaded_run_bundle())

    assert set(distribution_df["priority_label"].astype(str)) == {"high", "medium"}
    assert _require_float_scalar(distribution_df["share"].sum()) == 1.0


def test_build_ui_top_candidates_frame_sorts_by_priority_score() -> None:
    top_candidates_df = build_ui_top_candidates_frame(build_ui_loaded_run_bundle())

    assert list(top_candidates_df["source_id"]) == [101, 102]
    assert list(top_candidates_df["priority_label"].astype(str)) == ["high", "medium"]
    assert "spec_subclass" in top_candidates_df.columns


def _require_float_scalar(value: object) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, Real):
        return float(value)
    raise TypeError(f"Expected real-like scalar, got {type(value)!r}.")
