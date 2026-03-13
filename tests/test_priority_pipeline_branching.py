"""Точечные тесты helper-ветвления production pipeline."""

from __future__ import annotations

import pandas as pd

from priority_pipeline.branching import (
    is_unknown_router_output,
    known_low_reason_code,
    split_branches,
)


def test_is_unknown_router_output_is_case_and_whitespace_tolerant() -> None:
    """Unknown-detection должна быть устойчивой к регистру и пробелам."""
    assert is_unknown_router_output(" unknown ", "dwarf", "M_dwarf")
    assert is_unknown_router_output("K", " unknown ", "K_dwarf")
    assert is_unknown_router_output("K", "dwarf", " unknown ")
    assert not is_unknown_router_output("K", "dwarf", "K_dwarf")


def test_known_low_reason_code_separates_hot_evolved_and_filtered_cases() -> None:
    """Reason helper должен различать hot, evolved и прочий filtered-out low-known."""
    assert known_low_reason_code("A", "dwarf") == "HOT_STAR"
    assert known_low_reason_code("K", "evolved") == "EVOLVED_STAR"
    assert known_low_reason_code("L", "dwarf") == "FILTERED_OUT"


def test_split_branches_keeps_low_branch_order_and_includes_unknown_rows() -> None:
    """Совместимый wrapper должен возвращать host и combined low без смены порядка."""
    df_router = pd.DataFrame(
        [
            {
                "source_id": 11,
                "predicted_spec_class": "K",
                "predicted_evolution_stage": "dwarf",
                "router_label": "K_dwarf",
            },
            {
                "source_id": 22,
                "predicted_spec_class": "UNKNOWN",
                "predicted_evolution_stage": "unknown",
                "router_label": "UNKNOWN",
            },
            {
                "source_id": 33,
                "predicted_spec_class": "A",
                "predicted_evolution_stage": "dwarf",
                "router_label": "A_dwarf",
            },
        ]
    )

    host_df, low_df = split_branches(df_router)

    assert host_df["source_id"].tolist() == [11]
    assert low_df["source_id"].tolist() == [22, 33]
