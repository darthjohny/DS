# Тестовый файл `test_host_training_review.py` домена `reporting`.
#
# Этот файл проверяет только:
# - проверку логики домена: review-helper, сводки и notebook display-слой;
# - регрессионные сценарии и ожидаемые контракты целевого слоя.
#
# Следующий слой:
# - реализация домена `reporting` в `src/exohost`;
# - соседние тесты и testkit этого же пакета.

from __future__ import annotations

from numbers import Integral

import pandas as pd

from exohost.reporting.host_training_review import (
    build_host_class_distribution_frame,
    build_host_class_stage_crosstab_frame,
    build_host_stage_distribution_frame,
    build_host_training_summary_frame,
)


def _require_int_scalar(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"Expected integer-like scalar, got {type(value).__name__}.")
    return int(value)


def build_host_review_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_id": [1, 2, 3, 4],
            "hostname": ["Host A", "Host B", "Host B", pd.NA],
            "spec_class": ["G", "K", "K", "M"],
            "evolution_stage": ["dwarf", "evolved", "evolved", "dwarf"],
            "spec_subclass": ["G2", pd.NA, "K3", pd.NA],
        }
    )


def test_build_host_training_summary_frame_returns_compact_source_summary() -> None:
    summary_df = build_host_training_summary_frame(build_host_review_df())

    assert _require_int_scalar(summary_df.loc[0, "n_rows"]) == 4
    assert _require_int_scalar(summary_df.loc[0, "n_unique_source_id"]) == 4
    assert _require_int_scalar(summary_df.loc[0, "n_unique_hostname"]) == 2
    assert _require_int_scalar(summary_df.loc[0, "n_spec_subclass_rows"]) == 2


def test_host_review_distribution_helpers_cover_class_and_stage_balance() -> None:
    host_df = build_host_review_df()

    class_df = build_host_class_distribution_frame(host_df)
    stage_df = build_host_stage_distribution_frame(host_df)
    crosstab_df = build_host_class_stage_crosstab_frame(host_df)

    assert set(class_df["spec_class"]) == {"G", "K", "M"}
    assert set(stage_df["evolution_stage"]) == {"dwarf", "evolved"}
    assert _require_int_scalar(crosstab_df.loc["K", "evolved"]) == 2
