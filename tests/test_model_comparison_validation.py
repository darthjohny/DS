"""Тесты для dataset validation слоя comparison-layer."""

from __future__ import annotations

from numbers import Real
from pathlib import Path

import pandas as pd
from analysis.model_comparison import (
    BenchmarkSplit,
    ComparisonProtocol,
    SplitConfig,
    save_benchmark_validation_artifacts,
    split_benchmark_dataset,
    validate_benchmark_split,
)


def scalar_to_float(value: object) -> float:
    """Преобразовать pandas-скаляр в `float` с явной runtime-проверкой."""
    if isinstance(value, Real) and not isinstance(value, bool):
        return float(value)
    raise TypeError(f"Value is not float-compatible: {value!r}")


def scalar_to_int(value: object) -> int:
    """Преобразовать pandas-скаляр в `int` с явной runtime-проверкой."""
    if isinstance(value, Real) and not isinstance(value, bool):
        value_float = float(value)
        if value_float.is_integer():
            return int(value_float)
    raise TypeError(f"Value is not int-compatible: {value!r}")


def make_benchmark_df(*, rows_per_group: int = 16) -> pd.DataFrame:
    """Собрать synthetic benchmark dataset для validation tests."""
    rows: list[dict[str, object]] = []
    source_id = 40000
    class_offsets = {"M": 0.0, "K": 500.0, "G": 1000.0, "F": 1500.0}
    for spec_class, offset in class_offsets.items():
        for is_host in (False, True):
            for index in range(rows_per_group):
                rows.append(
                    {
                        "source_id": source_id,
                        "spec_class": spec_class,
                        "is_host": is_host,
                        "teff_gspphot": 3200.0 + offset + index,
                        "logg_gspphot": 4.0 + index * 0.01,
                        "radius_gspphot": 0.5 + index * 0.02,
                    }
                )
                source_id += 1
    return pd.DataFrame(rows)


def test_validate_benchmark_split_returns_clean_result_for_balanced_split() -> None:
    """Validation gate должен давать clean result для корректного split."""
    split = split_benchmark_dataset(
        make_benchmark_df(),
        split_config=SplitConfig(test_size=0.25, random_state=7),
    )

    result = validate_benchmark_split(split)

    assert result.errors == ()
    assert result.warnings == ()
    assert scalar_to_int(result.summary_df.at[0, "error_count"]) == 0
    assert scalar_to_int(result.summary_df.at[0, "warning_count"]) == 0
    assert set(result.stratify_df["scope_name"]) == {"full", "train", "test"}
    assert sorted(result.feature_drift_df["feature_name"].tolist()) == [
        "logg_gspphot",
        "radius_gspphot",
        "teff_gspphot",
    ]


def test_validate_benchmark_split_rejects_train_test_overlap() -> None:
    """Validation gate должен ловить overlap между train и test."""
    split = split_benchmark_dataset(
        make_benchmark_df(),
        split_config=SplitConfig(test_size=0.25, random_state=7),
    )
    overlap_row = split.train_df.head(1).copy()
    broken_test_df = pd.concat([split.test_df, overlap_row], ignore_index=True)
    broken_full_df = pd.concat([split.train_df, broken_test_df], ignore_index=True)

    result = validate_benchmark_split(
        BenchmarkSplit(
            full_df=broken_full_df,
            train_df=split.train_df,
            test_df=broken_test_df,
        )
    )

    assert result.has_errors is True
    assert any("train/test overlap" in message for message in result.errors)


def test_validate_benchmark_split_warns_on_large_feature_drift() -> None:
    """Validation gate должен предупреждать о сильном train/test drift."""
    split = split_benchmark_dataset(
        make_benchmark_df(),
        split_config=SplitConfig(test_size=0.25, random_state=11),
    )
    drifted_test_df = split.test_df.copy()
    drifted_test_df["teff_gspphot"] = drifted_test_df["teff_gspphot"] + 5000.0
    drifted_full_df = pd.concat([split.train_df, drifted_test_df], ignore_index=True)

    result = validate_benchmark_split(
        BenchmarkSplit(
            full_df=drifted_full_df,
            train_df=split.train_df,
            test_df=drifted_test_df,
        )
    )

    assert result.errors == ()
    assert any("feature drift exceeded" in message for message in result.warnings)
    assert scalar_to_float(result.summary_df.at[0, "max_abs_feature_smd"]) > 1.0


def test_save_benchmark_validation_artifacts_writes_markdown_and_csv(
    tmp_path: Path,
) -> None:
    """Validation artifact saver должен писать markdown и CSV слой."""
    split = split_benchmark_dataset(
        make_benchmark_df(),
        split_config=SplitConfig(test_size=0.25, random_state=5),
    )
    result = validate_benchmark_split(split)

    markdown_path = save_benchmark_validation_artifacts(
        "smoke_validation",
        result,
        output_dir=tmp_path,
        protocol=ComparisonProtocol(),
        note="validation smoke",
    )

    assert markdown_path.exists()
    assert (tmp_path / "smoke_validation_dataset_validation_summary.csv").exists()
    assert (tmp_path / "smoke_validation_dataset_validation_stratify.csv").exists()
    assert (
        tmp_path / "smoke_validation_dataset_validation_feature_drift.csv"
    ).exists()
