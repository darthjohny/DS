# Unit-тесты train-time support review для true `O` rows в coarse source.

from __future__ import annotations

from pathlib import Path

import pandas as pd

from exohost.reporting.archive_research.coarse_o_train_support_review import (
    CoarseOTrainSupportReviewBundle,
    CoarseOTrainSupportReviewConfig,
    build_benchmark_alignment_frame,
    build_coarse_o_train_support_review_bundle,
    build_hot_ob_boundary_support_frame,
    build_hottest_true_o_preview_frame,
    build_true_o_physics_summary_frame,
    build_true_o_split_support_frame,
    build_true_o_stage_support_frame,
    build_true_o_teff_band_support_frame,
)
from exohost.reporting.model_pipeline_review import BenchmarkReviewBundle


def _build_coarse_source_df() -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    o_temperatures = (
        8000.0,
        9000.0,
        11000.0,
        12000.0,
        16000.0,
        17000.0,
        21000.0,
        22000.0,
        26000.0,
        27000.0,
    )
    b_temperatures = (
        10500.0,
        11500.0,
        12500.0,
        13500.0,
        14500.0,
        15500.0,
        16500.0,
        17500.0,
        18500.0,
        19500.0,
    )

    for index, teff_value in enumerate(o_temperatures, start=1):
        rows.append(
            {
                "source_id": index,
                "spec_class": "O",
                "evolution_stage": "dwarf",
                "is_evolved": False,
                "teff_gspphot": teff_value,
                "logg_gspphot": 4.0,
                "mh_gspphot": 0.0,
                "bp_rp": -0.1,
                "parallax": 1.5,
                "parallax_over_error": 10.0,
                "ruwe": 1.0,
                "radius_feature": 5.0 + index,
                "radius_flame": 5.5 + index,
            }
        )

    for index, teff_value in enumerate(b_temperatures, start=101):
        rows.append(
            {
                "source_id": index,
                "spec_class": "B",
                "evolution_stage": "dwarf",
                "is_evolved": False,
                "teff_gspphot": teff_value,
                "logg_gspphot": 4.1,
                "mh_gspphot": -0.1,
                "bp_rp": 0.0,
                "parallax": 2.0,
                "parallax_over_error": 12.0,
                "ruwe": 1.0,
                "radius_feature": 3.0 + (index - 100),
                "radius_flame": 3.5 + (index - 100),
            }
        )

    return pd.DataFrame.from_records(rows)


def test_true_o_support_review_bundle_reconstructs_split_and_hot_support() -> None:
    bundle = build_coarse_o_train_support_review_bundle(_build_coarse_source_df())

    support_df = build_true_o_split_support_frame(bundle)
    stage_df = build_true_o_stage_support_frame(bundle)
    teff_band_df = build_true_o_teff_band_support_frame(bundle)
    hot_boundary_df = build_hot_ob_boundary_support_frame(bundle)
    physics_df = build_true_o_physics_summary_frame(bundle)
    preview_df = build_hottest_true_o_preview_frame(bundle, top_n=5)

    support_by_split = {
        str(row["split_name"]): int(row["n_rows_true_o"])
        for row in support_df.to_dict(orient="records")
    }
    assert support_by_split == {"full": 10, "train": 7, "test": 3}
    assert set(stage_df["evolution_stage"].astype(str)) == {"dwarf"}
    assert {
        "< 10000 K",
        "10000-14999 K",
        "15000-19999 K",
        "20000-24999 K",
        ">= 25000 K",
    }.issubset(set(teff_band_df["teff_band"].astype(str)))

    full_hot_boundary_df = hot_boundary_df.loc[hot_boundary_df["split_name"] == "full"].copy()
    full_hot_counts = {
        str(row["spec_class"]): int(row["n_rows"])
        for row in full_hot_boundary_df.to_dict(orient="records")
    }
    assert full_hot_counts == {"B": 10, "O": 8}
    assert set(physics_df["split_name"].astype(str)) == {"full", "train", "test"}
    assert list(preview_df["source_id"])[:2] == [10, 9]
    assert set(preview_df["split_name"].astype(str)).issubset({"full", "train", "test"})


def test_benchmark_alignment_frame_marks_matching_reconstructed_counts() -> None:
    base_bundle = build_coarse_o_train_support_review_bundle(_build_coarse_source_df())
    benchmark_bundle = BenchmarkReviewBundle(
        run_dir=Path("artifacts/benchmarks/fake"),
        metrics_df=pd.DataFrame(),
        cv_summary_df=pd.DataFrame(),
        target_distribution_df=pd.DataFrame(),
        metadata={
            "n_rows_full": int(base_bundle.split.full_df.shape[0]),
            "n_rows_train": int(base_bundle.split.train_df.shape[0]),
            "n_rows_test": int(base_bundle.split.test_df.shape[0]),
        },
    )
    bundle = CoarseOTrainSupportReviewBundle(
        config=CoarseOTrainSupportReviewConfig(),
        source_df=base_bundle.source_df,
        split=base_bundle.split,
        benchmark_bundle=benchmark_bundle,
    )

    alignment_df = build_benchmark_alignment_frame(bundle)

    assert list(alignment_df["split_name"]) == ["full", "train", "test"]
    assert alignment_df["is_match"].tolist() == [True, True, True]
