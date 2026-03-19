"""Тесты для slide-ready presentation assets comparison-layer."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from analysis.model_comparison.presentation_assets import (
    PresentationAssetConfig,
    build_operational_shortlist_summary_table,
    build_operational_shortlist_top_table,
    generate_assets,
    load_frames,
)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    """Сохранить CSV-файл для asset-builder smoke tests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def make_config(tmp_path: Path) -> PresentationAssetConfig:
    """Собрать минимальный конфиг для тестов presentation assets."""
    return PresentationAssetConfig(
        benchmark_run_name="benchmark_smoke",
        snapshot_run_name="benchmark_smoke",
        input_dir=tmp_path / "model_comparison",
        output_dir=tmp_path / "presentation_assets" / "benchmark_smoke",
        production_input_dir=tmp_path / "production_runs",
        operational_run_name="production_smoke",
        top_rank_limit=2,
    )


def seed_minimal_assets(config: PresentationAssetConfig) -> None:
    """Записать минимальный набор benchmark/snapshot/production CSV."""
    write_csv(
        config.input_dir / f"{config.benchmark_run_name}_summary.csv",
        pd.DataFrame(
            [
                {
                    "model_name": "main_contrastive_v1",
                    "split_name": "test",
                    "n_rows": 10,
                    "n_host": 3,
                    "n_field": 7,
                    "roc_auc": 0.8,
                    "pr_auc": 0.7,
                    "brier": 0.2,
                    "precision_at_k": 0.5,
                },
                {
                    "model_name": "baseline_random_forest",
                    "split_name": "test",
                    "n_rows": 10,
                    "n_host": 3,
                    "n_field": 7,
                    "roc_auc": 0.9,
                    "pr_auc": 0.8,
                    "brier": 0.1,
                    "precision_at_k": 0.6,
                },
                {
                    "model_name": "baseline_mlp_small",
                    "split_name": "test",
                    "n_rows": 10,
                    "n_host": 3,
                    "n_field": 7,
                    "roc_auc": 0.88,
                    "pr_auc": 0.78,
                    "brier": 0.12,
                    "precision_at_k": 0.58,
                },
                {
                    "model_name": "baseline_legacy_gaussian",
                    "split_name": "test",
                    "n_rows": 10,
                    "n_host": 3,
                    "n_field": 7,
                    "roc_auc": 0.75,
                    "pr_auc": 0.62,
                    "brier": 0.22,
                    "precision_at_k": 0.45,
                },
            ]
        ),
    )
    write_csv(
        config.input_dir / f"{config.benchmark_run_name}_classwise.csv",
        pd.DataFrame(
            [
                {
                    "model_name": model_name,
                    "split_name": "test",
                    "spec_class": spec_class,
                    "roc_auc": 0.8,
                }
                for model_name in (
                    "main_contrastive_v1",
                    "baseline_random_forest",
                    "baseline_mlp_small",
                    "baseline_legacy_gaussian",
                )
                for spec_class in ("F", "G", "K", "M")
            ]
        ),
    )
    write_csv(
        config.input_dir / f"{config.benchmark_run_name}_search_summary.csv",
        pd.DataFrame(
            [
                {
                    "model_name": "main_contrastive_v1",
                    "search_scope": "model",
                    "spec_class": "all",
                    "refit_metric": "roc_auc",
                    "cv_folds": 10,
                    "candidate_count": 4,
                    "best_cv_score": 0.81,
                    "best_params_json": "{\"alpha\": 0.1}",
                }
            ]
        ),
    )
    write_csv(
        config.input_dir / f"{config.snapshot_run_name}_snapshot_summary.csv",
        pd.DataFrame(
            [
                {
                    "model_name": model_name,
                    "input_rows": 100,
                    "host_candidates": 80,
                    "high_rows": 10,
                    "medium_rows": 20,
                    "low_rows": 70,
                    "top_final_score": 0.9,
                }
                for model_name in (
                    "main_contrastive_v1",
                    "baseline_random_forest",
                    "baseline_mlp_small",
                    "baseline_legacy_gaussian",
                )
            ]
        ),
    )
    for model_name in (
        "main_contrastive_v1",
        "baseline_random_forest",
        "baseline_mlp_small",
        "baseline_legacy_gaussian",
    ):
        write_csv(
            config.input_dir / f"{config.snapshot_run_name}_snapshot_{model_name}_top.csv",
            pd.DataFrame(
                [
                    {
                        "source_id": 1,
                        "predicted_spec_class": "K",
                        "host_posterior": 0.91,
                        "final_score": 0.83,
                        "priority_tier": "HIGH",
                        "teff_gspphot": 5200.0,
                        "logg_gspphot": 4.5,
                        "radius_gspphot": 0.91,
                    },
                    {
                        "source_id": 2,
                        "predicted_spec_class": "M",
                        "host_posterior": 0.85,
                        "final_score": 0.77,
                        "priority_tier": "HIGH",
                        "teff_gspphot": 3600.0,
                        "logg_gspphot": 4.7,
                        "radius_gspphot": 0.58,
                    },
                ]
            ),
        )
    write_csv(
        config.production_input_dir / f"{config.operational_run_name}_shortlist.csv",
        pd.DataFrame(
            [
                {
                    "observation_priority": 1,
                    "rank_in_priority": 1,
                    "source_id": 11,
                    "predicted_spec_class": "K",
                    "host_like_percent": 94.1,
                    "final_score": 0.73,
                    "ra": 12.3,
                    "dec": -4.5,
                },
                {
                    "observation_priority": 2,
                    "rank_in_priority": 1,
                    "source_id": 22,
                    "predicted_spec_class": "M",
                    "host_like_percent": 88.5,
                    "final_score": 0.61,
                    "ra": 98.7,
                    "dec": 11.2,
                },
            ]
        ),
    )
    write_csv(
        config.production_input_dir / f"{config.operational_run_name}_shortlist_summary.csv",
        pd.DataFrame(
            [
                {"observation_priority": 1, "n_rows": 1},
                {"observation_priority": 2, "n_rows": 1},
            ]
        ),
    )


def test_build_operational_shortlist_tables_use_production_artifacts(
    tmp_path: Path,
) -> None:
    """Presentation tables должны читаться из dedicated production shortlist."""
    config = make_config(tmp_path)
    seed_minimal_assets(config)
    frames = load_frames(config)

    summary_table = build_operational_shortlist_summary_table(frames)
    top_table = build_operational_shortlist_top_table(frames, top_rank_limit=2)

    assert summary_table.to_dict(orient="records") == [
        {"Приоритет": 1, "Класс": "K", "Число кандидатов": 1},
        {"Приоритет": 2, "Класс": "M", "Число кандидатов": 1},
    ]
    assert top_table["source_id"].tolist() == [11, 22]
    assert top_table["Класс"].tolist() == ["K", "M"]


def test_generate_assets_writes_operational_shortlist_tables(tmp_path: Path) -> None:
    """Asset-builder должен сохранять slide-ready production shortlist tables."""
    config = make_config(tmp_path)
    seed_minimal_assets(config)

    created_paths = generate_assets(config)

    expected_paths = {
        config.output_dir / "operational_shortlist_summary_table.csv",
        config.output_dir / "operational_shortlist_top_table.csv",
    }
    assert expected_paths.issubset(set(created_paths))
    assert (config.output_dir / "operational_shortlist_summary_table.csv").exists()
    assert (config.output_dir / "operational_shortlist_top_table.csv").exists()
