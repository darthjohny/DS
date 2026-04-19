# Тестовый файл `test_ui_model_metrics.py` домена `ui`.
#
# Этот файл проверяет только:
# - поиск benchmark-артефактов для страницы метрик;
# - сборку компактной stage-level таблицы по слоям моделей.
#
# Следующий слой:
# - визуальный компонент страницы метрик;
# - страница качества моделей Streamlit.

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from exohost.ui.model_metrics import (
    find_latest_benchmark_run_dir,
    load_benchmark_stage_overview_uncached,
)


def test_find_latest_benchmark_run_dir_returns_latest_matching_prefix(
    tmp_path: Path,
) -> None:
    _build_benchmark_dir(tmp_path, "gaia_id_ood_classification_2026_01_01_000000_000001")
    latest_dir = _build_benchmark_dir(
        tmp_path,
        "gaia_id_ood_classification_2026_01_02_000000_000001",
    )

    resolved_dir = find_latest_benchmark_run_dir(
        "gaia_id_ood_classification",
        artifacts_root=tmp_path,
    )

    assert resolved_dir == latest_dir


def test_load_benchmark_stage_overview_uncached_builds_stage_rows(
    tmp_path: Path,
) -> None:
    _build_benchmark_dir(tmp_path, "gaia_id_ood_classification_2026_01_02_000000_000001")
    _build_benchmark_dir(tmp_path, "gaia_id_coarse_classification_2026_01_02_000000_000001")
    _build_benchmark_dir(tmp_path, "host_field_classification_2026_01_02_000000_000001")
    _build_benchmark_dir(
        tmp_path,
        "gaia_mk_refinement_classification_2026_01_02_000000_000001",
    )

    overview_df = load_benchmark_stage_overview_uncached(artifacts_root=tmp_path)

    assert set(overview_df["stage_key"].astype(str)) == {
        "id_ood",
        "coarse",
        "host",
        "refinement",
    }
    assert set(overview_df["test_accuracy"].dropna().astype(float)) == {0.91}
    assert set(overview_df["cv_mean_macro_f1"].dropna().astype(float)) == {0.88}


def _build_benchmark_dir(root: Path, directory_name: str) -> Path:
    run_dir = root / directory_name
    run_dir.mkdir(parents=True, exist_ok=False)

    pd.DataFrame(
        [
            {
                "model_name": "hist_gradient_boosting",
                "split_name": "test",
                "n_rows": 100,
                "n_classes": 2,
                "accuracy": 0.91,
                "balanced_accuracy": 0.89,
                "macro_precision": 0.90,
                "macro_recall": 0.89,
                "macro_f1": 0.90,
                "roc_auc_ovr": 0.97,
            }
        ]
    ).to_csv(run_dir / "metrics.csv", index=False)

    pd.DataFrame(
        [
            {
                "model_name": "hist_gradient_boosting",
                "cv_folds": 5,
                "mean_accuracy": 0.90,
                "mean_balanced_accuracy": 0.87,
                "mean_macro_f1": 0.88,
                "fit_seconds": 1.0,
                "cv_seconds": 2.0,
                "total_seconds": 3.0,
            }
        ]
    ).to_csv(run_dir / "cv_summary.csv", index=False)

    pd.DataFrame(
        [
            {
                "split_name": "test",
                "target_label": "a",
                "n_rows": 100,
                "share": 1.0,
            }
        ]
    ).to_csv(run_dir / "target_distribution.csv", index=False)

    metadata = {
        "task_name": directory_name.split("_2026_")[0],
        "created_at_utc": "2026-01-02T00:00:00+00:00",
        "n_rows_full": 300,
        "n_rows_train": 200,
        "n_rows_test": 100,
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return run_dir
