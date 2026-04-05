# Тестовый файл `test_host_calibration_bundle.py` домена `reporting`.
#
# Этот файл проверяет только:
# - проверку логики домена: review-helper, сводки и notebook display-слой;
# - регрессионные сценарии и ожидаемые контракты целевого слоя.
#
# Следующий слой:
# - реализация домена `reporting` в `src/exohost`;
# - соседние тесты и testkit этого же пакета.

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from exohost.evaluation.split import DatasetSplit
from exohost.reporting.host_calibration_bundle import (
    find_latest_host_model_run_dir,
    load_host_calibration_review_bundle,
)
from exohost.reporting.host_calibration_source import HostCalibrationSource


def _write_model_metadata(
    run_dir: Path,
    *,
    task_name: str,
    created_at_utc: str,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "task_name": task_name,
                "model_name": "hist_gradient_boosting",
                "created_at_utc": created_at_utc,
                "target_column": "host_label",
                "feature_columns": ["teff_gspphot", "radius_flame"],
            }
        ),
        encoding="utf-8",
    )


def _build_dummy_source() -> HostCalibrationSource:
    full_df = pd.DataFrame(
        [
            {
                "source_id": 1,
                "host_label": "host",
                "host_similarity_score": 0.95,
                "predicted_host_label": "host",
                "spec_class": "G",
                "evolution_stage": "dwarf",
            },
            {
                "source_id": 2,
                "host_label": "field",
                "host_similarity_score": 0.20,
                "predicted_host_label": "field",
                "spec_class": "K",
                "evolution_stage": "dwarf",
            },
            {
                "source_id": 3,
                "host_label": "host",
                "host_similarity_score": 0.85,
                "predicted_host_label": "host",
                "spec_class": "G",
                "evolution_stage": "dwarf",
            },
            {
                "source_id": 4,
                "host_label": "field",
                "host_similarity_score": 0.40,
                "predicted_host_label": "field",
                "spec_class": "K",
                "evolution_stage": "dwarf",
            },
        ]
    )
    split = DatasetSplit(
        full_df=full_df.copy(),
        train_df=full_df.iloc[[0, 1]].reset_index(drop=True),
        test_df=full_df.iloc[[2, 3]].reset_index(drop=True),
    )
    return HostCalibrationSource(
        task_name="host_field_classification",
        model_name="hist_gradient_boosting",
        target_column="host_label",
        positive_label="host",
        host_score_column="host_similarity_score",
        feature_columns=("teff_gspphot", "radius_flame"),
        split=split,
        train_scored_df=split.train_df.copy(),
        test_scored_df=split.test_df.copy(),
    )


def test_find_latest_host_model_run_dir_returns_newest_host_run(tmp_path: Path) -> None:
    _write_model_metadata(
        tmp_path / "host_old",
        task_name="host_field_classification",
        created_at_utc="2026-03-29T07:16:01+00:00",
    )
    _write_model_metadata(
        tmp_path / "coarse_new",
        task_name="gaia_id_coarse_classification",
        created_at_utc="2026-03-29T08:00:00+00:00",
    )
    _write_model_metadata(
        tmp_path / "host_new",
        task_name="host_field_classification",
        created_at_utc="2026-03-29T09:00:00+00:00",
    )

    latest = find_latest_host_model_run_dir(tmp_path)

    assert latest == tmp_path / "host_new"


def test_load_host_calibration_review_bundle_builds_review_tables(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_dir = tmp_path / "host_run"
    _write_model_metadata(
        run_dir,
        task_name="host_field_classification",
        created_at_utc="2026-03-29T09:00:00+00:00",
    )

    monkeypatch.setattr(
        "exohost.reporting.host_calibration_bundle.build_host_calibration_source_from_model_artifact",
        lambda *args, **kwargs: _build_dummy_source(),
    )

    bundle = load_host_calibration_review_bundle(run_dir)

    assert bundle.run_dir == run_dir
    assert list(bundle.metric_summary_df["n_rows"]) == [2]
    assert set(bundle.class_group_df["spec_class"]) == {"G", "K"}
    assert set(bundle.stage_group_df["evolution_stage"]) == {"dwarf"}
