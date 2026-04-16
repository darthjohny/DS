# Тестовый файл `test_host_calibration_review.py` домена `reporting`.
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

from exohost.evaluation.protocol import BenchmarkProtocol, SplitConfig
from exohost.features.training_frame import (
    prepare_host_training_frame,
    prepare_router_training_frame,
)
from exohost.reporting.host_calibration_review import (
    build_host_calibration_curve_review_frame,
    build_host_calibration_group_frame,
    build_host_calibration_metric_summary_frame,
    build_host_calibration_split_summary_frame,
    build_host_probability_bin_review_frame,
)
from exohost.reporting.host_calibration_source import (
    build_host_calibration_source_from_frames,
)


def _require_int_scalar(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"Expected integer-like scalar, got {type(value).__name__}.")
    return int(value)


def build_host_frame() -> pd.DataFrame:
    # Небольшой host-кадр имитирует подтвержденные звезды-хозяева, на которых
    # review-слой потом строит calibration и групповые сводки.
    return pd.DataFrame(
        [
            {
                "source_id": 101,
                "hostname": "Host-G1",
                "teff_gspphot": 5800.0,
                "logg_gspphot": 4.4,
                "radius_flame": 1.05,
                "radius_gspphot": 1.00,
                "lum_flame": 1.1,
                "dist_arcsec": 0.1,
                "parallax": 15.0,
                "parallax_over_error": 18.0,
                "ruwe": 1.01,
                "phot_g_mean_mag": 11.1,
                "bp_rp": 0.75,
                "mh_gspphot": 0.10,
                "validation_factor": 0.95,
                "classprob_dsc_combmod_star": 0.95,
                "spec_class": "G",
                "spec_subclass": "G2",
                "evolution_stage": "dwarf",
            },
            {
                "source_id": 102,
                "hostname": "Host-G2",
                "teff_gspphot": 5900.0,
                "logg_gspphot": 4.3,
                "radius_flame": 1.12,
                "radius_gspphot": 1.09,
                "lum_flame": 1.3,
                "dist_arcsec": 0.1,
                "parallax": 14.2,
                "parallax_over_error": 19.0,
                "ruwe": 1.02,
                "phot_g_mean_mag": 10.9,
                "bp_rp": 0.70,
                "mh_gspphot": 0.08,
                "validation_factor": 0.94,
                "classprob_dsc_combmod_star": 0.94,
                "spec_class": "G",
                "spec_subclass": "G1",
                "evolution_stage": "dwarf",
            },
            {
                "source_id": 103,
                "hostname": "Host-K1",
                "teff_gspphot": 4500.0,
                "logg_gspphot": 4.6,
                "radius_flame": 0.82,
                "radius_gspphot": 0.80,
                "lum_flame": 0.4,
                "dist_arcsec": 0.2,
                "parallax": 12.0,
                "parallax_over_error": 17.0,
                "ruwe": 1.02,
                "phot_g_mean_mag": 12.4,
                "bp_rp": 1.10,
                "mh_gspphot": -0.10,
                "validation_factor": 0.92,
                "classprob_dsc_combmod_star": 0.92,
                "spec_class": "K",
                "spec_subclass": "K4",
                "evolution_stage": "dwarf",
            },
            {
                "source_id": 104,
                "hostname": "Host-K2",
                "teff_gspphot": 4600.0,
                "logg_gspphot": 4.5,
                "radius_flame": 0.88,
                "radius_gspphot": 0.84,
                "lum_flame": 0.5,
                "dist_arcsec": 0.2,
                "parallax": 11.3,
                "parallax_over_error": 16.0,
                "ruwe": 1.03,
                "phot_g_mean_mag": 12.0,
                "bp_rp": 1.05,
                "mh_gspphot": -0.05,
                "validation_factor": 0.91,
                "classprob_dsc_combmod_star": 0.91,
                "spec_class": "K",
                "spec_subclass": "K3",
                "evolution_stage": "dwarf",
            },
        ]
    )


def build_router_frame() -> pd.DataFrame:
    # Router-кадр служит фоновым сравнительным доменом. Он нужен, чтобы
    # calibration-source мог собрать balanced host/field картину для review.
    return pd.DataFrame(
        [
            {
                "source_id": 201,
                "ra": 10.0,
                "dec": 20.0,
                "teff_gspphot": 5750.0,
                "logg_gspphot": 4.4,
                "radius_gspphot": 0.98,
                "parallax": 15.2,
                "parallax_over_error": 17.5,
                "ruwe": 1.03,
                "bp_rp": 0.76,
                "mh_gspphot": 0.00,
                "validation_factor": 0.90,
                "spec_class": "G",
                "spec_subclass": "G3",
                "evolution_stage": "dwarf",
            },
            {
                "source_id": 202,
                "ra": 11.0,
                "dec": 21.0,
                "teff_gspphot": 5850.0,
                "logg_gspphot": 4.3,
                "radius_gspphot": 1.03,
                "parallax": 14.8,
                "parallax_over_error": 18.0,
                "ruwe": 1.01,
                "bp_rp": 0.71,
                "mh_gspphot": 0.02,
                "validation_factor": 0.89,
                "spec_class": "G",
                "spec_subclass": "G1",
                "evolution_stage": "dwarf",
            },
            {
                "source_id": 203,
                "ra": 12.0,
                "dec": 22.0,
                "teff_gspphot": 4480.0,
                "logg_gspphot": 4.7,
                "radius_gspphot": 0.78,
                "parallax": 11.7,
                "parallax_over_error": 16.5,
                "ruwe": 1.02,
                "bp_rp": 1.12,
                "mh_gspphot": -0.12,
                "validation_factor": 0.88,
                "spec_class": "K",
                "spec_subclass": "K5",
                "evolution_stage": "dwarf",
            },
            {
                "source_id": 204,
                "ra": 13.0,
                "dec": 23.0,
                "teff_gspphot": 4550.0,
                "logg_gspphot": 4.6,
                "radius_gspphot": 0.81,
                "parallax": 12.5,
                "parallax_over_error": 16.8,
                "ruwe": 1.04,
                "bp_rp": 1.08,
                "mh_gspphot": -0.08,
                "validation_factor": 0.87,
                "spec_class": "K",
                "spec_subclass": "K4",
                "evolution_stage": "dwarf",
            },
        ]
    )


def test_host_calibration_review_frames_build_expected_outputs() -> None:
    # Один тест покрывает весь главный набор таблиц host-calibration review:
    # split summary, metric summary, curve, probability bins и групповую сводку.
    host_frame = prepare_host_training_frame(build_host_frame())
    router_frame = prepare_router_training_frame(build_router_frame())
    source = build_host_calibration_source_from_frames(
        host_frame,
        router_frame,
        task_name="host_field_classification",
        model_name="hist_gradient_boosting",
        field_to_host_ratio=1,
        protocol=BenchmarkProtocol(split=SplitConfig(test_size=0.5, random_state=42)),
    )

    split_summary_df = build_host_calibration_split_summary_frame(source)
    metric_summary_df = build_host_calibration_metric_summary_frame(source)
    curve_df = build_host_calibration_curve_review_frame(source)
    bin_df = build_host_probability_bin_review_frame(source)
    class_df = build_host_calibration_group_frame(source, group_column="spec_class")

    assert _require_int_scalar(split_summary_df.loc[0, "n_rows_test"]) == 4
    assert "brier_score" in metric_summary_df.columns
    assert "fraction_of_positives" in curve_df.columns
    assert _require_int_scalar(bin_df["n_rows"].sum()) == 4
    assert set(class_df["spec_class"]) == {"G", "K"}
