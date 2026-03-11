"""Логика decision-layer scoring для офлайн-калибровки."""

from __future__ import annotations

from typing import Any

import pandas as pd

from decision_calibration.config import CalibrationConfig
from priority_pipeline import (
    clip_unit_interval,
    priority_tier_from_score,
    stub_reason_code,
)


def class_prior(spec_class: Any, config: CalibrationConfig) -> float:
    """Вернуть calibration prior для спектрального класса."""
    mapping = {
        "K": config.class_prior.k,
        "G": config.class_prior.g,
        "M": config.class_prior.m,
        "F": config.class_prior.f,
    }
    return float(mapping.get(str(spec_class), 0.90))


def metallicity_factor(value: Any, config: CalibrationConfig) -> float:
    """Вернуть calibration factor по metallicity."""
    if pd.isna(value):
        return config.metallicity.neutral_factor
    mh = float(value)
    if mh <= config.metallicity.low_threshold:
        return config.metallicity.low_factor
    if mh < config.metallicity.solar_threshold:
        return config.metallicity.neutral_factor
    if mh < config.metallicity.high_threshold:
        return config.metallicity.positive_factor
    return config.metallicity.high_factor


def distance_pc_from_parallax(parallax: Any) -> float | None:
    """Преобразовать параллакс в расстояние в парсеках."""
    if pd.isna(parallax):
        return None
    plx = float(parallax)
    if plx <= 0.0:
        return None
    return 1000.0 / plx


def distance_factor(parallax: Any, config: CalibrationConfig) -> float:
    """Вернуть calibration factor по расстоянию."""
    distance_pc = distance_pc_from_parallax(parallax)
    if distance_pc is None:
        return config.distance.invalid_factor
    if distance_pc <= config.distance.near_max_pc:
        return config.distance.near_factor
    if distance_pc <= config.distance.moderate_max_pc:
        return config.distance.moderate_factor
    if distance_pc <= config.distance.distant_max_pc:
        return config.distance.distant_factor
    if distance_pc <= config.distance.far_max_pc:
        return config.distance.far_factor
    return config.distance.very_far_factor


def ruwe_factor(value: Any, config: CalibrationConfig) -> float:
    """Вернуть calibration factor по RUWE."""
    ruwe = config.quality.ruwe
    if pd.isna(value):
        return ruwe.missing_factor
    current = float(value)
    if current <= ruwe.good_max:
        return ruwe.good_factor
    if current <= ruwe.warning_max:
        return ruwe.warning_factor
    if current <= ruwe.alert_max:
        return ruwe.alert_factor
    if current <= ruwe.bad_max:
        return ruwe.bad_factor
    return ruwe.very_bad_factor


def parallax_precision_factor(
    value: Any,
    config: CalibrationConfig,
) -> float:
    """Вернуть calibration factor по `parallax_over_error`."""
    precision = config.quality.parallax_precision
    if pd.isna(value):
        return precision.missing_factor
    current = float(value)
    if current >= precision.excellent_min:
        return precision.excellent_factor
    if current >= precision.good_min:
        return precision.good_factor
    if current >= precision.acceptable_min:
        return precision.acceptable_factor
    if current >= precision.weak_min:
        return precision.weak_factor
    return precision.poor_factor


def quality_factor(
    ruwe_value: Any,
    parallax_over_error: Any,
    config: CalibrationConfig,
) -> float:
    """Объединить RUWE и precision в quality factor для калибровки."""
    value = (
        ruwe_factor(ruwe_value, config)
        * parallax_precision_factor(parallax_over_error, config)
    )
    return clip_unit_interval(float(value))


def build_low_priority_preview(df_low: pd.DataFrame) -> pd.DataFrame:
    """Собрать preview-ветку для A/B/O и evolved stars.

    В офлайн-калибровке эта ветка не проходит host-scoring и нужна для
    того, чтобы итоговый ranking включал весь исходный batch.
    """
    if df_low.empty:
        return df_low.copy()

    result = df_low.copy()
    result["gauss_label"] = None
    result["host_log_likelihood"] = None
    result["field_log_likelihood"] = None
    result["host_log_lr"] = None
    result["host_posterior"] = None
    result["d_mahal"] = None
    result["similarity"] = None
    result["class_prior"] = None
    result["distance_factor"] = None
    result["quality_factor"] = None
    result["metallicity_factor"] = None
    result["final_score"] = 0.0
    result["priority_tier"] = "LOW"
    result["reason_code"] = [
        stub_reason_code(spec_class, stage)
        for spec_class, stage in result[
            ["predicted_spec_class", "predicted_evolution_stage"]
        ].itertuples(index=False, name=None)
    ]
    result["host_model_version"] = None
    return result


def apply_calibration_config(
    df_scored: pd.DataFrame,
    config: CalibrationConfig,
    host_model_version_value: str,
) -> pd.DataFrame:
    """Применить офлайн calibration formula к host-ветке.

    В отличие от production decision layer, здесь формула включает
    явный `distance_factor` и используется только для offline
    калибровочных итераций.
    """
    if df_scored.empty:
        return df_scored.copy()

    result = df_scored.copy()
    result["class_prior"] = [
        class_prior(spec_class, config)
        for spec_class in result["predicted_spec_class"]
    ]
    result["distance_factor"] = [
        distance_factor(parallax, config)
        for parallax in result["parallax"]
    ]
    result["quality_factor"] = [
        quality_factor(ruwe_value, plx_error, config)
        for ruwe_value, plx_error in result[
            ["ruwe", "parallax_over_error"]
        ].itertuples(index=False, name=None)
    ]
    result["metallicity_factor"] = [
        metallicity_factor(value, config)
        for value in result["mh_gspphot"]
    ]

    score_rows = result[
        [
            "host_posterior",
            "class_prior",
            "distance_factor",
            "quality_factor",
            "metallicity_factor",
        ]
    ].itertuples(index=False, name=None)
    result["final_score"] = [
        clip_unit_interval(
            float(host_posterior)
            * float(prior_value)
            * float(distance_value)
            * float(quality_value)
            * float(metallicity_value)
        )
        for (
            host_posterior,
            prior_value,
            distance_value,
            quality_value,
            metallicity_value,
        ) in score_rows
    ]
    result["d_mahal"] = None
    result["similarity"] = None
    result["priority_tier"] = [
        priority_tier_from_score(float(score))
        for score in result["final_score"]
    ]
    result["reason_code"] = "HOST_SCORING"
    result["host_model_version"] = host_model_version_value
    return result


__all__ = [
    "apply_calibration_config",
    "build_low_priority_preview",
    "class_prior",
    "distance_factor",
    "distance_pc_from_parallax",
    "metallicity_factor",
    "parallax_precision_factor",
    "quality_factor",
    "ruwe_factor",
]
