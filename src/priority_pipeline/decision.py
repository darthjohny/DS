"""Decision-layer helpers для production ranking pipeline.

Модуль отвечает за:

- расчёт физических и quality-based soft factors;
- применение contrastive host-score к допустимой ветке MKGF dwarf;
- сборку итогового `final_score` и low-priority stub-веток.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, cast

import pandas as pd

from host_model import score_df_contrastive as score_host_df
from priority_pipeline.branching import known_low_reason_code
from priority_pipeline.constants import (
    CLASS_PRIOR_BY_SPEC_CLASS,
    DEFAULT_CLASS_PRIOR,
    HOST_MODEL_VERSION,
    HOST_SCORING_REASON,
    PRIORITY_TIER_HIGH_THRESHOLD,
    PRIORITY_TIER_MEDIUM_THRESHOLD,
    ROUTER_UNKNOWN_REASON,
)
from priority_pipeline.frame_contract import ensure_decision_columns


def clip_unit_interval(value: float) -> float:
    """Ограничить числовое значение диапазоном `[0, 1]`."""
    return max(0.0, min(1.0, float(value)))


def class_prior(spec_class: Any) -> float:
    """Вернуть физический prior для спектрального класса."""
    return float(
        CLASS_PRIOR_BY_SPEC_CLASS.get(
            str(spec_class),
            DEFAULT_CLASS_PRIOR,
        )
    )


def ruwe_factor(value: Any) -> float:
    """Преобразовать RUWE в фактор астрометрического качества."""
    if pd.isna(value):
        return 0.85
    ruwe = float(value)
    if ruwe <= 1.10:
        return 1.00
    if ruwe <= 1.40:
        return 0.92
    if ruwe <= 2.00:
        return 0.70
    return 0.45


def parallax_precision_factor(value: Any) -> float:
    """Преобразовать `parallax_over_error` в фактор точности расстояния."""
    if pd.isna(value):
        return 0.75
    ratio = float(value)
    if ratio >= 20.0:
        return 1.00
    if ratio >= 10.0:
        return 0.92
    if ratio >= 5.0:
        return 0.78
    if ratio > 0.0:
        return 0.60
    return 0.40


def distance_factor(parallax: Any) -> float:
    """Использовать параллакс как мягкий proxy-фактор близости."""
    if pd.isna(parallax):
        return 0.75
    plx = float(parallax)
    if plx >= 20.0:
        return 1.00
    if plx >= 10.0:
        return 0.92
    if plx >= 5.0:
        return 0.82
    if plx > 0.0:
        return 0.65
    return 0.45


def quality_factor(
    ruwe: Any,
    parallax_over_error: Any,
) -> float:
    """Объединить факторы надёжности астрометрии без distance penalty."""
    values = (
        ruwe_factor(ruwe),
        parallax_precision_factor(parallax_over_error),
    )
    return clip_unit_interval(sum(values) / float(len(values)))


def reliability_factor(
    ruwe: Any,
    parallax_over_error: Any,
) -> float:
    """Вернуть отдельный фактор качества и надёжности астрометрии."""
    return quality_factor(ruwe, parallax_over_error)


def followup_factor(parallax: Any) -> float:
    """Вернуть отдельный фактор наблюдательной пригодности follow-up."""
    return distance_factor(parallax)


def metallicity_factor(value: Any) -> float:
    """Преобразовать `[M/H]` в консервативный фактор приоритета."""
    if pd.isna(value):
        return 1.00
    mh = float(value)
    if mh >= 0.20:
        return 1.00
    if mh >= -0.10:
        return 0.95
    if mh >= -0.40:
        return 0.85
    return 0.70


def color_factor(value: Any) -> float:
    """Использовать Gaia `BP-RP` как мягкое предпочтение к более холодным звёздам."""
    if pd.isna(value):
        return 1.00
    bp_rp = float(value)
    if bp_rp >= 1.30:
        return 1.00
    if bp_rp >= 0.90:
        return 0.90
    if bp_rp >= 0.50:
        return 0.75
    return 0.60


def normalized_validation_factor(value: Any) -> float:
    """Нормализовать `validation_factor` в диапазон `[0, 1]`."""
    if pd.isna(value):
        return 1.00
    return clip_unit_interval(float(value))


def priority_tier_from_score(score: float) -> str:
    """Преобразовать непрерывный score в operational priority tier."""
    if score >= PRIORITY_TIER_HIGH_THRESHOLD:
        return "HIGH"
    if score >= PRIORITY_TIER_MEDIUM_THRESHOLD:
        return "MEDIUM"
    return "LOW"


def apply_common_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Добавить общие decision-layer факторы для обеих веток.

    Функция ожидает колонки router output и decision metadata, после чего
    добавляет в DataFrame `class_prior`, `quality_factor`,
    `reliability_factor`, `followup_factor`, `metallicity_factor`,
    `color_factor` и нормализованный `validation_factor`.
    """
    result = ensure_decision_columns(df)
    result["class_prior"] = [
        class_prior(spec_class)
        for spec_class in result["predicted_spec_class"]
    ]
    reliability_rows = cast(
        Iterable[tuple[Any, Any]],
        result[
            ["ruwe", "parallax_over_error"]
        ].itertuples(index=False, name=None),
    )
    result["reliability_factor"] = [
        reliability_factor(ruwe, plx_err)
        for ruwe, plx_err in reliability_rows
    ]
    result["quality_factor"] = result["reliability_factor"]
    result["followup_factor"] = [
        followup_factor(parallax) for parallax in result["parallax"]
    ]
    result["metallicity_factor"] = [
        metallicity_factor(value) for value in result["mh_gspphot"]
    ]
    result["color_factor"] = [
        color_factor(value) for value in result["bp_rp"]
    ]
    result["validation_factor"] = [
        normalized_validation_factor(value)
        for value in result["validation_factor"]
    ]
    return result


def host_model_version(host_model: Mapping[str, Any]) -> str:
    """Собрать человекочитаемую строку версии host-модели."""
    meta_raw = host_model.get("meta", {})
    meta = cast(Mapping[str, Any], meta_raw)
    shrink = meta.get("shrink_alpha")
    use_m_subclasses = meta.get("use_m_subclasses")
    model_version = str(meta.get("model_version", HOST_MODEL_VERSION))
    score_mode = str(meta.get("score_mode", "legacy"))
    if shrink is None or use_m_subclasses is None:
        return f"{model_version}_{score_mode}"
    return (
        f"{model_version}_{score_mode}_"
        f"msub_{bool(use_m_subclasses)}_"
        f"shrink_{float(shrink):.2f}"
    )


def run_host_similarity(
    df_host: pd.DataFrame,
    host_model: Mapping[str, Any],
) -> pd.DataFrame:
    """Посчитать host-score только для физически допустимой MKGF dwarf ветки.

    Источник модели
    ---------------
    Использует current production scorer `host_model.score_df_contrastive`,
    а затем применяет общий decision-layer контракт для расчёта
    `final_score`.
    """
    if df_host.empty:
        return df_host.copy()

    scored = score_host_df(
        model=host_model,
        df=df_host,
        spec_class_col="predicted_spec_class",
    )
    scored = apply_common_factors(scored)

    scoring_rows = scored[
        [
            "host_posterior",
            "class_prior",
            "reliability_factor",
            "followup_factor",
            "metallicity_factor",
            "color_factor",
            "validation_factor",
        ]
    ].itertuples(index=False, name=None)
    scored["host_score"] = [
        clip_unit_interval(
            float(host_posterior)
            * float(class_prior_value)
            * float(metallicity_value)
        )
        for (
            host_posterior,
            class_prior_value,
            reliability_value,
            followup_value,
            metallicity_value,
            color_value,
            validation_value,
        ) in scoring_rows
    ]
    scoring_rows = scored[
        [
            "host_score",
            "reliability_factor",
            "followup_factor",
            "color_factor",
            "validation_factor",
        ]
    ].itertuples(index=False, name=None)
    scored["final_score"] = [
        clip_unit_interval(
            float(host_score_value)
            * float(reliability_value)
            * float(followup_value)
            * float(color_value)
            * float(validation_value)
        )
        for (
            host_score_value,
            reliability_value,
            followup_value,
            color_value,
            validation_value,
        ) in scoring_rows
    ]
    scored["d_mahal"] = None
    scored["similarity"] = None
    scored["priority_tier"] = [
        priority_tier_from_score(float(score))
        for score in scored["final_score"]
    ]
    scored["reason_code"] = HOST_SCORING_REASON
    scored["host_model_version"] = host_model_version(host_model)
    return scored


def build_low_priority_stub(df_low: pd.DataFrame) -> pd.DataFrame:
    """Собрать low-priority результат для known non-host объектов.

    Эта ветка предназначена для известных, но нецелевых объектов:
    горячих звёзд, evolved-объектов и других filtered known cases.
    """
    if df_low.empty:
        return df_low.copy()

    result = apply_common_factors(df_low)
    result["gauss_label"] = None
    result["host_log_likelihood"] = None
    result["field_log_likelihood"] = None
    result["host_log_lr"] = None
    result["host_posterior"] = None
    result["d_mahal"] = None
    result["similarity"] = None
    result["final_score"] = 0.0
    result["priority_tier"] = "LOW"
    result["reason_code"] = [
        known_low_reason_code(spec_class, stage)
        for spec_class, stage in result[
            ["predicted_spec_class", "predicted_evolution_stage"]
        ].itertuples(index=False, name=None)
    ]
    result["host_model_version"] = None
    return result


def build_unknown_priority_stub(df_unknown: pd.DataFrame) -> pd.DataFrame:
    """Собрать low-priority результат для canonical `UNKNOWN` ветки."""
    if df_unknown.empty:
        return df_unknown.copy()

    result = apply_common_factors(df_unknown)
    result["gauss_label"] = None
    result["host_log_likelihood"] = None
    result["field_log_likelihood"] = None
    result["host_log_lr"] = None
    result["host_posterior"] = None
    result["d_mahal"] = None
    result["similarity"] = None
    result["final_score"] = 0.0
    result["priority_tier"] = "LOW"
    result["reason_code"] = ROUTER_UNKNOWN_REASON
    result["host_model_version"] = None
    return result


def order_priority_results(df: pd.DataFrame) -> pd.DataFrame:
    """Отсортировать итоговые результаты в operational ranking order."""
    return df.sort_values(
        by=["final_score", "router_similarity"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)


__all__ = [
    "apply_common_factors",
    "build_low_priority_stub",
    "build_unknown_priority_stub",
    "class_prior",
    "clip_unit_interval",
    "color_factor",
    "distance_factor",
    "followup_factor",
    "host_model_version",
    "metallicity_factor",
    "normalized_validation_factor",
    "order_priority_results",
    "parallax_precision_factor",
    "priority_tier_from_score",
    "quality_factor",
    "reliability_factor",
    "ruwe_factor",
    "run_host_similarity",
]
