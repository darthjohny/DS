# Файл `candidate_overview.py` слоя `ui`.
#
# Этот файл отвечает только за:
# - компактную summary-модель карточки объекта;
# - сборку route-таблицы и верхней overview-сводки по одной звезде.
#
# Следующий слой:
# - визуальный компонент карточки объекта;
# - unit-тесты helper-слоя candidate overview.

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import pandas as pd


@dataclass(frozen=True, slots=True)
class UiCandidateOverview:
    # Компактная one-object summary для верхнего блока карточки.
    source_id: str
    final_domain_state: str | None
    final_quality_state: str | None
    final_coarse_class: str | None
    final_refinement_label: str | None
    final_refinement_state: str | None
    review_bucket: str | None
    priority_label: str | None
    priority_score: float | None
    host_similarity_score: float | None
    observability_score: float | None
    spec_class: str | None
    spec_subclass: str | None
    evolution_stage: str | None
    teff_gspphot: float | None
    logg_gspphot: float | None
    bp_rp: float | None
    phot_g_mean_mag: float | None
    parallax: float | None
    ruwe: float | None
    radius_flame: float | None
    lum_flame: float | None
    final_decision_reason: str | None
    quality_reason: str | None
    priority_reason: str | None
    overview_note: str


def build_ui_candidate_overview(
    summary_df: pd.DataFrame,
    physics_df: pd.DataFrame,
) -> UiCandidateOverview | None:
    # Верхний overview собираем из одной строки summary и при наличии physics preview.
    if summary_df.empty:
        return None

    summary_row = summary_df.iloc[0]
    physics_row = physics_df.iloc[0] if not physics_df.empty else pd.Series(dtype=object)

    source_id = _to_required_string(summary_row.get("source_id"))
    final_domain_state = _to_optional_string(summary_row.get("final_domain_state"))
    final_quality_state = _to_optional_string(summary_row.get("final_quality_state"))
    final_coarse_class = _to_optional_string(summary_row.get("final_coarse_class"))
    final_refinement_label = _to_optional_string(
        summary_row.get("final_refinement_label")
    )
    final_refinement_state = _to_optional_string(
        summary_row.get("final_refinement_state")
    )
    review_bucket = _to_optional_string(summary_row.get("review_bucket"))
    priority_label = _to_optional_string(summary_row.get("priority_label"))
    priority_score = _to_optional_float(summary_row.get("priority_score"))
    host_similarity_score = _to_optional_float(
        summary_row.get("host_similarity_score")
    )
    observability_score = _to_optional_float(summary_row.get("observability_score"))
    spec_class = _to_optional_string(physics_row.get("spec_class"))
    spec_subclass = _to_optional_string(physics_row.get("spec_subclass"))
    evolution_stage = _to_optional_string(physics_row.get("evolution_stage"))
    teff_gspphot = _to_optional_float(physics_row.get("teff_gspphot"))
    logg_gspphot = _to_optional_float(physics_row.get("logg_gspphot"))
    bp_rp = _to_optional_float(physics_row.get("bp_rp"))
    phot_g_mean_mag = _to_optional_float(physics_row.get("phot_g_mean_mag"))
    parallax = _to_optional_float(physics_row.get("parallax"))
    ruwe = _to_optional_float(physics_row.get("ruwe"))
    radius_flame = _to_optional_float(physics_row.get("radius_flame"))
    lum_flame = _to_optional_float(physics_row.get("lum_flame"))
    final_decision_reason = _to_optional_string(
        summary_row.get("final_decision_reason")
    )
    quality_reason = _to_optional_string(summary_row.get("quality_reason"))
    priority_reason = _to_optional_string(summary_row.get("priority_reason"))

    return UiCandidateOverview(
        source_id=source_id,
        final_domain_state=final_domain_state,
        final_quality_state=final_quality_state,
        final_coarse_class=final_coarse_class,
        final_refinement_label=final_refinement_label,
        final_refinement_state=final_refinement_state,
        review_bucket=review_bucket,
        priority_label=priority_label,
        priority_score=priority_score,
        host_similarity_score=host_similarity_score,
        observability_score=observability_score,
        spec_class=spec_class,
        spec_subclass=spec_subclass,
        evolution_stage=evolution_stage,
        teff_gspphot=teff_gspphot,
        logg_gspphot=logg_gspphot,
        bp_rp=bp_rp,
        phot_g_mean_mag=phot_g_mean_mag,
        parallax=parallax,
        ruwe=ruwe,
        radius_flame=radius_flame,
        lum_flame=lum_flame,
        final_decision_reason=final_decision_reason,
        quality_reason=quality_reason,
        priority_reason=priority_reason,
        overview_note=_build_overview_note(
            source_id=source_id,
            final_domain_state=final_domain_state,
            final_refinement_label=final_refinement_label,
            final_coarse_class=final_coarse_class,
            final_quality_state=final_quality_state,
            priority_label=priority_label,
            final_decision_reason=final_decision_reason,
        ),
    )


def build_ui_candidate_route_frame(summary_df: pd.DataFrame) -> pd.DataFrame:
    # Route-таблица нужна как короткая расшифровка шагов pipeline без чтения raw CSV.
    if summary_df.empty:
        return pd.DataFrame(
            columns=[
                "stage_name",
                "stage_state",
                "stage_reason",
            ]
        )

    summary_row = summary_df.iloc[0]
    route_rows = [
        {
            "stage_name": "Фильтр качества",
            "stage_state": _to_optional_string(summary_row.get("final_quality_state")),
            "stage_reason": _to_optional_string(summary_row.get("quality_reason")),
        },
        {
            "stage_name": "Уточнение класса",
            "stage_state": _to_optional_string(
                summary_row.get("final_refinement_state")
            ),
            "stage_reason": _to_optional_string(
                summary_row.get("final_refinement_label")
            ),
        },
        {
            "stage_name": "Итоговая маршрутизация",
            "stage_state": _to_optional_string(summary_row.get("final_domain_state")),
            "stage_reason": _to_optional_string(
                summary_row.get("final_decision_reason")
            ),
        },
        {
            "stage_name": "Ранжирование приоритета",
            "stage_state": _to_optional_string(summary_row.get("priority_label")),
            "stage_reason": _to_optional_string(summary_row.get("priority_reason")),
        },
    ]
    return pd.DataFrame(route_rows)


def _build_overview_note(
    *,
    source_id: str,
    final_domain_state: str | None,
    final_refinement_label: str | None,
    final_coarse_class: str | None,
    final_quality_state: str | None,
    priority_label: str | None,
    final_decision_reason: str | None,
) -> str:
    final_label = final_refinement_label or final_coarse_class or "n/a"
    message = (
        f"Объект `{source_id}` завершил маршрут со статусом "
        f"`{final_domain_state or 'n/a'}` и итоговым классом `{final_label}`. "
        f"Фильтр качества: `{final_quality_state or 'n/a'}`."
    )
    if priority_label is not None:
        message += f" Приоритет: `{priority_label}`."
    else:
        message += " Для объекта не рассчитано ранжирование приоритета."
    if final_decision_reason is not None:
        message += f" Причина итогового решения: `{final_decision_reason}`."
    return message


def _to_required_string(value: object) -> str:
    optional_value = _to_optional_string(value)
    if optional_value is None:
        return "n/a"
    return optional_value


def _to_optional_string(value: object) -> str | None:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, str):
        stripped_value = value.strip()
        return stripped_value or None
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, Real):
        if pd.isna(value):
            return None
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)
    return None


def _to_optional_float(value: object) -> float | None:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, Real):
        if pd.isna(value):
            return None
        return float(value)
    return None


__all__ = [
    "UiCandidateOverview",
    "build_ui_candidate_overview",
    "build_ui_candidate_route_frame",
]
