# Файл `candidate_card.py` слоя `ui/components`.
#
# Этот файл отвечает только за:
# - визуальный вывод карточки одной звезды;
# - отдельные таблицы по маршруту pipeline и физическим параметрам объекта.
#
# Следующий слой:
# - страница карточки объекта;
# - helper-модуль подготовки summary и physical preview.

from __future__ import annotations

import pandas as pd
import streamlit as st

from exohost.ui.candidate_overview import UiCandidateOverview


def render_candidate_card(
    overview: UiCandidateOverview,
    route_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    physics_df: pd.DataFrame,
) -> None:
    # Карточка должна показывать одну звезду без перегруза и без ручного чтения сырых CSV.
    if summary_df.empty:
        st.warning("Не удалось найти выбранный `source_id` в текущем запуске.")
        return

    _render_candidate_overview(overview)

    content_columns = st.columns((1.05, 0.95), gap="medium")
    with content_columns[0]:
        st.subheader("Маршрут пайплайна")
        st.dataframe(
            route_df.rename(columns=_ROUTE_LABELS),
            width="stretch",
            hide_index=True,
        )
    with content_columns[1]:
        st.subheader("Ключевые физические параметры")
        _render_physical_snapshot(overview)

    st.subheader("Подробные таблицы")
    st.markdown("**Итоговое решение**")
    st.dataframe(
        summary_df.rename(columns=_SUMMARY_LABELS),
        width="stretch",
        hide_index=True,
    )

    st.markdown("**Физические параметры**")
    if physics_df.empty:
        st.info("Для выбранного объекта нет физического предпросмотра в текущем запуске.")
        return

    st.dataframe(
        physics_df.rename(columns=_PHYSICS_LABELS),
        width="stretch",
        hide_index=True,
    )


def _render_candidate_overview(overview: UiCandidateOverview) -> None:
    overview_row = st.columns(4, gap="small")
    with overview_row[0]:
        st.metric("source_id", overview.source_id, border=True)
    with overview_row[1]:
        st.metric(
            "Итоговый домен",
            overview.final_domain_state or "n/a",
            border=True,
        )
    with overview_row[2]:
        st.metric(
            "Итоговый класс",
            overview.final_refinement_label or overview.final_coarse_class or "n/a",
            border=True,
        )
    with overview_row[3]:
        st.metric("Приоритет", overview.priority_label or "n/a", border=True)

    secondary_row = st.columns(4, gap="small")
    with secondary_row[0]:
        st.metric(
            "Фильтр качества",
            overview.final_quality_state or "n/a",
            border=True,
        )
    with secondary_row[1]:
        st.metric(
            "Уточнение класса",
            overview.final_refinement_state or "n/a",
            border=True,
        )
    with secondary_row[2]:
        st.metric(
            "Сходство со звездами-хозяевами",
            _format_metric_value(overview.host_similarity_score),
            border=True,
        )
    with secondary_row[3]:
        st.metric(
            "Наблюдаемость",
            _format_metric_value(overview.observability_score),
            border=True,
        )

    st.caption(overview.overview_note)


def _render_physical_snapshot(overview: UiCandidateOverview) -> None:
    upper_row = st.columns(2, gap="small")
    with upper_row[0]:
        st.metric("Температура", _format_metric_value(overview.teff_gspphot), border=True)
    with upper_row[1]:
        st.metric("logg", _format_metric_value(overview.logg_gspphot), border=True)

    middle_row = st.columns(2, gap="small")
    with middle_row[0]:
        st.metric("BP-RP", _format_metric_value(overview.bp_rp), border=True)
    with middle_row[1]:
        st.metric(
            "Звездная величина G",
            _format_metric_value(overview.phot_g_mean_mag),
            border=True,
        )

    lower_row = st.columns(2, gap="small")
    with lower_row[0]:
        st.metric(
            "Параллакс",
            _format_metric_value(overview.parallax),
            border=True,
        )
    with lower_row[1]:
        st.metric("RUWE", _format_metric_value(overview.ruwe), border=True)

    st.caption(
        "Спектральный класс: "
        f"`{overview.spec_class or 'n/a'}` / `{overview.spec_subclass or 'n/a'}`. "
        "Стадия эволюции: "
        f"`{overview.evolution_stage or 'n/a'}`. "
        "FLAME: радиус "
        f"`{_format_metric_value(overview.radius_flame)}`, "
        "светимость "
        f"`{_format_metric_value(overview.lum_flame)}`."
    )


def _format_metric_value(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


_SUMMARY_LABELS = {
    "source_id": "source_id",
    "final_domain_state": "Итоговое состояние",
    "final_quality_state": "Состояние качества",
    "final_coarse_class": "Итоговый крупный класс",
    "final_refinement_label": "Итоговый подкласс",
    "final_refinement_state": "Состояние подкласса",
    "final_decision_reason": "Причина итогового решения",
    "quality_reason": "Причина фильтра качества",
    "review_bucket": "Корзина проверки",
    "priority_label": "Приоритет",
    "priority_score": "Итоговый приоритет",
    "priority_reason": "Причина приоритета",
    "host_similarity_score": "Сходство со звездами-хозяевами",
    "observability_score": "Наблюдаемость",
}

_PHYSICS_LABELS = {
    "source_id": "source_id",
    "spec_class": "Класс",
    "spec_subclass": "Подкласс",
    "evolution_stage": "Стадия эволюции",
    "teff_gspphot": "Температура GSP-Phot",
    "logg_gspphot": "logg GSP-Phot",
    "mh_gspphot": "Металличность GSP-Phot",
    "bp_rp": "BP-RP",
    "parallax": "Параллакс",
    "parallax_over_error": "Параллакс / ошибка",
    "ruwe": "RUWE",
    "phot_g_mean_mag": "Звездная величина G",
    "radius_flame": "Радиус FLAME",
    "lum_flame": "Светимость FLAME",
    "evolstage_flame": "Стадия эволюции FLAME",
}

_ROUTE_LABELS = {
    "stage_name": "Этап пайплайна",
    "stage_state": "Состояние",
    "stage_reason": "Причина / результат",
}


__all__ = ["render_candidate_card"]
