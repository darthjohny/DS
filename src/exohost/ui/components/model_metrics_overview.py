# Файл `model_metrics_overview.py` слоя `ui/components`.
#
# Этот файл отвечает только за:
# - верхний trust-summary блок страницы качества моделей;
# - короткие stage-cards по benchmark-слоям без логики классификации.
#
# Следующий слой:
# - page-модуль `metrics_page`;
# - helper-слой `model_metrics_summary`.

from __future__ import annotations

import pandas as pd
import streamlit as st

from exohost.ui.model_metrics_summary import UiModelMetricsOverview


def render_model_metrics_overview(
    overview: UiModelMetricsOverview,
    assessment_df: pd.DataFrame,
) -> None:
    # Overview-блок нужен как быстрый слой доверия перед подробной таблицей benchmark-метрик.
    if assessment_df.empty:
        st.warning(overview.overview_message)
        return

    st.subheader("Короткий вывод")
    st.info(overview.overview_message)

    summary_columns = st.columns(4)
    summary_columns[0].metric("Сильные слои", str(overview.n_strong_stages), border=True)
    summary_columns[1].metric("Стабильные слои", str(overview.n_stable_stages), border=True)
    summary_columns[2].metric(
        "Нужна осторожность",
        str(overview.n_caution_stages),
        border=True,
    )
    summary_columns[3].metric(
        "Без данных",
        str(overview.n_missing_stages),
        border=True,
    )

    stage_columns = st.columns(2, gap="large")
    for index, (_, row) in enumerate(assessment_df.iterrows()):
        with stage_columns[index % 2]:
            st.markdown(f"**{row['stage_name']}**")
            st.caption(
                f"{row['trust_label']}. "
                f"Macro F1: {_format_metric(row['test_macro_f1'])}. "
                f"Balanced accuracy: {_format_metric(row['test_balanced_accuracy'])}."
            )
            st.markdown(str(row["trust_summary"]))
            if row["benchmark_run_dir"] is not None:
                st.caption(f"Контрольный запуск: `{row['benchmark_run_dir']}`")


def _format_metric(value: object) -> str:
    if value is None or value is pd.NA:
        return "n/a"
    if isinstance(value, bool):
        return f"{float(value):.3f}"
    if isinstance(value, (int, float)):
        if pd.isna(value):
            return "n/a"
        return f"{float(value):.3f}"
    return "n/a"


__all__ = ["render_model_metrics_overview"]
