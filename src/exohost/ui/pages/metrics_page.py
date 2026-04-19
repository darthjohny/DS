# Файл `metrics_page.py` слоя `ui/pages`.
#
# Этот файл отвечает только за:
# - страницу метрик качества моделей;
# - краткий слой доверия к итоговому прикладному результату.
#
# Следующий слой:
# - отдельные компоненты вывода метрик;
# - loader-слой model-artifacts и run summary.

from __future__ import annotations

import streamlit as st

from exohost.ui.components.model_metrics import render_model_metrics_table
from exohost.ui.components.model_metrics_overview import render_model_metrics_overview
from exohost.ui.model_metrics import load_benchmark_stage_overview
from exohost.ui.model_metrics_summary import (
    build_ui_metric_stage_assessment_frame,
    build_ui_model_metrics_overview,
)


def render_metrics_page() -> None:
    # Страница метрик должна показать силу и ограничения слоев без обращения к notebook.
    st.title("Качество моделей")
    st.caption(
        "Эта страница читает benchmark-артефакты и показывает краткую сводку по "
        "рабочим слоям моделей."
    )
    metrics_df = load_benchmark_stage_overview()
    assessment_df = build_ui_metric_stage_assessment_frame(metrics_df)
    render_model_metrics_overview(
        build_ui_model_metrics_overview(assessment_df),
        assessment_df,
    )
    render_model_metrics_table(metrics_df)
