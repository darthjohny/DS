# Файл `model_metrics.py` слоя `ui/components`.
#
# Этот файл отвечает только за:
# - визуальный вывод benchmark-метрик по слоям моделей;
# - компактную таблицу качества без notebook и без ручного чтения CSV.
#
# Следующий слой:
# - страница метрик интерфейса;
# - helper-модуль подготовки benchmark summary.

from __future__ import annotations

import pandas as pd
import streamlit as st


def render_model_metrics_table(metrics_df: pd.DataFrame) -> None:
    # Если benchmark-артефакты не найдены, страница должна честно сказать об этом, а не падать.
    if metrics_df.empty:
        st.warning("Не удалось найти контрольные артефакты качества для страницы метрик.")
        return

    st.subheader("Подробная контрольная сводка")
    display_df = metrics_df.loc[
        :,
        [
            "stage_name",
            "test_accuracy",
            "test_balanced_accuracy",
            "test_macro_f1",
            "test_roc_auc_ovr",
            "cv_mean_accuracy",
            "cv_mean_balanced_accuracy",
            "cv_mean_macro_f1",
            "n_rows_test",
            "benchmark_run_dir",
        ],
    ].copy()
    display_df = display_df.rename(
        columns={
            "stage_name": "Слой",
            "test_accuracy": "Accuracy",
            "test_balanced_accuracy": "Balanced accuracy",
            "test_macro_f1": "Macro F1",
            "test_roc_auc_ovr": "ROC AUC OvR",
            "cv_mean_accuracy": "CV accuracy",
            "cv_mean_balanced_accuracy": "CV balanced accuracy",
            "cv_mean_macro_f1": "CV Macro F1",
            "n_rows_test": "Размер теста",
            "benchmark_run_dir": "Контрольный запуск",
        }
    )
    st.dataframe(display_df, width="stretch", hide_index=True)

    with st.expander("Пояснения по слоям", expanded=False):
        for _, row in metrics_df.iterrows():
            stage_name = str(row["stage_name"])
            note = str(row["note"])
            st.markdown(f"**{stage_name}.** {note}")
