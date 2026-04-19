# Файл `overview_metrics.py` слоя `ui/components`.
#
# Этот файл отвечает только за:
# - блок краткой сводки по одному рабочему прогону;
# - визуальный вывод главных чисел на домашней странице интерфейса.
#
# Следующий слой:
# - домашняя страница и соседние компоненты интерфейса;
# - helper-модули сводок и загрузки артефактов.

from __future__ import annotations

import streamlit as st

from exohost.ui.run_overview import UiRunOverview


def render_run_overview_metrics(overview: UiRunOverview) -> None:
    # Показываем только главные числа, чтобы домашняя страница не превращалась в отчетный notebook.
    first_row = st.columns(4)
    first_row[0].metric("Всего объектов", f"{overview.n_rows_final_decision:,}".replace(",", " "))
    first_row[1].metric("ID", f"{overview.id_count:,}".replace(",", " "))
    first_row[2].metric(
        "Высокий приоритет",
        f"{overview.high_priority_count:,}".replace(",", " "),
    )
    first_row[3].metric("OOD", f"{overview.ood_count:,}".replace(",", " "))

    second_row = st.columns(3)
    second_row[0].metric("Проверить", f"{overview.unknown_count:,}".replace(",", " "))
    second_row[1].metric("Средний приоритет", f"{overview.medium_priority_count:,}".replace(",", " "))
    second_row[2].metric("Низкий приоритет", f"{overview.low_priority_count:,}".replace(",", " "))

    st.caption(
        f"Прогон: `{overview.run_dir_name}`. Создан: `{overview.created_at_utc}`. "
        f"Контур: `{overview.pipeline_name}`."
    )
