# Файл `home_result.py` слоя `ui/components`.
#
# Этот файл отвечает только за:
# - визуальный блок главного прикладного результата на домашней странице;
# - краткий preview верхнего shortlist без логики загрузки данных.
#
# Следующий слой:
# - page-модуль `home_page`;
# - helper-модуль `home_summary`.

from __future__ import annotations

import pandas as pd
import streamlit as st

from exohost.ui.home_summary import UiHomeMainResult


def render_home_main_result(
    main_result: UiHomeMainResult,
    top_candidates_df: pd.DataFrame,
) -> None:
    # Домашняя страница должна быстро отвечать на прикладной вопрос: что именно получилось на latest run.
    st.subheader("Главный прикладной результат")
    st.success(
        "Система воспроизводимо формирует shortlist целей для последующих наблюдений. "
        f"В latest run высокий приоритет получили `{_format_int(main_result.high_priority_count)}` "
        "объектов."
    )
    st.caption(
        f"Источник блока: `{main_result.run_dir_name}` от `{main_result.created_at_utc}`."
    )

    metric_columns = st.columns(3)
    metric_columns[0].metric(
        "Ранжировано",
        _format_int(main_result.ranked_count),
        border=True,
    )
    metric_columns[1].metric(
        "Высокий приоритет",
        _format_int(main_result.high_priority_count),
        border=True,
    )
    metric_columns[2].metric(
        "Доля high среди ranked",
        _format_share(main_result.high_priority_share),
        border=True,
    )

    st.markdown(
        "Текущий latest run прошел через тот же пайплайн final decision и priority, "
        "поэтому home-screen показывает не рекламный текст, а реальный итог рабочего контура."
    )

    st.subheader("Preview верхнего shortlist")
    if top_candidates_df.empty:
        st.info("В latest run нет доступных строк для preview верхнего shortlist.")
        return

    st.dataframe(
        top_candidates_df.rename(columns=_HOME_TOP_CANDIDATE_LABELS),
        width="stretch",
        hide_index=True,
    )


def _format_int(value: int) -> str:
    return f"{value:,}".replace(",", " ")


def _format_share(value: float) -> str:
    return f"{value * 100.0:.1f}%"


_HOME_TOP_CANDIDATE_LABELS = {
    "source_id": "source_id",
    "spec_class": "Класс",
    "spec_subclass": "Подкласс",
    "priority_label": "Приоритет",
    "priority_score": "Итоговый приоритет",
    "host_similarity_score": "Сходство с host",
}


__all__ = ["render_home_main_result"]
