# Файл `system_overview.py` слоя `ui/components`.
#
# Этот файл отвечает только за:
# - визуальный вывод схемы системы на домашней странице;
# - аккуратный fallback, если локальный SVG недоступен.
#
# Следующий слой:
# - page-модуль `home_page`;
# - helper поиска пути к диаграмме.

from __future__ import annotations

from pathlib import Path

import streamlit as st


def render_system_overview(diagram_path: Path | None) -> None:
    # Диаграмма нужна как быстрый визуальный контекст: от источников данных до shortlist.
    st.subheader("Схема системы")
    if diagram_path is None:
        st.info("Не удалось найти локальный SVG со схемой системы.")
        return

    st.image(
        diagram_path,
        width="stretch",
    )
    st.caption(
        "Схема показывает путь от источников данных и model layers к итоговому shortlist "
        "для follow-up наблюдений."
    )


__all__ = ["render_system_overview"]
