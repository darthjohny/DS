# Файл `home_page.py` слоя `ui/pages`.
#
# Этот файл отвечает только за:
# - главную страницу интерфейса;
# - краткую витрину проекта и переход к следующим страницам.
#
# Следующий слой:
# - более предметные страницы метрик, запуска и просмотра run_dir;
# - компоненты summary-слоя.

from __future__ import annotations

import streamlit as st

from exohost.ui.components.home_result import render_home_main_result
from exohost.ui.components.overview_metrics import render_run_overview_metrics
from exohost.ui.components.system_overview import render_system_overview
from exohost.ui.home_summary import (
    build_ui_home_main_result,
    build_ui_home_top_candidates_preview,
    resolve_ui_system_overview_path,
)
from exohost.ui.loaders import list_available_run_dirs, load_ui_run_bundle
from exohost.ui.run_overview import build_ui_run_overview


def render_home_page() -> None:
    # Главная страница должна быстро объяснить, что делает интерфейс и куда идти дальше.
    st.title("Интерфейс проекта")
    st.caption(
        "Тонкая витрина поверх готового прикладного контура: артефакты, метрики, "
        "кандидаты и внешний запуск `decide`."
    )
    st.info(
        "Интерфейс показывает готовые прогоны, качество моделей и итоговый "
        "список кандидатов без открытия исследовательских ноутбуков."
    )
    st.markdown(
        "Главный прикладной результат проекта — воспроизводимый короткий список целей "
        "для последующих наблюдений, а не объявление обнаружения экзопланет."
    )

    available_run_dirs = list_available_run_dirs()
    if not available_run_dirs:
        st.warning("Не удалось найти ни одного готового `run_dir` в `artifacts/decisions`.")
        return

    try:
        latest_run_bundle = load_ui_run_bundle(str(available_run_dirs[0]))
    except RuntimeError as exc:
        st.error(f"Не удалось загрузить последний `run_dir`: {exc}")
        return

    render_run_overview_metrics(build_ui_run_overview(latest_run_bundle))

    content_columns = st.columns([1.1, 0.9], gap="large")
    with content_columns[0]:
        render_home_main_result(
            build_ui_home_main_result(latest_run_bundle),
            build_ui_home_top_candidates_preview(latest_run_bundle),
        )

    with content_columns[1]:
        render_system_overview(resolve_ui_system_overview_path())

    st.subheader("Навигация по витрине")
    st.markdown(
        "- Страница `Метрики` показывает контрольную сводку по основным слоям моделей.\n"
        "- Страница `Запуск` открывает готовый `run_dir` и итоговые распределения.\n"
        "- Страница `Объект` показывает маршрут по пайплайну и физические параметры одной звезды.\n"
        "- Страница `CSV-запуск` принимает внешний файл и запускает существующий `decide` без CLI."
    )
