# Точка входа `streamlit_app.py` интерфейсного слоя.
#
# Этот файл отвечает только за:
# - общий каркас multipage-приложения Streamlit;
# - маршрутизацию страниц и инициализацию общего состояния.
#
# Следующий слой:
# - сами страницы интерфейса и их helper-модули;
# - smoke-проверка entrypoint после сборки страниц.

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Direct `streamlit run streamlit_app.py` does not add `src` to Python path automatically.
# Поднимаем путь проекта здесь, чтобы entrypoint работал из корня репозитория без ручного `PYTHONPATH`.
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

APP_TITLE = "Экзопланетный shortlist"
APP_ICON = "🔭"


def build_navigation():
    # Навигацию держим в одном месте, чтобы страницы не размазывали маршрутную логику.
    from exohost.ui.pages.candidate_page import render_candidate_page
    from exohost.ui.pages.csv_decide_page import render_csv_decide_page
    from exohost.ui.pages.home_page import render_home_page
    from exohost.ui.pages.metrics_page import render_metrics_page
    from exohost.ui.pages.run_browser_page import render_run_browser_page

    return st.navigation(
        pages=[
            st.Page(
                render_home_page,
                title="Главная",
                icon="🏠",
                url_path="home",
                default=True,
            ),
            st.Page(
                render_metrics_page,
                title="Метрики",
                icon="📊",
                url_path="metrics",
            ),
            st.Page(
                render_run_browser_page,
                title="Запуск",
                icon="📁",
                url_path="run-browser",
            ),
            st.Page(
                render_candidate_page,
                title="Объект",
                icon="⭐",
                url_path="candidate",
            ),
            st.Page(
                render_csv_decide_page,
                title="CSV-запуск",
                icon="🚀",
                url_path="csv-decide",
            ),
        ],
        position="sidebar",
    )


def main() -> None:
    # Entry point только поднимает каркас приложения и передает выполнение выбранной странице.
    from exohost.ui.session_state import initialize_ui_session_state

    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    initialize_ui_session_state(st.session_state)
    current_page = build_navigation()
    current_page.run()


if __name__ == "__main__":
    main()
