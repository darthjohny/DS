# Тестовый файл `test_streamlit_pages_smoke.py` домена `smoke/ui`.
#
# Этот файл проверяет только:
# - smoke-рендер multipage entrypoint и ключевых страниц Streamlit-интерфейса;
# - базовую доступность главных виджетов и отсутствие необработанных исключений.
#
# Следующий слой:
# - более узкие сценарные smoke-проверки фильтров, CSV-запуска и навигации;
# - при необходимости app-level регрессии UI.

from __future__ import annotations

from .streamlit_testkit import (
    run_streamlit_entrypoint_smoke,
    run_streamlit_page_smoke,
)


def test_streamlit_entrypoint_smoke_renders_default_page() -> None:
    app_test = run_streamlit_entrypoint_smoke()

    assert not app_test.exception
    assert app_test.title[0].value == "Интерфейс проекта"


def test_home_page_smoke_renders_main_sections() -> None:
    app_test = run_streamlit_page_smoke(
        page_module="exohost.ui.pages.home_page",
        page_function="render_home_page",
    )

    assert not app_test.exception
    assert app_test.title[0].value == "Интерфейс проекта"
    assert "Главный прикладной результат" in [item.value for item in app_test.subheader]
    assert "Схема системы" in [item.value for item in app_test.subheader]


def test_metrics_page_smoke_renders_quality_summary() -> None:
    app_test = run_streamlit_page_smoke(
        page_module="exohost.ui.pages.metrics_page",
        page_function="render_metrics_page",
    )

    assert not app_test.exception
    assert app_test.title[0].value == "Качество моделей"
    assert len(app_test.dataframe) >= 1


def test_run_browser_page_smoke_renders_filters_and_preview() -> None:
    app_test = run_streamlit_page_smoke(
        page_module="exohost.ui.pages.run_browser_page",
        page_function="render_run_browser_page",
    )

    assert not app_test.exception
    assert app_test.title[0].value == "Просмотр запуска"
    assert len(app_test.selectbox) >= 2
    assert len(app_test.multiselect) >= 3
    assert len(app_test.text_input) >= 1
    assert len(app_test.number_input) >= 1
    assert len(app_test.expander) >= 1
    assert len(app_test.dataframe) >= 3


def test_candidate_page_smoke_renders_selection_controls() -> None:
    app_test = run_streamlit_page_smoke(
        page_module="exohost.ui.pages.candidate_page",
        page_function="render_candidate_page",
    )

    assert not app_test.exception
    assert app_test.title[0].value == "Карточка объекта"
    assert len(app_test.selectbox) >= 2
    assert "Маршрут pipeline" in [item.value for item in app_test.subheader]
    assert "Подробные таблицы" in [item.value for item in app_test.subheader]


def test_csv_decide_page_smoke_renders_upload_flow_shell() -> None:
    app_test = run_streamlit_page_smoke(
        page_module="exohost.ui.pages.csv_decide_page",
        page_function="render_csv_decide_page",
    )

    assert not app_test.exception
    assert app_test.title[0].value == "Запуск по внешнему CSV"
    assert len(app_test.selectbox) >= 1
    assert len(app_test.expander) >= 1
