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

from typing import Any, cast

import pytest

from .streamlit_testkit import (
    run_streamlit_entrypoint_smoke,
    run_streamlit_page_smoke,
)


def test_streamlit_entrypoint_smoke_renders_default_page() -> None:
    app_test = run_streamlit_entrypoint_smoke()

    assert not app_test.exception
    assert app_test.title[0].value == "Интерфейс проекта"


def test_streamlit_entrypoint_navigation_declares_expected_pages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import streamlit_app

    captured_navigation: dict[str, object] = {}
    fake_navigation = object()

    def fake_page(
        page: object,
        *,
        title: str,
        icon: str,
        url_path: str,
        default: bool = False,
    ) -> dict[str, object]:
        return {
            "page": page,
            "title": title,
            "icon": icon,
            "url_path": url_path,
            "default": default,
        }

    def fake_navigation_factory(
        *,
        pages: list[dict[str, object]],
        position: str,
    ) -> object:
        captured_navigation["pages"] = pages
        captured_navigation["position"] = position
        return fake_navigation

    monkeypatch.setattr(streamlit_app.st, "Page", fake_page)
    monkeypatch.setattr(streamlit_app.st, "navigation", fake_navigation_factory)

    navigation = streamlit_app.build_navigation()

    assert navigation is fake_navigation
    assert captured_navigation["position"] == "sidebar"
    pages = cast(list[dict[str, Any]], captured_navigation["pages"])
    assert [
        (page["title"], page["icon"], page["url_path"], page["default"])
        for page in pages
    ] == [
        ("Главная", "🏠", "home", True),
        ("Метрики", "📊", "metrics", False),
        ("Запуск", "📁", "run-browser", False),
        ("Объект", "⭐", "candidate", False),
        ("CSV-запуск", "🚀", "csv-decide", False),
    ]
    assert all(callable(page["page"]) for page in pages)


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
    assert "Маршрут пайплайна" in [item.value for item in app_test.subheader]
    assert "Подробные таблицы" in [item.value for item in app_test.subheader]


def test_csv_decide_page_smoke_renders_upload_flow_shell() -> None:
    app_test = run_streamlit_page_smoke(
        page_module="exohost.ui.pages.csv_decide_page",
        page_function="render_csv_decide_page",
        setup_code="""
        from pathlib import Path

        from exohost.ui.run_service import UiCsvDecideDefaults
        from tests.unit.ui.ui_testkit import build_ui_loaded_run_bundle

        page_module_under_test.list_available_run_dirs = lambda: (
            Path("artifacts/decisions/hierarchical_final_decision_demo"),
        )
        page_module_under_test.load_ui_run_bundle = (
            lambda selected_run_dir: build_ui_loaded_run_bundle()
        )
        page_module_under_test.build_ui_csv_decide_defaults = lambda bundle: UiCsvDecideDefaults(
            ood_model_run_dir="artifacts/models/ood",
            ood_threshold_run_dir="artifacts/thresholds/ood",
            coarse_model_run_dir="artifacts/models/coarse",
            refinement_model_run_dirs=("artifacts/models/refinement_g",),
            host_model_run_dir="artifacts/models/host",
            decision_policy_version="final_decision_v2",
            candidate_ood_disposition="keep",
            host_score_column="host_similarity_score",
            min_refinement_confidence=None,
            min_coarse_probability=0.60,
            min_coarse_margin=None,
            quality_ruwe_unknown_threshold=1.4,
            quality_parallax_snr_unknown_threshold=10.0,
            quality_require_flame_for_pass=False,
            priority_high_min=0.85,
            priority_medium_min=0.55,
            output_dir="artifacts/decisions",
            dotenv_path=".env",
            connect_timeout=10,
        )
        """,
    )

    assert not app_test.exception
    assert app_test.title[0].value == "Запуск по внешнему CSV"
    assert len(app_test.selectbox) >= 1
    assert len(app_test.expander) >= 1
    assert len(app_test.file_uploader) == 1
