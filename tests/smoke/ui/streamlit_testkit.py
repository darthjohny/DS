# Тестовый helper `streamlit_testkit.py` домена `smoke/ui`.
#
# Этот файл хранит только:
# - обертки вокруг официального `streamlit.testing.v1.AppTest`;
# - единый способ запускать multipage entrypoint и отдельные page-функции в pytest.
#
# Следующий слой:
# - smoke-тесты страниц интерфейса;
# - более узкие сценарные проверки отдельных виджетов.

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STREAMLIT_SMOKE_TIMEOUT = 15.0


def run_streamlit_entrypoint_smoke(
    *,
    timeout: float = DEFAULT_STREAMLIT_SMOKE_TIMEOUT,
) -> AppTest:
    # Entry point поднимаем как обычный multipage-app, но в headless AppTest-контуре.
    script = dedent(
        f"""
        from pathlib import Path
        import os
        import sys

        project_root = Path({str(PROJECT_ROOT)!r})
        os.chdir(project_root)
        sys.path.insert(0, str(project_root))
        sys.path.insert(0, str(project_root / "src"))

        import streamlit_app

        streamlit_app.main()
        """
    )
    return _run_streamlit_script(script, timeout=timeout)


def run_streamlit_page_smoke(
    *,
    page_module: str,
    page_function: str,
    setup_code: str = "",
    timeout: float = DEFAULT_STREAMLIT_SMOKE_TIMEOUT,
) -> AppTest:
    # Отдельные страницы запускаем в изолированном AppTest-контуре без переключения multipage navigation.
    setup_block = dedent(setup_code).strip()
    script = dedent(
        f"""
        from pathlib import Path
        import os
        import sys

        project_root = Path({str(PROJECT_ROOT)!r})
        os.chdir(project_root)
        sys.path.insert(0, str(project_root))
        sys.path.insert(0, str(project_root / "src"))

        import {page_module} as page_module_under_test
        """
    )
    if setup_block:
        script = f"{script}\n{setup_block}\n"
    script = f"{script}\npage_module_under_test.{page_function}()\n"
    return _run_streamlit_script(script, timeout=timeout)


def _run_streamlit_script(
    script: str,
    *,
    timeout: float,
) -> AppTest:
    app_test = AppTest.from_string(script, default_timeout=timeout)
    app_test.run(timeout=timeout)
    return app_test


__all__ = [
    "DEFAULT_STREAMLIT_SMOKE_TIMEOUT",
    "PROJECT_ROOT",
    "run_streamlit_entrypoint_smoke",
    "run_streamlit_page_smoke",
]
