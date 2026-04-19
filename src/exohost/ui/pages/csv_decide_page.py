# Файл `csv_decide_page.py` слоя `ui/pages`.
#
# Этот файл отвечает только за:
# - страницу запуска по внешнему `CSV`;
# - пользовательский вход в существующий контур `decide`.
#
# Следующий слой:
# - run_service и валидация входного контракта;
# - страницы просмотра нового run_dir.

from __future__ import annotations

from pathlib import Path

import streamlit as st

from exohost.ui.components.overview_metrics import render_run_overview_metrics
from exohost.ui.contracts import UI_EXTERNAL_CSV_CONTRACT
from exohost.ui.loaders import (
    clear_ui_run_caches,
    list_available_run_dirs,
    load_ui_run_bundle,
)
from exohost.ui.pages.support import resolve_selected_index
from exohost.ui.run_overview import build_ui_run_overview
from exohost.ui.run_service import (
    build_ui_csv_decide_defaults,
    run_ui_csv_decide,
    save_uploaded_csv_bytes,
    validate_uploaded_csv_bytes,
)
from exohost.ui.session_state import (
    clear_ui_errors,
    remember_generated_run_dir,
    remember_selected_run_dir,
    remember_selected_source_id,
    remember_uploaded_csv_path,
    set_csv_validation_error,
    set_run_load_error,
)


def render_csv_decide_page() -> None:
    # Кнопочный запуск держим отдельной страницей, чтобы read-only и write-сценарии не мешали друг другу.
    st.title("Запуск по внешнему CSV")
    st.caption(
        "Страница валидирует внешний `CSV`, переиспользует артефакты выбранного "
        "рабочего запуска и сохраняет новый `run_dir` без ручного CLI."
    )
    st.info(
        "Для повторяемости UI не придумывает свои defaults. Он берет model artifacts "
        "и policy из уже готового `run_dir`, выбранного ниже."
    )

    available_run_dirs = list_available_run_dirs()
    if not available_run_dirs:
        st.warning(
            "Не удалось найти базовый `run_dir` в `artifacts/decisions`, "
            "от которого можно унаследовать model artifacts и policy."
        )
        return

    _render_contract_reference()

    run_dir_options = tuple(str(path) for path in available_run_dirs)
    selected_run_dir = st.selectbox(
        "Базовый запуск для defaults",
        options=run_dir_options,
        index=resolve_selected_index(
            options=run_dir_options,
            selected_value=st.session_state.get("selected_run_dir"),
        ),
        format_func=lambda value: Path(value).name,
    )
    remember_selected_run_dir(st.session_state, selected_run_dir)

    try:
        base_bundle = load_ui_run_bundle(selected_run_dir)
        defaults = build_ui_csv_decide_defaults(base_bundle)
    except RuntimeError as exc:
        set_run_load_error(st.session_state, str(exc))
        st.error(f"Не удалось собрать defaults из выбранного запуска: {exc}")
        _render_last_generated_run()
        return

    set_run_load_error(st.session_state, None)
    st.caption(
        "Будут повторно использованы: "
        f"`{defaults.decision_policy_version}`, пороги priority и model artifacts "
        f"из `{Path(selected_run_dir).name}`."
    )

    uploaded_file = st.file_uploader(
        "Внешний CSV",
        type=["csv"],
        help=(
            "Файл должен соответствовать минимальному входному контракту проекта. "
            "После загрузки UI сначала проверит обязательные колонки."
        ),
    )
    if uploaded_file is None:
        _render_last_generated_run()
        return

    uploaded_bytes = uploaded_file.getvalue()
    try:
        validation_preview = validate_uploaded_csv_bytes(uploaded_bytes)
    except RuntimeError as exc:
        set_csv_validation_error(st.session_state, str(exc))
        st.error(f"Не удалось провалидировать внешний CSV: {exc}")
        _render_last_generated_run()
        return

    set_csv_validation_error(st.session_state, None)
    _render_validation_preview(validation_preview)

    if not st.button(
        "Запустить decide по этому CSV",
        type="primary",
        width="stretch",
    ):
        _render_last_generated_run()
        return

    clear_ui_errors(st.session_state)
    try:
        saved_csv_path = save_uploaded_csv_bytes(
            filename=uploaded_file.name,
            uploaded_bytes=uploaded_bytes,
        )
        remember_uploaded_csv_path(st.session_state, str(saved_csv_path))
        with st.spinner("Выполняю `decide` на внешнем CSV..."):
            run_result = run_ui_csv_decide(
                csv_path=saved_csv_path,
                defaults=defaults,
            )
        clear_ui_run_caches()
        remember_generated_run_dir(
            st.session_state,
            str(run_result.artifact_paths.run_dir),
        )
        remember_selected_run_dir(
            st.session_state,
            str(run_result.artifact_paths.run_dir),
        )
        remember_selected_source_id(st.session_state, None)
    except Exception as exc:
        st.error(f"Не удалось завершить CSV-запуск: {exc}")
        _render_last_generated_run()
        return

    st.success("Новый `run_dir` успешно сохранен. Его можно сразу открыть на странице «Запуск».")
    _render_generated_run_summary(str(run_result.artifact_paths.run_dir))


def _render_contract_reference() -> None:
    with st.expander("Минимальный контракт внешнего CSV", expanded=False):
        st.markdown(
            "Обязательные колонки: "
            + ", ".join(f"`{column_name}`" for column_name in UI_EXTERNAL_CSV_CONTRACT.required_columns)
        )
        st.markdown(
            "Рекомендуемые колонки: "
            + ", ".join(
                f"`{column_name}`" for column_name in UI_EXTERNAL_CSV_CONTRACT.recommended_columns
            )
        )
        st.markdown(
            "Если quality-слой проекта не воспроизводится отдельно, "
            f"допустим технический сценарий с `quality_state = '{UI_EXTERNAL_CSV_CONTRACT.quality_state_default}'`."
        )
        st.caption(
            "Подробный контракт описан в "
            f"`{UI_EXTERNAL_CSV_CONTRACT.contract_doc_path}`."
        )


def _render_validation_preview(validation_preview) -> None:
    preview_columns = st.columns(3)
    preview_columns[0].metric("Строк в CSV", f"{validation_preview.n_rows:,}".replace(",", " "))
    preview_columns[1].metric("Колонок", str(len(validation_preview.column_names)))

    missing_recommended = [
        column_name
        for column_name in UI_EXTERNAL_CSV_CONTRACT.recommended_columns
        if column_name not in validation_preview.column_names
    ]
    preview_columns[2].metric(
        "Не хватает рекомендуемых",
        str(len(missing_recommended)),
    )

    if missing_recommended:
        missing_sql = ", ".join(f"`{column_name}`" for column_name in missing_recommended)
        st.caption(
            "CSV пройдет минимальную валидацию, но в нем отсутствуют рекомендуемые поля: "
            f"{missing_sql}."
        )

    st.subheader("Preview входного CSV")
    st.dataframe(
        validation_preview.validated_df.head(10),
        width="stretch",
        hide_index=True,
    )


def _render_last_generated_run() -> None:
    generated_run_dir = st.session_state.get("generated_run_dir")
    if generated_run_dir is None:
        return
    _render_generated_run_summary(str(generated_run_dir))


def _render_generated_run_summary(run_dir: str) -> None:
    try:
        result_bundle = load_ui_run_bundle(run_dir)
    except RuntimeError as exc:
        st.warning(f"Не удалось перечитать последний сохраненный `run_dir`: {exc}")
        return

    st.subheader("Последний сохраненный запуск")
    render_run_overview_metrics(build_ui_run_overview(result_bundle))
    st.caption(f"Каталог артефактов: `{Path(run_dir).resolve()}`")

    uploaded_csv_path = st.session_state.get("uploaded_csv_path")
    if uploaded_csv_path is not None:
        st.caption(f"Исходный сохраненный CSV: `{Path(str(uploaded_csv_path)).resolve()}`")
