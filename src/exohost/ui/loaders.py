# Файл `loaders.py` слоя `ui`.
#
# Этот файл отвечает только за:
# - загрузку и валидацию готовых final-decision artifacts для интерфейса;
# - тонкий read-only доступ к run_dir без дублирования прикладной логики.
#
# Следующий слой:
# - страницы и компоненты интерфейса;
# - unit-тесты UI loader-слоя.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from exohost.reporting.final_decision_artifacts import (
    LoadedFinalDecisionArtifacts,
    load_final_decision_artifacts,
)
from exohost.ui.contracts import (
    UI_FINAL_DECISION_RUN_CONTRACT,
    UiArtifactTableContract,
    UiRunArtifactContract,
    build_ui_artifact_table_map,
)
from exohost.ui.streamlit_compat import cache_data, clear_cached_call

DEFAULT_UI_RUNS_DIR = Path("artifacts/decisions")


@dataclass(frozen=True, slots=True)
class UiLoadedRunBundle:
    # Загруженный и провалидированный run bundle для read-only страниц интерфейса.
    run_dir: Path
    loaded_artifacts: LoadedFinalDecisionArtifacts


def validate_ui_run_dir(
    run_dir: str | Path,
    *,
    contract: UiRunArtifactContract = UI_FINAL_DECISION_RUN_CONTRACT,
) -> Path:
    # Интерфейс принимает только run_dir с полным обязательным набором файлов.
    resolved_run_dir = Path(run_dir).expanduser().resolve()
    if not resolved_run_dir.exists() or not resolved_run_dir.is_dir():
        raise RuntimeError(f"UI run directory does not exist: {resolved_run_dir}")

    missing_filenames = [
        filename
        for filename in contract.required_filenames
        if not (resolved_run_dir / filename).exists()
    ]
    if missing_filenames:
        missing_filenames_sql = ", ".join(missing_filenames)
        raise RuntimeError(
            "UI run directory is missing required files: "
            f"{missing_filenames_sql}"
        )
    return resolved_run_dir


def load_ui_run_bundle_uncached(
    run_dir: str | Path,
    *,
    contract: UiRunArtifactContract = UI_FINAL_DECISION_RUN_CONTRACT,
) -> UiLoadedRunBundle:
    # Загружаем готовый run и валидируем его по UI-контракту, прежде чем отдавать страницам.
    resolved_run_dir = validate_ui_run_dir(run_dir, contract=contract)
    loaded_artifacts = load_final_decision_artifacts(resolved_run_dir)

    _validate_ui_metadata(loaded_artifacts.metadata, contract=contract)
    _validate_ui_loaded_tables(loaded_artifacts, contract=contract)

    return UiLoadedRunBundle(
        run_dir=resolved_run_dir,
        loaded_artifacts=loaded_artifacts,
    )


@cache_data(show_spinner=False)
def load_ui_run_bundle(
    run_dir: str,
    *,
    contract: UiRunArtifactContract = UI_FINAL_DECISION_RUN_CONTRACT,
) -> UiLoadedRunBundle:
    # Кэшируем read-only загрузку, чтобы rerun страниц не перечитывал большие CSV без нужды.
    return load_ui_run_bundle_uncached(run_dir, contract=contract)


@cache_data(show_spinner=False)
def list_available_run_dirs(
    base_dir: str = str(DEFAULT_UI_RUNS_DIR),
    *,
    contract: UiRunArtifactContract = UI_FINAL_DECISION_RUN_CONTRACT,
) -> tuple[Path, ...]:
    # Для read-only выбора показываем только те каталоги, которые уже похожи на валидный run_dir.
    resolved_base_dir = Path(base_dir).expanduser().resolve()
    if not resolved_base_dir.exists() or not resolved_base_dir.is_dir():
        return ()

    valid_run_dirs: list[Path] = []
    for child_path in resolved_base_dir.iterdir():
        if not child_path.is_dir():
            continue
        try:
            validate_ui_run_dir(child_path, contract=contract)
        except RuntimeError:
            continue
        valid_run_dirs.append(child_path)
    return tuple(sorted(valid_run_dirs, key=lambda path: path.name, reverse=True))


def clear_ui_run_caches() -> None:
    # После сохранения нового запуска очищаем read-only кэш, чтобы свежий `run_dir`
    # сразу попадал в списки страниц интерфейса.
    clear_cached_call(load_ui_run_bundle)
    clear_cached_call(list_available_run_dirs)


def _validate_ui_metadata(
    metadata: dict[str, Any],
    *,
    contract: UiRunArtifactContract,
) -> None:
    missing_metadata_keys = [
        key for key in contract.required_metadata_keys if key not in metadata
    ]
    if missing_metadata_keys:
        missing_metadata_keys_sql = ", ".join(missing_metadata_keys)
        raise RuntimeError(
            "UI run metadata is missing required keys: "
            f"{missing_metadata_keys_sql}"
        )

    pipeline_name = metadata.get("pipeline_name")
    if pipeline_name != contract.pipeline_name:
        raise RuntimeError(
            "UI run metadata points to unexpected pipeline: "
            f"{pipeline_name!r}"
        )

    raw_context = metadata.get("context")
    if not isinstance(raw_context, dict):
        raise RuntimeError("UI run metadata must contain object-valued context.")

    missing_context_keys = [
        key for key in contract.required_metadata_context_keys if key not in raw_context
    ]
    if missing_context_keys:
        missing_context_keys_sql = ", ".join(missing_context_keys)
        raise RuntimeError(
            "UI run metadata context is missing required keys: "
            f"{missing_context_keys_sql}"
        )


def _validate_ui_loaded_tables(
    loaded_artifacts: LoadedFinalDecisionArtifacts,
    *,
    contract: UiRunArtifactContract,
) -> None:
    table_contract_map = build_ui_artifact_table_map(contract)

    _validate_table_columns(
        available_columns=set(loaded_artifacts.decision_input_df.columns.astype(str)),
        table_contract=table_contract_map["decision_input"],
    )
    _validate_table_columns(
        available_columns=set(loaded_artifacts.final_decision_df.columns.astype(str)),
        table_contract=table_contract_map["final_decision"],
    )
    _validate_table_columns(
        available_columns=set(loaded_artifacts.priority_input_df.columns.astype(str)),
        table_contract=table_contract_map["priority_input"],
    )
    _validate_table_columns(
        available_columns=set(loaded_artifacts.priority_ranking_df.columns.astype(str)),
        table_contract=table_contract_map["priority_ranking"],
    )


def _validate_table_columns(
    *,
    available_columns: set[str],
    table_contract: UiArtifactTableContract,
) -> None:
    missing_columns = [
        column_name
        for column_name in table_contract.required_columns
        if column_name not in available_columns
    ]
    if not missing_columns:
        return

    missing_columns_sql = ", ".join(missing_columns)
    raise RuntimeError(
        f"UI table '{table_contract.filename}' is missing required columns: "
        f"{missing_columns_sql}"
    )


__all__ = [
    "clear_ui_run_caches",
    "DEFAULT_UI_RUNS_DIR",
    "UiLoadedRunBundle",
    "list_available_run_dirs",
    "load_ui_run_bundle",
    "load_ui_run_bundle_uncached",
    "validate_ui_run_dir",
]
