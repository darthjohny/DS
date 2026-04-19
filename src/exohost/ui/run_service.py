# Файл `run_service.py` слоя `ui`.
#
# Этот файл отвечает только за:
# - программный запуск существующего `decide` из интерфейса Streamlit;
# - валидацию внешнего `CSV` и извлечение defaults из выбранного `run_dir`.
#
# Следующий слой:
# - страница `CSV`-запуска интерфейса;
# - unit-тесты service-слоя UI.

from __future__ import annotations

import argparse
import io
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any

import pandas as pd

from exohost.cli.decide.priority_support import (
    build_priority_integration_config_from_namespace,
)
from exohost.cli.decide.quality_gate_support import (
    build_quality_gate_tuning_config_from_namespace,
)
from exohost.cli.decide.support import (
    build_decide_context,
    build_final_decision_policy_from_namespace,
    load_decision_input_frame,
    resolve_decision_output_dir,
)
from exohost.db.engine import make_read_only_engine
from exohost.posthoc.decision_model_bundle import load_final_decision_model_bundle
from exohost.posthoc.final_decision_artifact_runner import (
    FinalDecisionArtifactRunResult,
    run_final_decision_with_artifacts,
)
from exohost.posthoc.quality_gate_tuning import apply_quality_gate_tuning
from exohost.ranking.priority_score import DEFAULT_HOST_SCORE_COLUMN
from exohost.reporting.benchmark_artifacts import build_run_stamp, sanitize_artifact_name
from exohost.reporting.final_decision_artifacts import (
    DEFAULT_FINAL_DECISION_OUTPUT_DIR,
    FinalDecisionArtifactPaths,
    save_final_decision_artifacts,
)
from exohost.ui.contracts import UI_EXTERNAL_CSV_CONTRACT
from exohost.ui.loaders import UiLoadedRunBundle

DEFAULT_UI_UPLOADS_DIR = Path("tmp/streamlit_uploads")


@dataclass(frozen=True, slots=True)
class UiCsvDecideDefaults:
    # Defaults берутся из выбранного рабочего run и повторяют его artifact-контур и policy.
    ood_model_run_dir: str
    ood_threshold_run_dir: str
    coarse_model_run_dir: str
    refinement_model_run_dirs: tuple[str, ...]
    host_model_run_dir: str | None
    decision_policy_version: str
    candidate_ood_disposition: str
    host_score_column: str
    min_refinement_confidence: float | None
    min_coarse_probability: float | None
    min_coarse_margin: float | None
    quality_ruwe_unknown_threshold: float | None
    quality_parallax_snr_unknown_threshold: float | None
    quality_require_flame_for_pass: bool | None
    priority_high_min: float | None
    priority_medium_min: float | None
    output_dir: str
    dotenv_path: str
    connect_timeout: int
    preview_rows: int = 10


@dataclass(frozen=True, slots=True)
class UiCsvValidationPreview:
    # Перед запуском показываем компактный preview внешнего CSV и его shape.
    validated_df: pd.DataFrame
    n_rows: int
    column_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class UiCsvDecideRunResult:
    # Страница запуска получает новый run_dir и может сразу перевести пользователя к просмотру.
    uploaded_csv_path: Path
    artifact_paths: FinalDecisionArtifactPaths
    decision_result: FinalDecisionArtifactRunResult


def build_ui_csv_decide_defaults(bundle: UiLoadedRunBundle) -> UiCsvDecideDefaults:
    # Все defaults вытаскиваем из metadata.context выбранного run, а не держим отдельной магией.
    raw_context = bundle.loaded_artifacts.metadata.get("context")
    if not isinstance(raw_context, dict):
        raise RuntimeError("В выбранном `run_dir` отсутствует корректный `metadata.context`.")

    return UiCsvDecideDefaults(
        ood_model_run_dir=_require_string(raw_context, "ood_model_run_dir"),
        ood_threshold_run_dir=_require_string(raw_context, "ood_threshold_run_dir"),
        coarse_model_run_dir=_require_string(raw_context, "coarse_model_run_dir"),
        refinement_model_run_dirs=_require_string_tuple(raw_context, "refinement_model_run_dirs"),
        host_model_run_dir=_optional_string(raw_context.get("host_model_run_dir")),
        decision_policy_version=_require_string(raw_context, "decision_policy_version"),
        candidate_ood_disposition=_require_string(raw_context, "candidate_ood_disposition"),
        host_score_column=_optional_string(raw_context.get("host_score_column"))
        or DEFAULT_HOST_SCORE_COLUMN,
        min_refinement_confidence=_optional_float(
            raw_context.get("min_refinement_confidence")
        ),
        min_coarse_probability=_optional_float(raw_context.get("min_coarse_probability")),
        min_coarse_margin=_optional_float(raw_context.get("min_coarse_margin")),
        quality_ruwe_unknown_threshold=_optional_float(
            raw_context.get("quality_ruwe_unknown_threshold")
        ),
        quality_parallax_snr_unknown_threshold=_optional_float(
            raw_context.get("quality_parallax_snr_unknown_threshold")
        ),
        quality_require_flame_for_pass=_optional_bool(
            raw_context.get("quality_require_flame_for_pass")
        ),
        priority_high_min=_optional_float(raw_context.get("priority_high_min")),
        priority_medium_min=_optional_float(raw_context.get("priority_medium_min")),
        output_dir=_optional_string(raw_context.get("output_dir"))
        or str(DEFAULT_FINAL_DECISION_OUTPUT_DIR),
        dotenv_path=_optional_string(raw_context.get("dotenv_path")) or ".env",
        connect_timeout=_optional_int(raw_context.get("connect_timeout")) or 10,
    )


def validate_uploaded_csv_bytes(uploaded_bytes: bytes) -> UiCsvValidationPreview:
    # CSV сначала читаем и валидируем в памяти, чтобы не плодить мусорные временные файлы.
    try:
        validated_df = pd.read_csv(io.BytesIO(uploaded_bytes))
    except Exception as exc:
        raise RuntimeError(f"Не удалось прочитать загруженный CSV: {exc}") from exc

    missing_columns = [
        column_name
        for column_name in UI_EXTERNAL_CSV_CONTRACT.required_columns
        if column_name not in validated_df.columns.astype(str)
    ]
    if missing_columns:
        missing_columns_sql = ", ".join(missing_columns)
        raise RuntimeError(
            "Во внешнем CSV отсутствуют обязательные колонки: "
            f"{missing_columns_sql}"
        )

    return UiCsvValidationPreview(
        validated_df=validated_df.copy(),
        n_rows=int(validated_df.shape[0]),
        column_names=tuple(validated_df.columns.astype(str).tolist()),
    )


def save_uploaded_csv_bytes(
    *,
    filename: str,
    uploaded_bytes: bytes,
    uploads_dir: str | Path = DEFAULT_UI_UPLOADS_DIR,
) -> Path:
    # Загруженный CSV сохраняем в отдельный demo-контур, чтобы page rerun не терял файл.
    target_dir = Path(uploads_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    source_name = Path(filename).stem or "uploaded_input"
    suffix = Path(filename).suffix or ".csv"
    target_name = f"{sanitize_artifact_name(source_name)}_{build_run_stamp()}{suffix}"
    target_path = target_dir / target_name
    target_path.write_bytes(uploaded_bytes)
    return target_path.resolve()


def run_ui_csv_decide(
    *,
    csv_path: str | Path,
    defaults: UiCsvDecideDefaults,
) -> UiCsvDecideRunResult:
    # Программно запускаем тот же artifact-based `decide`, который использует CLI.
    namespace = _build_namespace_for_ui(csv_path=csv_path, defaults=defaults)

    bundle = load_final_decision_model_bundle(
        ood_model_run_dir=namespace.ood_model_run_dir,
        ood_threshold_run_dir=namespace.ood_threshold_run_dir,
        coarse_model_run_dir=namespace.coarse_model_run_dir,
        refinement_model_run_dirs=namespace.refinement_model_run_dir,
        host_model_run_dir=namespace.host_model_run_dir,
    )
    quality_gate_config = build_quality_gate_tuning_config_from_namespace(namespace)
    base_df = load_decision_input_frame(
        namespace,
        bundle=bundle,
        engine_factory=make_read_only_engine,
        dataset_loader=_unsupported_dataset_loader,
    )
    base_df = apply_quality_gate_tuning(base_df, config=quality_gate_config)

    decision_result = run_final_decision_with_artifacts(
        base_df,
        bundle=bundle,
        final_decision_policy=build_final_decision_policy_from_namespace(namespace),
        priority_config=build_priority_integration_config_from_namespace(namespace),
    )
    output_dir = resolve_decision_output_dir(namespace)
    artifact_paths = save_final_decision_artifacts(
        pipeline_name="hierarchical_final_decision",
        decision_input_df=decision_result.decision_input_df,
        final_decision_df=decision_result.final_decision_df,
        priority_input_df=decision_result.priority_input_df,
        priority_ranking_df=decision_result.priority_ranking_df,
        output_dir=output_dir,
        extra_metadata=build_decide_context(
            namespace,
            output_dir=output_dir,
            bundle=bundle,
            quality_gate_policy_name=quality_gate_config.policy_name,
        ),
    )
    return UiCsvDecideRunResult(
        uploaded_csv_path=Path(csv_path).expanduser().resolve(),
        artifact_paths=artifact_paths,
        decision_result=decision_result,
    )


def _build_namespace_for_ui(
    *,
    csv_path: str | Path,
    defaults: UiCsvDecideDefaults,
) -> argparse.Namespace:
    return argparse.Namespace(
        input_csv=str(Path(csv_path).expanduser().resolve()),
        relation_name=None,
        ood_model_run_dir=defaults.ood_model_run_dir,
        ood_threshold_run_dir=defaults.ood_threshold_run_dir,
        coarse_model_run_dir=defaults.coarse_model_run_dir,
        refinement_model_run_dir=list(defaults.refinement_model_run_dirs),
        host_model_run_dir=defaults.host_model_run_dir,
        decision_policy_version=defaults.decision_policy_version,
        candidate_ood_disposition=defaults.candidate_ood_disposition,
        min_refinement_confidence=defaults.min_refinement_confidence,
        min_coarse_probability=defaults.min_coarse_probability,
        min_coarse_margin=defaults.min_coarse_margin,
        host_score_column=defaults.host_score_column,
        quality_ruwe_unknown_threshold=defaults.quality_ruwe_unknown_threshold,
        quality_parallax_snr_unknown_threshold=defaults.quality_parallax_snr_unknown_threshold,
        quality_require_flame_for_pass=defaults.quality_require_flame_for_pass,
        priority_high_min=defaults.priority_high_min,
        priority_medium_min=defaults.priority_medium_min,
        output_dir=defaults.output_dir,
        preview_rows=defaults.preview_rows,
        limit=None,
        dotenv_path=defaults.dotenv_path,
        connect_timeout=defaults.connect_timeout,
    )


def _unsupported_dataset_loader(*args: Any, **kwargs: Any) -> pd.DataFrame:
    raise RuntimeError("UI CSV launch must not call DB-backed dataset loader.")


def _require_string(context: dict[str, Any], key: str) -> str:
    value = context.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"В metadata.context отсутствует обязательная строка `{key}`.")
    return value


def _require_string_tuple(context: dict[str, Any], key: str) -> tuple[str, ...]:
    value = context.get(key)
    if value is None:
        raise RuntimeError(f"В metadata.context отсутствует обязательный список `{key}`.")
    if not isinstance(value, list):
        raise RuntimeError(f"Поле `{key}` в metadata.context должно быть списком строк.")

    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise RuntimeError(f"Поле `{key}` содержит нестроковое значение.")
        result.append(item)
    return tuple(result)


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _optional_float(value: object) -> float | None:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, Real):
        return float(value)
    return None


def _optional_bool(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    return None


def _optional_int(value: object) -> int | None:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    return None


__all__ = [
    "DEFAULT_UI_UPLOADS_DIR",
    "UiCsvDecideDefaults",
    "UiCsvDecideRunResult",
    "UiCsvValidationPreview",
    "build_ui_csv_decide_defaults",
    "run_ui_csv_decide",
    "save_uploaded_csv_bytes",
    "validate_uploaded_csv_bytes",
]
