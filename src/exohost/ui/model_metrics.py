# Файл `model_metrics.py` слоя `ui`.
#
# Этот файл отвечает только за:
# - загрузку benchmark-артефактов для страницы качества моделей;
# - подготовку компактной таблицы по слоям `ID/OOD`, `coarse`, `host` и `refinement`.
#
# Следующий слой:
# - page- и component-слой интерфейса;
# - unit-тесты helper-модуля метрик.

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from exohost.reporting.model_pipeline_review import (
    build_benchmark_summary_frame,
    load_benchmark_review_bundle,
)
from exohost.ui.contracts import (
    UI_BENCHMARK_STAGE_CONTRACTS,
    UiBenchmarkStageContract,
)
from exohost.ui.streamlit_compat import cache_data

DEFAULT_UI_BENCHMARKS_DIR = Path("artifacts/benchmarks")


@dataclass(frozen=True, slots=True)
class UiBenchmarkStageSummary:
    # Нормализованная строка по одному benchmark-этапу для UI-таблицы.
    stage_key: str
    stage_name: str
    benchmark_run_dir: str | None
    task_name: str
    test_accuracy: float | None
    test_balanced_accuracy: float | None
    test_macro_f1: float | None
    test_roc_auc_ovr: float | None
    cv_mean_accuracy: float | None
    cv_mean_balanced_accuracy: float | None
    cv_mean_macro_f1: float | None
    n_rows_test: int | None
    note: str


def find_latest_benchmark_run_dir(
    task_name_prefix: str,
    *,
    artifacts_root: str | Path = DEFAULT_UI_BENCHMARKS_DIR,
) -> Path | None:
    # Ищем самый свежий benchmark-dir по task prefix, чтобы UI не держал жесткие пути.
    root = Path(artifacts_root)
    if not root.exists() or not root.is_dir():
        return None

    candidate_dirs = [
        path
        for path in root.iterdir()
        if path.is_dir() and path.name.startswith(task_name_prefix)
    ]
    if not candidate_dirs:
        return None
    return sorted(candidate_dirs, key=lambda path: path.name)[-1]


def load_benchmark_stage_overview_uncached(
    *,
    artifacts_root: str | Path = DEFAULT_UI_BENCHMARKS_DIR,
) -> pd.DataFrame:
    # Строим компактную stage-level таблицу для страницы качества моделей.
    rows = [
        _build_stage_summary_row(stage_contract, artifacts_root=artifacts_root)
        for stage_contract in UI_BENCHMARK_STAGE_CONTRACTS
    ]
    return pd.DataFrame.from_records([asdict(row) for row in rows])


@cache_data(show_spinner=False)
def load_benchmark_stage_overview(
    *,
    artifacts_root: str = str(DEFAULT_UI_BENCHMARKS_DIR),
) -> pd.DataFrame:
    # Benchmark-артефакты меняются редко, поэтому read-only сводку можно кэшировать.
    return load_benchmark_stage_overview_uncached(artifacts_root=artifacts_root)


def _build_stage_summary_row(
    stage_contract: UiBenchmarkStageContract,
    *,
    artifacts_root: str | Path,
) -> UiBenchmarkStageSummary:
    benchmark_run_dir = find_latest_benchmark_run_dir(
        stage_contract.task_name_prefix,
        artifacts_root=artifacts_root,
    )
    if benchmark_run_dir is None:
        return UiBenchmarkStageSummary(
            stage_key=stage_contract.stage_key,
            stage_name=stage_contract.display_name,
            benchmark_run_dir=None,
            task_name=stage_contract.task_name_prefix,
            test_accuracy=None,
            test_balanced_accuracy=None,
            test_macro_f1=None,
            test_roc_auc_ovr=None,
            cv_mean_accuracy=None,
            cv_mean_balanced_accuracy=None,
            cv_mean_macro_f1=None,
            n_rows_test=None,
            note=stage_contract.interpretation_note,
        )

    summary_df = build_benchmark_summary_frame(
        load_benchmark_review_bundle(benchmark_run_dir)
    )
    summary_row = summary_df.iloc[0]

    return UiBenchmarkStageSummary(
        stage_key=stage_contract.stage_key,
        stage_name=stage_contract.display_name,
        benchmark_run_dir=benchmark_run_dir.name,
        task_name=str(summary_row["task_name"]),
        test_accuracy=_to_optional_float(summary_row["test_accuracy"]),
        test_balanced_accuracy=_to_optional_float(summary_row["test_balanced_accuracy"]),
        test_macro_f1=_to_optional_float(summary_row["test_macro_f1"]),
        test_roc_auc_ovr=_to_optional_float(summary_row["test_roc_auc_ovr"]),
        cv_mean_accuracy=_to_optional_float(summary_row["cv_mean_accuracy"]),
        cv_mean_balanced_accuracy=_to_optional_float(
            summary_row["cv_mean_balanced_accuracy"]
        ),
        cv_mean_macro_f1=_to_optional_float(summary_row["cv_mean_macro_f1"]),
        n_rows_test=_to_optional_int(summary_row["n_rows_test"]),
        note=stage_contract.interpretation_note,
    )


def _to_optional_float(value: object) -> float | None:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        if pd.isna(value):
            return None
        return float(value)
    return None


def _to_optional_int(value: object) -> int | None:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if pd.isna(value):
            return None
        return int(value)
    return None


__all__ = [
    "DEFAULT_UI_BENCHMARKS_DIR",
    "UiBenchmarkStageSummary",
    "find_latest_benchmark_run_dir",
    "load_benchmark_stage_overview",
    "load_benchmark_stage_overview_uncached",
]
