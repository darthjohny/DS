# Файл `model_metrics_summary.py` слоя `ui`.
#
# Этот файл отвечает только за:
# - интерпретацию benchmark-метрик для страницы качества моделей;
# - компактную trust-summary сводку по слоям `ID/OOD`, `coarse`, `host` и `refinement`.
#
# Следующий слой:
# - визуальные компоненты overview-страницы метрик;
# - unit-тесты helper-слоя интерпретации benchmark quality.

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class UiModelMetricsOverview:
    # Короткая сводка по слоям моделей для верхнего блока страницы метрик.
    n_strong_stages: int
    n_stable_stages: int
    n_caution_stages: int
    n_missing_stages: int
    best_stage_name: str | None
    best_stage_macro_f1: float | None
    weakest_stage_name: str | None
    weakest_stage_macro_f1: float | None
    overview_message: str


def build_ui_metric_stage_assessment_frame(metrics_df: pd.DataFrame) -> pd.DataFrame:
    # Преобразуем raw benchmark summary в явную trust-layer таблицу для UI и smoke-проверок.
    if metrics_df.empty:
        return pd.DataFrame(
            columns=[
                "stage_name",
                "benchmark_run_dir",
                "trust_level",
                "trust_label",
                "trust_summary",
                "test_macro_f1",
                "test_balanced_accuracy",
                "test_roc_auc_ovr",
                "cv_mean_macro_f1",
                "n_rows_test",
                "note",
            ]
        )

    assessment_rows: list[dict[str, object]] = []
    for _, row in metrics_df.iterrows():
        trust_level = _classify_stage_trust_level(
            macro_f1=_to_optional_float(row.get("test_macro_f1")),
            balanced_accuracy=_to_optional_float(row.get("test_balanced_accuracy")),
        )
        assessment_rows.append(
            {
                "stage_name": str(row.get("stage_name", "unknown")),
                "benchmark_run_dir": _to_optional_string(row.get("benchmark_run_dir")),
                "trust_level": trust_level,
                "trust_label": _build_trust_label(trust_level),
                "trust_summary": _build_trust_summary(
                    stage_name=str(row.get("stage_name", "unknown")),
                    trust_level=trust_level,
                    macro_f1=_to_optional_float(row.get("test_macro_f1")),
                    balanced_accuracy=_to_optional_float(row.get("test_balanced_accuracy")),
                ),
                "test_macro_f1": _to_optional_float(row.get("test_macro_f1")),
                "test_balanced_accuracy": _to_optional_float(row.get("test_balanced_accuracy")),
                "test_roc_auc_ovr": _to_optional_float(row.get("test_roc_auc_ovr")),
                "cv_mean_macro_f1": _to_optional_float(row.get("cv_mean_macro_f1")),
                "n_rows_test": _to_optional_int(row.get("n_rows_test")),
                "note": str(row.get("note", "")),
            }
        )
    return pd.DataFrame.from_records(assessment_rows)


def build_ui_model_metrics_overview(
    assessment_df: pd.DataFrame,
) -> UiModelMetricsOverview:
    # Верхний обзор страницы должен быстро показать сильные и рискованные зоны model stack.
    if assessment_df.empty:
        return UiModelMetricsOverview(
            n_strong_stages=0,
            n_stable_stages=0,
            n_caution_stages=0,
            n_missing_stages=0,
            best_stage_name=None,
            best_stage_macro_f1=None,
            weakest_stage_name=None,
            weakest_stage_macro_f1=None,
            overview_message=(
                "Контрольные артефакты не найдены, поэтому доверительный слой по моделям пока "
                "нельзя построить."
            ),
        )

    strong_mask = assessment_df["trust_level"].astype(str) == "strong"
    stable_mask = assessment_df["trust_level"].astype(str) == "stable"
    caution_mask = assessment_df["trust_level"].astype(str) == "caution"
    missing_mask = assessment_df["trust_level"].astype(str) == "missing"

    scored_df = assessment_df.loc[
        assessment_df["test_macro_f1"].notna(),
        ["stage_name", "test_macro_f1"],
    ].copy()
    best_stage_name: str | None = None
    best_stage_macro_f1: float | None = None
    weakest_stage_name: str | None = None
    weakest_stage_macro_f1: float | None = None
    if not scored_df.empty:
        best_row = scored_df.sort_values(
            ["test_macro_f1", "stage_name"],
            ascending=[False, True],
            kind="mergesort",
            ignore_index=True,
        ).iloc[0]
        weakest_row = scored_df.sort_values(
            ["test_macro_f1", "stage_name"],
            ascending=[True, True],
            kind="mergesort",
            ignore_index=True,
        ).iloc[0]
        best_stage_name = str(best_row["stage_name"])
        best_stage_macro_f1 = _to_optional_float(best_row["test_macro_f1"])
        weakest_stage_name = str(weakest_row["stage_name"])
        weakest_stage_macro_f1 = _to_optional_float(weakest_row["test_macro_f1"])

    return UiModelMetricsOverview(
        n_strong_stages=int(strong_mask.sum()),
        n_stable_stages=int(stable_mask.sum()),
        n_caution_stages=int(caution_mask.sum()),
        n_missing_stages=int(missing_mask.sum()),
        best_stage_name=best_stage_name,
        best_stage_macro_f1=best_stage_macro_f1,
        weakest_stage_name=weakest_stage_name,
        weakest_stage_macro_f1=weakest_stage_macro_f1,
        overview_message=_build_overview_message(
            n_strong_stages=int(strong_mask.sum()),
            n_stable_stages=int(stable_mask.sum()),
            n_caution_stages=int(caution_mask.sum()),
            n_missing_stages=int(missing_mask.sum()),
            best_stage_name=best_stage_name,
            weakest_stage_name=weakest_stage_name,
        ),
    )


def _classify_stage_trust_level(
    *,
    macro_f1: float | None,
    balanced_accuracy: float | None,
) -> str:
    if macro_f1 is None or balanced_accuracy is None:
        return "missing"
    if macro_f1 >= 0.90 and balanced_accuracy >= 0.90:
        return "strong"
    if macro_f1 >= 0.75 and balanced_accuracy >= 0.75:
        return "stable"
    return "caution"


def _build_trust_label(trust_level: str) -> str:
    trust_labels = {
        "strong": "Сильный слой",
        "stable": "Стабильный слой",
        "caution": "Нужна осторожность",
        "missing": "Нет контрольных данных",
    }
    return trust_labels.get(trust_level, "Неизвестный статус")


def _build_trust_summary(
    *,
    stage_name: str,
    trust_level: str,
    macro_f1: float | None,
    balanced_accuracy: float | None,
) -> str:
    if trust_level == "missing":
        return (
            f"Для слоя `{stage_name}` нет полного набора контрольных метрик, поэтому "
            "его нельзя уверенно интерпретировать в UI."
        )

    metric_sql = (
        f"Macro F1 = {_format_metric(macro_f1)}, "
        f"balanced accuracy = {_format_metric(balanced_accuracy)}."
    )
    if trust_level == "strong":
        return (
            f"Слой `{stage_name}` выглядит надежно по основным контрольным метрикам: "
            f"{metric_sql}"
        )
    if trust_level == "stable":
        return (
            f"Слой `{stage_name}` выглядит рабочим, но его стоит интерпретировать без "
            f"лишнего оптимизма: {metric_sql}"
        )
    return (
        f"Слой `{stage_name}` полезен, но требует осторожной интерпретации в прикладных "
        f"выводах: {metric_sql}"
    )


def _build_overview_message(
    *,
    n_strong_stages: int,
    n_stable_stages: int,
    n_caution_stages: int,
    n_missing_stages: int,
    best_stage_name: str | None,
    weakest_stage_name: str | None,
) -> str:
    message = (
        f"Сильных слоев: `{n_strong_stages}`, стабильных: `{n_stable_stages}`, "
        f"зон осторожной интерпретации: `{n_caution_stages}`."
    )
    if n_missing_stages > 0:
        message += f" Без контрольных данных осталось `{n_missing_stages}` слоев."
    if best_stage_name is not None and weakest_stage_name is not None:
        message += (
            f" Лучший слой по Macro F1 сейчас — `{best_stage_name}`, "
            f"самый слабый — `{weakest_stage_name}`."
        )
    return message


def _format_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


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


def _to_optional_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


__all__ = [
    "UiModelMetricsOverview",
    "build_ui_metric_stage_assessment_frame",
    "build_ui_model_metrics_overview",
]
