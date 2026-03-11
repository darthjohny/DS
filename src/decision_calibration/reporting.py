"""Сборка отчётов и артефактов для калибровки `decision layer`.

Модуль отвечает за представление результатов калибровочной итерации:

- агрегированную сводную структуру;
- таблицы top-N и распределение по классам;
- markdown-отчёт;
- CSV- и JSON-артефакты, сохраняемые в журнал.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from decision_calibration.config import CalibrationConfig
from decision_calibration.constants import CALIBRATOR_VERSION
from decision_calibration.runtime import BaseScoringResult, ReadyDatasetRecord

type SummaryRecord = tuple[str, object]


@dataclass(frozen=True)
class IterationSummary:
    """Короткая агрегированная сводка одной калибровочной итерации."""

    run_id: str
    relation_name: str
    source_name: str
    input_rows: int
    router_rows: int
    host_rows: int
    low_rows: int
    router_score_mode: str
    host_score_mode: str
    host_model_version: str
    final_score_min: float | None
    final_score_mean: float | None
    final_score_max: float | None
    top_n: int
    calibrator_version: str = CALIBRATOR_VERSION


def build_iteration_summary(
    run_id: str,
    dataset: ReadyDatasetRecord,
    base_result: BaseScoringResult,
    ordered_results: pd.DataFrame,
    top_n: int,
    router_score_mode: str,
    host_score_mode: str,
    host_model_version_value: str,
) -> IterationSummary:
    """Собрать короткую агрегированную сводку по результату итерации."""
    if ordered_results.empty:
        min_score = None
        mean_score = None
        max_score = None
    else:
        scores = ordered_results["final_score"].astype(float)
        min_score = float(scores.min())
        mean_score = float(scores.mean())
        max_score = float(scores.max())

    return IterationSummary(
        run_id=run_id,
        relation_name=dataset.relation_name,
        source_name=dataset.source_name,
        input_rows=int(len(base_result.input_df)),
        router_rows=int(len(base_result.router_df)),
        host_rows=int(len(base_result.host_input_df)),
        low_rows=int(len(base_result.low_input_df)),
        router_score_mode=str(router_score_mode),
        host_score_mode=str(host_score_mode),
        host_model_version=str(host_model_version_value),
        final_score_min=min_score,
        final_score_mean=mean_score,
        final_score_max=max_score,
        top_n=int(top_n),
    )


def top_candidates_frame(
    ordered_results: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    """Собрать таблицу top-N кандидатов с ключевыми диагностическими полями."""
    columns = [
        "source_id",
        "predicted_spec_class",
        "predicted_evolution_stage",
        "gauss_label",
        "router_similarity",
        "router_log_posterior",
        "posterior_margin",
        "host_log_lr",
        "host_posterior",
        "class_prior",
        "distance_factor",
        "quality_factor",
        "metallicity_factor",
        "final_score",
        "priority_tier",
        "reason_code",
        "ra",
        "dec",
        "teff_gspphot",
        "logg_gspphot",
        "radius_gspphot",
        "mh_gspphot",
        "parallax",
        "parallax_over_error",
        "ruwe",
    ]
    existing = [column for column in columns if column in ordered_results.columns]
    return ordered_results.loc[:, existing].head(top_n).copy()


def class_distribution_frame(
    top_candidates: pd.DataFrame,
) -> pd.DataFrame:
    """Посчитать распределение спектральных классов внутри top-N."""
    if top_candidates.empty:
        return pd.DataFrame(
            columns=["predicted_spec_class", "count"]
        )
    counts = (
        top_candidates["predicted_spec_class"]
        .astype(str)
        .value_counts(dropna=False)
        .rename_axis("predicted_spec_class")
        .reset_index(name="count")
    )
    return counts


def score_summary_frame(summary: IterationSummary) -> pd.DataFrame:
    """Собрать компактную CSV-таблицу сводки в формате `metric/value`."""
    records: list[SummaryRecord] = [
        ("run_id", summary.run_id),
        ("relation_name", summary.relation_name),
        ("source_name", summary.source_name),
        ("input_rows", summary.input_rows),
        ("router_rows", summary.router_rows),
        ("host_rows", summary.host_rows),
        ("low_rows", summary.low_rows),
        ("router_score_mode", summary.router_score_mode),
        ("host_score_mode", summary.host_score_mode),
        ("host_model_version", summary.host_model_version),
        ("final_score_min", summary.final_score_min),
        ("final_score_mean", summary.final_score_mean),
        ("final_score_max", summary.final_score_max),
        ("top_n", summary.top_n),
        ("calibrator_version", summary.calibrator_version),
    ]
    return pd.DataFrame.from_records(records, columns=["metric", "value"])


def frame_to_text(df: pd.DataFrame) -> str:
    """Преобразовать DataFrame в компактный текстовый блок для markdown."""
    if df.empty:
        return "Пусто"
    return df.to_string(index=False)


def build_iteration_markdown(
    iteration_id: str,
    config: CalibrationConfig,
    summary: IterationSummary,
    top_candidates: pd.DataFrame,
    class_distribution: pd.DataFrame,
    iteration_note: str,
) -> str:
    """Собрать markdown-отчёт для одной калибровочной итерации."""
    created_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    note = iteration_note.strip() or "-"
    return f"""# {iteration_id}

Дата: {created_at}
Идентификатор запуска: {summary.run_id}
Статус: выполнено

## Что меняем
- class_prior
- metallicity_factor
- distance_factor
- quality_factor
- host_score semantics

## Формула
`final_score = host_posterior × class_prior × distance_factor × quality_factor × metallicity_factor`

## Режимы скоринга
- router_score_mode: `{summary.router_score_mode}`
- host_score_mode: `{summary.host_score_mode}`
- host_model_version: `{summary.host_model_version}`

## Параметры итерации
### class_prior
- K: {config.class_prior.k:.3f}
- G: {config.class_prior.g:.3f}
- M: {config.class_prior.m:.3f}
- F: {config.class_prior.f:.3f}

### metallicity_factor
- mh <= {config.metallicity.low_threshold:.2f}: {config.metallicity.low_factor:.2f}
- mh < {config.metallicity.solar_threshold:.2f}: {config.metallicity.neutral_factor:.2f}
- mh < {config.metallicity.high_threshold:.2f}: {config.metallicity.positive_factor:.2f}
- mh >= {config.metallicity.high_threshold:.2f}: {config.metallicity.high_factor:.2f}

### distance_factor
- distance <= {config.distance.near_max_pc:.0f} pc: {config.distance.near_factor:.2f}
- distance <= {config.distance.moderate_max_pc:.0f} pc: {config.distance.moderate_factor:.2f}
- distance <= {config.distance.distant_max_pc:.0f} pc: {config.distance.distant_factor:.2f}
- distance <= {config.distance.far_max_pc:.0f} pc: {config.distance.far_factor:.2f}
- distance > {config.distance.far_max_pc:.0f} pc: {config.distance.very_far_factor:.2f}
- invalid distance: {config.distance.invalid_factor:.2f}

### quality_factor
- quality_factor = ruwe_factor × parallax_precision_factor
- ruwe <= {config.quality.ruwe.good_max:.1f}: {config.quality.ruwe.good_factor:.2f}
- ruwe <= {config.quality.ruwe.warning_max:.1f}: {config.quality.ruwe.warning_factor:.2f}
- ruwe <= {config.quality.ruwe.alert_max:.1f}: {config.quality.ruwe.alert_factor:.2f}
- ruwe <= {config.quality.ruwe.bad_max:.1f}: {config.quality.ruwe.bad_factor:.2f}
- ruwe > {config.quality.ruwe.bad_max:.1f}: {config.quality.ruwe.very_bad_factor:.2f}
- parallax_over_error >= {config.quality.parallax_precision.excellent_min:.0f}: {config.quality.parallax_precision.excellent_factor:.2f}
- parallax_over_error >= {config.quality.parallax_precision.good_min:.0f}: {config.quality.parallax_precision.good_factor:.2f}
- parallax_over_error >= {config.quality.parallax_precision.acceptable_min:.0f}: {config.quality.parallax_precision.acceptable_factor:.2f}
- parallax_over_error >= {config.quality.parallax_precision.weak_min:.0f}: {config.quality.parallax_precision.weak_factor:.2f}
- parallax_over_error < {config.quality.parallax_precision.weak_min:.0f}: {config.quality.parallax_precision.poor_factor:.2f}

## Фактический результат
- relation: `{summary.relation_name}`
- source_name: `{summary.source_name}`
- input_rows: {summary.input_rows}
- router_rows: {summary.router_rows}
- host_rows: {summary.host_rows}
- low_rows: {summary.low_rows}
- router_score_mode: {summary.router_score_mode}
- host_score_mode: {summary.host_score_mode}
- host_model_version: {summary.host_model_version}
- final_score_min: {summary.final_score_min}
- final_score_mean: {summary.final_score_mean}
- final_score_max: {summary.final_score_max}

## Итог
- {note}

## Сводка по top-{summary.top_n}
```text
{frame_to_text(class_distribution)}
```

## Кандидаты top-{summary.top_n}
```text
{frame_to_text(top_candidates)}
```
"""


def save_iteration_artifacts(
    logbook_dir: Path,
    config: CalibrationConfig,
    summary: IterationSummary,
    ordered_results: pd.DataFrame,
    top_n: int,
    iteration_note: str,
    next_iteration_number: int,
) -> Path:
    """Сохранить markdown-, CSV- и JSON-артефакты одной итерации.

    Побочные эффекты
    ----------------
    Создаёт в каталоге журнала:
    - markdown-отчёт;
    - JSON с конфигурацией;
    - CSV с top-N кандидатами;
    - CSV со сводкой score;
    - CSV с распределением классов.
    """
    iteration_id = f"iteration_{next_iteration_number:03d}"
    prefix = logbook_dir / iteration_id

    top_candidates = top_candidates_frame(ordered_results, top_n)
    class_distribution = class_distribution_frame(top_candidates)
    score_summary = score_summary_frame(summary)

    markdown_path = prefix.with_suffix(".md")
    config_path = prefix.parent / f"{iteration_id}_config.json"
    top_path = prefix.parent / f"{iteration_id}_top_candidates.csv"
    score_path = prefix.parent / f"{iteration_id}_score_summary.csv"
    class_path = prefix.parent / f"{iteration_id}_class_distribution.csv"

    markdown_path.write_text(
        build_iteration_markdown(
            iteration_id=iteration_id,
            config=config,
            summary=summary,
            top_candidates=top_candidates,
            class_distribution=class_distribution,
            iteration_note=iteration_note,
        ),
        encoding="utf-8",
    )
    config_path.write_text(
        json.dumps(
            {
                "config": asdict(config),
                "router_score_mode": summary.router_score_mode,
                "host_score_mode": summary.host_score_mode,
                "host_model_version": summary.host_model_version,
                "calibrator_version": summary.calibrator_version,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    top_candidates.to_csv(top_path, index=False)
    score_summary.to_csv(score_path, index=False)
    class_distribution.to_csv(class_path, index=False)

    return markdown_path


def print_summary(
    summary: IterationSummary,
    markdown_path: Path,
    top_candidates: pd.DataFrame,
) -> None:
    """Напечатать короткую терминальную сводку по итерации калибровки."""
    print("=== КАЛИБРОВКА DECISION LAYER ===")
    print(f"Идентификатор запуска: {summary.run_id}")
    print(f"Relation: {summary.relation_name}")
    print(f"Источник: {summary.source_name}")
    print(f"Входных строк: {summary.input_rows}")
    print(f"Строк host-ветки: {summary.host_rows}")
    print(f"Строк low-ветки: {summary.low_rows}")
    print(f"Режим router-score: {summary.router_score_mode}")
    print(f"Режим host-score: {summary.host_score_mode}")
    print(f"Версия host-модели: {summary.host_model_version}")
    print(
        f"final_score min/mean/max: "
        f"{summary.final_score_min} / "
        f"{summary.final_score_mean} / "
        f"{summary.final_score_max}"
    )
    print(f"Markdown-отчёт: {markdown_path}")
    print("Preview top-кандидатов:")
    print(frame_to_text(top_candidates.head(min(10, len(top_candidates)))))


__all__ = [
    "IterationSummary",
    "build_iteration_markdown",
    "build_iteration_summary",
    "class_distribution_frame",
    "frame_to_text",
    "print_summary",
    "save_iteration_artifacts",
    "score_summary_frame",
    "top_candidates_frame",
]
