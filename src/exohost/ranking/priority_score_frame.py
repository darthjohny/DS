# Файл `priority_score_frame.py` слоя `ranking`.
#
# Этот файл отвечает только за:
# - логики приоритизации и наблюдательной пригодности;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - соседние модули слоя `ranking` и orchestration поверх них;
# - unit-тесты и более верхние слои проекта.

from __future__ import annotations

from dataclasses import asdict

import pandas as pd

from exohost.contracts.label_contract import LOW_PRIORITY_SPECTRAL_CLASSES
from exohost.ranking.observability_score import build_observability_score_record, clamp_score
from exohost.ranking.priority_score_contracts import (
    DEFAULT_HOST_SCORE_COLUMN,
    DEFAULT_PRIORITY_THRESHOLDS,
    DEFAULT_RANKING_WEIGHTS,
    PriorityScoreRecord,
    PriorityThresholds,
    RankingWeights,
)
from exohost.ranking.priority_score_rules import (
    assign_priority_label,
    build_priority_reason,
    compute_class_priority_score,
    compute_host_similarity_score,
    require_ranking_columns,
)
from exohost.ranking.priority_score_scalars import coerce_optional_float, is_missing_scalar


def build_priority_score_record(
    row: pd.Series,
    *,
    host_score_column: str = DEFAULT_HOST_SCORE_COLUMN,
    weights: RankingWeights = DEFAULT_RANKING_WEIGHTS,
    thresholds: PriorityThresholds = DEFAULT_PRIORITY_THRESHOLDS,
) -> PriorityScoreRecord:
    # Собираем explainable ranking-запись по одной строке.
    spec_class = str(row["spec_class"]).strip().upper()
    evolution_stage_value = row.get("evolution_stage")
    evolution_stage = (
        None
        if is_missing_scalar(evolution_stage_value)
        else str(evolution_stage_value)
    )
    class_priority_score = compute_class_priority_score(
        spec_class,
        evolution_stage,
        thresholds=thresholds,
    )
    host_similarity_score = compute_host_similarity_score(
        coerce_optional_float(row.get(host_score_column))
    )
    observability_record = build_observability_score_record(
        phot_g_mean_mag=coerce_optional_float(row.get("phot_g_mean_mag")),
        parallax=coerce_optional_float(row.get("parallax")),
        parallax_over_error=coerce_optional_float(row.get("parallax_over_error")),
        ruwe=coerce_optional_float(row.get("ruwe")),
        validation_factor=coerce_optional_float(row.get("validation_factor")),
    )

    priority_score = clamp_score(
        weights.class_priority_weight * class_priority_score
        + weights.host_similarity_weight * host_similarity_score
        + weights.observability_weight * observability_record.observability_score
    )

    if spec_class in LOW_PRIORITY_SPECTRAL_CLASSES:
        priority_score = min(priority_score, thresholds.low_priority_class_cap)

    return PriorityScoreRecord(
        source_id=str(row["source_id"]),
        spec_class=spec_class,
        evolution_stage=evolution_stage,
        class_priority_score=class_priority_score,
        host_similarity_score=host_similarity_score,
        brightness_score=observability_record.brightness_score,
        distance_score=observability_record.distance_score,
        astrometry_score=observability_record.astrometry_score,
        observability_score=observability_record.observability_score,
        brightness_available=observability_record.brightness_available,
        distance_available=observability_record.distance_available,
        astrometry_available=observability_record.astrometry_available,
        observability_evidence_count=observability_record.observability_evidence_count,
        priority_score=priority_score,
        priority_label=assign_priority_label(priority_score, thresholds=thresholds),
        priority_reason=build_priority_reason(
            spec_class=spec_class,
            evolution_stage=evolution_stage,
            host_similarity_score=host_similarity_score,
            observability_record=observability_record,
        ),
    )


def priority_record_to_dict(record: PriorityScoreRecord) -> dict[str, object]:
    # Преобразуем dataclass в табличный вид для DataFrame-контура.
    return asdict(record)


def build_priority_ranking_frame(
    df: pd.DataFrame,
    *,
    host_score_column: str = DEFAULT_HOST_SCORE_COLUMN,
    weights: RankingWeights = DEFAULT_RANKING_WEIGHTS,
    thresholds: PriorityThresholds = DEFAULT_PRIORITY_THRESHOLDS,
) -> pd.DataFrame:
    # Строим итоговую ranking-таблицу поверх model outputs и observability-признаков.
    require_ranking_columns(df, host_score_column=host_score_column)

    records = [
        build_priority_score_record(
            row,
            host_score_column=host_score_column,
            weights=weights,
            thresholds=thresholds,
        )
        for _, row in df.iterrows()
    ]
    ranking_frame = pd.DataFrame.from_records(
        [priority_record_to_dict(record) for record in records]
    )
    return ranking_frame.sort_values(
        ["priority_score", "host_similarity_score", "observability_score", "source_id"],
        ascending=[False, False, False, True],
        kind="mergesort",
        ignore_index=True,
    )


__all__ = [
    "build_priority_ranking_frame",
    "build_priority_score_record",
    "priority_record_to_dict",
]
