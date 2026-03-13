"""Ветвление runtime-потока для production ranking pipeline.

Модуль отвечает только за маршрутизацию router output по operational
веткам и не занимается расчётом factors или host-scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from priority_pipeline.constants import (
    EVOLVED_STAR_REASON,
    FILTERED_OUT_REASON,
    HOT_STAR_REASON,
    MKGF_CLASSES,
)
from router_model.labels import (
    UNKNOWN_EVOLUTION_STAGE,
    UNKNOWN_ROUTER_LABEL,
    UNKNOWN_SPEC_CLASS,
)


@dataclass(frozen=True)
class RouterBranchFrames:
    """Три operational-ветки после router scoring."""

    host_df: pd.DataFrame
    low_known_df: pd.DataFrame
    unknown_df: pd.DataFrame


def is_unknown_router_output(
    spec_class: Any,
    evolution_stage: Any,
    router_label: Any,
) -> bool:
    """Проверить, что router output соответствует canonical `UNKNOWN`."""
    spec = str(spec_class).strip().upper()
    stage = str(evolution_stage).strip().lower()
    label = str(router_label).strip().upper()
    return (
        spec == UNKNOWN_SPEC_CLASS
        or stage == UNKNOWN_EVOLUTION_STAGE
        or label == UNKNOWN_ROUTER_LABEL
    )


def is_host_candidate(
    spec_class: Any,
    evolution_stage: Any,
) -> bool:
    """Проверить, что объект может идти в host-scoring ветку."""
    return (
        str(spec_class).strip().upper() in MKGF_CLASSES
        and str(evolution_stage).strip().lower() == "dwarf"
    )


def known_low_reason_code(spec_class: Any, evolution_stage: Any) -> str:
    """Вернуть reason-code для known non-host ветки."""
    spec = str(spec_class).strip().upper()
    stage = str(evolution_stage).strip().lower()
    if spec in {"A", "B", "O"}:
        return HOT_STAR_REASON
    if stage == "evolved":
        return EVOLVED_STAR_REASON
    return FILTERED_OUT_REASON


def split_router_branches(df_router: pd.DataFrame) -> RouterBranchFrames:
    """Разделить router output на `host`, `low_known` и `unknown`."""
    if df_router.empty:
        empty = df_router.copy()
        return RouterBranchFrames(
            host_df=empty.copy(),
            low_known_df=empty.copy(),
            unknown_df=empty.copy(),
        )

    if "router_label" in df_router.columns:
        router_label_series = df_router["router_label"]
    else:
        router_label_series = pd.Series(
            [None] * len(df_router),
            index=df_router.index,
            dtype=object,
        )

    unknown_mask = [
        is_unknown_router_output(spec_class, evolution_stage, router_label)
        for spec_class, evolution_stage, router_label in zip(
            df_router["predicted_spec_class"],
            df_router["predicted_evolution_stage"],
            router_label_series,
            strict=True,
        )
    ]
    host_mask = [
        is_host_candidate(spec_class, evolution_stage)
        for spec_class, evolution_stage in df_router[
            ["predicted_spec_class", "predicted_evolution_stage"]
        ].itertuples(index=False, name=None)
    ]

    unknown_series = pd.Series(unknown_mask, index=df_router.index)
    host_series = pd.Series(host_mask, index=df_router.index) & ~unknown_series
    low_known_series = ~unknown_series & ~host_series

    return RouterBranchFrames(
        host_df=df_router.loc[host_series].copy(),
        low_known_df=df_router.loc[low_known_series].copy(),
        unknown_df=df_router.loc[unknown_series].copy(),
    )


def split_branches(df_router: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Совместимый wrapper: вернуть host и объединённую low-ветку.

    Важно: combined low-frame сохраняет исходный порядок router output,
    чтобы старые callsite и notebook-артефакты не меняли семантику после
    появления отдельной unknown-ветки.
    """
    branches = split_router_branches(df_router)
    low_mask = ~df_router.index.isin(branches.host_df.index)
    low_df = df_router.loc[low_mask].copy()
    return branches.host_df, low_df


__all__ = [
    "RouterBranchFrames",
    "is_host_candidate",
    "is_unknown_router_output",
    "known_low_reason_code",
    "split_branches",
    "split_router_branches",
]
