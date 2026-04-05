# Файл `quality_gate_tuning.py` слоя `posthoc`.
#
# Этот файл отвечает только за:
# - post-hoc scoring, routing и final decision policy;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - соседние модули слоя `posthoc` и orchestration поверх них;
# - unit-тесты и более верхние слои проекта.

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class QualityGateTuningConfig:
    # Конфиг project-threshold override для review-слоя quality-gate.
    policy_name: str = "baseline"
    ruwe_unknown_threshold: float | None = 1.4
    parallax_snr_unknown_threshold: float | None = 5.0
    require_flame_for_pass: bool = True

    def __post_init__(self) -> None:
        if self.ruwe_unknown_threshold is not None and self.ruwe_unknown_threshold <= 0.0:
            raise ValueError("ruwe_unknown_threshold must be positive or None.")
        if (
            self.parallax_snr_unknown_threshold is not None
            and self.parallax_snr_unknown_threshold < 0.0
        ):
            raise ValueError(
                "parallax_snr_unknown_threshold must be non-negative or None."
            )


DEFAULT_QUALITY_GATE_TUNING_CONFIG = QualityGateTuningConfig()


def apply_quality_gate_tuning(
    df: pd.DataFrame,
    *,
    config: QualityGateTuningConfig = DEFAULT_QUALITY_GATE_TUNING_CONFIG,
) -> pd.DataFrame:
    # Пересчитываем review-часть quality-gate поверх уже загруженного frame.
    _require_series_column(df, "quality_state")

    if df.empty:
        return df.copy()

    result = df.copy()
    _ensure_quality_columns(result)

    original_quality_state = _require_series_column(result, "quality_state").astype("string")
    reject_mask = original_quality_state.eq("reject")
    non_reject_mask = ~reject_mask.fillna(False)

    result.loc[non_reject_mask, "quality_state"] = "pass"
    result.loc[non_reject_mask, "quality_reason"] = "pass"
    result.loc[non_reject_mask, "review_bucket"] = "pass"

    _apply_unknown_mask(
        result,
        mask=_resolve_threshold_mask(
            result,
            column_name="ruwe",
            threshold=config.ruwe_unknown_threshold,
            comparator="gt",
        ),
        reason="review_high_ruwe",
    )
    _apply_unknown_mask(
        result,
        mask=_resolve_threshold_mask(
            result,
            column_name="parallax_over_error",
            threshold=config.parallax_snr_unknown_threshold,
            comparator="lt",
        ),
        reason="review_low_parallax_snr",
    )
    if config.require_flame_for_pass:
        _apply_unknown_mask(
            result,
            mask=_resolve_signal_mask(
                result,
                signal_column_name="has_missing_flame_features",
                fallback_column_name="radius_flame",
                invert_fallback=False,
            ),
            reason="review_missing_radius_flame",
        )

    return result


def _ensure_quality_columns(df: pd.DataFrame) -> None:
    if "quality_reason" not in df.columns:
        df["quality_reason"] = pd.NA
    if "review_bucket" not in df.columns:
        df["review_bucket"] = pd.NA


def _apply_unknown_mask(
    df: pd.DataFrame,
    *,
    mask: pd.Series,
    reason: str,
) -> None:
    active_mask = mask & _require_series_column(df, "quality_state").astype("string").eq("pass")
    if not bool(active_mask.any()):
        return
    df.loc[active_mask, "quality_state"] = "unknown"
    df.loc[active_mask, "quality_reason"] = reason
    df.loc[active_mask, "review_bucket"] = reason


def _resolve_signal_mask(
    df: pd.DataFrame,
    *,
    signal_column_name: str,
    fallback_column_name: str,
    invert_fallback: bool,
) -> pd.Series:
    if signal_column_name in df.columns:
        return _require_series_column(df, signal_column_name).fillna(False).astype(bool)

    if fallback_column_name not in df.columns:
        return pd.Series(False, index=df.index, dtype=bool)

    fallback_series = _require_series_column(df, fallback_column_name)
    fallback_mask = fallback_series.isna()
    if fallback_series.dtype == bool:
        fallback_mask = fallback_series.fillna(False).astype(bool)
        if invert_fallback:
            fallback_mask = ~fallback_mask
    return fallback_mask


def _resolve_threshold_mask(
    df: pd.DataFrame,
    *,
    column_name: str,
    threshold: float | None,
    comparator: str,
) -> pd.Series:
    if threshold is None or column_name not in df.columns:
        return pd.Series(False, index=df.index, dtype=bool)

    numeric_series = _require_numeric_series(df, column_name)
    if comparator == "gt":
        return numeric_series > float(threshold)
    if comparator == "lt":
        return numeric_series < float(threshold)
    raise ValueError(f"Unsupported comparator: {comparator}")


def _require_series_column(df: pd.DataFrame, column_name: str) -> pd.Series:
    column = df.loc[:, column_name]
    if not isinstance(column, pd.Series):
        raise TypeError(f"{column_name} must resolve to a pandas Series.")
    return column


def _require_numeric_series(df: pd.DataFrame, column_name: str) -> pd.Series:
    numeric_column = pd.to_numeric(_require_series_column(df, column_name), errors="coerce")
    if not isinstance(numeric_column, pd.Series):
        raise TypeError(f"{column_name} must resolve to a numeric pandas Series.")
    return numeric_column


__all__ = [
    "DEFAULT_QUALITY_GATE_TUNING_CONFIG",
    "QualityGateTuningConfig",
    "apply_quality_gate_tuning",
]
