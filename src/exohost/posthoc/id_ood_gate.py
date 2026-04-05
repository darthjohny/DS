# Файл `id_ood_gate.py` слоя `posthoc`.
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
from typing import Final, Literal, cast

import pandas as pd

from exohost.models.inference import build_probability_frame, require_feature_columns
from exohost.models.protocol import ClassifierModel

IdOodDecision = Literal["in_domain", "candidate_ood", "ood"]
IdOodLabel = Literal["id", "ood"]

ID_OOD_IN_DOMAIN_STATE: Final[IdOodDecision] = "in_domain"
ID_OOD_CANDIDATE_OOD_STATE: Final[IdOodDecision] = "candidate_ood"
ID_OOD_OOD_STATE: Final[IdOodDecision] = "ood"

ID_OOD_ID_LABEL: Final[IdOodLabel] = "id"
ID_OOD_OOD_LABEL: Final[IdOodLabel] = "ood"


@dataclass(frozen=True, slots=True)
class IdOodThresholdPolicy:
    # Версионированный threshold policy для ID/OOD gate.
    threshold_name: str
    threshold_value: float
    threshold_metric: str
    threshold_fit_scope: str
    threshold_policy_version: str
    candidate_ood_threshold: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.threshold_value <= 1.0:
            raise ValueError("IdOodThresholdPolicy.threshold_value must be between 0 and 1.")
        if self.candidate_ood_threshold is None:
            return
        if not 0.0 <= self.candidate_ood_threshold <= self.threshold_value:
            raise ValueError(
                "IdOodThresholdPolicy.candidate_ood_threshold must satisfy "
                "0 <= candidate_ood_threshold <= threshold_value."
            )


def build_id_ood_threshold_policy(
    *,
    tuned_threshold: float,
    threshold_policy_version: str,
    candidate_ood_threshold: float | None = None,
    threshold_name: str = "tuned_threshold",
    threshold_metric: str = "balanced_accuracy",
    threshold_fit_scope: str = "internal_cv",
) -> IdOodThresholdPolicy:
    # Собираем explicit threshold policy из fitted tuner output и project metadata.
    return IdOodThresholdPolicy(
        threshold_name=threshold_name,
        threshold_value=float(tuned_threshold),
        threshold_metric=threshold_metric,
        threshold_fit_scope=threshold_fit_scope,
        threshold_policy_version=threshold_policy_version,
        candidate_ood_threshold=candidate_ood_threshold,
    )


def decide_id_ood_state(
    ood_probability: float,
    *,
    policy: IdOodThresholdPolicy,
) -> IdOodDecision:
    # Преобразуем calibrated OOD probability в gate-state без скрытых magic numbers.
    if ood_probability >= policy.threshold_value:
        return ID_OOD_OOD_STATE
    if (
        policy.candidate_ood_threshold is not None
        and ood_probability >= policy.candidate_ood_threshold
    ):
        return ID_OOD_CANDIDATE_OOD_STATE
    return ID_OOD_IN_DOMAIN_STATE


def build_id_ood_gate_scored_frame(
    df: pd.DataFrame,
    *,
    estimator: object,
    feature_columns: tuple[str, ...],
    policy: IdOodThresholdPolicy,
) -> pd.DataFrame:
    # Добавляем к входному frame calibrated probabilities и threshold-based decisions.
    require_feature_columns(df, feature_columns=feature_columns)
    model = cast(ClassifierModel, estimator)
    feature_frame = df.loc[:, list(feature_columns)].copy()
    probability_frame = build_probability_frame(model, feature_frame)
    if probability_frame is None:
        raise ValueError("ID/OOD gate estimator must expose predict_proba and classes_.")

    _validate_binary_probability_frame(probability_frame)
    result = df.copy()
    result["id_probability"] = probability_frame[ID_OOD_ID_LABEL].astype(float)
    result["ood_probability"] = probability_frame[ID_OOD_OOD_LABEL].astype(float)
    result["ood_threshold"] = float(policy.threshold_value)
    result["candidate_ood_threshold"] = (
        float(policy.candidate_ood_threshold)
        if policy.candidate_ood_threshold is not None
        else float("nan")
    )
    result["ood_threshold_name"] = policy.threshold_name
    result["ood_threshold_metric"] = policy.threshold_metric
    result["ood_threshold_fit_scope"] = policy.threshold_fit_scope
    result["ood_threshold_policy_version"] = policy.threshold_policy_version
    ood_probability_series = _coerce_numeric_series(result, "ood_probability")
    result["predicted_domain_target"] = ood_probability_series.ge(policy.threshold_value).map(
        {
            True: ID_OOD_OOD_LABEL,
            False: ID_OOD_ID_LABEL,
        }
    )
    result["ood_decision"] = pd.Series(
        [
            decide_id_ood_state(ood_probability, policy=policy)
            for ood_probability in ood_probability_series.tolist()
        ],
        index=result.index,
    )
    return result


def build_id_ood_metrics_probability_frame(df: pd.DataFrame) -> pd.DataFrame:
    # Возвращаем probability frame в label-совместимом виде для metrics layer.
    return pd.DataFrame(
        {
            ID_OOD_ID_LABEL: df["id_probability"].astype(float),
            ID_OOD_OOD_LABEL: df["ood_probability"].astype(float),
        },
        index=df.index,
    )


def _validate_binary_probability_frame(probability_frame: pd.DataFrame) -> None:
    missing_columns = [
        column_name
        for column_name in (ID_OOD_ID_LABEL, ID_OOD_OOD_LABEL)
        if column_name not in probability_frame.columns
    ]
    if missing_columns:
        missing_columns_sql = ", ".join(missing_columns)
        raise ValueError(
            "ID/OOD gate probability frame is missing binary class columns: "
            f"{missing_columns_sql}"
        )


def _coerce_numeric_series(df: pd.DataFrame, column_name: str) -> pd.Series:
    column = df.loc[:, column_name]
    if not isinstance(column, pd.Series):
        raise TypeError(f"{column_name} must resolve to a pandas Series.")
    numeric_column = pd.to_numeric(column, errors="raise")
    if not isinstance(numeric_column, pd.Series):
        raise TypeError(f"{column_name} must resolve to a numeric pandas Series.")
    return numeric_column.astype(float)
