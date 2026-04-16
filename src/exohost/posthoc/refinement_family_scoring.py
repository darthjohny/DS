# Файл `refinement_family_scoring.py` слоя `posthoc`.
#
# Этот файл отвечает только за:
# - post-hoc scoring, routing и final decision policy;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - соседние модули слоя `posthoc` и orchestration поверх них;
# - unit-тесты и более верхние слои проекта.

from __future__ import annotations

from typing import cast

import pandas as pd

from exohost.contracts.refinement_family_dataset_contracts import (
    validate_refinement_family_class,
)
from exohost.models.inference import build_probability_frame, require_feature_columns
from exohost.models.protocol import ClassifierModel
from exohost.posthoc.probability_summary import build_probability_summary_frame


def build_refinement_family_scored_frame(
    df: pd.DataFrame,
    *,
    estimator: object,
    feature_columns: tuple[str, ...],
    family_name: str,
    model_name: str | None = None,
) -> pd.DataFrame:
    # Строим compact refinement-family output для final routing.
    _require_source_id_column(df)
    normalized_family = validate_refinement_family_class(family_name)
    if df.empty:
        return _build_empty_refinement_frame(model_name=model_name, family_name=normalized_family)

    require_feature_columns(df, feature_columns=feature_columns)
    # На вход family-модели должен прийти уже отфильтрованный срез одной coarse-семьи.
    # Иначе даже хороший классификатор начнет смешивать ярлыки соседних семейств.
    _validate_input_family(df, family_name=normalized_family)

    model = cast(ClassifierModel, estimator)
    probability_frame = build_probability_frame(
        model,
        df.loc[:, list(feature_columns)].copy(),
    )
    if probability_frame is None:
        raise ValueError("Refinement estimator must expose predict_proba and classes_.")

    summary_frame = build_probability_summary_frame(
        probability_frame,
        prediction_column_name="refinement_predicted_label",
        confidence_column_name="refinement_probability_max",
        margin_column_name="refinement_probability_margin",
    )
    # Часть моделей может вернуть только число подкласса без буквенной семьи.
    # Здесь приводим прогноз к единому виду вроде `G2` или `K7`.
    summary_frame["refinement_predicted_label"] = summary_frame[
        "refinement_predicted_label"
    ].map(
        lambda value: _normalize_predicted_family_label(
            value,
            family_name=normalized_family,
        )
    )
    _validate_predicted_family(summary_frame, family_name=normalized_family)

    result = df.loc[:, ["source_id"]].copy().join(summary_frame)
    result["refinement_family_name"] = normalized_family
    if model_name is not None:
        result["refinement_model_name"] = model_name
    return result


def _require_source_id_column(df: pd.DataFrame) -> None:
    if "source_id" not in df.columns:
        raise ValueError(
            "Refinement family scoring frame is missing required column: source_id"
        )


def _validate_input_family(df: pd.DataFrame, *, family_name: str) -> None:
    if "spectral_class" not in df.columns:
        return

    # Проверка нужна для раннего отлова ошибок handoff между coarse и refinement.
    # Если сюда попали строки нескольких семейств, виноват уже upstream routing.
    normalized_classes = {
        str(value).strip().upper()
        for value in df["spectral_class"].dropna().tolist()
        if str(value).strip()
    }
    if not normalized_classes:
        return
    if normalized_classes != {family_name}:
        classes_sql = ", ".join(sorted(normalized_classes))
        raise ValueError(
            "Refinement family scoring frame contains rows outside requested family: "
            f"{classes_sql}"
        )


def _validate_predicted_family(summary_frame: pd.DataFrame, *, family_name: str) -> None:
    # Даже после нормализации модель не должна выдавать метки чужого семейства.
    # Это защищает итоговый pipeline от скрытого разъезда классов в артефактах.
    invalid_labels = [
        str(value)
        for value in summary_frame["refinement_predicted_label"].dropna().tolist()
        if not str(value).strip().upper().startswith(family_name)
    ]
    if invalid_labels:
        sample_sql = ", ".join(sorted(set(invalid_labels))[:5])
        raise ValueError(
            "Refinement family estimator predicted labels outside requested family: "
            f"{sample_sql}"
        )


def _normalize_predicted_family_label(
    value: object,
    *,
    family_name: str,
) -> object:
    # Нормализуем разные формы выхода модели к одному контракту.
    # Пустые значения сохраняем как пропуск, а числовой подкласс привязываем к семье.
    if value is None or value is pd.NA:
        return pd.NA
    if isinstance(value, float) and pd.isna(value):
        return pd.NA

    normalized_value = str(value).strip().upper()
    if not normalized_value:
        return pd.NA
    if normalized_value.isdigit():
        return f"{family_name}{normalized_value}"
    return normalized_value


def _build_empty_refinement_frame(
    *,
    model_name: str | None,
    family_name: str,
) -> pd.DataFrame:
    # Пустой результат тоже должен сохранять схему scoring-слоя, чтобы
    # final decision мог безопасно объединять его с другими кадрами.
    result = pd.DataFrame(
        {
            "source_id": pd.Series(dtype="int64"),
            "refinement_predicted_label": pd.Series(dtype="string"),
            "refinement_probability_max": pd.Series(dtype="float64"),
            "refinement_probability_margin": pd.Series(dtype="float64"),
            "refinement_family_name": pd.Series(dtype="string"),
        }
    )
    result["refinement_family_name"] = family_name
    if model_name is not None:
        result["refinement_model_name"] = pd.Series(dtype="string")
    return result
