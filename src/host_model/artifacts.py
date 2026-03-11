"""Контракты артефактов, валидация и JSON IO для host-моделей.

Модуль описывает два семейства artifact-контрактов:

- legacy Gaussian artifact для расстояния до центроидов;
- contrastive artifact для `host-vs-field` scoring внутри routed classes.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, NotRequired, TypedDict, cast


class ScoreResult(TypedDict):
    """Типизированный результат legacy Gaussian scoring."""

    label: str
    d_mahal: float
    similarity: float


class ContrastiveScoreResult(TypedDict):
    """Типизированный результат contrastive `host-vs-field` scoring."""

    label: str
    host_log_likelihood: float
    field_log_likelihood: float
    host_log_lr: float
    host_posterior: float


class ContrastivePopulationParams(TypedDict):
    """Параметры одной популяции внутри contrastive класса.

    Описывает либо `host`, либо `field` Gaussian-компоненту для одной
    routed stellar class.
    """

    n: int
    mu: list[float]
    cov: list[list[float]]
    effective_cov: list[list[float]]
    inv_cov: list[list[float]]
    log_det_cov: float


class ContrastiveClassParams(TypedDict):
    """Пара Gaussian-компонент `host/field` для одного class label."""

    host: ContrastivePopulationParams
    field: ContrastivePopulationParams


class LegacyClassParams(TypedDict):
    """Параметры одной Gaussian-компоненты legacy host-модели."""

    n: int
    mu: list[float]
    cov: list[list[float]]
    inv_cov: list[list[float]]


class LegacyModelMeta(TypedDict):
    """Метаданные legacy artifact, сохранённые рядом с моделью."""

    logg_dwarf_min: float
    use_m_subclasses: bool
    shrink_alpha: float


class ContrastiveModelMeta(TypedDict):
    """Метаданные contrastive `host-vs-field` artifact."""

    model_version: str
    score_mode: str
    population_col: str
    use_m_subclasses: bool
    shrink_alpha: float
    min_population_size: int
    source_view: NotRequired[str]


class LegacyGaussianModel(TypedDict):
    """Полная сериализуемая структура legacy host artifact."""

    global_mu: list[float]
    global_sigma: list[float]
    classes: dict[str, LegacyClassParams]
    features: list[str]
    meta: LegacyModelMeta


class ContrastiveGaussianModel(TypedDict):
    """Полная сериализуемая структура contrastive host artifact."""

    global_mu: list[float]
    global_sigma: list[float]
    classes: dict[str, ContrastiveClassParams]
    features: list[str]
    meta: ContrastiveModelMeta


type HostModelArtifact = LegacyGaussianModel | ContrastiveGaussianModel


def is_contrastive_model(model: Mapping[str, Any]) -> bool:
    """Проверить, соответствует ли artifact контракту `host-vs-field`."""
    meta_raw = model.get("meta", {})
    if not isinstance(meta_raw, Mapping):
        return False
    meta = cast(Mapping[str, object], meta_raw)
    return str(meta.get("model_version", "")) == "gaussian_host_field_v1"


def require_legacy_scoring_model(model: Mapping[str, Any]) -> None:
    """Запретить применение legacy scorer к contrastive artifact.

    Функция используется как защитный барьер, чтобы старый scoring-path
    не запускался на production artifact нового формата.
    """
    if is_contrastive_model(model):
        raise ValueError(
            "Legacy Gaussian scoring does not support contrastive host-field "
            "models. Use score_df_contrastive() or score_one_contrastive()."
        )


def validate_host_model_artifact(model: Mapping[str, Any]) -> None:
    """Проверить, что artifact совместим с текущим production runtime.

    Текущий pipeline ожидает именно contrastive модель. Функция валидирует
    наличие обязательных `host/field` payload для каждого class label и
    выбрасывает ошибку, если в рантайм попал legacy artifact.
    """
    if not is_contrastive_model(model):
        raise ValueError(
            "Host model artifact is legacy and incompatible with the current "
            "pipeline. Rebuild data/model_gaussian_params.json with "
            "python src/model_gaussian.py --mode contrastive --view <host_field_view>."
        )

    classes_raw = model.get("classes", {})
    if not isinstance(classes_raw, dict) or not classes_raw:
        raise ValueError("Host model artifact has no contrastive classes.")
    classes = cast(Mapping[str, Any], classes_raw)

    broken_labels: list[str] = []
    for label, params_raw in classes.items():
        if not isinstance(params_raw, dict):
            broken_labels.append(str(label))
            continue
        params = cast(Mapping[str, Any], params_raw)
        if {"host", "field"} - set(params.keys()):
            broken_labels.append(str(label))
            continue
        for population_name in ("host", "field"):
            population_raw = params.get(population_name)
            if not isinstance(population_raw, dict):
                broken_labels.append(str(label))
                continue
            population = cast(Mapping[str, Any], population_raw)
            required = {
                "n",
                "mu",
                "cov",
                "effective_cov",
                "inv_cov",
                "log_det_cov",
            }
            if required - set(population.keys()):
                broken_labels.append(str(label))
                break

    if broken_labels:
        labels = ", ".join(sorted(set(broken_labels)))
        raise ValueError(
            "Host model artifact is missing required contrastive payload for "
            f"labels: {labels}"
        )


def require_contrastive_scoring_model(model: Mapping[str, Any]) -> None:
    """Требовать contrastive artifact перед запуском нового scoring-path."""
    validate_host_model_artifact(model)


def save_model(model: Mapping[str, Any], path: str) -> None:
    """Сохранить host-model artifact в JSON.

    Побочные эффекты
    ----------------
    Перезаписывает файл по указанному пути и сохраняет artifact в формате,
    совместимом с `data/model_gaussian_params.json`.
    """
    with open(path, "w", encoding="utf-8") as file:
        json.dump(model, file, ensure_ascii=False, indent=2)


def load_model(path: str) -> dict[str, Any]:
    """Загрузить host-model artifact из JSON без дополнительной валидации."""
    with open(path, encoding="utf-8") as file:
        return cast(dict[str, Any], json.load(file))


def load_contrastive_model(path: str) -> ContrastiveGaussianModel:
    """Загрузить и провалидировать contrastive artifact для production scoring."""
    model = load_model(path)
    validate_host_model_artifact(model)
    return cast(ContrastiveGaussianModel, model)


__all__ = [
    "ContrastiveClassParams",
    "ContrastiveGaussianModel",
    "ContrastiveModelMeta",
    "ContrastivePopulationParams",
    "ContrastiveScoreResult",
    "HostModelArtifact",
    "LegacyClassParams",
    "LegacyGaussianModel",
    "LegacyModelMeta",
    "ScoreResult",
    "is_contrastive_model",
    "load_contrastive_model",
    "load_model",
    "require_contrastive_scoring_model",
    "require_legacy_scoring_model",
    "save_model",
    "validate_host_model_artifact",
]
