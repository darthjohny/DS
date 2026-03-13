"""Контракты артефактов и JSON IO для Gaussian router.

Модуль описывает сериализуемую структуру router-модели и предоставляет
единый способ сохранить или загрузить production artifact с диска.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, TypedDict, cast

DEFAULT_OOD_POLICY_VERSION = "posterior_reject_v1"
DISABLED_OOD_POLICY_VERSION = "disabled"


class RouterClassParams(TypedDict):
    """Параметры одной Gaussian-компоненты router-модели.

    Соответствуют одному физическому классу вида `A_dwarf` или
    `K_evolved` и содержат центр, ковариации и численные поля,
    необходимые для posterior-aware scoring.
    """

    n: int
    spec_class: str
    evolution_stage: str
    mu: list[float]
    cov: list[list[float]]
    effective_cov: list[list[float]]
    inv_cov: list[list[float]]
    log_det_cov: float


class RouterOODMeta(TypedDict, total=False):
    """Дополнительные OOD-поля metadata для open-set режима router."""

    allow_unknown: bool
    ood_policy_version: str
    min_router_log_posterior: float | None
    min_posterior_margin: float | None
    min_router_similarity: float | None


class RouterMeta(TypedDict):
    """Полный metadata contract, сериализуемый рядом с router-артефактом.

    В отличие от частичного `RouterOODMeta`, этот TypedDict описывает уже
    нормализованный payload, где OOD-поля присутствуют всегда, даже если
    open-set режим выключен.
    """

    model_version: str
    source_view: str
    shrink_alpha: float
    min_class_size: int
    score_mode: str
    prior_mode: str
    allow_unknown: bool
    ood_policy_version: str
    min_router_log_posterior: float | None
    min_posterior_margin: float | None
    min_router_similarity: float | None


class RouterModel(TypedDict):
    """Полная сериализуемая структура production router-модели."""

    global_mu: list[float]
    global_sigma: list[float]
    classes: dict[str, RouterClassParams]
    features: list[str]
    meta: RouterMeta


class RouterScoreResult(TypedDict):
    """Типизированный результат scoring одной строки через router."""

    predicted_spec_class: str
    predicted_evolution_stage: str
    router_label: str
    d_mahal_router: float
    router_similarity: float
    router_log_likelihood: float
    router_log_posterior: float
    second_best_label: str
    margin: float
    posterior_margin: float
    model_version: str


def _optional_float(value: Any) -> float | None:
    """Преобразовать произвольное значение metadata в `float | None`."""
    if value is None:
        return None
    return float(value)


def build_router_meta(
    *,
    model_version: str,
    source_view: str,
    shrink_alpha: float,
    min_class_size: int,
    score_mode: str,
    prior_mode: str,
    allow_unknown: bool = False,
    ood_policy_version: str | None = None,
    min_router_log_posterior: float | None = None,
    min_posterior_margin: float | None = None,
    min_router_similarity: float | None = None,
) -> RouterMeta:
    """Собрать канонический metadata payload для router artifact.

    Даже при выключенном open-set режиме metadata остаётся структурно
    полной и сериализуется предсказуемо, чтобы JSON artifact был
    самодокументируемым и типово стабильным.
    """
    resolved_policy_version = ood_policy_version
    if resolved_policy_version is None:
        resolved_policy_version = (
            DEFAULT_OOD_POLICY_VERSION
            if allow_unknown
            else DISABLED_OOD_POLICY_VERSION
        )

    return {
        "model_version": model_version,
        "source_view": source_view,
        "shrink_alpha": float(shrink_alpha),
        "min_class_size": int(min_class_size),
        "score_mode": score_mode,
        "prior_mode": prior_mode,
        "allow_unknown": bool(allow_unknown),
        "ood_policy_version": str(resolved_policy_version),
        "min_router_log_posterior": _optional_float(
            min_router_log_posterior
        ),
        "min_posterior_margin": _optional_float(min_posterior_margin),
        "min_router_similarity": _optional_float(min_router_similarity),
    }


def normalize_router_meta(meta: Mapping[str, Any]) -> RouterMeta:
    """Нормализовать metadata к актуальному router contract.

    Для legacy-artifact, где OOD-поля ещё не были сохранены, функция
    достраивает их с безопасными значениями по умолчанию.
    """
    allow_unknown = bool(meta.get("allow_unknown", False))
    policy_version_raw = meta.get("ood_policy_version")
    policy_version = (
        str(policy_version_raw)
        if policy_version_raw is not None
        else None
    )
    return build_router_meta(
        model_version=str(meta["model_version"]),
        source_view=str(meta["source_view"]),
        shrink_alpha=float(meta["shrink_alpha"]),
        min_class_size=int(meta["min_class_size"]),
        score_mode=str(meta["score_mode"]),
        prior_mode=str(meta["prior_mode"]),
        allow_unknown=allow_unknown,
        ood_policy_version=policy_version,
        min_router_log_posterior=_optional_float(
            meta.get("min_router_log_posterior")
        ),
        min_posterior_margin=_optional_float(meta.get("min_posterior_margin")),
        min_router_similarity=_optional_float(meta.get("min_router_similarity")),
    )


def normalize_router_model(model: Mapping[str, Any]) -> RouterModel:
    """Нормализовать загруженный artifact к актуальному контракту."""
    normalized = dict(model)
    meta_raw = cast(Mapping[str, Any], model["meta"])
    normalized["meta"] = normalize_router_meta(meta_raw)
    return cast(RouterModel, normalized)


def save_router_model(model: RouterModel, path: str) -> None:
    """Сохранить router-модель в JSON-файл.

    Побочные эффекты
    ----------------
    Перезаписывает файл по указанному пути и сохраняет модель в формате,
    совместимом с production artifact в `data/router_gaussian_params.json`.
    """
    normalized = normalize_router_model(model)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(normalized, file, ensure_ascii=False, indent=2)


def load_router_model(path: str) -> RouterModel:
    """Загрузить router-модель из JSON-файла.

    Возвращаемая структура должна соответствовать контракту `RouterModel`
    и далее может быть напрямую использована в scoring-функциях пакета.
    """
    with open(path, encoding="utf-8") as file:
        loaded = json.load(file)
    return normalize_router_model(loaded)


__all__ = [
    "DEFAULT_OOD_POLICY_VERSION",
    "DISABLED_OOD_POLICY_VERSION",
    "RouterClassParams",
    "RouterMeta",
    "RouterModel",
    "RouterOODMeta",
    "RouterScoreResult",
    "build_router_meta",
    "load_router_model",
    "normalize_router_meta",
    "normalize_router_model",
    "save_router_model",
]
