"""Контракты артефактов и JSON IO для Gaussian router.

Модуль описывает сериализуемую структуру router-модели и предоставляет
единый способ сохранить или загрузить production artifact с диска.
"""

from __future__ import annotations

import json
from typing import TypedDict


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


class RouterMeta(TypedDict):
    """Метаданные, записываемые рядом с router-артефактом."""

    model_version: str
    source_view: str
    shrink_alpha: float
    min_class_size: int
    score_mode: str
    prior_mode: str


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


def save_router_model(model: RouterModel, path: str) -> None:
    """Сохранить router-модель в JSON-файл.

    Побочные эффекты
    ----------------
    Перезаписывает файл по указанному пути и сохраняет модель в формате,
    совместимом с production artifact в `data/router_gaussian_params.json`.
    """
    with open(path, "w", encoding="utf-8") as file:
        json.dump(model, file, ensure_ascii=False, indent=2)


def load_router_model(path: str) -> RouterModel:
    """Загрузить router-модель из JSON-файла.

    Возвращаемая структура должна соответствовать контракту `RouterModel`
    и далее может быть напрямую использована в scoring-функциях пакета.
    """
    with open(path, encoding="utf-8") as file:
        return json.load(file)
