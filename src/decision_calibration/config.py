"""Контракты конфигурации и загрузка JSON для decision calibration."""

from __future__ import annotations

import json
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from decision_calibration.constants import DEFAULT_TOP_N
from input_layer import DEFAULT_INPUT_RELATION


@dataclass(frozen=True)
class ClassPriorConfig:
    """Настройки soft prior по спектральным классам для калибровки."""

    k: float = 1.08
    g: float = 1.05
    m: float = 1.02
    f: float = 0.97

    @classmethod
    def from_mapping(
        cls,
        data: dict[str, Any] | None,
    ) -> ClassPriorConfig:
        """Собрать конфигурацию class prior из JSON-подобного mapping."""
        default = cls()
        if data is None:
            return default
        return cls(
            k=float(data.get("K", default.k)),
            g=float(data.get("G", default.g)),
            m=float(data.get("M", default.m)),
            f=float(data.get("F", default.f)),
        )


@dataclass(frozen=True)
class MetallicityConfig:
    """Пороговая схема metallicity factor для офлайн-калибровки."""

    low_threshold: float = -0.3
    solar_threshold: float = 0.2
    high_threshold: float = 0.5
    low_factor: float = 0.96
    neutral_factor: float = 1.00
    positive_factor: float = 1.03
    high_factor: float = 1.05

    @classmethod
    def from_mapping(
        cls,
        data: dict[str, Any] | None,
    ) -> MetallicityConfig:
        """Собрать конфигурацию metallicity factor из mapping."""
        default = cls()
        if data is None:
            return default
        return cls(
            low_threshold=float(
                data.get("low_threshold", default.low_threshold)
            ),
            solar_threshold=float(
                data.get("solar_threshold", default.solar_threshold)
            ),
            high_threshold=float(
                data.get("high_threshold", default.high_threshold)
            ),
            low_factor=float(data.get("low_factor", default.low_factor)),
            neutral_factor=float(
                data.get("neutral_factor", default.neutral_factor)
            ),
            positive_factor=float(
                data.get("positive_factor", default.positive_factor)
            ),
            high_factor=float(data.get("high_factor", default.high_factor)),
        )


@dataclass(frozen=True)
class DistanceConfig:
    """Пороговая схема distance factor в парсеках."""

    near_max_pc: float = 50.0
    moderate_max_pc: float = 100.0
    distant_max_pc: float = 200.0
    far_max_pc: float = 500.0
    near_factor: float = 1.00
    moderate_factor: float = 0.96
    distant_factor: float = 0.90
    far_factor: float = 0.80
    very_far_factor: float = 0.65
    invalid_factor: float = 0.70

    @classmethod
    def from_mapping(
        cls,
        data: dict[str, Any] | None,
    ) -> DistanceConfig:
        """Собрать конфигурацию distance factor из mapping."""
        default = cls()
        if data is None:
            return default
        return cls(
            near_max_pc=float(data.get("near_max_pc", default.near_max_pc)),
            moderate_max_pc=float(
                data.get("moderate_max_pc", default.moderate_max_pc)
            ),
            distant_max_pc=float(
                data.get("distant_max_pc", default.distant_max_pc)
            ),
            far_max_pc=float(data.get("far_max_pc", default.far_max_pc)),
            near_factor=float(data.get("near_factor", default.near_factor)),
            moderate_factor=float(
                data.get("moderate_factor", default.moderate_factor)
            ),
            distant_factor=float(
                data.get("distant_factor", default.distant_factor)
            ),
            far_factor=float(data.get("far_factor", default.far_factor)),
            very_far_factor=float(
                data.get("very_far_factor", default.very_far_factor)
            ),
            invalid_factor=float(
                data.get("invalid_factor", default.invalid_factor)
            ),
        )


@dataclass(frozen=True)
class RuweConfig:
    """Пороговая схема RUWE quality factor."""

    good_max: float = 1.1
    warning_max: float = 1.2
    alert_max: float = 1.4
    bad_max: float = 1.8
    good_factor: float = 1.00
    warning_factor: float = 0.98
    alert_factor: float = 0.93
    bad_factor: float = 0.80
    very_bad_factor: float = 0.60
    missing_factor: float = 0.90

    @classmethod
    def from_mapping(
        cls,
        data: dict[str, Any] | None,
    ) -> RuweConfig:
        """Собрать конфигурацию RUWE factor из mapping."""
        default = cls()
        if data is None:
            return default
        return cls(
            good_max=float(data.get("good_max", default.good_max)),
            warning_max=float(data.get("warning_max", default.warning_max)),
            alert_max=float(data.get("alert_max", default.alert_max)),
            bad_max=float(data.get("bad_max", default.bad_max)),
            good_factor=float(data.get("good_factor", default.good_factor)),
            warning_factor=float(
                data.get("warning_factor", default.warning_factor)
            ),
            alert_factor=float(
                data.get("alert_factor", default.alert_factor)
            ),
            bad_factor=float(data.get("bad_factor", default.bad_factor)),
            very_bad_factor=float(
                data.get("very_bad_factor", default.very_bad_factor)
            ),
            missing_factor=float(
                data.get("missing_factor", default.missing_factor)
            ),
        )


@dataclass(frozen=True)
class ParallaxPrecisionConfig:
    """Пороговая схема качества по `parallax_over_error`."""

    excellent_min: float = 20.0
    good_min: float = 10.0
    acceptable_min: float = 5.0
    weak_min: float = 3.0
    excellent_factor: float = 1.00
    good_factor: float = 0.97
    acceptable_factor: float = 0.90
    weak_factor: float = 0.75
    poor_factor: float = 0.55
    missing_factor: float = 0.85

    @classmethod
    def from_mapping(
        cls,
        data: dict[str, Any] | None,
    ) -> ParallaxPrecisionConfig:
        """Собрать конфигурацию precision factor из mapping."""
        default = cls()
        if data is None:
            return default
        return cls(
            excellent_min=float(
                data.get("excellent_min", default.excellent_min)
            ),
            good_min=float(data.get("good_min", default.good_min)),
            acceptable_min=float(
                data.get("acceptable_min", default.acceptable_min)
            ),
            weak_min=float(data.get("weak_min", default.weak_min)),
            excellent_factor=float(
                data.get("excellent_factor", default.excellent_factor)
            ),
            good_factor=float(data.get("good_factor", default.good_factor)),
            acceptable_factor=float(
                data.get("acceptable_factor", default.acceptable_factor)
            ),
            weak_factor=float(data.get("weak_factor", default.weak_factor)),
            poor_factor=float(data.get("poor_factor", default.poor_factor)),
            missing_factor=float(
                data.get("missing_factor", default.missing_factor)
            ),
        )


@dataclass(frozen=True)
class QualityConfig:
    """Сводная конфигурация quality factor.

    Объединяет подконфигурации для `ruwe` и `parallax_over_error`.
    """

    ruwe: RuweConfig = RuweConfig()
    parallax_precision: ParallaxPrecisionConfig = (
        ParallaxPrecisionConfig()
    )

    @classmethod
    def from_mapping(
        cls,
        data: dict[str, Any] | None,
    ) -> QualityConfig:
        """Собрать quality config из JSON-подобного mapping."""
        if data is None:
            return cls()
        return cls(
            ruwe=RuweConfig.from_mapping(data.get("ruwe")),
            parallax_precision=ParallaxPrecisionConfig.from_mapping(
                data.get("parallax_precision")
            ),
        )


@dataclass(frozen=True)
class CalibrationConfig:
    """Полная конфигурация офлайн-калибровки decision layer."""

    class_prior: ClassPriorConfig = ClassPriorConfig()
    metallicity: MetallicityConfig = MetallicityConfig()
    distance: DistanceConfig = DistanceConfig()
    quality: QualityConfig = QualityConfig()

    @classmethod
    def from_mapping(
        cls,
        data: dict[str, Any] | None,
    ) -> CalibrationConfig:
        """Собрать полную calibration config из mapping."""
        if data is None:
            return cls()
        return cls(
            class_prior=ClassPriorConfig.from_mapping(
                data.get("class_prior")
            ),
            metallicity=MetallicityConfig.from_mapping(
                data.get("metallicity")
            ),
            distance=DistanceConfig.from_mapping(data.get("distance")),
            quality=QualityConfig.from_mapping(data.get("quality")),
        )


def parse_args() -> Namespace:
    """Разобрать CLI-аргументы офлайн-калибратора."""
    parser = ArgumentParser(
        description="Калибровка decision layer без записи в боевые таблицы."
    )
    parser.add_argument(
        "--relation",
        default=DEFAULT_INPUT_RELATION,
        help="Relation со статусом READY в registry-таблице.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Опциональный лимит строк для preview-прогона.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help="Сколько лучших кандидатов сохранять в summary.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Путь к JSON-файлу с переопределением коэффициентов.",
    )
    parser.add_argument(
        "--iteration-note",
        default="",
        help="Короткая заметка о цели текущей итерации.",
    )
    return parser.parse_args()


def normalize_json_object(raw_obj: object) -> dict[str, Any]:
    """Нормализовать сырой JSON-объект в mapping со строковыми ключами."""
    if not isinstance(raw_obj, dict):
        raise ValueError("Calibration config must be a JSON object.")

    normalized: dict[str, Any] = {}
    for raw_key, raw_value in raw_obj.items():
        if not isinstance(raw_key, str):
            raise ValueError("Calibration config keys must be strings.")
        normalized[raw_key] = raw_value
    return normalized


def load_calibration_config(path: Path | None) -> CalibrationConfig:
    """Загрузить calibration config из JSON или вернуть baseline-настройки."""
    if path is None:
        return CalibrationConfig()
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    raw_any = json.loads(path.read_text(encoding="utf-8"))
    raw_mapping = normalize_json_object(raw_any)
    return CalibrationConfig.from_mapping(raw_mapping)


__all__ = [
    "CalibrationConfig",
    "ClassPriorConfig",
    "DistanceConfig",
    "MetallicityConfig",
    "ParallaxPrecisionConfig",
    "QualityConfig",
    "RuweConfig",
    "load_calibration_config",
    "normalize_json_object",
    "parse_args",
]
