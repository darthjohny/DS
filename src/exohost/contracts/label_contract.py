# Файл `label_contract.py` слоя `contracts`.
#
# Этот файл отвечает только за:
# - контракты колонок, датасетов и ролей правил;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - соседние модули слоя `contracts` и orchestration поверх них;
# - unit-тесты и более верхние слои проекта.

from __future__ import annotations

import re
from typing import Literal

type SpectralClass = Literal["O", "B", "A", "F", "G", "K", "M"]
type EvolutionStage = Literal["dwarf", "evolved"]
type HostLabel = Literal["host", "field"]

SPECTRAL_CLASSES: tuple[SpectralClass, ...] = ("O", "B", "A", "F", "G", "K", "M")
EVOLUTION_STAGES: tuple[EvolutionStage, ...] = ("dwarf", "evolved")
HOST_LABELS: tuple[HostLabel, ...] = ("host", "field")
HOST_FIELD_TARGET_COLUMN = "host_label"

# На первой волне эти классы остаются в упрощенной low-priority ветке.
LOW_PRIORITY_SPECTRAL_CLASSES: tuple[SpectralClass, ...] = ("O", "B", "A")

# Основные целевые классы для host-priority контура.
TARGET_SPECTRAL_CLASSES: tuple[SpectralClass, ...] = ("F", "G", "K", "M")

SPECTRAL_SUBCLASS_PATTERN = re.compile(r"^[OBAFGKM][0-9]$")
EVOLUTION_STAGE_ALIASES: dict[str, EvolutionStage] = {
    "dwarf": "dwarf",
    "evolved": "evolved",
    "subgiant": "evolved",
    "giant": "evolved",
}
SOURCE_EVOLUTION_STAGES: tuple[str, ...] = tuple(EVOLUTION_STAGE_ALIASES)


def is_supported_spectral_class(value: str) -> bool:
    # Проверяем, что класс входит в зафиксированный доменный контракт.
    return value in SPECTRAL_CLASSES


def is_supported_evolution_stage(value: str) -> bool:
    # Проверяем, что стадия эволюции относится к поддерживаемой схеме V2.
    return normalize_evolution_stage(value) in EVOLUTION_STAGES


def normalize_evolution_stage(value: str) -> str:
    # Приводим стадии источников к канонической схеме dwarf/evolved.
    normalized_value = value.strip().lower()
    return EVOLUTION_STAGE_ALIASES.get(normalized_value, normalized_value)


def normalize_spectral_subclass(value: str) -> str:
    # Приводим подкласс к каноническому виду без лишних пробелов и регистра.
    return value.strip().upper()


def is_valid_spectral_subclass(value: str) -> bool:
    # Валидируем подкласс по простой и читаемой схеме вида G2, K7, M4.
    normalized_value = normalize_spectral_subclass(value)
    return bool(SPECTRAL_SUBCLASS_PATTERN.fullmatch(normalized_value))
