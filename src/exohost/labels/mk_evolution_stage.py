# Файл `mk_evolution_stage.py` слоя `labels`.
#
# Этот файл отвечает только за:
# - правила семантики меток и производных label-helper;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - соседние модули слоя `labels` и orchestration поверх них;
# - unit-тесты и более верхние слои проекта.

from __future__ import annotations

from exohost.contracts.label_contract import EvolutionStage

DWARF_LUMINOSITY_CLASS_CODES: tuple[str, ...] = ("V", "VI", "VII")
EVOLVED_LUMINOSITY_CLASS_CODES: tuple[str, ...] = (
    "0",
    "IA+",
    "IA",
    "IAB",
    "IB",
    "I",
    "II",
    "III",
    "IV",
)


def normalize_luminosity_class_code(value: str) -> str:
    # Для coarse stage mapping достаточно trim + upper без усложнения.
    return value.strip().upper()


def map_luminosity_class_to_evolution_stage(
    luminosity_class: str | None,
) -> EvolutionStage | None:
    # Преобразуем MK luminosity class в legacy-compatible dwarf/evolved слой.
    if luminosity_class is None:
        return None

    normalized_value = normalize_luminosity_class_code(luminosity_class)
    if not normalized_value:
        return None
    if normalized_value in DWARF_LUMINOSITY_CLASS_CODES:
        return "dwarf"
    if normalized_value in EVOLVED_LUMINOSITY_CLASS_CODES:
        return "evolved"
    return None


def has_supported_evolution_stage_mapping(luminosity_class: str | None) -> bool:
    # Явно отделяем отсутствие coarse stage от неподдержанного/пустого кода.
    return map_luminosity_class_to_evolution_stage(luminosity_class) is not None
