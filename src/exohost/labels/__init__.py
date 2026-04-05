# Пакетный файл слоя `labels`.
#
# Этот файл отвечает только за:
# - пакетный marker/export-layer для домена `labels`;
# - короткую навигацию по модульной структуре правила семантики меток и производных label-helper.
#
# Следующий слой:
# - конкретные модули этого пакета;
# - слои выше, которые импортируют этот пакет дальше.

from exohost.labels.mk_evolution_stage import (
    DWARF_LUMINOSITY_CLASS_CODES,
    EVOLVED_LUMINOSITY_CLASS_CODES,
    has_supported_evolution_stage_mapping,
    map_luminosity_class_to_evolution_stage,
    normalize_luminosity_class_code,
)

__all__ = [
    "DWARF_LUMINOSITY_CLASS_CODES",
    "EVOLVED_LUMINOSITY_CLASS_CODES",
    "has_supported_evolution_stage_mapping",
    "map_luminosity_class_to_evolution_stage",
    "normalize_luminosity_class_code",
]
