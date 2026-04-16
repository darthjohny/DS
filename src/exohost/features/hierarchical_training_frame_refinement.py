# Файл `hierarchical_training_frame_refinement.py` слоя `features`.
#
# Этот файл отвечает только за:
# - подготовку признаков и training frame-слой;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - соседние модули слоя `features` и orchestration поверх них;
# - unit-тесты и более верхние слои проекта.

from __future__ import annotations

import pandas as pd

from exohost.contracts.hierarchical_dataset_contracts import (
    GAIA_MK_REFINEMENT_TRAINING_CONTRACT,
)
from exohost.features.hierarchical_training_frame_common import (
    filter_minimum_label_support,
    normalize_optional_text,
    normalize_refinement_subclass,
    select_available_columns,
)
from exohost.features.hierarchical_training_frame_contracts import (
    MIN_REFINEMENT_SUBCLASS_SUPPORT,
    REFINEMENT_NUMERIC_COLUMNS,
)
from exohost.features.training_frame import (
    cast_numeric_columns,
    ensure_unique_source_id,
    require_columns,
    sort_by_source_id,
    validate_label_columns,
)
from exohost.labels.mk_evolution_stage import map_luminosity_class_to_evolution_stage


def prepare_gaia_mk_refinement_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    # Нормализуем refinement source до train-frame с каноническими label-полями.
    contract = GAIA_MK_REFINEMENT_TRAINING_CONTRACT
    require_columns(
        df,
        contract.required_columns,
        frame_name="gaia mk refinement training frame",
    )

    result = select_available_columns(
        df,
        contract=contract,
    )
    result = result.rename(
        columns={
            "spectral_class": "spec_class",
            "spectral_subclass": "spec_subclass",
        }
    )
    # Coarse-класс и подкласс нормализуем отдельно: coarse остается буквенной семьей,
    # а подкласс приводится к виду, пригодному для family-specific обучения.
    result["spec_class"] = result["spec_class"].astype(str).str.strip().str.upper()
    result["spec_subclass"] = result.apply(
        lambda row: normalize_refinement_subclass(
            row["spec_class"],
            row["spec_subclass"],
        ),
        axis=1,
    )
    if "luminosity_class" in result.columns:
        result["luminosity_class"] = result["luminosity_class"].map(
            normalize_optional_text
        )
        # Эволюционную стадию не тащим из сырого текста напрямую.
        # Вместо этого пересчитываем ее из luminosity class, чтобы разметка была единообразной.
        result["evolution_stage"] = result["luminosity_class"].map(
            lambda value: map_luminosity_class_to_evolution_stage(
                str(value) if pd.notna(value) else None
            )
        )
    else:
        result["evolution_stage"] = pd.NA

    result["radius_gspphot"] = result["radius_flame"]
    result = cast_numeric_columns(result, REFINEMENT_NUMERIC_COLUMNS)
    # Для refinement оставляем только строки с полным набором ключевых числовых признаков.
    # Иначе family-модели начнут учиться на неоднородном и частично пустом наборе.
    result = result.dropna(
        subset=(
            "source_id",
            "spec_class",
            "spec_subclass",
            "teff_gspphot",
            "logg_gspphot",
            "mh_gspphot",
            "bp_rp",
            "parallax",
            "parallax_over_error",
            "ruwe",
            "radius_flame",
        )
    ).reset_index(drop=True)
    validate_label_columns(result, frame_name="gaia mk refinement training frame")
    # Минимальная поддержка подкласса защищает модель от единичных редких меток,
    # которые дают красивую физику, но нестабильны для обучения.
    result = filter_minimum_label_support(
        result,
        target_column="spec_subclass",
        min_count=MIN_REFINEMENT_SUBCLASS_SUPPORT,
    )
    ensure_unique_source_id(result, frame_name="gaia mk refinement training frame")
    return sort_by_source_id(result)


__all__ = ["prepare_gaia_mk_refinement_training_frame"]
