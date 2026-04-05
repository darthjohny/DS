# Файл `hierarchical_training_frame_coarse.py` слоя `features`.
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
    GAIA_ID_COARSE_TRAINING_CONTRACT,
)
from exohost.features.hierarchical_training_frame_common import (
    bool_to_evolution_stage,
    select_available_columns,
)
from exohost.features.hierarchical_training_frame_contracts import (
    COARSE_NUMERIC_COLUMNS,
)
from exohost.features.training_frame import (
    cast_numeric_columns,
    ensure_unique_source_id,
    require_columns,
    sort_by_source_id,
    validate_label_columns,
)


def prepare_gaia_id_coarse_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    # Нормализуем coarse training source до устойчивого train-frame.
    contract = GAIA_ID_COARSE_TRAINING_CONTRACT
    require_columns(df, contract.required_columns, frame_name="gaia id coarse training frame")

    result = select_available_columns(
        df,
        contract=contract,
    )
    result["spec_class"] = result["spec_class"].astype(str).str.strip().str.upper()
    result["evolution_stage"] = result["is_evolved"].map(bool_to_evolution_stage)
    result = cast_numeric_columns(result, COARSE_NUMERIC_COLUMNS)
    result = result.dropna(
        subset=(
            "source_id",
            "spec_class",
            "teff_gspphot",
            "logg_gspphot",
            "mh_gspphot",
            "bp_rp",
            "parallax",
            "parallax_over_error",
            "ruwe",
        )
    ).reset_index(drop=True)
    validate_label_columns(result, frame_name="gaia id coarse training frame")
    ensure_unique_source_id(result, frame_name="gaia id coarse training frame")
    return sort_by_source_id(result)


__all__ = ["prepare_gaia_id_coarse_training_frame"]
