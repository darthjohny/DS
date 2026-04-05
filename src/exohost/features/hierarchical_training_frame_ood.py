# Файл `hierarchical_training_frame_ood.py` слоя `features`.
#
# Этот файл отвечает только за:
# - подготовку признаков и training frame-слой;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - соседние модули слоя `features` и orchestration поверх них;
# - unit-тесты и более верхние слои проекта.

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from exohost.contracts.hierarchical_dataset_contracts import (
    GAIA_ID_OOD_TRAINING_CONTRACT,
)
from exohost.features.hierarchical_training_frame_common import (
    max_numeric_value,
    normalize_domain_target,
    normalize_record_mapping,
    select_available_columns,
    validate_domain_target_values,
)
from exohost.features.hierarchical_training_frame_contracts import (
    ID_OOD_NUMERIC_COLUMNS,
)
from exohost.features.training_frame import (
    cast_numeric_columns,
    ensure_unique_source_id,
    require_columns,
    sort_by_source_id,
)


def build_multi_ood_row(group: pd.DataFrame) -> dict[str, Any]:
    # Для binary ID/OOD training схлопываем multi-membership,
    # но сохраняем признак overlap и максимальные selector-score.
    domain_targets = {
        normalize_domain_target(value)
        for value in group["domain_target"].dropna().tolist()
    }
    if len(domain_targets) > 1:
        sample = ", ".join(sorted(domain_targets))
        raise ValueError(
            "ID/OOD training frame contains conflicting domain_target values "
            f"for one source_id: {sample}"
        )

    first_row = normalize_record_mapping(group.iloc[0].to_dict())
    ood_groups = sorted(
        {
            str(value)
            for value in group["ood_group"].dropna().tolist()
            if str(value).strip()
        }
    )
    selector_score_1 = max_numeric_value(group.loc[:, "selector_score_1"])
    selector_score_2 = max_numeric_value(group.loc[:, "selector_score_2"])
    membership_count = max_numeric_value(group.loc[:, "ood_membership_count"])

    first_row["ood_group"] = "multi_ood" if len(ood_groups) > 1 else first_row["ood_group"]
    first_row["ood_group_members"] = ",".join(ood_groups) if ood_groups else pd.NA
    first_row["ood_membership_count"] = (
        int(membership_count) if membership_count is not None else len(ood_groups)
    )
    first_row["has_multi_ood_membership"] = True
    first_row["selector_score_1"] = (
        float(selector_score_1) if selector_score_1 is not None else float("nan")
    )
    first_row["selector_score_2"] = (
        float(selector_score_2) if selector_score_2 is not None else float("nan")
    )
    return first_row


def collapse_multi_membership_ood_rows(df: pd.DataFrame) -> pd.DataFrame:
    # Убираем duplicate source_id для binary ID/OOD training,
    # но сохраняем факт overlap в явных колонках.
    duplicate_mask = df["source_id"].astype(str).duplicated(keep=False)
    if not bool(duplicate_mask.to_numpy().any()):
        result = df.copy()
        if "ood_group_members" not in result.columns:
            result["ood_group_members"] = pd.NA
        return result

    rows: list[Mapping[str, Any]] = []
    grouped = df.groupby("source_id", sort=False, dropna=False)
    for _, group in grouped:
        if int(group.shape[0]) == 1:
            row = normalize_record_mapping(group.iloc[0].to_dict())
            row["ood_group_members"] = pd.NA
            rows.append(row)
            continue
        rows.append(build_multi_ood_row(group))

    return pd.DataFrame.from_records(rows)


def prepare_gaia_id_ood_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    # Нормализуем binary ID/OOD source и явно схлопываем multi-membership.
    contract = GAIA_ID_OOD_TRAINING_CONTRACT
    require_columns(df, contract.required_columns, frame_name="gaia id ood training frame")

    result = select_available_columns(
        df,
        contract=contract,
    )
    result["domain_target"] = result["domain_target"].map(normalize_domain_target)
    result = cast_numeric_columns(result, ID_OOD_NUMERIC_COLUMNS)
    result = result.dropna(
        subset=(
            "source_id",
            "domain_target",
            "teff_gspphot",
            "logg_gspphot",
            "mh_gspphot",
            "bp_rp",
            "parallax",
            "parallax_over_error",
            "ruwe",
        )
    ).reset_index(drop=True)
    validate_domain_target_values(result, frame_name="gaia id ood training frame")
    result = collapse_multi_membership_ood_rows(result)
    ensure_unique_source_id(result, frame_name="gaia id ood training frame")
    return sort_by_source_id(result)


__all__ = [
    "collapse_multi_membership_ood_rows",
    "prepare_gaia_id_ood_training_frame",
]
