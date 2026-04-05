# Файл `quality_gate_dataset_contracts.py` слоя `contracts`.
#
# Этот файл отвечает только за:
# - контракты колонок, датасетов и ролей правил;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - соседние модули слоя `contracts` и orchestration поверх них;
# - unit-тесты и более верхние слои проекта.

from __future__ import annotations

from exohost.contracts.dataset_contracts import DatasetContract
from exohost.contracts.feature_contract import IDENTIFIER_COLUMNS, unique_columns

QUALITY_GATE_SIGNAL_COLUMNS: tuple[str, ...] = (
    "has_core_features",
    "has_flame_features",
    "has_non_single_star_flag",
    "has_low_single_star_probability",
    "has_missing_core_features",
    "has_missing_flame_features",
    "has_high_ruwe",
    "has_low_parallax_snr",
)

QUALITY_GATE_REQUIRED_COLUMNS: tuple[str, ...] = unique_columns(
    IDENTIFIER_COLUMNS,
    (
        "quality_state",
        "ood_state",
        "quality_reason",
        "ood_reason",
        "review_bucket",
        *QUALITY_GATE_SIGNAL_COLUMNS,
    ),
)

QUALITY_GATE_OPTIONAL_COLUMNS: tuple[str, ...] = (
    "spectral_class",
    "spectral_subclass",
    "luminosity_class",
    "ruwe",
    "parallax",
    "parallax_over_error",
    "non_single_star",
    "classprob_dsc_combmod_star",
    "radius_flame",
    "radius_gspphot",
    "lum_flame",
    "evolstage_flame",
    "phot_g_mean_mag",
    "bp_rp",
    "teff_gspphot",
    "logg_gspphot",
    "mh_gspphot",
    "in_ood_reference",
    "ood_membership_count",
    "ood_group_list",
    "quality_gate_version",
    "quality_gated_at_utc",
    "random_index",
)

GAIA_MK_QUALITY_GATE_AUDIT_CONTRACT = DatasetContract(
    relation_name="lab.gaia_mk_quality_gated",
    required_columns=QUALITY_GATE_REQUIRED_COLUMNS,
    optional_columns=QUALITY_GATE_OPTIONAL_COLUMNS,
)

GAIA_MK_UNKNOWN_REVIEW_AUDIT_CONTRACT = DatasetContract(
    relation_name="lab.gaia_mk_unknown_review",
    required_columns=QUALITY_GATE_REQUIRED_COLUMNS,
    optional_columns=QUALITY_GATE_OPTIONAL_COLUMNS,
)
