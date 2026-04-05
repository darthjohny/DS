# Файл `hierarchical_dataset_contracts.py` слоя `contracts`.
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

GAIA_ID_COARSE_TRAINING_CONTRACT = DatasetContract(
    relation_name="lab.v_gaia_id_coarse_training",
    required_columns=(
        "source_id",
        "ra",
        "dec",
        "teff_gspphot",
        "logg_gspphot",
        "mh_gspphot",
        "bp_rp",
        "parallax",
        "parallax_over_error",
        "ruwe",
        "spec_class",
        "is_evolved",
    ),
    optional_columns=(
        "random_index",
        "radius_feature",
        "radius_flame",
        "radius_gspphot",
        "lum_flame",
        "evolstage_flame",
        "reference_membership_count",
        "has_reference_overlap",
    ),
)

GAIA_MK_REFINEMENT_TRAINING_CONTRACT = DatasetContract(
    relation_name="lab.v_gaia_mk_refinement_training",
    required_columns=(
        "source_id",
        "spectral_class",
        "spectral_subclass",
        "teff_gspphot",
        "logg_gspphot",
        "mh_gspphot",
        "bp_rp",
        "parallax",
        "parallax_over_error",
        "ruwe",
        "radius_flame",
    ),
    optional_columns=(
        "random_index",
        "ra",
        "dec",
        "phot_g_mean_mag",
        "luminosity_class",
        "peculiarity_suffix",
        "lum_flame",
        "evolstage_flame",
        "raw_sptype",
        "label_parse_status",
        "label_parse_notes",
        "xmatch_batch_id",
        "external_row_id",
        "external_catalog_name",
        "external_object_id",
        "xmatch_separation_arcsec",
        "quality_state",
        "ood_state",
        "quality_reason",
        "ood_reason",
        "quality_gate_version",
        "quality_gated_at_utc",
    ),
)

GAIA_ID_OOD_TRAINING_CONTRACT = DatasetContract(
    relation_name="lab.v_gaia_id_ood_training",
    required_columns=(
        "source_id",
        "domain_target",
        "teff_gspphot",
        "logg_gspphot",
        "mh_gspphot",
        "bp_rp",
        "parallax",
        "parallax_over_error",
        "ruwe",
    ),
    optional_columns=(
        "ood_group",
        "ood_membership_count",
        "has_multi_ood_membership",
        "random_index",
        "ra",
        "dec",
        "phot_g_mean_mag",
        "radius_flame",
        "selector_score_1",
        "selector_score_2",
    ),
)
