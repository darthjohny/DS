# Файл `refinement_family_dataset_contracts.py` слоя `contracts`.
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

REFINEMENT_ENABLED_SPECTRAL_CLASSES: tuple[str, ...] = ("A", "B", "F", "G", "K", "M")
REFINEMENT_FAMILY_VIEW_PREFIX = "lab.v_gaia_mk_refinement_training_"

REFINEMENT_FAMILY_TARGET_CARDINALITY: dict[str, int] = {
    "A": 10,
    "B": 10,
    "F": 10,
    "G": 10,
    "K": 9,
    "M": 10,
}


def validate_refinement_family_class(spectral_class: str) -> str:
    # Оставляем только second-wave coarse classes, для которых refinement разрешен.
    normalized_class = str(spectral_class).strip().upper()
    if normalized_class not in REFINEMENT_ENABLED_SPECTRAL_CLASSES:
        supported_classes = ", ".join(REFINEMENT_ENABLED_SPECTRAL_CLASSES)
        raise ValueError(
            f"Unsupported refinement family class: {spectral_class}. "
            f"Supported classes: {supported_classes}"
        )
    return normalized_class


def build_refinement_family_view_name(spectral_class: str) -> str:
    # Собираем каноническое имя family-view в `lab`.
    normalized_class = validate_refinement_family_class(spectral_class)
    return f"{REFINEMENT_FAMILY_VIEW_PREFIX}{normalized_class.lower()}"


def build_gaia_mk_refinement_family_training_contract(
    spectral_class: str,
) -> DatasetContract:
    # Строим dataset contract для одной family-view.
    return DatasetContract(
        relation_name=build_refinement_family_view_name(spectral_class),
        required_columns=(
            "source_id",
            "spectral_class",
            "spectral_subclass",
            "full_subclass_label",
            "teff_gspphot",
            "logg_gspphot",
            "mh_gspphot",
            "bp_rp",
            "parallax",
            "parallax_over_error",
            "ruwe",
            "radius_flame",
            "lum_flame",
            "evolstage_flame",
            "phot_g_mean_mag",
        ),
        optional_columns=(
            "random_index",
            "ra",
            "dec",
            "luminosity_class",
            "peculiarity_suffix",
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
