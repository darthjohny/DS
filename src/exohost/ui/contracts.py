# Файл `contracts.py` слоя `ui`.
#
# Этот файл отвечает только за:
# - контракты интерфейса Streamlit;
# - описание run-артефактов, внешнего CSV и минимального состояния UI.
#
# Следующий слой:
# - loader-, state- и page-модули интерфейса;
# - unit-тесты UI-пакета.

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

type UiPageKey = Literal["home", "metrics", "run_browser", "candidate", "csv_decide"]
type UiArtifactTableName = Literal[
    "decision_input",
    "final_decision",
    "priority_input",
    "priority_ranking",
]
type UiMetricStageKey = Literal["id_ood", "coarse", "host", "refinement"]


@dataclass(frozen=True, slots=True)
class UiArtifactTableContract:
    # Контракт одной таблицы внутри final-decision run_dir.
    table_name: UiArtifactTableName
    filename: str
    required_columns: tuple[str, ...]
    optional_columns: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class UiRunArtifactContract:
    # Контракт набора артефактов, который интерфейс может читать без notebook.
    pipeline_name: str
    required_filenames: tuple[str, ...]
    optional_filenames: tuple[str, ...]
    required_metadata_keys: tuple[str, ...]
    required_metadata_context_keys: tuple[str, ...]
    table_contracts: tuple[UiArtifactTableContract, ...]


@dataclass(frozen=True, slots=True)
class UiExternalCsvContract:
    # Минимальный входной CSV-контур для запуска `decide` из интерфейса.
    required_columns: tuple[str, ...]
    recommended_columns: tuple[str, ...]
    quality_state_default: str
    contract_doc_path: str


@dataclass(frozen=True, slots=True)
class UiSessionStateContract:
    # Минимальный набор session_state-ключей для первой версии интерфейса.
    default_page: UiPageKey
    selected_run_dir_key: str
    selected_source_id_key: str
    uploaded_csv_path_key: str
    generated_run_dir_key: str
    run_load_error_key: str
    csv_validation_error_key: str


@dataclass(frozen=True, slots=True)
class UiBenchmarkStageContract:
    # Контракт одного benchmark-этапа для страницы качества моделей.
    stage_key: UiMetricStageKey
    display_name: str
    task_name_prefix: str
    interpretation_note: str


UI_FINAL_DECISION_RUN_CONTRACT = UiRunArtifactContract(
    pipeline_name="hierarchical_final_decision",
    required_filenames=(
        "decision_input.csv",
        "final_decision.csv",
        "priority_input.csv",
        "priority_ranking.csv",
        "metadata.json",
    ),
    optional_filenames=(),
    required_metadata_keys=(
        "pipeline_name",
        "created_at_utc",
        "n_rows_input",
        "n_rows_final_decision",
        "n_rows_priority_input",
        "n_rows_priority_ranking",
        "decision_input_columns",
        "final_decision_columns",
        "priority_input_columns",
        "priority_ranking_columns",
        "final_domain_distribution",
        "priority_label_distribution",
        "context",
    ),
    required_metadata_context_keys=(
        "candidate_ood_disposition",
        "coarse_model_run_dir",
        "connect_timeout",
        "decision_policy_version",
        "dotenv_path",
        "host_model_run_dir",
        "host_score_column",
        "input_csv",
        "ood_model_run_dir",
        "ood_threshold_run_dir",
        "output_dir",
        "preview_rows",
        "priority_high_min",
        "priority_medium_min",
        "quality_gate_policy_name",
        "quality_require_flame_for_pass",
        "refinement_families",
        "refinement_model_run_dirs",
        "relation_name",
    ),
    table_contracts=(
        UiArtifactTableContract(
            table_name="decision_input",
            filename="decision_input.csv",
            required_columns=(
                "source_id",
                "quality_state",
                "quality_reason",
                "review_bucket",
                "ood_state",
                "ood_reason",
                "ood_decision",
                "coarse_predicted_label",
                "coarse_probability_max",
            ),
            optional_columns=(
                "spec_class",
                "spec_subclass",
                "evolution_stage",
                "teff_gspphot",
                "logg_gspphot",
                "mh_gspphot",
                "bp_rp",
                "parallax",
                "parallax_over_error",
                "ruwe",
                "phot_g_mean_mag",
            ),
        ),
        UiArtifactTableContract(
            table_name="final_decision",
            filename="final_decision.csv",
            required_columns=(
                "source_id",
                "final_domain_state",
                "final_quality_state",
                "final_coarse_class",
                "final_refinement_state",
                "final_decision_reason",
                "final_decision_policy_version",
                "priority_state",
            ),
            optional_columns=(
                "final_refinement_label",
                "quality_reason",
                "review_bucket",
                "priority_label",
            ),
        ),
        UiArtifactTableContract(
            table_name="priority_input",
            filename="priority_input.csv",
            required_columns=(
                "source_id",
                "spec_class",
                "host_similarity_score",
            ),
            optional_columns=(
                "observability_score",
                "bp_rp",
                "phot_g_mean_mag",
            ),
        ),
        UiArtifactTableContract(
            table_name="priority_ranking",
            filename="priority_ranking.csv",
            required_columns=(
                "source_id",
                "spec_class",
                "class_priority_score",
                "host_similarity_score",
                "priority_score",
                "priority_label",
                "priority_reason",
            ),
            optional_columns=(
                "observability_score",
                "phot_g_mean_mag",
                "bp_rp",
            ),
        ),
    ),
)

UI_EXTERNAL_CSV_CONTRACT = UiExternalCsvContract(
    required_columns=(
        "source_id",
        "quality_state",
        "teff_gspphot",
        "logg_gspphot",
        "mh_gspphot",
        "bp_rp",
        "parallax",
        "parallax_over_error",
        "ruwe",
        "phot_g_mean_mag",
        "radius_flame",
        "lum_flame",
        "evolstage_flame",
    ),
    recommended_columns=(
        "radius_gspphot",
        "ra",
        "dec",
        "random_index",
    ),
    quality_state_default="pass",
    contract_doc_path=(
        "docs/methodology/contracts/external_decide_input_contract_ru.md"
    ),
)

UI_SESSION_STATE_CONTRACT = UiSessionStateContract(
    default_page="home",
    selected_run_dir_key="selected_run_dir",
    selected_source_id_key="selected_source_id",
    uploaded_csv_path_key="uploaded_csv_path",
    generated_run_dir_key="generated_run_dir",
    run_load_error_key="run_load_error",
    csv_validation_error_key="csv_validation_error",
)

UI_BENCHMARK_STAGE_CONTRACTS: tuple[UiBenchmarkStageContract, ...] = (
    UiBenchmarkStageContract(
        stage_key="id_ood",
        display_name="ID/OOD",
        task_name_prefix="gaia_id_ood_classification",
        interpretation_note=(
            "Первый фильтр домена. Этот слой должен надежно отделять рабочий контур "
            "от явного OOD."
        ),
    ),
    UiBenchmarkStageContract(
        stage_key="coarse",
        display_name="Coarse",
        task_name_prefix="gaia_id_coarse_classification",
        interpretation_note=(
            "Грубая классификация по крупным спектральным классам. Это основной "
            "маршрутизирующий слой перед более тонкой логикой."
        ),
    ),
    UiBenchmarkStageContract(
        stage_key="host",
        display_name="Host",
        task_name_prefix="host_field_classification",
        interpretation_note=(
            "Прикладной слой сходства с host-популяцией. Он влияет на итоговый "
            "наблюдательный приоритет."
        ),
    ),
    UiBenchmarkStageContract(
        stage_key="refinement",
        display_name="Refinement",
        task_name_prefix="gaia_mk_refinement_classification",
        interpretation_note=(
            "Тонкая научная надстройка по подклассам. Этот слой полезен, но требует "
            "более осторожной интерпретации."
        ),
    ),
)


def build_ui_artifact_table_map(
    contract: UiRunArtifactContract,
) -> dict[UiArtifactTableName, UiArtifactTableContract]:
    # Даем page- и loader-слою быстрый доступ к контрактам по logical table name.
    return {table_contract.table_name: table_contract for table_contract in contract.table_contracts}


__all__ = [
    "UI_EXTERNAL_CSV_CONTRACT",
    "UI_FINAL_DECISION_RUN_CONTRACT",
    "UI_SESSION_STATE_CONTRACT",
    "UI_BENCHMARK_STAGE_CONTRACTS",
    "UiArtifactTableContract",
    "UiArtifactTableName",
    "UiBenchmarkStageContract",
    "UiExternalCsvContract",
    "UiMetricStageKey",
    "UiPageKey",
    "UiRunArtifactContract",
    "UiSessionStateContract",
    "build_ui_artifact_table_map",
]
