"""Общие константы production ranking pipeline.

Модуль содержит:

- пути к production artifact-файлам;
- имена входных и выходных DB relations;
- канонический порядок колонок для persist;
- базовые соглашения о ветках pipeline и поддерживаемых классах.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

DEFAULT_INPUT_SOURCE = "public.gaia_dr3_training"
DEFAULT_ROUTER_RESULTS_TABLE = "lab.gaia_router_results"
DEFAULT_PRIORITY_RESULTS_TABLE = "lab.gaia_priority_results"

ROUTER_MODEL_PATH = DATA_DIR / "router_gaussian_params.json"
HOST_MODEL_PATH = DATA_DIR / "model_gaussian_params.json"
HOST_MODEL_VERSION = "gaussian_host_field_v1"

MKGF_CLASSES = {"M", "K", "G", "F"}

INPUT_COLUMNS: tuple[str, ...] = (
    "source_id",
    "ra",
    "dec",
    "parallax",
    "parallax_over_error",
    "ruwe",
    "bp_rp",
    "teff_gspphot",
    "logg_gspphot",
    "radius_gspphot",
    "mh_gspphot",
    "validation_factor",
)

ROUTER_RESULTS_COLUMNS: tuple[str, ...] = (
    "run_id",
    "source_id",
    "ra",
    "dec",
    "teff_gspphot",
    "logg_gspphot",
    "radius_gspphot",
    "predicted_spec_class",
    "predicted_evolution_stage",
    "router_label",
    "d_mahal_router",
    "router_similarity",
    "router_log_likelihood",
    "router_log_posterior",
    "second_best_label",
    "margin",
    "posterior_margin",
    "router_model_version",
)

ROUTER_REQUIRED_DB_COLUMNS: tuple[str, ...] = (
    "run_id",
    "source_id",
    "predicted_spec_class",
    "predicted_evolution_stage",
    "router_label",
)

PRIORITY_RESULTS_COLUMNS: tuple[str, ...] = (
    "run_id",
    "source_id",
    "ra",
    "dec",
    "predicted_spec_class",
    "predicted_evolution_stage",
    "router_label",
    "d_mahal_router",
    "router_similarity",
    "router_log_likelihood",
    "router_log_posterior",
    "gauss_label",
    "host_log_likelihood",
    "field_log_likelihood",
    "host_log_lr",
    "host_posterior",
    "d_mahal",
    "similarity",
    "class_prior",
    "quality_factor",
    "metallicity_factor",
    "color_factor",
    "validation_factor",
    "final_score",
    "priority_tier",
    "reason_code",
    "posterior_margin",
    "router_model_version",
    "host_model_version",
)

PRIORITY_REQUIRED_DB_COLUMNS: tuple[str, ...] = (
    "run_id",
    "source_id",
    "predicted_spec_class",
    "predicted_evolution_stage",
    "router_label",
    "final_score",
    "priority_tier",
    "reason_code",
)

__all__ = [
    "DATA_DIR",
    "DEFAULT_INPUT_SOURCE",
    "DEFAULT_PRIORITY_RESULTS_TABLE",
    "DEFAULT_ROUTER_RESULTS_TABLE",
    "HOST_MODEL_PATH",
    "HOST_MODEL_VERSION",
    "INPUT_COLUMNS",
    "MKGF_CLASSES",
    "PRIORITY_REQUIRED_DB_COLUMNS",
    "PRIORITY_RESULTS_COLUMNS",
    "PROJECT_ROOT",
    "ROUTER_MODEL_PATH",
    "ROUTER_REQUIRED_DB_COLUMNS",
    "ROUTER_RESULTS_COLUMNS",
]
