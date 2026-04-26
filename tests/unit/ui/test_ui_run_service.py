# Тестовый файл `test_ui_run_service.py` домена `ui`.
#
# Этот файл проверяет только:
# - валидацию внешнего `CSV` для страницы кнопочного запуска;
# - извлечение defaults из `metadata.context` выбранного рабочего run.
#
# Следующий слой:
# - сама Streamlit-страница `CSV`-запуска;
# - интеграционный сценарий реального `decide` поверх UI.

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from exohost.ui.contracts import UI_EXTERNAL_CSV_CONTRACT
from exohost.ui.run_service import (
    build_ui_csv_decide_defaults,
    save_uploaded_csv_bytes,
    validate_uploaded_csv_bytes,
)

from .ui_testkit import build_ui_loaded_run_bundle


def test_validate_uploaded_csv_bytes_accepts_minimal_contract() -> None:
    csv_bytes = _build_csv_bytes(_build_minimal_csv_frame())

    preview = validate_uploaded_csv_bytes(csv_bytes)

    assert preview.n_rows == 1
    assert "source_id" in preview.column_names
    assert float(preview.validated_df.loc[0, "teff_gspphot"]) == 5750.0


def test_validate_uploaded_csv_bytes_rejects_empty_csv() -> None:
    _assert_csv_validation_error(
        b"",
        expected_message_part="пустой",
    )


def test_validate_uploaded_csv_bytes_rejects_header_only_csv() -> None:
    csv_bytes = _build_csv_bytes(
        pd.DataFrame(columns=UI_EXTERNAL_CSV_CONTRACT.required_columns)
    )

    _assert_csv_validation_error(
        csv_bytes,
        expected_message_part="не содержит строк данных",
    )


def test_validate_uploaded_csv_bytes_rejects_missing_required_columns() -> None:
    csv_bytes = _build_csv_bytes(pd.DataFrame([{"source_id": 101, "quality_state": "pass"}]))

    _assert_csv_validation_error(
        csv_bytes,
        expected_message_part="teff_gspphot",
    )


def test_validate_uploaded_csv_bytes_rejects_missing_required_values() -> None:
    frame = _build_minimal_csv_frame()
    frame.loc[0, "radius_flame"] = pd.NA

    _assert_csv_validation_error(
        _build_csv_bytes(frame),
        expected_message_part="radius_flame",
    )


def test_validate_uploaded_csv_bytes_rejects_unsupported_quality_state() -> None:
    frame = _build_minimal_csv_frame()
    frame.loc[0, "quality_state"] = "maybe"

    _assert_csv_validation_error(
        _build_csv_bytes(frame),
        expected_message_part="quality_state",
    )


def test_validate_uploaded_csv_bytes_rejects_non_numeric_physical_value() -> None:
    frame = _build_minimal_csv_frame()
    frame["teff_gspphot"] = frame["teff_gspphot"].astype("object")
    frame.loc[0, "teff_gspphot"] = "hot"

    _assert_csv_validation_error(
        _build_csv_bytes(frame),
        expected_message_part="teff_gspphot",
    )


def test_build_ui_csv_decide_defaults_reads_required_context_from_bundle() -> None:
    bundle = build_ui_loaded_run_bundle()
    bundle.loaded_artifacts.metadata["context"] = _build_defaults_context()

    defaults = build_ui_csv_decide_defaults(bundle)

    assert defaults.ood_model_run_dir == "artifacts/models/ood"
    assert defaults.refinement_model_run_dirs == ("artifacts/models/refinement_g",)
    assert defaults.host_model_run_dir == "artifacts/models/host"
    assert defaults.priority_high_min == 0.85


def test_build_ui_csv_decide_defaults_rejects_missing_context() -> None:
    bundle = build_ui_loaded_run_bundle()
    bundle.loaded_artifacts.metadata.pop("context", None)

    try:
        build_ui_csv_decide_defaults(bundle)
    except RuntimeError as exc:
        assert "metadata.context" in str(exc)
    else:
        raise AssertionError("Expected UI defaults builder to reject missing context.")


def test_build_ui_csv_decide_defaults_rejects_missing_required_model_path() -> None:
    bundle = build_ui_loaded_run_bundle()
    context = _build_defaults_context()
    context.pop("ood_model_run_dir")
    bundle.loaded_artifacts.metadata["context"] = context

    try:
        build_ui_csv_decide_defaults(bundle)
    except RuntimeError as exc:
        assert "ood_model_run_dir" in str(exc)
    else:
        raise AssertionError("Expected UI defaults builder to reject missing model paths.")


def test_build_ui_csv_decide_defaults_rejects_malformed_refinement_run_dirs() -> None:
    bundle = build_ui_loaded_run_bundle()
    context = _build_defaults_context()
    context["refinement_model_run_dirs"] = "artifacts/models/refinement_g"
    bundle.loaded_artifacts.metadata["context"] = context

    try:
        build_ui_csv_decide_defaults(bundle)
    except RuntimeError as exc:
        assert "refinement_model_run_dirs" in str(exc)
        assert "списком строк" in str(exc)
    else:
        raise AssertionError("Expected UI defaults builder to reject malformed refinement dirs.")


def test_save_uploaded_csv_bytes_persists_uploaded_file(tmp_path: Path) -> None:
    csv_bytes = _build_csv_bytes(pd.DataFrame([{"source_id": 101, "quality_state": "pass"}]))

    saved_path = save_uploaded_csv_bytes(
        filename="Gaia demo.csv",
        uploaded_bytes=csv_bytes,
        uploads_dir=tmp_path,
    )

    assert saved_path.exists()
    assert saved_path.suffix == ".csv"
    assert "gaia_demo" in saved_path.name


def _build_csv_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def _build_minimal_csv_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_id": 101,
                "quality_state": "pass",
                "teff_gspphot": 5750.0,
                "logg_gspphot": 4.3,
                "mh_gspphot": 0.0,
                "bp_rp": 0.8,
                "parallax": 10.0,
                "parallax_over_error": 100.0,
                "ruwe": 1.0,
                "phot_g_mean_mag": 9.5,
                "radius_flame": 1.0,
                "lum_flame": 1.1,
                "evolstage_flame": "main_sequence",
            }
        ]
    )


def _build_defaults_context() -> dict[str, object]:
    return {
        "ood_model_run_dir": "artifacts/models/ood",
        "ood_threshold_run_dir": "artifacts/thresholds/ood",
        "coarse_model_run_dir": "artifacts/models/coarse",
        "refinement_model_run_dirs": ["artifacts/models/refinement_g"],
        "host_model_run_dir": "artifacts/models/host",
        "decision_policy_version": "final_decision_v2",
        "candidate_ood_disposition": "keep",
        "host_score_column": "host_similarity_score",
        "min_refinement_confidence": 0.55,
        "min_coarse_probability": 0.60,
        "min_coarse_margin": 0.10,
        "quality_ruwe_unknown_threshold": 1.4,
        "quality_parallax_snr_unknown_threshold": 10.0,
        "quality_require_flame_for_pass": False,
        "priority_high_min": 0.85,
        "priority_medium_min": 0.55,
        "output_dir": "artifacts/decisions",
        "dotenv_path": ".env",
        "connect_timeout": 10,
    }


def _assert_csv_validation_error(
    csv_bytes: bytes,
    *,
    expected_message_part: str,
) -> None:
    try:
        validate_uploaded_csv_bytes(csv_bytes)
    except RuntimeError as exc:
        assert expected_message_part in str(exc)
    else:
        raise AssertionError("Expected CSV validator to reject invalid input.")
