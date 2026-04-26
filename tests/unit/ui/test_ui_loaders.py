# Тестовый файл `test_ui_loaders.py` домена `ui`.
#
# Этот файл проверяет только:
# - загрузку и валидацию готовых run artifacts для интерфейса;
# - защиту read-only слоя от дрейфа файлов, колонок и metadata.
#
# Следующий слой:
# - страницы интерфейса, использующие готовый bundle;
# - smoke-проверка entrypoint и навигации.

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from exohost.reporting.final_decision_artifacts import save_final_decision_artifacts
from exohost.ui.loaders import (
    list_available_run_dirs,
    load_ui_run_bundle_uncached,
)


def test_load_ui_run_bundle_uncached_reads_valid_run_dir(tmp_path: Path) -> None:
    run_dir = _build_valid_ui_run_dir(tmp_path)

    loaded_bundle = load_ui_run_bundle_uncached(run_dir)

    assert loaded_bundle.run_dir == run_dir.resolve()
    assert loaded_bundle.loaded_artifacts.metadata["pipeline_name"] == "hierarchical_final_decision"
    assert list(loaded_bundle.loaded_artifacts.final_decision_df["final_domain_state"]) == ["id"]


def test_list_available_run_dirs_returns_only_valid_runs(tmp_path: Path) -> None:
    valid_run_dir = _build_valid_ui_run_dir(tmp_path)
    (tmp_path / "broken_run").mkdir()

    run_dirs = list_available_run_dirs(str(tmp_path))

    assert run_dirs == (valid_run_dir.resolve(),)


def test_load_ui_run_bundle_uncached_rejects_missing_required_file(
    tmp_path: Path,
) -> None:
    run_dir = _build_valid_ui_run_dir(tmp_path)
    (run_dir / "priority_input.csv").unlink()

    try:
        load_ui_run_bundle_uncached(run_dir)
    except RuntimeError as exc:
        assert "priority_input.csv" in str(exc)
    else:
        raise AssertionError("Expected UI loader to reject run without priority_input.csv.")


def test_load_ui_run_bundle_uncached_rejects_missing_required_column(
    tmp_path: Path,
) -> None:
    run_dir = _build_valid_ui_run_dir(tmp_path)
    priority_ranking_df = pd.read_csv(run_dir / "priority_ranking.csv").drop(
        columns=["priority_reason"]
    )
    priority_ranking_df.to_csv(run_dir / "priority_ranking.csv", index=False)

    try:
        load_ui_run_bundle_uncached(run_dir)
    except RuntimeError as exc:
        assert "priority_reason" in str(exc)
    else:
        raise AssertionError("Expected UI loader to reject run with broken priority_ranking.csv.")


def test_load_ui_run_bundle_uncached_rejects_missing_metadata_context_key(
    tmp_path: Path,
) -> None:
    run_dir = _build_valid_ui_run_dir(tmp_path)
    metadata = _read_metadata(run_dir)
    context = metadata["context"]
    assert isinstance(context, dict)
    context.pop("priority_high_min")
    _write_metadata(run_dir, metadata)

    try:
        load_ui_run_bundle_uncached(run_dir)
    except RuntimeError as exc:
        assert "priority_high_min" in str(exc)
    else:
        raise AssertionError("Expected UI loader to reject metadata with missing context keys.")


def test_load_ui_run_bundle_uncached_rejects_missing_metadata_key(
    tmp_path: Path,
) -> None:
    run_dir = _build_valid_ui_run_dir(tmp_path)
    metadata = _read_metadata(run_dir)
    metadata.pop("final_domain_distribution")
    _write_metadata(run_dir, metadata)

    try:
        load_ui_run_bundle_uncached(run_dir)
    except RuntimeError as exc:
        assert "final_domain_distribution" in str(exc)
    else:
        raise AssertionError("Expected UI loader to reject metadata with missing top-level keys.")


def test_load_ui_run_bundle_uncached_rejects_unexpected_pipeline_name(
    tmp_path: Path,
) -> None:
    run_dir = _build_valid_ui_run_dir(tmp_path)
    metadata = _read_metadata(run_dir)
    metadata["pipeline_name"] = "experimental_pipeline"
    _write_metadata(run_dir, metadata)

    try:
        load_ui_run_bundle_uncached(run_dir)
    except RuntimeError as exc:
        assert "unexpected pipeline" in str(exc)
        assert "experimental_pipeline" in str(exc)
    else:
        raise AssertionError("Expected UI loader to reject metadata from another pipeline.")


def _build_valid_ui_run_dir(tmp_path: Path) -> Path:
    artifact_paths = save_final_decision_artifacts(
        pipeline_name="hierarchical_final_decision",
        decision_input_df=_build_decision_input_df(),
        final_decision_df=_build_final_decision_df(),
        priority_input_df=_build_priority_input_df(),
        priority_ranking_df=_build_priority_ranking_df(),
        output_dir=tmp_path,
        extra_metadata=_build_metadata_context(),
    )
    return artifact_paths.run_dir


def _read_metadata(run_dir: Path) -> dict[str, object]:
    return json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))


def _write_metadata(run_dir: Path, metadata: dict[str, object]) -> None:
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _build_decision_input_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_id": 1,
                "quality_state": "pass",
                "quality_reason": "pass",
                "review_bucket": "pass",
                "ood_state": "in_domain",
                "ood_reason": "strong_single_star",
                "ood_decision": "in_domain",
                "coarse_predicted_label": "G",
                "coarse_probability_max": 0.98,
                "spec_class": "G",
                "spec_subclass": "G2",
            }
        ]
    )


def _build_final_decision_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_id": 1,
                "final_domain_state": "id",
                "final_quality_state": "pass",
                "final_coarse_class": "G",
                "final_refinement_state": "accepted",
                "final_decision_reason": "refinement_accepted",
                "final_decision_policy_version": "final_decision_v2",
                "priority_state": "ranked",
                "priority_label": "high",
            }
        ]
    )


def _build_priority_input_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_id": 1,
                "spec_class": "G",
                "host_similarity_score": 0.98,
                "observability_score": 0.77,
            }
        ]
    )


def _build_priority_ranking_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_id": 1,
                "spec_class": "G",
                "class_priority_score": 0.83,
                "host_similarity_score": 0.98,
                "priority_score": 0.89,
                "priority_label": "high",
                "priority_reason": "host_like_and_observable",
                "observability_score": 0.77,
            }
        ]
    )


def _build_metadata_context() -> dict[str, object]:
    # Здесь держим полный обязательный context-пакет, чтобы loader действительно страховал контракт.
    return {
        "candidate_ood_disposition": "keep",
        "coarse_model_run_dir": "artifacts/models/coarse",
        "connect_timeout": 10,
        "decision_policy_version": "final_decision_v2",
        "dotenv_path": ".env",
        "host_model_run_dir": "artifacts/models/host",
        "host_score_column": "host_similarity_score",
        "input_csv": None,
        "min_coarse_probability": 0.6,
        "ood_model_run_dir": "artifacts/models/ood",
        "ood_threshold_run_dir": "artifacts/thresholds/ood",
        "output_dir": "artifacts/decisions",
        "preview_rows": 10,
        "priority_high_min": 0.85,
        "priority_medium_min": 0.55,
        "quality_gate_policy_name": "relaxed_radius",
        "quality_require_flame_for_pass": False,
        "refinement_families": ["fgk"],
        "refinement_model_run_dirs": [
            "artifacts/models/refinement_fgk",
        ],
        "relation_name": "lab.gaia_mk_quality_gated",
    }
