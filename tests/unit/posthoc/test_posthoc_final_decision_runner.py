# Тестовый файл `test_posthoc_final_decision_runner.py` домена `posthoc`.
#
# Этот файл проверяет только:
# - проверку логики домена: post-hoc routing, gate и final decision policy;
# - регрессионные сценарии и ожидаемые контракты целевого слоя.
#
# Следующий слой:
# - реализация домена `posthoc` в `src/exohost`;
# - соседние тесты и testkit этого же пакета.

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

from exohost.posthoc.candidate_ood_policy import (
    CANDIDATE_OOD_MAP_TO_UNKNOWN_DISPOSITION,
    CandidateOodPolicy,
)
from exohost.posthoc.final_decision import (
    FINAL_DOMAIN_ID,
    FINAL_DOMAIN_UNKNOWN,
    FinalDecisionPolicy,
)
from exohost.posthoc.final_decision_runner import (
    FinalDecisionRunnerConfig,
    run_final_decision_pipeline,
)
from exohost.posthoc.id_ood_gate import build_id_ood_threshold_policy
from exohost.posthoc.refinement_handoff import RefinementHandoffPolicy


class DummyIdOodEstimator:
    classes_ = np.asarray(["id", "ood"], dtype=object)

    def predict_proba(self, X: pd.DataFrame) -> npt.NDArray[np.float64]:
        probabilities: list[list[float]] = []
        for source_id in X["source_id"].astype(int).tolist():
            if source_id == 1:
                probabilities.append([0.95, 0.05])
            elif source_id == 2:
                probabilities.append([0.45, 0.55])
            else:
                probabilities.append([0.20, 0.80])
        return np.asarray(probabilities, dtype=float)


class DummyCoarseEstimator:
    classes_ = np.asarray(["G", "K"], dtype=object)

    def predict_proba(self, X: pd.DataFrame) -> npt.NDArray[np.float64]:
        probabilities: list[list[float]] = []
        for source_id in X["source_id"].astype(int).tolist():
            if source_id == 1:
                probabilities.append([0.90, 0.10])
            else:
                probabilities.append([0.20, 0.80])
        return np.asarray(probabilities, dtype=float)


class DummyGRefinementEstimator:
    classes_ = np.asarray(["G2", "G4"], dtype=object)

    def predict_proba(self, X: pd.DataFrame) -> npt.NDArray[np.float64]:
        return np.asarray([[0.15, 0.85] for _ in range(int(X.shape[0]))], dtype=float)


def build_base_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_id": [1, 2, 3],
            "quality_state": ["pass", "pass", "pass"],
            "priority_state": ["high", "medium", "low"],
            "teff_gspphot": [5700.0, 5200.0, 4800.0],
            "logg_gspphot": [4.3, 4.1, 4.0],
            "mh_gspphot": [0.0, -0.1, 0.2],
            "bp_rp": [0.8, 1.1, 1.4],
            "parallax": [10.0, 9.0, 8.0],
            "parallax_over_error": [15.0, 13.0, 11.0],
            "ruwe": [1.0, 1.1, 1.2],
            "phot_g_mean_mag": [11.0, 12.0, 13.0],
        }
    )


def test_run_final_decision_pipeline_routes_id_candidate_and_ood() -> None:
    config = FinalDecisionRunnerConfig(
        ood_estimator=DummyIdOodEstimator(),
        ood_feature_columns=(
            "source_id",
            "teff_gspphot",
            "logg_gspphot",
            "mh_gspphot",
            "bp_rp",
            "parallax",
            "parallax_over_error",
            "ruwe",
            "phot_g_mean_mag",
        ),
        ood_threshold_policy=build_id_ood_threshold_policy(
            tuned_threshold=0.7,
            threshold_policy_version="id_ood_threshold_v1",
            candidate_ood_threshold=0.4,
        ),
        coarse_estimator=DummyCoarseEstimator(),
        coarse_feature_columns=("source_id", "teff_gspphot"),
        final_decision_policy=FinalDecisionPolicy(
            decision_policy_version="final_decision_v2",
            refinement_handoff_policy=RefinementHandoffPolicy(min_coarse_probability=0.8),
            min_refinement_confidence=0.6,
        ),
        refinement_estimators_by_family={"G": DummyGRefinementEstimator()},
        refinement_feature_columns=("source_id", "teff_gspphot"),
        coarse_model_name="hist_gradient_boosting",
        refinement_model_names_by_family={"G": "hist_gradient_boosting"},
    )

    result = run_final_decision_pipeline(build_base_frame(), config=config)

    assert result.final_decision_df["final_domain_state"].tolist() == [
        FINAL_DOMAIN_ID,
        "candidate_ood",
        "ood",
    ]
    assert result.final_decision_df.loc[0, "final_refinement_label"] == "G4"
    assert pd.isna(result.final_decision_df.loc[1, "final_refinement_label"])
    assert pd.isna(result.final_decision_df.loc[2, "final_refinement_label"])
    assert result.input_df.loc[result.input_df["source_id"] == 1, "coarse_predicted_label"].item() == "G"


def test_run_final_decision_pipeline_can_map_candidate_ood_to_unknown() -> None:
    config = FinalDecisionRunnerConfig(
        ood_estimator=DummyIdOodEstimator(),
        ood_feature_columns=(
            "source_id",
            "teff_gspphot",
            "logg_gspphot",
            "mh_gspphot",
            "bp_rp",
            "parallax",
            "parallax_over_error",
            "ruwe",
            "phot_g_mean_mag",
        ),
        ood_threshold_policy=build_id_ood_threshold_policy(
            tuned_threshold=0.7,
            threshold_policy_version="id_ood_threshold_v1",
            candidate_ood_threshold=0.4,
        ),
        coarse_estimator=DummyCoarseEstimator(),
        coarse_feature_columns=("source_id", "teff_gspphot"),
        final_decision_policy=FinalDecisionPolicy(
            decision_policy_version="final_decision_v2",
            candidate_ood_policy=CandidateOodPolicy(
                disposition=CANDIDATE_OOD_MAP_TO_UNKNOWN_DISPOSITION
            ),
        ),
    )

    result = run_final_decision_pipeline(build_base_frame(), config=config)

    assert result.final_decision_df.loc[1, "final_domain_state"] == FINAL_DOMAIN_UNKNOWN
    assert result.final_decision_df.loc[1, "final_decision_reason"] == "candidate_ood_mapped_to_unknown"


def test_run_final_decision_pipeline_requires_refinement_columns_when_family_estimators_set() -> None:
    config = FinalDecisionRunnerConfig(
        ood_estimator=DummyIdOodEstimator(),
        ood_feature_columns=(
            "source_id",
            "teff_gspphot",
            "logg_gspphot",
            "mh_gspphot",
            "bp_rp",
            "parallax",
            "parallax_over_error",
            "ruwe",
            "phot_g_mean_mag",
        ),
        ood_threshold_policy=build_id_ood_threshold_policy(
            tuned_threshold=0.7,
            threshold_policy_version="id_ood_threshold_v1",
        ),
        coarse_estimator=DummyCoarseEstimator(),
        coarse_feature_columns=("source_id", "teff_gspphot"),
        final_decision_policy=FinalDecisionPolicy(decision_policy_version="final_decision_v2"),
        refinement_estimators_by_family={"G": DummyGRefinementEstimator()},
    )

    try:
        run_final_decision_pipeline(build_base_frame(), config=config)
    except ValueError as exc:
        assert "refinement_feature_columns" in str(exc)
    else:
        raise AssertionError("Expected ValueError when refinement columns are not provided.")
