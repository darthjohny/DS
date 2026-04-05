# Тестовый файл `test_posthoc_final_decision_artifact_runner.py` домена `posthoc`.
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

from exohost.posthoc.decision_model_bundle import FinalDecisionModelBundle
from exohost.posthoc.final_decision import FinalDecisionPolicy
from exohost.posthoc.final_decision_artifact_runner import (
    run_final_decision_with_artifacts,
)
from exohost.posthoc.id_ood_gate import build_id_ood_threshold_policy
from exohost.posthoc.priority_integration import PriorityIntegrationConfig
from exohost.reporting.id_ood_threshold_artifacts import LoadedIdOodThresholdArtifact
from exohost.reporting.model_artifacts import LoadedModelArtifact


class DummyIdOodEstimator:
    classes_ = np.asarray(["id", "ood"], dtype=object)

    def predict_proba(self, X: pd.DataFrame) -> npt.NDArray[np.float64]:
        probabilities: list[list[float]] = []
        for source_id in X["source_id"].astype(int).tolist():
            if source_id == 1:
                probabilities.append([0.94, 0.06])
            elif source_id == 2:
                probabilities.append([0.55, 0.45])
            else:
                probabilities.append([0.20, 0.80])
        return np.asarray(probabilities, dtype=float)


class DummyCoarseEstimator:
    classes_ = np.asarray(["G", "K"], dtype=object)

    def predict_proba(self, X: pd.DataFrame) -> npt.NDArray[np.float64]:
        probabilities: list[list[float]] = []
        for source_id in X["source_id"].astype(int).tolist():
            if source_id == 1:
                probabilities.append([0.88, 0.12])
            else:
                probabilities.append([0.30, 0.70])
        return np.asarray(probabilities, dtype=float)


class DummyGFamilyEstimator:
    classes_ = np.asarray(["G2", "G4"], dtype=object)

    def predict_proba(self, X: pd.DataFrame) -> npt.NDArray[np.float64]:
        return np.asarray([[0.10, 0.90]], dtype=float)


class DummyHostEstimator:
    classes_ = np.asarray(["field", "host"], dtype=object)

    def predict(self, X: pd.DataFrame) -> npt.NDArray[np.object_]:
        return np.asarray(["host" for _ in range(int(X.shape[0]))], dtype=object)

    def predict_proba(self, X: pd.DataFrame) -> npt.NDArray[np.float64]:
        return np.asarray([[0.10, 0.90], [0.40, 0.60], [0.85, 0.15]], dtype=float)


def build_base_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_id": [1, 2, 3],
            "quality_state": ["pass", "pass", "pass"],
            "teff_gspphot": [5700.0, 5100.0, 4700.0],
            "logg_gspphot": [4.3, 4.4, 4.7],
            "mh_gspphot": [0.0, -0.1, -0.2],
            "bp_rp": [0.8, 1.0, 1.3],
            "parallax": [10.0, 9.0, 8.0],
            "parallax_over_error": [15.0, 13.0, 11.0],
            "ruwe": [1.0, 1.1, 1.2],
            "phot_g_mean_mag": [11.0, 12.0, 13.0],
            "radius_feature": [1.0, 0.9, 0.8],
            "radius_gspphot": [1.0, 0.9, 0.8],
            "validation_factor": [0.95, 0.82, 0.60],
        }
    )


def build_bundle() -> FinalDecisionModelBundle:
    return FinalDecisionModelBundle(
        ood_artifact=LoadedModelArtifact(
            estimator=DummyIdOodEstimator(),
            metadata={},
            task_name="gaia_id_ood_classification",
            target_column="domain_target",
            feature_columns=(
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
            model_name="dummy_ood",
        ),
        ood_threshold_artifact=LoadedIdOodThresholdArtifact(
            policy=build_id_ood_threshold_policy(
                tuned_threshold=0.7,
                threshold_policy_version="id_ood_threshold_v1",
                candidate_ood_threshold=0.4,
            ),
            metadata={},
            task_name="gaia_id_ood_classification",
            model_name="dummy_ood",
        ),
        coarse_artifact=LoadedModelArtifact(
            estimator=DummyCoarseEstimator(),
            metadata={},
            task_name="gaia_id_coarse_classification",
            target_column="spec_class",
            feature_columns=("source_id", "teff_gspphot"),
            model_name="dummy_coarse",
        ),
        refinement_artifacts_by_family={
            "G": LoadedModelArtifact(
                estimator=DummyGFamilyEstimator(),
                metadata={},
                task_name="gaia_mk_refinement_g_classification",
                target_column="spectral_subclass",
                feature_columns=("source_id", "teff_gspphot"),
                model_name="dummy_refinement",
            )
        },
        host_artifact=LoadedModelArtifact(
            estimator=DummyHostEstimator(),
            metadata={},
            task_name="host_field_classification",
            target_column="host_label",
            feature_columns=(
                "teff_gspphot",
                "logg_gspphot",
                "radius_gspphot",
                "parallax",
                "parallax_over_error",
                "ruwe",
                "bp_rp",
                "mh_gspphot",
            ),
            model_name="dummy_host",
        ),
    )


def test_run_final_decision_with_artifacts_builds_priority_for_clean_id_rows() -> None:
    result = run_final_decision_with_artifacts(
        build_base_frame(),
        bundle=build_bundle(),
        final_decision_policy=FinalDecisionPolicy(
            decision_policy_version="final_decision_v2",
            min_refinement_confidence=0.6,
        ),
        priority_config=PriorityIntegrationConfig(),
    )

    assert result.final_decision_df["final_domain_state"].tolist() == [
        "id",
        "candidate_ood",
        "ood",
    ]
    assert result.final_decision_df.loc[0, "final_refinement_label"] == "G4"
    assert result.priority_input_df["source_id"].astype(int).tolist() == [1]
    assert result.priority_ranking_df["source_id"].astype(str).tolist() == ["1"]
    assert result.final_decision_df.loc[0, "priority_state"] in {"high", "medium", "low"}
    assert pd.isna(result.final_decision_df.loc[1, "priority_state"])
    assert pd.isna(result.final_decision_df.loc[2, "priority_state"])
