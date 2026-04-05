# Тестовый файл `test_posthoc_id_ood_gate.py` домена `posthoc`.
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
import pytest

from exohost.posthoc.id_ood_gate import (
    ID_OOD_CANDIDATE_OOD_STATE,
    ID_OOD_IN_DOMAIN_STATE,
    ID_OOD_OOD_STATE,
    IdOodThresholdPolicy,
    build_id_ood_gate_scored_frame,
    build_id_ood_metrics_probability_frame,
    build_id_ood_threshold_policy,
    decide_id_ood_state,
)


class DummyIdOodClassifier:
    # Минимальная binary probability model для проверки gate-contract.

    def __init__(self) -> None:
        self.model_name = "dummy_id_ood_classifier"
        self.classes_ = np.asarray(["id", "ood"], dtype=object)

    def predict(self, X: pd.DataFrame) -> npt.NDArray[np.object_]:
        return np.asarray(
            [
                "ood" if float(value) >= 0.5 else "id"
                for value in X["bp_rp"].astype(float).tolist()
            ],
            dtype=object,
        )

    def predict_proba(self, X: pd.DataFrame) -> npt.NDArray[np.float64]:
        probabilities: list[list[float]] = []
        for value in X["bp_rp"].astype(float).tolist():
            ood_probability = min(max(float(value), 0.0), 1.0)
            probabilities.append([1.0 - ood_probability, ood_probability])
        return np.asarray(probabilities, dtype=float)


def test_decide_id_ood_state_supports_candidate_band() -> None:
    policy = IdOodThresholdPolicy(
        threshold_name="manual",
        threshold_value=0.7,
        threshold_metric="balanced_accuracy",
        threshold_fit_scope="validation",
        threshold_policy_version="v1",
        candidate_ood_threshold=0.4,
    )

    assert decide_id_ood_state(0.2, policy=policy) == ID_OOD_IN_DOMAIN_STATE
    assert decide_id_ood_state(0.5, policy=policy) == ID_OOD_CANDIDATE_OOD_STATE
    assert decide_id_ood_state(0.8, policy=policy) == ID_OOD_OOD_STATE


def test_build_id_ood_gate_scored_frame_adds_probabilities_and_states() -> None:
    policy = build_id_ood_threshold_policy(
        tuned_threshold=0.7,
        threshold_policy_version="id_ood_threshold_v1",
        candidate_ood_threshold=0.4,
    )
    scored = build_id_ood_gate_scored_frame(
        pd.DataFrame(
            [
                {
                    "source_id": 1,
                    "teff_gspphot": 5000.0,
                    "logg_gspphot": 4.5,
                    "mh_gspphot": 0.0,
                    "bp_rp": 0.2,
                    "parallax": 10.0,
                    "parallax_over_error": 12.0,
                    "ruwe": 1.0,
                    "phot_g_mean_mag": 11.0,
                },
                {
                    "source_id": 2,
                    "teff_gspphot": 5200.0,
                    "logg_gspphot": 4.4,
                    "mh_gspphot": 0.1,
                    "bp_rp": 0.5,
                    "parallax": 9.0,
                    "parallax_over_error": 11.0,
                    "ruwe": 1.1,
                    "phot_g_mean_mag": 12.0,
                },
                {
                    "source_id": 3,
                    "teff_gspphot": 5400.0,
                    "logg_gspphot": 4.3,
                    "mh_gspphot": 0.2,
                    "bp_rp": 0.9,
                    "parallax": 8.0,
                    "parallax_over_error": 10.0,
                    "ruwe": 1.2,
                    "phot_g_mean_mag": 13.0,
                },
            ]
        ),
        estimator=DummyIdOodClassifier(),
        feature_columns=(
            "teff_gspphot",
            "logg_gspphot",
            "mh_gspphot",
            "bp_rp",
            "parallax",
            "parallax_over_error",
            "ruwe",
            "phot_g_mean_mag",
        ),
        policy=policy,
    )

    assert scored["predicted_domain_target"].tolist() == ["id", "id", "ood"]
    assert scored["ood_decision"].tolist() == [
        ID_OOD_IN_DOMAIN_STATE,
        ID_OOD_CANDIDATE_OOD_STATE,
        ID_OOD_OOD_STATE,
    ]
    probability_frame = build_id_ood_metrics_probability_frame(scored)
    assert tuple(probability_frame.columns) == ("id", "ood")


def test_id_ood_threshold_policy_rejects_invalid_candidate_threshold() -> None:
    with pytest.raises(ValueError, match="candidate_ood_threshold"):
        IdOodThresholdPolicy(
            threshold_name="manual",
            threshold_value=0.3,
            threshold_metric="balanced_accuracy",
            threshold_fit_scope="validation",
            threshold_policy_version="v1",
            candidate_ood_threshold=0.5,
        )
