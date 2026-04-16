# Тестовый файл `test_posthoc_decision_model_bundle.py` домена `posthoc`.
#
# Этот файл проверяет только:
# - проверку логики домена: post-hoc routing, gate и final decision policy;
# - регрессионные сценарии и ожидаемые контракты целевого слоя.
#
# Следующий слой:
# - реализация домена `posthoc` в `src/exohost`;
# - соседние тесты и testkit этого же пакета.

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from exohost.evaluation.hierarchical_tasks import (
    GAIA_ID_COARSE_CLASSIFICATION_TASK,
    GAIA_ID_OOD_CLASSIFICATION_TASK,
)
from exohost.models.hgb_classifier import HGBClassifier
from exohost.models.protocol import ModelSpec
from exohost.posthoc.decision_model_bundle import (
    build_final_decision_feature_union,
    load_final_decision_model_bundle,
)
from exohost.posthoc.id_ood_gate import build_id_ood_threshold_policy
from exohost.reporting.id_ood_threshold_artifacts import save_id_ood_threshold_artifact
from exohost.reporting.model_artifacts import save_model_artifacts
from exohost.training.train_runner import TrainRunResult, run_training


def build_id_ood_training_frame() -> pd.DataFrame:
    # Минимальный train-frame для OOD-артефакта. Его задача — дать bundle-тесту
    # воспроизводимый источник модели и порогов, а не реалистичный benchmark.
    return pd.DataFrame(
        [
            {
                "source_id": 1,
                "teff_gspphot": 5800.0,
                "logg_gspphot": 4.4,
                "mh_gspphot": 0.1,
                "bp_rp": 0.75,
                "parallax": 15.0,
                "parallax_over_error": 18.0,
                "ruwe": 1.01,
                "phot_g_mean_mag": 10.8,
                "domain_target": "id",
            },
            {
                "source_id": 2,
                "teff_gspphot": 4700.0,
                "logg_gspphot": 4.7,
                "mh_gspphot": -0.2,
                "bp_rp": 1.20,
                "parallax": 8.0,
                "parallax_over_error": 6.0,
                "ruwe": 1.40,
                "phot_g_mean_mag": 14.0,
                "domain_target": "ood",
            },
            {
                "source_id": 3,
                "teff_gspphot": 5900.0,
                "logg_gspphot": 4.3,
                "mh_gspphot": 0.0,
                "bp_rp": 0.70,
                "parallax": 14.0,
                "parallax_over_error": 17.0,
                "ruwe": 1.02,
                "phot_g_mean_mag": 11.0,
                "domain_target": "id",
            },
            {
                "source_id": 4,
                "teff_gspphot": 4300.0,
                "logg_gspphot": 4.8,
                "mh_gspphot": -0.3,
                "bp_rp": 1.35,
                "parallax": 7.5,
                "parallax_over_error": 5.5,
                "ruwe": 1.50,
                "phot_g_mean_mag": 14.2,
                "domain_target": "ood",
            },
        ]
    )


def build_coarse_training_frame() -> pd.DataFrame:
    # Небольшой coarse train-frame с двумя классами нужен для проверки,
    # что bundle корректно поднимает feature union и metadata обеих моделей.
    return pd.DataFrame(
        [
            {
                "source_id": 10,
                "teff_gspphot": 5800.0,
                "logg_gspphot": 4.4,
                "mh_gspphot": 0.1,
                "bp_rp": 0.75,
                "parallax": 15.0,
                "parallax_over_error": 18.0,
                "ruwe": 1.01,
                "radius_feature": 1.0,
                "spec_class": "G",
            },
            {
                "source_id": 11,
                "teff_gspphot": 4500.0,
                "logg_gspphot": 4.7,
                "mh_gspphot": -0.2,
                "bp_rp": 1.10,
                "parallax": 11.0,
                "parallax_over_error": 16.0,
                "ruwe": 1.02,
                "radius_feature": 0.8,
                "spec_class": "K",
            },
            {
                "source_id": 12,
                "teff_gspphot": 5900.0,
                "logg_gspphot": 4.3,
                "mh_gspphot": 0.0,
                "bp_rp": 0.70,
                "parallax": 14.0,
                "parallax_over_error": 19.0,
                "ruwe": 1.03,
                "radius_feature": 1.1,
                "spec_class": "G",
            },
            {
                "source_id": 13,
                "teff_gspphot": 4400.0,
                "logg_gspphot": 4.8,
                "mh_gspphot": -0.1,
                "bp_rp": 1.15,
                "parallax": 10.0,
                "parallax_over_error": 15.0,
                "ruwe": 1.04,
                "radius_feature": 0.7,
                "spec_class": "K",
            },
        ]
    )


def build_train_result(
    *,
    frame: pd.DataFrame,
    task_name: str,
) -> TrainRunResult:
    # Один helper на обучение не дает тесту размножать почти одинаковый setup
    # и держит bundle-сценарии на одном способе подготовки артефактов.
    if task_name == GAIA_ID_OOD_CLASSIFICATION_TASK.name:
        task = GAIA_ID_OOD_CLASSIFICATION_TASK
    elif task_name == GAIA_ID_COARSE_CLASSIFICATION_TASK.name:
        task = GAIA_ID_COARSE_CLASSIFICATION_TASK
    else:
        raise ValueError(f"Unsupported task_name: {task_name}")

    return run_training(
        frame,
        task=task,
        model_spec=ModelSpec(
            model_name="hist_gradient_boosting",
            estimator=HGBClassifier(
                feature_columns=task.feature_columns,
                max_iter=20,
                min_samples_leaf=1,
                model_name="hist_gradient_boosting",
            ),
        ),
    )


def test_load_final_decision_model_bundle_reads_valid_bundle(tmp_path: Path) -> None:
    # Проверяем основной happy path: артефакты моделей и порогов читаются вместе,
    # а итоговый bundle возвращает согласованный набор feature columns.
    ood_paths = save_model_artifacts(
        build_train_result(
            frame=build_id_ood_training_frame(),
            task_name=GAIA_ID_OOD_CLASSIFICATION_TASK.name,
        ),
        output_dir=tmp_path / "models",
    )
    coarse_paths = save_model_artifacts(
        build_train_result(
            frame=build_coarse_training_frame(),
            task_name=GAIA_ID_COARSE_CLASSIFICATION_TASK.name,
        ),
        output_dir=tmp_path / "models",
    )
    threshold_paths = save_id_ood_threshold_artifact(
        build_id_ood_threshold_policy(
            tuned_threshold=0.61,
            threshold_policy_version="id_ood_threshold_v1",
            candidate_ood_threshold=0.42,
        ),
        task_name=GAIA_ID_OOD_CLASSIFICATION_TASK.name,
        model_name="hist_gradient_boosting",
        output_dir=tmp_path / "thresholds",
    )

    bundle = load_final_decision_model_bundle(
        ood_model_run_dir=ood_paths.run_dir,
        ood_threshold_run_dir=threshold_paths.run_dir,
        coarse_model_run_dir=coarse_paths.run_dir,
    )

    assert bundle.ood_artifact.task_name == GAIA_ID_OOD_CLASSIFICATION_TASK.name
    assert bundle.coarse_artifact.task_name == GAIA_ID_COARSE_CLASSIFICATION_TASK.name
    assert bundle.ood_threshold_artifact.policy.threshold_value == 0.61
    feature_union = build_final_decision_feature_union(bundle)
    assert "teff_gspphot" in feature_union
    assert "radius_feature" in feature_union


def test_load_final_decision_model_bundle_rejects_misaligned_threshold_artifact(
    tmp_path: Path,
) -> None:
    # Если threshold artifact собран для другого `model_name`, bundle должен
    # упасть сразу, а не тащить несогласованную пару в боевой pipeline.
    ood_paths = save_model_artifacts(
        build_train_result(
            frame=build_id_ood_training_frame(),
            task_name=GAIA_ID_OOD_CLASSIFICATION_TASK.name,
        ),
        output_dir=tmp_path / "models",
    )
    coarse_paths = save_model_artifacts(
        build_train_result(
            frame=build_coarse_training_frame(),
            task_name=GAIA_ID_COARSE_CLASSIFICATION_TASK.name,
        ),
        output_dir=tmp_path / "models",
    )
    threshold_paths = save_id_ood_threshold_artifact(
        build_id_ood_threshold_policy(
            tuned_threshold=0.61,
            threshold_policy_version="id_ood_threshold_v1",
        ),
        task_name=GAIA_ID_OOD_CLASSIFICATION_TASK.name,
        model_name="mlp_classifier",
        output_dir=tmp_path / "thresholds",
    )

    with pytest.raises(ValueError, match="model_name must match"):
        load_final_decision_model_bundle(
            ood_model_run_dir=ood_paths.run_dir,
            ood_threshold_run_dir=threshold_paths.run_dir,
            coarse_model_run_dir=coarse_paths.run_dir,
        )
