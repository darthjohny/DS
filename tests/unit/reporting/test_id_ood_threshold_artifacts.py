# Тестовый файл `test_id_ood_threshold_artifacts.py` домена `reporting`.
#
# Этот файл проверяет только:
# - проверку логики домена: review-helper, сводки и notebook display-слой;
# - регрессионные сценарии и ожидаемые контракты целевого слоя.
#
# Следующий слой:
# - реализация домена `reporting` в `src/exohost`;
# - соседние тесты и testkit этого же пакета.

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from exohost.posthoc.id_ood_gate import build_id_ood_threshold_policy
from exohost.reporting.id_ood_threshold_artifacts import (
    build_id_ood_threshold_artifact_paths,
    load_id_ood_threshold_artifact,
    save_id_ood_threshold_artifact,
)


def build_threshold_policy():
    return build_id_ood_threshold_policy(
        tuned_threshold=0.73,
        threshold_policy_version="id_ood_threshold_v1",
        candidate_ood_threshold=0.41,
    )


def test_build_id_ood_threshold_artifact_paths_creates_stable_layout(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 3, 28, 19, 0, 0, tzinfo=UTC)
    paths = build_id_ood_threshold_artifact_paths(
        output_dir=tmp_path,
        task_name="gaia_id_ood_classification",
        model_name="hist_gradient_boosting",
        now=now,
    )

    assert paths.run_dir.parent == tmp_path
    assert paths.threshold_policy_json_path.name == "threshold_policy.json"
    assert paths.metadata_json_path.name == "metadata.json"


def test_save_id_ood_threshold_artifact_writes_json_and_metadata(
    tmp_path: Path,
) -> None:
    paths = save_id_ood_threshold_artifact(
        build_threshold_policy(),
        task_name="gaia_id_ood_classification",
        model_name="hist_gradient_boosting",
        output_dir=tmp_path,
        now=datetime(2026, 3, 28, 19, 0, 0, tzinfo=UTC),
        extra_metadata={"stage": "posthoc_gate"},
    )

    threshold_payload = json.loads(paths.threshold_policy_json_path.read_text(encoding="utf-8"))
    metadata = json.loads(paths.metadata_json_path.read_text(encoding="utf-8"))

    assert threshold_payload["threshold_value"] == 0.73
    assert threshold_payload["candidate_ood_threshold"] == 0.41
    assert metadata["task_name"] == "gaia_id_ood_classification"
    assert metadata["model_name"] == "hist_gradient_boosting"
    assert metadata["context"]["stage"] == "posthoc_gate"


def test_load_id_ood_threshold_artifact_restores_typed_contract(
    tmp_path: Path,
) -> None:
    paths = save_id_ood_threshold_artifact(
        build_threshold_policy(),
        task_name="gaia_id_ood_classification",
        model_name="hist_gradient_boosting",
        output_dir=tmp_path,
    )

    loaded_artifact = load_id_ood_threshold_artifact(paths.run_dir)

    assert loaded_artifact.task_name == "gaia_id_ood_classification"
    assert loaded_artifact.model_name == "hist_gradient_boosting"
    assert loaded_artifact.policy.threshold_value == 0.73
    assert loaded_artifact.policy.candidate_ood_threshold == 0.41
