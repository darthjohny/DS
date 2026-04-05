# Файл `id_ood_threshold_artifacts.py` слоя `reporting`.
#
# Этот файл отвечает только за:
# - review-слой, notebook helper и отчетные dataframe;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - active notebook, docs и review-слой поверх этих helper;
# - unit-тесты reporting-пакета.

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from exohost.posthoc.id_ood_gate import IdOodThresholdPolicy
from exohost.reporting.benchmark_artifacts import build_run_stamp, sanitize_artifact_name

DEFAULT_ID_OOD_THRESHOLD_OUTPUT_DIR = Path("artifacts/thresholds")


@dataclass(frozen=True, slots=True)
class IdOodThresholdArtifactPaths:
    # Пути к threshold-policy artifact одного post-hoc прогона.
    run_dir: Path
    threshold_policy_json_path: Path
    metadata_json_path: Path


@dataclass(frozen=True, slots=True)
class LoadedIdOodThresholdArtifact:
    # Загруженный threshold-policy artifact вместе с metadata.
    policy: IdOodThresholdPolicy
    metadata: dict[str, Any]
    task_name: str
    model_name: str


def build_id_ood_threshold_artifact_paths(
    *,
    output_dir: str | Path,
    task_name: str,
    model_name: str,
    now: datetime | None = None,
) -> IdOodThresholdArtifactPaths:
    # Собираем стандартную файловую структуру threshold-policy artifact.
    base_dir = Path(output_dir)
    run_name = (
        f"{sanitize_artifact_name(task_name)}__"
        f"{sanitize_artifact_name(model_name)}__"
        f"threshold__{build_run_stamp(now)}"
    )
    run_dir = base_dir / run_name
    return IdOodThresholdArtifactPaths(
        run_dir=run_dir,
        threshold_policy_json_path=run_dir / "threshold_policy.json",
        metadata_json_path=run_dir / "metadata.json",
    )


def build_id_ood_threshold_metadata(
    policy: IdOodThresholdPolicy,
    *,
    task_name: str,
    model_name: str,
    created_at: datetime,
    extra_metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    # Собираем metadata threshold-policy artifact.
    payload: dict[str, object] = {
        "task_name": task_name,
        "model_name": model_name,
        "created_at_utc": created_at.astimezone(UTC).isoformat(),
        "threshold_name": policy.threshold_name,
        "threshold_value": float(policy.threshold_value),
        "threshold_metric": policy.threshold_metric,
        "threshold_fit_scope": policy.threshold_fit_scope,
        "threshold_policy_version": policy.threshold_policy_version,
        "candidate_ood_threshold": (
            None
            if policy.candidate_ood_threshold is None
            else float(policy.candidate_ood_threshold)
        ),
    }
    if extra_metadata:
        payload["context"] = dict(extra_metadata)
    return payload


def save_id_ood_threshold_artifact(
    policy: IdOodThresholdPolicy,
    *,
    task_name: str,
    model_name: str,
    output_dir: str | Path = DEFAULT_ID_OOD_THRESHOLD_OUTPUT_DIR,
    now: datetime | None = None,
    extra_metadata: Mapping[str, object] | None = None,
) -> IdOodThresholdArtifactPaths:
    # Сохраняем threshold-policy artifact в отдельный run_dir.
    created_at = now or datetime.now(UTC)
    paths = build_id_ood_threshold_artifact_paths(
        output_dir=output_dir,
        task_name=task_name,
        model_name=model_name,
        now=created_at,
    )
    paths.run_dir.mkdir(parents=True, exist_ok=False)
    paths.threshold_policy_json_path.write_text(
        json.dumps(
            _build_threshold_policy_payload(policy),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    metadata = build_id_ood_threshold_metadata(
        policy,
        task_name=task_name,
        model_name=model_name,
        created_at=created_at,
        extra_metadata=extra_metadata,
    )
    paths.metadata_json_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return paths


def load_id_ood_threshold_artifact(run_dir: str | Path) -> LoadedIdOodThresholdArtifact:
    # Загружаем threshold-policy artifact из run_dir и валидируем contract.
    artifact_dir = Path(run_dir)
    threshold_payload = json.loads(
        (artifact_dir / "threshold_policy.json").read_text(encoding="utf-8")
    )
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
    return LoadedIdOodThresholdArtifact(
        policy=_build_threshold_policy_from_payload(threshold_payload),
        metadata=metadata,
        task_name=_require_metadata_string(metadata, field_name="task_name"),
        model_name=_require_metadata_string(metadata, field_name="model_name"),
    )


def _build_threshold_policy_payload(policy: IdOodThresholdPolicy) -> dict[str, object]:
    return {
        "threshold_name": policy.threshold_name,
        "threshold_value": float(policy.threshold_value),
        "threshold_metric": policy.threshold_metric,
        "threshold_fit_scope": policy.threshold_fit_scope,
        "threshold_policy_version": policy.threshold_policy_version,
        "candidate_ood_threshold": (
            None
            if policy.candidate_ood_threshold is None
            else float(policy.candidate_ood_threshold)
        ),
    }


def _build_threshold_policy_from_payload(payload: Mapping[str, Any]) -> IdOodThresholdPolicy:
    return IdOodThresholdPolicy(
        threshold_name=_require_metadata_string(payload, field_name="threshold_name"),
        threshold_value=_require_metadata_float(payload, field_name="threshold_value"),
        threshold_metric=_require_metadata_string(payload, field_name="threshold_metric"),
        threshold_fit_scope=_require_metadata_string(payload, field_name="threshold_fit_scope"),
        threshold_policy_version=_require_metadata_string(
            payload,
            field_name="threshold_policy_version",
        ),
        candidate_ood_threshold=_require_optional_metadata_float(
            payload,
            field_name="candidate_ood_threshold",
        ),
    )


def _require_metadata_string(
    metadata: Mapping[str, Any],
    *,
    field_name: str,
) -> str:
    value = metadata.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"ID/OOD threshold artifact field '{field_name}' must be a non-empty string."
        )
    return value


def _require_metadata_float(
    metadata: Mapping[str, Any],
    *,
    field_name: str,
) -> float:
    value = metadata.get(field_name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"ID/OOD threshold artifact field '{field_name}' must be numeric."
        )
    return float(value)


def _require_optional_metadata_float(
    metadata: Mapping[str, Any],
    *,
    field_name: str,
) -> float | None:
    value = metadata.get(field_name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"ID/OOD threshold artifact field '{field_name}' must be numeric or null."
        )
    return float(value)
