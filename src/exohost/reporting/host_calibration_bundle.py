# Файл `host_calibration_bundle.py` слоя `reporting`.
#
# Этот файл отвечает только за:
# - review-слой, notebook helper и отчетные dataframe;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - active notebook, docs и review-слой поверх этих helper;
# - unit-тесты reporting-пакета.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from exohost.reporting.binary_calibration_review import (
    DEFAULT_BINARY_CALIBRATION_CONFIG,
    BinaryCalibrationConfig,
)
from exohost.reporting.host_calibration_review import (
    build_host_calibration_curve_review_frame,
    build_host_calibration_group_frame,
    build_host_calibration_metric_summary_frame,
    build_host_calibration_split_summary_frame,
    build_host_probability_bin_review_frame,
)
from exohost.reporting.host_calibration_source import (
    HostCalibrationSource,
    build_host_calibration_source_from_model_artifact,
)
from exohost.reporting.model_artifacts import (
    DEFAULT_MODEL_OUTPUT_DIR,
    load_model_artifact_metadata,
    require_metadata_string,
)

HOST_CALIBRATION_TASK_NAME = "host_field_classification"


@dataclass(frozen=True, slots=True)
class HostCalibrationReviewBundle:
    # Полный пакет табличного calibration-review для host model run.
    run_dir: Path
    metadata: dict[str, Any]
    source: HostCalibrationSource
    split_summary_df: pd.DataFrame
    metric_summary_df: pd.DataFrame
    curve_df: pd.DataFrame
    probability_bin_df: pd.DataFrame
    class_group_df: pd.DataFrame
    stage_group_df: pd.DataFrame


def find_latest_host_model_run_dir(
    artifacts_root: str | Path = DEFAULT_MODEL_OUTPUT_DIR,
) -> Path | None:
    # Возвращаем самый свежий host model artifact по metadata и имени run_dir.
    candidate_rows: list[tuple[str, str, Path]] = []
    root = Path(artifacts_root)
    if not root.exists():
        return None

    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        metadata_path = run_dir / "metadata.json"
        if not metadata_path.exists():
            continue

        try:
            metadata = load_model_artifact_metadata(run_dir)
            task_name = require_metadata_string(metadata, field_name="task_name")
            created_at_utc = require_metadata_string(metadata, field_name="created_at_utc")
        except (ValueError, TypeError):
            continue

        if task_name != HOST_CALIBRATION_TASK_NAME:
            continue
        candidate_rows.append((created_at_utc, run_dir.name, run_dir))

    if not candidate_rows:
        return None

    candidate_rows.sort(key=lambda item: (item[0], item[1]))
    return candidate_rows[-1][2]


def load_host_calibration_review_bundle(
    model_run_dir: str | Path,
    *,
    host_limit: int | None = None,
    router_limit: int | None = None,
    field_to_host_ratio: int = 1,
    calibration_config: BinaryCalibrationConfig = DEFAULT_BINARY_CALIBRATION_CONFIG,
    dotenv_path: str = ".env",
    connect_timeout: int | None = 10,
) -> HostCalibrationReviewBundle:
    # Собираем все review-frame'ы для выбранного host model artifact.
    run_dir = Path(model_run_dir)
    metadata = load_model_artifact_metadata(run_dir)
    source = build_host_calibration_source_from_model_artifact(
        run_dir,
        host_limit=host_limit,
        router_limit=router_limit,
        field_to_host_ratio=field_to_host_ratio,
        dotenv_path=dotenv_path,
        connect_timeout=connect_timeout,
    )
    return HostCalibrationReviewBundle(
        run_dir=run_dir,
        metadata=metadata,
        source=source,
        split_summary_df=build_host_calibration_split_summary_frame(source),
        metric_summary_df=build_host_calibration_metric_summary_frame(
            source,
            config=calibration_config,
        ),
        curve_df=build_host_calibration_curve_review_frame(
            source,
            config=calibration_config,
        ),
        probability_bin_df=build_host_probability_bin_review_frame(
            source,
            config=calibration_config,
        ),
        class_group_df=build_host_calibration_group_frame(source, group_column="spec_class"),
        stage_group_df=build_host_calibration_group_frame(
            source,
            group_column="evolution_stage",
        ),
    )
