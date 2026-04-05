# Файл `model_pipeline_review_bundle.py` слоя `reporting`.
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
from pathlib import Path
from typing import Any

import pandas as pd

from exohost.reporting.benchmark_report import load_benchmark_artifacts
from exohost.reporting.model_pipeline_review_contracts import BenchmarkReviewBundle


def load_benchmark_review_bundle(run_dir: str | Path) -> BenchmarkReviewBundle:
    # Загружаем benchmark-таблицы, metadata и target distribution из одного run_dir.
    benchmark_dir = Path(run_dir)
    metrics_df, cv_summary_df, metadata = load_benchmark_artifacts(benchmark_dir)
    target_distribution_df = pd.read_csv(benchmark_dir / "target_distribution.csv")
    return BenchmarkReviewBundle(
        run_dir=benchmark_dir,
        metrics_df=metrics_df,
        cv_summary_df=cv_summary_df,
        target_distribution_df=target_distribution_df,
        metadata=metadata,
    )


def load_benchmark_metadata_only(run_dir: str | Path) -> dict[str, Any]:
    # Нужен для легкой проверки notebook-конфига без чтения тяжелых frame-ов.
    benchmark_dir = Path(run_dir)
    return json.loads((benchmark_dir / "metadata.json").read_text(encoding="utf-8"))


__all__ = [
    "load_benchmark_metadata_only",
    "load_benchmark_review_bundle",
]
