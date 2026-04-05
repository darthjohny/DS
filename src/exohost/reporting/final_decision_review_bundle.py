# Файл `final_decision_review_bundle.py` слоя `reporting`.
#
# Этот файл отвечает только за:
# - review-слой, notebook helper и отчетные dataframe;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - active notebook, docs и review-слой поверх этих helper;
# - unit-тесты reporting-пакета.

from __future__ import annotations

from pathlib import Path

import pandas as pd

from exohost.reporting.final_decision_artifacts import load_final_decision_artifacts
from exohost.reporting.final_decision_review_contracts import FinalDecisionReviewBundle


def load_final_decision_review_bundle(run_dir: str | Path) -> FinalDecisionReviewBundle:
    # Загружаем final-decision bundle из одного run_dir.
    artifact_dir = Path(run_dir)
    loaded_bundle = load_final_decision_artifacts(artifact_dir)
    return FinalDecisionReviewBundle(
        run_dir=artifact_dir,
        decision_input_df=loaded_bundle.decision_input_df,
        final_decision_df=loaded_bundle.final_decision_df,
        priority_input_df=loaded_bundle.priority_input_df,
        priority_ranking_df=loaded_bundle.priority_ranking_df,
        metadata=loaded_bundle.metadata,
    )


def build_final_decision_summary_frame(bundle: FinalDecisionReviewBundle) -> pd.DataFrame:
    # Собираем компактный one-row summary по final decision run.
    return pd.DataFrame(
        [
            {
                "run_dir": bundle.run_dir.name,
                "pipeline_name": bundle.metadata.get("pipeline_name", "unknown"),
                "created_at_utc": bundle.metadata.get("created_at_utc", "unknown"),
                "n_rows_input": bundle.metadata.get(
                    "n_rows_input",
                    int(bundle.decision_input_df.shape[0]),
                ),
                "n_rows_final_decision": bundle.metadata.get(
                    "n_rows_final_decision",
                    int(bundle.final_decision_df.shape[0]),
                ),
                "n_rows_priority_input": bundle.metadata.get(
                    "n_rows_priority_input",
                    int(bundle.priority_input_df.shape[0]),
                ),
                "n_rows_priority_ranking": bundle.metadata.get(
                    "n_rows_priority_ranking",
                    int(bundle.priority_ranking_df.shape[0]),
                ),
                "n_unique_source_id": int(
                    bundle.final_decision_df.loc[:, "source_id"].nunique(dropna=False)
                ),
            }
        ]
    )


__all__ = [
    "build_final_decision_summary_frame",
    "load_final_decision_review_bundle",
]
