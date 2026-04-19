# Тестовый файл `test_ui_run_overview.py` домена `ui`.
#
# Этот файл проверяет только:
# - компактную сводку по готовому run bundle;
# - сохранение главных прикладных счетчиков для домашней страницы интерфейса.
#
# Следующий слой:
# - компонент домашней страницы;
# - loader-слой и связка с read-only run_dir.

from __future__ import annotations

from pathlib import Path

import pandas as pd

from exohost.reporting.final_decision_artifacts import LoadedFinalDecisionArtifacts
from exohost.ui.loaders import UiLoadedRunBundle
from exohost.ui.run_overview import build_ui_run_overview


def test_build_ui_run_overview_extracts_main_counts() -> None:
    bundle = UiLoadedRunBundle(
        run_dir=Path("artifacts/decisions/hierarchical_final_decision_demo"),
        loaded_artifacts=LoadedFinalDecisionArtifacts(
            decision_input_df=pd.DataFrame({"source_id": [1, 2, 3]}),
            final_decision_df=pd.DataFrame(
                {
                    "source_id": [1, 2, 3],
                    "final_domain_state": ["id", "unknown", "ood"],
                }
            ),
            priority_input_df=pd.DataFrame({"source_id": [1, 2, 3]}),
            priority_ranking_df=pd.DataFrame(
                {
                    "source_id": [1, 2, 3],
                    "priority_label": ["high", "medium", "low"],
                }
            ),
            metadata={
                "pipeline_name": "hierarchical_final_decision",
                "created_at_utc": "2026-04-17T12:00:00+00:00",
                "n_rows_input": 3,
                "n_rows_final_decision": 3,
            },
        ),
    )

    overview = build_ui_run_overview(bundle)

    assert overview.run_dir_name == "hierarchical_final_decision_demo"
    assert overview.pipeline_name == "hierarchical_final_decision"
    assert overview.id_count == 1
    assert overview.unknown_count == 1
    assert overview.ood_count == 1
    assert overview.high_priority_count == 1
    assert overview.medium_priority_count == 1
    assert overview.low_priority_count == 1
