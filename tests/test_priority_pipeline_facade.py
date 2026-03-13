"""Точечные тесты публичного API пакета `priority_pipeline`."""

from __future__ import annotations

import pandas as pd

import priority_pipeline as pipeline_pkg
import priority_pipeline.contracts as pipeline_contracts


def test_priority_pipeline_package_reexports_core_public_symbols() -> None:
    """Пакет должен публиковать canonical facade поверх внутренних модулей."""
    assert pipeline_pkg.PipelineRunResult is pipeline_contracts.PipelineRunResult
    assert pipeline_pkg.ensure_decision_columns is not None
    assert pipeline_pkg.split_router_branches is not None
    assert pipeline_pkg.run_pipeline is not None
    assert pipeline_pkg.save_priority_results is not None


def test_priority_pipeline_run_result_contract_is_simple_dataclass() -> None:
    """Pipeline result contract должен оставаться простым типизированным контейнером."""
    result = pipeline_pkg.PipelineRunResult(
        run_id="run_1",
        router_results=pd.DataFrame(),
        priority_results=pd.DataFrame(),
    )

    assert result.run_id == "run_1"
    assert result.router_results.empty
    assert result.priority_results.empty
