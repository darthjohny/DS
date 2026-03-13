"""Точечные тесты compatibility facade `src/model_comparison.py`."""

from __future__ import annotations

import analysis.model_comparison as comparison_pkg

import model_comparison as comparison_facade


def test_model_comparison_facade_reexports_canonical_public_symbols() -> None:
    """Facade должен указывать на те же публичные символы, что и comparison package."""
    assert comparison_facade.SplitConfig is comparison_pkg.SplitConfig
    assert comparison_facade.ComparisonProtocol is comparison_pkg.ComparisonProtocol
    assert comparison_facade.run_main_contrastive_model is comparison_pkg.run_main_contrastive_model
    assert comparison_facade.run_snapshot_comparison is comparison_pkg.run_snapshot_comparison
    assert comparison_facade.DEFAULT_MLP_BASELINE_CONFIG == comparison_pkg.DEFAULT_MLP_BASELINE_CONFIG


def test_model_comparison_facade_declares_expected_cli_exports() -> None:
    """Facade `__all__` должен явно держать ключевые entrypoints benchmark-layer."""
    exported = set(comparison_facade.__all__)

    assert "main" in exported
    assert "parse_args" in exported
    assert "run_model_comparison" in exported
    assert "run_snapshot_comparison" in exported
    assert "ComparisonProtocol" in exported
