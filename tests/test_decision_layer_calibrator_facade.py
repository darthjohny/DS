"""Точечные тесты compatibility facade `decision_layer_calibrator.py`."""

from __future__ import annotations

import decision_calibration as calibrator_pkg
import decision_layer_calibrator as calibrator_facade


def test_decision_layer_calibrator_facade_reexports_canonical_symbols() -> None:
    """Facade должен реэкспортировать тот же публичный API, что и пакет калибратора."""
    assert calibrator_facade.CalibrationConfig is calibrator_pkg.CalibrationConfig
    assert calibrator_facade.IterationSummary is calibrator_pkg.IterationSummary
    assert calibrator_facade.build_iteration_summary is calibrator_pkg.build_iteration_summary
    assert calibrator_facade.save_iteration_artifacts is calibrator_pkg.save_iteration_artifacts
    assert calibrator_facade.DEFAULT_TOP_N == calibrator_pkg.DEFAULT_TOP_N


def test_decision_layer_calibrator_facade_declares_expected_entrypoints() -> None:
    """Facade `__all__` должен явно держать ключевые public entrypoints."""
    exported = set(calibrator_facade.__all__)

    assert "main" in exported
    assert "parse_args" in exported
    assert "CalibrationConfig" in exported
    assert "build_iteration_summary" in exported
    assert "save_iteration_artifacts" in exported
