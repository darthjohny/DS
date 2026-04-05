# Тестовый файл `test_load_coarse_ob_downstream_boundary_dataset.py` домена `datasets`.
#
# Этот файл проверяет только:
# - проверку логики домена: loader-слой и shape рабочих dataframe;
# - регрессионные сценарии и ожидаемые контракты целевого слоя.
#
# Следующий слой:
# - реализация домена `datasets` в `src/exohost`;
# - соседние тесты и testkit этого же пакета.

from __future__ import annotations

from exohost.datasets.load_coarse_ob_downstream_boundary_dataset import (
    build_coarse_ob_downstream_boundary_query,
)


def test_build_coarse_ob_downstream_boundary_query_contains_expected_filters() -> None:
    query = build_coarse_ob_downstream_boundary_query(
        "lab.gaia_mk_quality_gated",
        ("source_id", "spectral_class", "teff_gspphot", "quality_state"),
        quality_state="pass",
        teff_min_k=10_000.0,
        limit=100,
    )

    assert "spectral_class IN ('O', 'B')" in query
    assert "quality_state = 'pass'" in query
    assert "teff_gspphot >= 10000.0" in query
    assert "LIMIT 100" in query
