# Тестовый файл `test_load_host_training_dataset.py` домена `datasets`.
#
# Этот файл проверяет только:
# - проверку логики домена: loader-слой и shape рабочих dataframe;
# - регрессионные сценарии и ожидаемые контракты целевого слоя.
#
# Следующий слой:
# - реализация домена `datasets` в `src/exohost`;
# - соседние тесты и testkit этого же пакета.

from __future__ import annotations

from exohost.contracts.dataset_contracts import HOST_TRAINING_CONTRACT
from exohost.datasets.load_host_training_dataset import build_host_training_query


def test_build_host_training_query_contains_required_filters() -> None:
    # Проверяем базовые фильтры и привязку к host training relation.
    query = build_host_training_query(
        HOST_TRAINING_CONTRACT.relation_name,
        HOST_TRAINING_CONTRACT.required_columns,
        limit=50,
    )

    assert "FROM lab.nasa_gaia_host_training_enriched" in query
    assert "radius_flame IS NOT NULL" in query
    assert "spec_class IN ('O', 'B', 'A', 'F', 'G', 'K', 'M')" in query
    assert "ORDER BY source_id ASC" in query
    assert "LIMIT 50" in query
