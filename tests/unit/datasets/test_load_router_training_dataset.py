# Тестовый файл `test_load_router_training_dataset.py` домена `datasets`.
#
# Этот файл проверяет только:
# - проверку логики домена: loader-слой и shape рабочих dataframe;
# - регрессионные сценарии и ожидаемые контракты целевого слоя.
#
# Следующий слой:
# - реализация домена `datasets` в `src/exohost`;
# - соседние тесты и testkit этого же пакета.

from __future__ import annotations

from exohost.contracts.dataset_contracts import ROUTER_TRAINING_CONTRACT
from exohost.datasets.load_router_training_dataset import build_router_training_query


def test_build_router_training_query_contains_required_filters() -> None:
    # Проверяем базовые фильтры качества и label-контракт для router source.
    query = build_router_training_query(
        ROUTER_TRAINING_CONTRACT.relation_name,
        ROUTER_TRAINING_CONTRACT.required_columns,
        limit=100,
    )

    assert "FROM lab.v_gaia_router_training" in query
    assert "spec_class IN ('O', 'B', 'A', 'F', 'G', 'K', 'M')" in query
    assert "evolution_stage IN ('dwarf', 'evolved')" in query
    assert "ORDER BY random_index ASC, source_id ASC" in query
    assert "LIMIT 100" in query
