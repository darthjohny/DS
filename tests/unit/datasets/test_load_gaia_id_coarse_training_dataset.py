# Тестовый файл `test_load_gaia_id_coarse_training_dataset.py` домена `datasets`.
#
# Этот файл проверяет только:
# - проверку логики домена: loader-слой и shape рабочих dataframe;
# - регрессионные сценарии и ожидаемые контракты целевого слоя.
#
# Следующий слой:
# - реализация домена `datasets` в `src/exohost`;
# - соседние тесты и testkit этого же пакета.

from __future__ import annotations

from exohost.contracts.hierarchical_dataset_contracts import (
    GAIA_ID_COARSE_TRAINING_CONTRACT,
)
from exohost.datasets.load_gaia_id_coarse_training_dataset import (
    build_gaia_id_coarse_training_query,
)


def test_build_gaia_id_coarse_training_query_contains_required_filters() -> None:
    query = build_gaia_id_coarse_training_query(
        GAIA_ID_COARSE_TRAINING_CONTRACT.relation_name,
        GAIA_ID_COARSE_TRAINING_CONTRACT.required_columns,
        limit=100,
    )

    assert "FROM lab.v_gaia_id_coarse_training" in query
    assert "spec_class IN ('O', 'B', 'A', 'F', 'G', 'K', 'M')" in query
    assert "is_evolved IS NOT NULL" in query
    assert "ORDER BY source_id ASC" in query
    assert "LIMIT 100" in query
