# Тестовый файл `test_load_gaia_id_ood_training_dataset.py` домена `datasets`.
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
    GAIA_ID_OOD_TRAINING_CONTRACT,
)
from exohost.datasets.load_gaia_id_ood_training_dataset import (
    build_gaia_id_ood_training_query,
)


def test_build_gaia_id_ood_training_query_contains_required_filters() -> None:
    query = build_gaia_id_ood_training_query(
        GAIA_ID_OOD_TRAINING_CONTRACT.relation_name,
        GAIA_ID_OOD_TRAINING_CONTRACT.required_columns + ("random_index",),
        limit=75,
    )

    assert "FROM lab.v_gaia_id_ood_training" in query
    assert "domain_target IN ('id', 'ood')" in query
    assert "ORDER BY random_index ASC, domain_target ASC, source_id ASC" in query
    assert "LIMIT 75" in query
