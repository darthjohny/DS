# Тестовый файл `test_refinement_family_tasks.py` домена `evaluation`.
#
# Этот файл проверяет только:
# - проверку логики домена: метрики, split-логику и benchmark contracts;
# - регрессионные сценарии и ожидаемые контракты целевого слоя.
#
# Следующий слой:
# - реализация домена `evaluation` в `src/exohost`;
# - соседние тесты и testkit этого же пакета.

from __future__ import annotations

from exohost.evaluation.refinement_family_tasks import (
    REFINEMENT_FAMILY_TASK_BY_NAME,
    build_refinement_family_task_definition,
    build_refinement_family_task_name,
)


def test_refinement_family_task_registry_exposes_expected_names() -> None:
    assert tuple(sorted(REFINEMENT_FAMILY_TASK_BY_NAME)) == (
        "gaia_mk_refinement_a_classification",
        "gaia_mk_refinement_b_classification",
        "gaia_mk_refinement_f_classification",
        "gaia_mk_refinement_g_classification",
        "gaia_mk_refinement_k_classification",
        "gaia_mk_refinement_m_classification",
    )


def test_build_refinement_family_task_definition_uses_family_target() -> None:
    definition = build_refinement_family_task_definition("K")

    assert definition.spectral_class == "K"
    assert definition.target_cardinality == 9
    assert definition.task.name == "gaia_mk_refinement_k_classification"
    assert definition.task.target_column == "spectral_subclass"
    assert "evolstage_flame" in definition.task.feature_columns


def test_build_refinement_family_task_name_normalizes_case() -> None:
    assert build_refinement_family_task_name(" m ") == "gaia_mk_refinement_m_classification"
