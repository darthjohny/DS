# Тестовый файл `test_evaluation_split.py` домена `evaluation`.
#
# Этот файл проверяет только:
# - проверку логики домена: метрики, split-логику и benchmark contracts;
# - регрессионные сценарии и ожидаемые контракты целевого слоя.
#
# Следующий слой:
# - реализация домена `evaluation` в `src/exohost`;
# - соседние тесты и testkit этого же пакета.

from __future__ import annotations

import pandas as pd
import pytest

from exohost.evaluation.protocol import CrossValidationConfig, SplitConfig
from exohost.evaluation.split import (
    build_cv_splitter,
    build_stratify_labels,
    split_dataset,
    validate_split_inputs,
)


def build_split_frame() -> pd.DataFrame:
    # Synthetic dataset с балансом по двум стратификационным группам.
    return pd.DataFrame(
        {
            "source_id": list(range(8)),
            "spec_class": ["G", "G", "G", "G", "K", "K", "K", "K"],
            "evolution_stage": [
                "dwarf",
                "dwarf",
                "evolved",
                "evolved",
                "dwarf",
                "dwarf",
                "evolved",
                "evolved",
            ],
        }
    )


def test_build_stratify_labels_combines_columns_in_order() -> None:
    # Проверяем, что стратификационная метка собирается детерминированно.
    labels = build_stratify_labels(build_split_frame(), ("spec_class", "evolution_stage"))

    assert labels.iloc[0] == "G|dwarf"
    assert labels.iloc[-1] == "K|evolved"


def test_validate_split_inputs_rejects_small_stratify_groups() -> None:
    # Группы размером в одну строку нельзя стратифицировать.
    frame = pd.DataFrame(
        {
            "spec_class": ["G", "K"],
            "evolution_stage": ["dwarf", "evolved"],
        }
    )

    with pytest.raises(ValueError, match="at least two rows"):
        validate_split_inputs(
            frame,
            split_config=SplitConfig(test_size=0.3, random_state=42),
            stratify_columns=("spec_class", "evolution_stage"),
        )


def test_split_dataset_returns_train_and_test_parts() -> None:
    # Проверяем, что split дает непустые train/test части.
    dataset_split = split_dataset(
        build_split_frame(),
        split_config=SplitConfig(test_size=0.5, random_state=42),
        stratify_columns=("spec_class", "evolution_stage"),
    )

    assert dataset_split.full_df.shape[0] == 8
    assert dataset_split.train_df.shape[0] == 4
    assert dataset_split.test_df.shape[0] == 4


def test_build_cv_splitter_uses_protocol_values() -> None:
    # Проверяем, что CV splitter наследует значения из конфига.
    splitter = build_cv_splitter(
        CrossValidationConfig(
            n_splits=5,
            shuffle=True,
            random_state=42,
        )
    )

    assert splitter.get_n_splits() == 5
