# Файл `protocol.py` слоя `evaluation`.
#
# Этот файл отвечает только за:
# - метрики, split-логику и benchmark-task contracts;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - соседние модули слоя `evaluation` и orchestration поверх них;
# - unit-тесты и более верхние слои проекта.

from __future__ import annotations

from dataclasses import dataclass, field

from exohost.contracts.feature_contract import ROUTER_FEATURES
from exohost.contracts.label_contract import HOST_FIELD_TARGET_COLUMN


@dataclass(frozen=True, slots=True)
class SplitConfig:
    # Параметры фиксированного train/test split.
    test_size: float = 0.30
    random_state: int = 42

    def __post_init__(self) -> None:
        # Валидируем базовую корректность доли test split.
        if not 0.0 < self.test_size < 1.0:
            raise ValueError("SplitConfig.test_size must be between 0 and 1.")


@dataclass(frozen=True, slots=True)
class CrossValidationConfig:
    # Параметры стратифицированной cross-validation.
    n_splits: int = 10
    shuffle: bool = True
    random_state: int = 42

    def __post_init__(self) -> None:
        # Для CV нужна хотя бы пара фолдов.
        if self.n_splits < 2:
            raise ValueError("CrossValidationConfig.n_splits must be at least 2.")


@dataclass(frozen=True, slots=True)
class ClassificationTask:
    # Имя benchmark-задачи.
    name: str

    # Целевой столбец.
    target_column: str

    # Признаки модели.
    feature_columns: tuple[str, ...]

    # Колонки для стратификации split.
    stratify_columns: tuple[str, ...]


STAGE_CLASSIFICATION_TASK = ClassificationTask(
    name="stage_classification",
    target_column="evolution_stage",
    feature_columns=ROUTER_FEATURES,
    stratify_columns=("evolution_stage", "spec_class"),
)

SPECTRAL_CLASS_CLASSIFICATION_TASK = ClassificationTask(
    name="spectral_class_classification",
    target_column="spec_class",
    feature_columns=ROUTER_FEATURES,
    stratify_columns=("spec_class", "evolution_stage"),
)

SPECTRAL_SUBCLASS_CLASSIFICATION_TASK = ClassificationTask(
    name="spectral_subclass_classification",
    target_column="spec_subclass",
    feature_columns=ROUTER_FEATURES,
    stratify_columns=("spec_subclass",),
)

HOST_FIELD_CLASSIFICATION_TASK = ClassificationTask(
    name="host_field_classification",
    target_column=HOST_FIELD_TARGET_COLUMN,
    feature_columns=ROUTER_FEATURES,
    stratify_columns=("host_label", "spec_class", "evolution_stage"),
)


@dataclass(frozen=True, slots=True)
class BenchmarkProtocol:
    # Общий protocol benchmark-первой волны.
    split: SplitConfig = field(default_factory=SplitConfig)
    cv: CrossValidationConfig = field(default_factory=CrossValidationConfig)


DEFAULT_BENCHMARK_PROTOCOL = BenchmarkProtocol()
