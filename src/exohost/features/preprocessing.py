# Файл `preprocessing.py` слоя `features`.
#
# Этот файл отвечает только за:
# - подготовку признаков и training frame-слой;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - соседние модули слоя `features` и orchestration поверх них;
# - unit-тесты и более верхние слои проекта.

from __future__ import annotations

from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True, slots=True)
class NumericPreprocessingConfig:
    # Стратегия заполнения пропусков для числовых признаков.
    imputer_strategy: str = "median"

    # Нужна ли стандартизация числовых признаков.
    scale_numeric: bool = True


def build_numeric_preprocessor(
    feature_columns: tuple[str, ...],
    *,
    config: NumericPreprocessingConfig | None = None,
) -> ColumnTransformer:
    # Собираем минимальный и прозрачный preprocessing для числовых полей.
    actual_config = config or NumericPreprocessingConfig()
    numeric_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy=actual_config.imputer_strategy)),
    ]

    if actual_config.scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(steps=numeric_steps)
    return ColumnTransformer(
        transformers=[("numeric", numeric_pipeline, list(feature_columns))],
        remainder="drop",
        verbose_feature_names_out=False,
    )
