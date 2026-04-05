# Файл `registry.py` слоя `models`.
#
# Этот файл отвечает только за:
# - обертки моделей и inference-протоколы;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - соседние модули слоя `models` и orchestration поверх них;
# - unit-тесты и более верхние слои проекта.

from __future__ import annotations

from exohost.models.gmm_classifier import GMMClassifier
from exohost.models.hgb_classifier import HGBClassifier
from exohost.models.mlp_classifier import MLPClassifier
from exohost.models.protocol import ModelSpec

SUPPORTED_MODEL_NAMES: tuple[str, ...] = (
    "gmm_classifier",
    "hist_gradient_boosting",
    "mlp_classifier",
)


def build_router_model_specs(
    feature_columns: tuple[str, ...],
) -> tuple[ModelSpec, ...]:
    # Возвращаем компактный реестр baseline-моделей для router-задач.
    return (
        ModelSpec(
            model_name="gmm_classifier",
            estimator=GMMClassifier(
                feature_columns=feature_columns,
                n_components=2,
                covariance_type="diag",
                reg_covar=1e-5,
                max_iter=200,
                random_state=42,
                scale_numeric=True,
                model_name="gmm_classifier",
            ),
        ),
        ModelSpec(
            model_name="hist_gradient_boosting",
            estimator=HGBClassifier(
                feature_columns=feature_columns,
                learning_rate=0.1,
                max_iter=200,
                max_leaf_nodes=31,
                min_samples_leaf=10,
                random_state=42,
                model_name="hist_gradient_boosting",
            ),
        ),
        ModelSpec(
            model_name="mlp_classifier",
            estimator=MLPClassifier(
                feature_columns=feature_columns,
                hidden_layer_sizes=(32, 16),
                alpha=1e-4,
                learning_rate_init=1e-3,
                max_iter=300,
                random_state=42,
                model_name="mlp_classifier",
            ),
        ),
    )


def select_model_specs(
    model_specs: tuple[ModelSpec, ...],
    *,
    selected_model_names: tuple[str, ...] | None = None,
) -> tuple[ModelSpec, ...]:
    # Возвращаем либо полный набор моделей, либо выбранный поднабор.
    if selected_model_names is None:
        return model_specs

    available_specs = {model_spec.model_name: model_spec for model_spec in model_specs}
    unknown_model_names = [
        model_name
        for model_name in selected_model_names
        if model_name not in available_specs
    ]
    if unknown_model_names:
        unknown_names_sql = ", ".join(unknown_model_names)
        supported_names_sql = ", ".join(SUPPORTED_MODEL_NAMES)
        raise ValueError(
            f"Unsupported model names: {unknown_names_sql}. "
            f"Supported models: {supported_names_sql}"
        )

    return tuple(available_specs[model_name] for model_name in selected_model_names)


def get_model_spec(
    model_specs: tuple[ModelSpec, ...],
    *,
    model_name: str,
) -> ModelSpec:
    # Возвращаем одну модель по имени и не оставляем вызывающему tuple-обвязку.
    return select_model_specs(
        model_specs,
        selected_model_names=(model_name,),
    )[0]
