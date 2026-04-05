# Тестовый файл `test_model_classifier_semantics.py` домена `models`.
#
# Этот файл проверяет только:
# - проверку логики домена: обертки моделей и inference-контракты;
# - регрессионные сценарии и ожидаемые контракты целевого слоя.
#
# Следующий слой:
# - реализация домена `models` в `src/exohost`;
# - соседние тесты и testkit этого же пакета.

from __future__ import annotations

from sklearn.base import is_classifier

from exohost.models.gmm_classifier import GMMClassifier
from exohost.models.hgb_classifier import HGBClassifier
from exohost.models.mlp_classifier import MLPClassifier


def test_hgb_classifier_is_recognized_as_classifier() -> None:
    estimator = HGBClassifier(feature_columns=("feature_a", "feature_b"))

    assert is_classifier(estimator) is True


def test_mlp_classifier_is_recognized_as_classifier() -> None:
    estimator = MLPClassifier(feature_columns=("feature_a", "feature_b"))

    assert is_classifier(estimator) is True


def test_gmm_classifier_is_recognized_as_classifier() -> None:
    estimator = GMMClassifier(feature_columns=("feature_a", "feature_b"))

    assert is_classifier(estimator) is True
