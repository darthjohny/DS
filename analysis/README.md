# Слой анализа

Здесь живут только исследовательские материалы текущей версии проекта:

- ноутбуки;
- промежуточные графики;
- таблицы для EDA и обзорных сравнений.

Исходный код моделей, пайплайнов и контрактов сюда не кладем.

## Текущие ноутбуки

- `analysis/notebooks/eda/router_training.ipynb`
  EDA `router training` source под coarse-class контур.
- `analysis/notebooks/eda/host_training.ipynb`
  EDA `host training` source известных звезд-host.
- `analysis/notebooks/eda/label_coverage.ipynb`
  аудит покрытия классов, стадий и `spec_subclass` перед subclass-волной.
- `analysis/notebooks/technical/scoring_review.ipynb`
  обзор scoring- и ranking-артефактов с проверкой соответствия конечной задаче
  отбора целей для последующих наблюдений.

## Правило

- тяжелая логика остается в `src/exohost`;
- ноутбуки только читают готовые артефакты и помогают их интерпретировать;
- если notebook требует новую логику, сначала выносим ее в тестируемый модуль.
