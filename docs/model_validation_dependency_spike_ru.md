# Проверка внешней библиотеки для validation-layer

Дата: 14 марта 2026 года

## 1. Цель

Проверить, можно ли использовать внешнюю библиотеку с готовыми
train/test и overfitting-checks как часть validation-layer проекта.

На первом шаге в качестве основного кандидата рассматривался
`Deepchecks`, потому что он даёт:

- train-test validation suite;
- leakage и drift checks;
- model-evaluation checks для tabular classification.

## 2. Что именно проверялось

Проверка выполнялась в текущем рабочем окружении проекта:

- Python `3.13.2`
- `scikit-learn 1.8.0`
- основной `venv` проекта

Проверялись два уровня совместимости:

1. разрешение зависимостей через `pip install --dry-run`;
2. реальный импорт `deepchecks.tabular` API после установки.

## 3. Результат

### 3.1 Установка

Установка прошла успешно. `pip` смог разрешить и установить:

- `deepchecks 0.19.1`
- `category-encoders`
- `statsmodels`
- `plotly`
- вспомогательные зависимости

### 3.2 Реальный import

На import-layer библиотека в текущем окружении не проходит.

Проблема проявляется при импорте `deepchecks.tabular`:

- внутри `deepchecks` вызывается `sklearn.metrics.get_scorer("max_error")`;
- в текущем `scikit-learn 1.8.0` этот scorer больше не доступен в таком
  виде;
- из-за этого import падает ещё до практического использования checks.

Итог:

- библиотека устанавливается;
- но не является совместимой с текущим validation stack проекта "как есть".

## 4. Вывод

`Deepchecks` не должен входить в основной dependency stack проекта на
текущем этапе.

Причины:

- проект уже стабилизирован под `Python 3.13` и `scikit-learn 1.8.0`;
- попытка встроить `Deepchecks` прямо сейчас усложнит среду и может
  повредить воспроизводимости;
- основной validation-layer можно и нужно строить на уже используемом
  `scikit-learn`.

## 5. Архитектурное решение

Для проекта принимается следующий порядок:

1. основной validation-layer строится на `scikit-learn` и собственных
   контрактах проекта;
2. `Deepchecks` рассматривается только как optional external spike;
3. если позже понадобится отдельная среда под `Deepchecks`, она должна
   быть вынесена в отдельный optional workflow, а не в основной runtime.

## 6. Что используем вместо этого

До появления отдельного validation-env проект опирается на:

- `cross_validate(..., return_train_score=True)`
- `learning_curve`
- `validation_curve`
- `permutation_test_score`
- repeated split / repeated benchmark
- собственные train/test leakage checks
- classwise и calibration summaries

То есть validation-layer будет встроен в текущую архитектуру без новой
обязательной библиотеки.

## 7. Статус решения

Статус: `accepted`

Текущее решение:

- не добавлять `Deepchecks` в `requirements.txt`;
- не делать его обязательным для `pytest`, `ruff` или `mypy`;
- продолжать блок `Generalization Policy` на базе sklearn-native
  validation architecture.

## 8. Официальные источники

- [Deepchecks: installation](https://docs.deepchecks.com/stable/getting-started/installation.html)
- [Deepchecks: train-test validation suite](https://docs.deepchecks.com/stable/api/generated/deepchecks.tabular.suites.train_test_validation.html)
- [Deepchecks: train-test validation checks](https://docs.deepchecks.com/stable/api/generated/deepchecks.tabular.checks.train_test_validation.html)
- [scikit-learn: learning_curve](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.learning_curve.html)
- [scikit-learn: validation curves](https://scikit-learn.org/stable/modules/learning_curve.html)
- [scikit-learn: nested vs non-nested CV](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)
