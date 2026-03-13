# Ревью ноутбуков после обновления comparison-layer

Дата: 13 марта 2026 года

## Цель проверки

Проверить, какие `.ipynb` в проекте нужно синхронизировать после перехода comparative benchmark на контракт:
- `test_size = 0.30`;
- `10-fold CV`;
- search summary для всех сравниваемых моделей;
- новый snapshot preview `limit=5000`.

## Проверенные ноутбуки

### `notebooks/eda/00_data_extraction_and_preprocessing.ipynb`

Статус: без изменений.

Причина:
- ноутбук относится к извлечению и preprocessing данных;
- не содержит жёсткой привязки к старым comparison run-name;
- методически остаётся актуальным.

### `notebooks/eda/01_host_eda_overview.ipynb`

Статус: технически синхронизирован, аналитически без изменений.

Причина:
- ноутбук описывает upstream EDA host-population;
- не зависит от новых benchmark и snapshot-артефактов;
- bootstrap обновлён: notebook теперь явно добавляет в `sys.path` не только
  корень repo, но и `src/`, чтобы импорты оставались корректными после
  удаления import-time path side effects из EDA-пакетов.

### `notebooks/eda/02_router_readiness.ipynb`

Статус: технически синхронизирован, аналитически без изменений.

Причина:
- ноутбук относится к router readiness и подготовке входов;
- изменений comparison-layer для него недостаточно, чтобы требовать
  переписывания;
- bootstrap обновлён по той же причине: notebook теперь явно добавляет `src/`
  в `sys.path` для стабильного доступа к пакетам `infra` и `router_model`.

### `notebooks/eda/03_host_vs_field_contrastive.ipynb`

Статус: технически синхронизирован, аналитически без изменений.

Причина:
- ноутбук фокусируется на contrastive host-vs-field анализе;
- прямой зависимости от новых `vkr30_cv10` run-name не обнаружено;
- bootstrap обновлён по той же схеме, чтобы notebook корректно импортировал
  модули из `src/` и `analysis/` без скрытой зависимости от локальной среды.

### `notebooks/eda/04_model_comparison_summary.ipynb`

Статус: обновлён и переисполнен.

Что изменено:
- заменены устаревшие benchmark/snapshot run-name на:
  - `baseline_comparison_2026-03-13_vkr30_cv10`
  - `baseline_comparison_2026-03-13_vkr30_cv10_limit5000_snapshot`
- добавлено чтение `search_summary.csv`;
- добавлен отдельный блок с method contract:
  - `test_size = 0.30`
  - `cv_folds = 10`
  - `search_refit_metric = roc_auc`
- обновлены markdown-комментарии под новый comparison-контур;
- финальные выводы синхронизированы с новыми benchmark и snapshot preview.

Результат:
- ноутбук успешно исполнен top-to-bottom через `nbconvert --execute --inplace`.

## Итог

После обновления comparison-layer содержательного переписывания потребовал
только summary-ноутбук `04_model_comparison_summary.ipynb`.

Ноутбуки `01`, `02` и `03` также были синхронизированы технически:
в их bootstrap явно добавлен `src/` в `sys.path`, чтобы зафиксировать
корректный import-контур внутри Jupyter.

По аналитическому содержанию остальные EDA-ноутбуки остаются актуальными и
могут использоваться в текущем контуре ВКР без переписывания narrative.
