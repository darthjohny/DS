# ТЗ по packaging preprocessing и блоку сравнений

Дата: 12 марта 2026 года

Статус документа:
- historical planning document;
- фиксирует промежуточное ТЗ на packaging preprocessing и comparison-layer
  до завершения основной волны работ;
- не является главным current-state описанием этих контуров.

Где смотреть текущее состояние:
- `docs/preprocessing_pipeline_ru.md` — канонический preprocessing
  narrative;
- `docs/model_comparison_protocol_ru.md` — канонический comparison
  protocol;
- `docs/model_comparison_findings_ru.md` — текущие результаты и выводы;
- `docs/repository_state_policy_ru.md` — policy того, что считать
  versioned current state.

## 1. Цель

Подготовить следующий слой материалов для ВКР без изменения текущего
production-контура:

- зафиксировать preprocessing как воспроизводимый слой репозитория;
- вынести ADQL и SQL из DBeaver в канонические файлы проекта;
- добавить notebook, которым удобно пользоваться в пояснительной записке
  и презентации;
- подготовить базу для сравнительного блока по моделям;
- не смешивать research-артефакты и production-логику.

## 2. Входные материалы

Основание для этой итерации:

- `Требования к ВКР.docx`;
- текущее состояние `README.md` и пакетов `src/*`;
- существующие EDA-ноутбуки в `notebooks/eda/`;
- DBeaver SQL-скрипт пользователя, который уже покрывает:
  - crossmatch NASA x Gaia;
  - QA по пропускам и dist_arcsec;
  - сборку `lab.nasa_gaia_crossmatch`;
  - сборку `lab.nasa_gaia_train`;
  - rule-based classification views;
  - router reference layer views;
  - QA ковариаций и PD-checks;
  - DDL для result tables.

## 3. Ограничения и принципы

- Канонический production-контур не ломаем.
- Preprocessing оформляем как отдельный воспроизводимый слой.
- Не дублируем result-table migration, которая уже живёт в:
  `sql/2026-03-11_gaia_results_posterior_host_fields.sql`.
- Не внедряем в этой волне `GMM`, `router mixture` и иерархический `router`.
- Всё новое документируем на русском языке по принятому стандарту проекта.

## 4. Что считаем результатом

После завершения этой волны в проекте должны появиться:

- оформленный preprocessing-контур в `sql/` и `docs/`;
- отдельный notebook по извлечению и подготовке данных;
- сохранённый ADQL-запрос к Gaia Archive;
- карта сравнения основной модели и baseline;
- понятная трассировка между требованиями ВКР и артефактами repo.

## 5. Целевая файловая карта

### 5.1 Обязательные файлы до baseline-блока

| Файл | Назначение | Источник наполнения |
| --- | --- | --- |
| `sql/adql/01_nasa_hosts_crossmatch_batch_template.adql` | Batch-шаблон crossmatch NASA host-stars с Gaia DR3 | Реальные ADQL-запросы пользователя из Gaia Archive |
| `sql/adql/02_validation_physics_enrichment.adql` | Enrichment провалидированных match-строк физикой Gaia DR3 | Реальный ADQL-запрос пользователя |
| `sql/adql/03_gaia_reference_sampling_examples.adql` | Примеры ADQL-выгрузок reference stars для router-слоя | Реальные ADQL-запросы пользователя |
| `sql/preprocessing/01_nasa_gaia_crossmatch.sql` | Сборка `lab.nasa_gaia_crossmatch` и `lab.nasa_gaia_train`, QA по `dist_arcsec`, полнота физики | DBeaver SQL Part 1-3 |
| `sql/preprocessing/02_train_classification_views.sql` | `lab.v_nasa_gaia_train_classified`, `..._dwarfs`, `..._evolved` | DBeaver блок классификации |
| `sql/preprocessing/03_router_reference_layer.sql` | Нормализующие views и `lab.v_gaia_router_training` | DBeaver блок router reference layer |
| `sql/preprocessing/04_data_quality_checks.sql` | Корреляции, ковариации, PD-check и sanity-check запросы | DBeaver QA-блок |
| `docs/preprocessing_pipeline_ru.md` | Человеческое описание data lineage, источников, joins и выходных таблиц | SQL-артефакты + README |
| `notebooks/eda/00_data_extraction_and_preprocessing.ipynb` | Графики и narrative для пояснительной записки | SQL + сохранённые выборки + ADQL |

### 5.2 Файлы следующей волны для сравнений

| Файл | Назначение |
| --- | --- |
| `docs/model_comparison_protocol_ru.md` | Единый протокол сравнения моделей, split, метрики и артефакты |
| `analysis/model_comparison/` | Исследовательский контур для baseline-моделей |
| `experiments/model_comparison/` | Отчёты и таблицы сравнений для ВКР |

Примечание:

- отдельный файл `sql/schema/01_results_tables.sql` пока не нужен;
- действующий DDL result-таблиц уже есть в существующей migration и должен
  оставаться её канонической точкой.

## 6. Микро-ТЗ

### Микро-ТЗ 1

Разобрать DBeaver preprocessing-скрипт на канонические SQL-файлы.

Критерий готовности:

- в `sql/preprocessing/` лежат непересекающиеся SQL-файлы;
- из них убраны черновые повторы, альтернативные версии и сломанные комментарии;
- у каждого файла есть короткая шапка с назначением.

Статус:

- выполнено 12 марта 2026 года.

### Микро-ТЗ 2

Сохранить ADQL-запросы к Gaia Archive в отдельные канонические файлы.

Критерий готовности:

- запросы лежат в `sql/adql/`;
- сохранены отдельно crossmatch, validation-enrichment и reference-sampling;
- в шапке каждого файла написано, что именно он извлекает;
- в `docs/preprocessing_pipeline_ru.md` есть ссылки на эти файлы.

Статус:

- выполнено 12 марта 2026 года.

### Микро-ТЗ 3

Оформить markdown-документ по preprocessing и data lineage.

Критерий готовности:

- описаны источники NASA и Gaia;
- описан spatial crossmatch;
- описаны train-таблицы и reference-views;
- описаны ключевые признаки и quality-поля;
- явно написано, какие артефакты используются дальше в Python-контуре.

Статус:

- выполнено 12 марта 2026 года.

### Микро-ТЗ 4

Создать preprocessing-ноутбук для ВКР.

Критерий готовности:

- в ноутбуке есть narrative от источников данных до train-слоя;
- показаны хотя бы ключевые распределения, пропуски и sanity-check;
- есть ссылка на [Gaia Archive](https://gea.esac.esa.int/archive/) и,
  при наличии локального PNG, скрин интерфейса архива;
- ноутбук пригоден и для пояснительной записки, и для презентации.

Статус:

- выполнено 12 марта 2026 года.
- Gaia-скрины уже сохранены в `docs/assets/`.
- В notebook уже добавлены три сравнительных preprocessing-блока `до/после`.

### Микро-ТЗ 5

Добавить в `README.md` нижний раздел со ссылками на материалы ВКР.

Критерий готовности:

- `README` не раздувается деталями;
- внизу есть короткая карта: требования ВКР, preprocessing, comparative block.

### Микро-ТЗ 6

Зафиксировать протокол сравнения моделей.

Критерий готовности:

- определены единые split и random seed;
- определены router-, host- и ranking-метрики;
- определён формат итоговой сравнительной таблицы.

### Микро-ТЗ 7

Добавить `Baseline 1: Legacy Gaussian`.

Критерий готовности:

- baseline воспроизводимо считает метрики;
- baseline не вмешивается в production-контур;
- сравнение с `V1` формулируется честно и прозрачно.

### Микро-ТЗ 8

Добавить `Baseline 2: RandomForest`.

Критерий готовности:

- есть train/test результаты;
- baseline участвует в общем comparative report;
- baseline закрывает требование по классическим ML-моделям.

### Микро-ТЗ 9

Подготовить лёгкие tests/smoke-checks для baseline-контуров.

Критерий готовности:

- есть проверка детерминизма при фиксированном `random_state`;
- есть smoke-test на минимальном датасете;
- есть проверка формата итогового отчёта.

### Микро-ТЗ 10

Подготовить отдельный компактный ИНС-блок.

Критерий готовности:

- есть небольшая нейросетевая модель;
- она сравнивается по тому же протоколу;
- её роль в проекте обозначена как исследовательская, а не production-first.

## 7. Карта внедрения

### Этап A. Обязательный слой до baseline

Порядок:

1. Разобрать preprocessing SQL.
2. Сохранить ADQL.
3. Написать `docs/preprocessing_pipeline_ru.md`.
4. Сделать notebook `00_data_extraction_and_preprocessing.ipynb`.
5. Обновить `README` нижним разделом.

Почему этот этап идёт первым:

- он закрывает preprocessing-требование ВКР;
- он превращает локальный DBeaver-скрипт в репозиторный SSOT;
- он даёт чистую основу для baseline-сравнений.

### Этап B. Сравнительный блок

Порядок:

1. Зафиксировать comparison protocol.
2. Реализовать `Legacy Gaussian`.
3. Реализовать `RandomForest`.
4. Собрать единый comparative report.
5. Добавить лёгкие tests/smoke-checks.

### Этап C. Формальная полнота ВКР

Порядок:

1. Добавить компактный ИНС-блок.
2. Описать CLI как приложение.
3. Свести таблицу соответствия ВКР, пояснительную записку и презентацию.

## 8. Что делать раньше, а что позже

Нужно сделать раньше baseline:

- вынести preprocessing из DBeaver в repo;
- сохранить ADQL;
- оформить preprocessing-документацию;
- связать эти материалы через `README`.

Текущий статус:

- SQL и ADQL уже вынесены в repo;
- документация и `README` уже синхронизированы;
- preprocessing-ноутбук уже создан;
- Gaia-скрины уже добавлены в `docs/assets/`.

Можно делать после preprocessing:

- baseline-модели;
- benchmark-отчёт;
- ИНС;
- polishing материалов под презентацию.

## 9. Нужны ли тесты

Для preprocessing-артефактов:

- обязательных `pytest`-тестов не требуется, если это SQL и notebook;
- достаточно воспроизводимости и понятного data lineage.

Для baseline-контуров:

- smoke-тесты полезны и желательны;
- DB-backed тесты не нужны, если baseline не идёт в production.

## 10. Открытые входы от пользователя

Для завершения preprocessing-блока нужно ещё получить:

- при необходимости, дополнительные ADQL-запросы для остальных
  reference-классов;
- локальные заметки о том, какие SQL-блоки были рабочими, а какие
  экспериментальными.

## 11. Связанный документ

Формальная карта соответствия требованиям ВКР лежит в:

- [vkr_requirements_traceability_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/vkr_requirements_traceability_ru.md)
