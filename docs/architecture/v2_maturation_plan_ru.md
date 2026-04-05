# V2 Maturation Plan

## Цель

Этот план описывает следующую волну взросления проекта после базовой сборки `V2`.

Задача этой волны:

- перейти от грубой звездной схемы к составной MK-логике;
- не ломать существующую БД и уже полезные данные;
- аккуратно пересобрать data engineering и training source;
- сохранить прежний инженерный стандарт:
  - пишем по официальной документации;
  - проверяем по официальной документации;
  - каждый блок закрываем `QA`;
  - каждый законченный модуль проверяем целиком до следующего шага.

Связанные документы этой волны:

- [mk_label_contract_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/contracts/mk_label_contract_ru.md)
- [quality_ood_contract_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/contracts/quality_ood_contract_ru.md)
- [db_relation_policy_mk_wave_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/contracts/db_relation_policy_mk_wave_ru.md)
- [local_source_audit_mk_wave_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/plans/local_source_audit_mk_wave_ru.md)
- [external_mk_source_selection_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/plans/external_mk_source_selection_ru.md)
- [gaia_crossmatch_strategy_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/plans/gaia_crossmatch_strategy_ru.md)
- [gaia_upload_contract_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/contracts/gaia_upload_contract_ru.md)
- [mk_ingestion_schema_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/contracts/mk_ingestion_schema_ru.md)
- [mk_import_column_contract_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/contracts/mk_import_column_contract_ru.md)
- [mk_ingestion_workflow_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/plans/mk_ingestion_workflow_ru.md)
- [db_layer_closure_plan_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/plans/db_layer_closure_plan_ru.md)
- [training_view_contracts_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/contracts/training_view_contracts_ru.md)
- [bmk_parser_tz_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/plans/bmk_parser_tz_ru.md)

## Главный Принцип Волны

- Не делаем новый rewrite.
- Не сносим существующие таблицы БД.
- Не переписываем все ядро ради одной новой идеи.
- Расширяем текущую архитектуру `V2` новой data/task-веткой.

## Что Меняется Концептуально

Раньше рабочая схема была грубее:

- `stage`
- `spectral_class`
- `host-like`
- `observability`

Теперь проект взрослеет до более правильной астрофизической схемы:

- `quality_state`
- `ood_state`
- `luminosity_class` или укрупненная `luminosity_group`
- `spectral_class`
- `spectral_subclass`
- `host-like signal`
- `observability`
- `priority`

## Новый Целевой Pipeline

`quality/ood -> luminosity_or_stage -> spectral_class -> spectral_subclass -> host prior -> observability -> ranking`

## Что Считаем Label-Частью

- `spectral_class`
- `spectral_subclass`
- `luminosity_class`

Важно:

- не храним `G2V` как основной внутренний target одной строкой;
- храним его по составным частям;
- строковый MK-label допускается только как внешний import/export формат.

## Что Считаем Feature-Частью

Базовые физические признаки:

- `teff`
- `logg`
- `radius`
- `luminosity` или доступная прокси
- `bp_rp`

Качество и наблюдаемость:

- `ruwe`
- `parallax`
- `parallax_over_error`
- `phot_g_mean_mag`, если доступна

Дополнительные астрофизические:

- `mh_gspphot`

## Что Считаем Ранним Gate

До основной классификации проверяем:

- `ruwe`
- `parallax_over_error`
- наличие критических признаков
- базовую физическую правдоподобность значений

Плохие объекты:

- либо переводим в `unknown`;
- либо маркируем как `OOD`;
- либо не допускаем до обычного ranking-контура.

## Что Делаем С Металличностью

- `metallicity` не делаем магическим универсальным правилом.
- Выделяем отдельный scientific block:
  - обзор литературы;
  - выбор подтвержденной гипотезы;
  - фиксация правил в docs;
  - только потом внедрение в priors.

На этой волне:

- для giant planets связь с высокой металличностью рассматриваем как вероятно положительную;
- для rocky planets не вшиваем грубую ручную эвристику до отдельного review.

## Стратегия Работы С БД

БД остается прежней.

Правила:

- ничего не удаляем из существующего слоя;
- существующие таблицы и view не трогаем без крайней необходимости;
- добавляем новые таблицы и новые view рядом;
- переключение пайплайна делаем на новые relation names, а не через разрушение старых.

Это нужно, чтобы:

- не потерять уже полезные данные;
- не сломать существующие обучающие и аналитические контуры;
- иметь возможность сравнивать старые и новые data source.

## Предлагаемая Схема Новых Таблиц

Названия фиксируем заранее и не придумываем по ходу.

Черновой набор:

- `lab.gaia_mk_external_raw`
- `lab.gaia_mk_external_crossmatch`
- `lab.gaia_mk_external_labeled`
- `lab.gaia_mk_quality_gated`
- `lab.gaia_mk_training_reference`
- hierarchical/OOD decision layer для новой `MK`-волны
- `lab.v_gaia_mk_training_dwarfs`
- `lab.v_gaia_mk_training_evolved`
- `lab.v_gaia_mk_router_training`
- `lab.v_gaia_mk_candidate_scoring`

Если появится отдельный host-side enriched source:

- `lab.v_gaia_mk_host_training`

Правило:

- raw table хранит то, что пришло извне;
- crossmatch table хранит связь external source и Gaia;
- labeled/training tables хранят уже нормализованные labels;
- view отделяют training/scoring-сценарии.

## Источники Данных Для Нового Label-Контура

Приоритет такой:

1. локальные таблицы БД;
2. если их не хватает, внешний spectral catalog;
3. `Gaia` как физический и астрометрический backbone;
4. при необходимости host-side дополняющие данные из `NASA Exoplanet Archive`.

На этой волне:

- сначала проектируем схему таблиц;
- потом готовим ingestion plan;
- только потом идем за данными.

## Большой План Реализации

### Этап 1. Replan И Contracts

Цель:

- зафиксировать новый label-contract;
- зафиксировать новый quality/OOD contract;
- зафиксировать новую DB-стратегию без разрушения существующего слоя.

Результат:

- обновленные docs;
- обновленный roadmap;
- перечень relation names для новой волны.

QA Gate:

- все правила зафиксированы в docs;
- нет противоречий с текущим `V2`;
- БД-стратегия не требует удаления старых таблиц.

### Этап 2. Data Audit И Source Design

Цель:

- понять, что уже есть локально;
- понять, чего не хватает;
- зафиксировать минимальный внешний source для MK-labels.

Результат:

- audit по локальным таблицам;
- решение, что берем из БД, что тянем извне;
- описание target schema для новых таблиц.

QA Gate:

- для каждого нового поля понятно, откуда оно берется;
- нет "магических" колонок без источника;
- нет смешения raw/import/training/view слоев.

### Этап 3. External Catalog Ingestion Plan

Цель:

- спланировать загрузку внешнего spectral catalog;
- спланировать crossmatch с `Gaia`;
- спланировать загрузку результата в БД.

Результат:

- ingestion plan;
- SQL/ADQL/backfill plan;
- список колонок для сохранения.

QA Gate:

- источник выбран осознанно;
- crossmatch объяснен;
- схема raw/crossmatch/labeled tables определена заранее.

### Этап 4. New Data Engineering Layer

Цель:

- собрать новый training source под MK-схему;
- внедрить quality gate;
- подготовить candidate-scoring source.

Результат:

- новые dataset loaders;
- новые contracts;
- новые training views;
- quality/OOD flow.

QA Gate:

- старый контур не сломан;
- новый контур читает только новые relation names;
- unit/integration tests покрывают новые contracts.

### Этап 5. New Model Tasks

Цель:

- обучить новые модели на составных labels;
- не смешивать все в один target.

Результат:

- `luminosity/stage` task;
- `spectral_class` task;
- `spectral_subclass` task;
- обновленный host-like task при необходимости.

QA Gate:

- каждая задача имеет свой dataset contract;
- benchmark одинаково применим ко всем задачам;
- нет скрытого leakage между labels.

### Этап 6. Priority Science Layer

Цель:

- пересмотреть ranking под научную правдоподобность;
- добавить quality-aware и observability-aware decisions;
- отдельно внедрить metallicity priors после literature review.

Результат:

- новая priority policy;
- обновленный scoring review;
- обновленный reporting.

QA Gate:

- каждый приоритетный фактор объяснен;
- hard filters и soft priors разделены;
- docs и код не противоречат друг другу.

### Этап 7. Real Runs И Review

Цель:

- прогнать новый контур на реальных relation;
- посмотреть реальные top candidates;
- решить, где ranking ведет себя хорошо, а где нет.

Результат:

- реальные train/score/prioritize artifacts;
- EDA/review notebooks;
- список корректировок второй итерации.

QA Gate:

- сквозной прогон проходит;
- результаты интерпретируемы;
- видно, соответствует ли top списка задаче последующих наблюдений.

## План На Микро-ТЗ

Микро-ТЗ будем строить по тем же правилам, что и раньше.

Шаблон остается прежним:

- `цель`
- `результат`
- `зона ответственности`
- `владельцы файлов`
- `зависимости`
- `проверки`
- `критерий готовности`

Идем строго по порядку:

1. contracts
2. data audit
3. ingestion design
4. new data engineering
5. model tasks
6. priority science layer
7. real runs

## Правило Для Каждого Блока

- пишем маленький законченный блок;
- сразу делаем микропроверку;
- сверяем спорные места с официальной документацией;
- только потом берем следующий блок.

## Правило Для Каждого Модуля

- закончили модуль;
- прошли `ruff`, `mypy`, `pyright`, `pytest`;
- проверили логику и contracts;
- проверили, что модуль не усложнил архитектуру;
- только потом считаем этап закрытым.

## Что Делаем После Завершения Плана

- только после готового replan и contracts идем за новыми данными;
- только после этого стучимся к `Gaia` и внешнему spectral source;
- только после загрузки новых таблиц переключаем training/scoring на новые relation names.

## Текущая Точка Остановки Для Этой Волны

На текущем этапе:

- базовый `V2` собран;
- train/score/prioritize уже работают;
- notebooks и reporting уже есть;
- следующая большая волна начинается не с кода, а с `replan + data engineering redesign`.

## Напоминание После Закрытия Плана

План выполнен, идем за данными `Gaia`.
