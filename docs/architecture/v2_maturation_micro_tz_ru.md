# V2 Maturation Micro-TZ

## Правило Работы

Для этой волны сохраняем тот же цикл:

- блок;
- микропроверка;
- сверка с официальной документацией;
- только потом следующий блок.

После завершения модуля:

- `ruff`
- `mypy`
- `pyright`
- `pytest`
- проверка contracts и общей простоты кода

## Этап 1. Replan И Contracts

### MTZ-M01. Зафиксировать составной MK label contract

- Цель: описать `spectral_class`, `spectral_subclass`, `luminosity_class`, `peculiarity_suffix`.
- Результат: документированный target-contract новой волны.
- Зона ответственности: только labels.
- Владельцы файлов: `docs/architecture/`, `docs/methodology/`, позже `src/exohost/contracts/`.
- Зависимости: нет.
- Проверки:
  - MK не хранится как монолитный `G2V` target;
  - роли полей разделены явно.
- Критерий готовности: любой новый source можно проверить на соответствие label-contract.

### MTZ-M02. Зафиксировать quality/OOD contract

- Цель: описать ранний gate по `ruwe`, `parallax_over_error`, пропускам и физической правдоподобности.
- Результат: отдельный quality-contract для новой волны.
- Зона ответственности: только quality/OOD.
- Владельцы файлов: `docs/architecture/`, `docs/methodology/`, позже `src/exohost/contracts/`.
- Зависимости: `MTZ-M01`.
- Проверки:
  - quality и labels не смешаны в один слой;
  - `unknown` и `OOD` не используются как синонимы без определения.
- Критерий готовности: понятно, кого пускаем в обычный pipeline, а кого нет.

### MTZ-M03. Зафиксировать DB naming policy для новой волны

- Цель: заранее определить имена новых таблиц и view.
- Результат: перечень relation names для raw, crossmatch, labeled, training, candidate scoring.
- Зона ответственности: только naming и слой БД.
- Владельцы файлов: `docs/architecture/`.
- Зависимости: `MTZ-M01`, `MTZ-M02`.
- Проверки:
  - старые relation names не переиспользуются;
  - нет временных имен вроде `new`, `tmp`, `final`.
- Критерий готовности: ingestion и loaders потом не придумывают имена на ходу.

## Этап 2. Data Audit И Source Design

### MTZ-M04. Провести полный audit локальных таблиц БД

- Цель: понять текущее покрытие по `spectral_class`, `spectral_subclass`, `luminosity_class`, quality-полям.
- Результат: таблица покрытия локальных source.
- Зона ответственности: только audit.
- Владельцы файлов: `docs/methodology/`, `analysis/notebooks/`, возможно `docs/decisions/`.
- Зависимости: `MTZ-M01`, `MTZ-M02`, `MTZ-M03`.
- Проверки:
  - по каждому source видно, каких labels и quality-полей не хватает;
  - не делаем выводов без цифр.
- Критерий готовности: есть формальный ответ, что берем локально, а что нет.

### MTZ-M05. Зафиксировать минимальный внешний source для MK labels

- Цель: выбрать внешний spectral catalog и описать его роль.
- Результат: решение по внешнему label-source.
- Зона ответственности: только source selection.
- Владельцы файлов: `docs/architecture/`, `docs/decisions/`.
- Зависимости: `MTZ-M04`.
- Проверки:
  - источник выбран по документации, а не по памяти;
  - понятно, что в нем есть, а чего нет.
- Критерий готовности: можно переходить к ingestion design.

### MTZ-M06. Зафиксировать crossmatch strategy

- Цель: описать, как внешний spectral source связывается с `Gaia`.
- Результат: crossmatch plan.
- Зона ответственности: только связка источников.
- Владельцы файлов: `docs/methodology/`, `docs/architecture/`.
- Зависимости: `MTZ-M05`.
- Проверки:
  - понятно, какие ключи или координатные поля используются;
  - raw/import/crossmatch слои не смешаны.
- Критерий готовности: ingestion можно реализовывать без архитектурной догадки.

## Этап 3. External Ingestion Plan

### MTZ-M07. Описать схему новых таблиц БД

- Цель: определить столбцы и роль таблиц `raw`, `crossmatch`, `labeled`, `quality_gated`, `training`.
- Результат: схема новых relation.
- Зона ответственности: только структура БД новой волны.
- Владельцы файлов: `docs/architecture/`, позже SQL/DB-модули.
- Зависимости: `MTZ-M03`, `MTZ-M06`.
- Проверки:
  - raw данные не смешаны с нормализованными labels;
  - training views выводим поверх более низких слоев.
- Критерий готовности: можно писать ingestion и loaders.

### MTZ-M08. Зафиксировать список колонок для импорта

- Цель: определить минимально достаточный набор полей из внешнего source и `Gaia`.
- Результат: import-column contract.
- Зона ответственности: только колонки данных.
- Владельцы файлов: `docs/methodology/`.
- Зависимости: `MTZ-M07`.
- Проверки:
  - каждая колонка нужна для label, quality или ranking;
  - нет декоративных полей без причины.
- Критерий готовности: импорт не тянет лишний шум.

### MTZ-M09. Зафиксировать ingestion workflow

- Цель: описать, как данные скачиваются, загружаются, crossmatch-ятся и попадают в БД.
- Результат: пошаговый ingestion workflow.
- Зона ответственности: только процесс загрузки.
- Владельцы файлов: `docs/architecture/`, `docs/methodology/`.
- Зависимости: `MTZ-M07`, `MTZ-M08`.
- Проверки:
  - ясно, что делается через `ADQL`, а что локально;
  - workflow воспроизводим.
- Критерий готовности: можно идти за данными без импровизации.

## Этап 4. New Data Engineering

### MTZ-M10. Пересобрать dataset contracts под новую схему

- Цель: обновить contracts для training/scoring source под MK labels и quality gate.
- Результат: новые dataset contracts.
- Зона ответственности: contracts.
- Владельцы файлов: `src/exohost/contracts/`.
- Зависимости: `MTZ-M01`, `MTZ-M02`, `MTZ-M07`, `MTZ-M08`.
- Проверки:
  - labels и features не перепутаны;
  - quality-поля доступны явно.
- Критерий готовности: loaders можно писать без двусмысленности.

### MTZ-M11. Реализовать quality gate layer

- Цель: ввести раннюю фильтрацию/маркировку по quality/OOD.
- Результат: reusable quality gate logic.
- Зона ответственности: только quality/OOD.
- Владельцы файлов: `src/exohost/features/`, `src/exohost/contracts/`, `src/exohost/ranking/` при необходимости.
- Зависимости: `MTZ-M10`.
- Проверки:
  - gate работает до основной классификации;
  - fallback не маскирует плохие объекты как хорошие.
- Критерий готовности: pipeline понимает, кого не надо ранжировать как обычную цель.

### MTZ-M12. Реализовать новые loaders и training views

- Цель: переключить новую волну на новые relation names.
- Результат: loaders для нового training/scoring source.
- Зона ответственности: datasets/db.
- Владельцы файлов: `src/exohost/datasets/`, `src/exohost/db/`.
- Зависимости: `MTZ-M10`, `MTZ-M11`.
- Проверки:
  - старый контур не ломается;
  - новый контур использует только новую ветку relation.
- Критерий готовности: новые source читаются и валидируются.

## Этап 5. New Model Tasks

### MTZ-M13. Подготовить task-contract для luminosity/stage

- Цель: определить отдельную задачу по `luminosity_class` или укрупненной группе.
- Результат: task definition.
- Зона ответственности: labels/tasks.
- Владельцы файлов: `docs/methodology/`, `src/exohost/evaluation/`, `src/exohost/training/`.
- Зависимости: `MTZ-M10`, `MTZ-M12`.
- Проверки:
  - задача не дублирует старый `stage` без смысла;
  - target определен явно.
- Критерий готовности: задачу можно benchmark-ить отдельно.

### MTZ-M14. Подготовить task-contract для spectral_subclass

- Цель: определить отдельную subclass-задачу на новом source.
- Результат: task definition для `spectral_subclass`.
- Зона ответственности: labels/tasks.
- Владельцы файлов: `docs/methodology/`, `src/exohost/evaluation/`, `src/exohost/training/`.
- Зависимости: `MTZ-M10`, `MTZ-M12`.
- Проверки:
  - subclass не смешан с luminosity;
  - task не публикуется раньше готовности source.
- Критерий готовности: subclass task можно обучать и сравнивать отдельно.

### MTZ-M15. Пересобрать benchmark protocol для новой волны

- Цель: расширить benchmark на новые tasks.
- Результат: benchmark protocol новой волны.
- Зона ответственности: evaluation/training.
- Владельцы файлов: `src/exohost/evaluation/`, `src/exohost/training/`.
- Зависимости: `MTZ-M13`, `MTZ-M14`.
- Проверки:
  - одинаковые правила split/CV для новых задач;
  - metrics подходят под multiclass labels.
- Критерий готовности: benchmark новой волны воспроизводим.

## Этап 6. Priority Science Layer

### MTZ-M16. Зафиксировать hard filters и soft priors

- Цель: разделить quality filters, luminosity filters, observability factors и астрофизические priors.
- Результат: priority policy новой волны.
- Зона ответственности: ranking science layer.
- Владельцы файлов: `docs/methodology/`, `src/exohost/ranking/`.
- Зависимости: `MTZ-M11`, `MTZ-M13`, `MTZ-M15`.
- Проверки:
  - hard filters не замаскированы под score;
  - priors объяснимы.
- Критерий готовности: ranking layer можно обсуждать как научную схему, а не набор случайных чисел.

### MTZ-M17. Провести literature review по metallicity priors

- Цель: собрать подтвержденную картину зависимости `metallicity -> planet type`.
- Результат: scientific note по metallicity.
- Зона ответственности: только literature review.
- Владельцы файлов: `docs/methodology/`, `docs/decisions/`.
- Зависимости: нет.
- Проверки:
  - используются первичные источники;
  - выводы разделены по типам планет.
- Критерий готовности: можно внедрять metallicity priors без ручной фантазии.

### MTZ-M18. Внедрить metallicity priors после review

- Цель: добавить научно обоснованный metallicity block в ranking.
- Результат: обновленный priority layer.
- Зона ответственности: ranking.
- Владельцы файлов: `src/exohost/ranking/`, `src/exohost/reporting/`.
- Зависимости: `MTZ-M16`, `MTZ-M17`.
- Проверки:
  - prior не конфликтует с docs;
  - влияние priors интерпретируемо.
- Критерий готовности: металличность участвует как осмысленный scientific factor.

## Этап 7. Real Runs И Review

### MTZ-M19. Прогнать новую волну train/score/prioritize

- Цель: проверить новый контур на реальных relation names.
- Результат: реальные artifacts новой волны.
- Зона ответственности: training/scoring.
- Владельцы файлов: `src/exohost/training/`, `src/exohost/cli/`, `src/exohost/reporting/`.
- Зависимости: `MTZ-M12`, `MTZ-M15`, `MTZ-M18`.
- Проверки:
  - сквозной прогон успешен;
  - outputs сохраняются корректно.
- Критерий готовности: новый pipeline живет на реальных данных.

### MTZ-M20. Провести review топ-кандидатов

- Цель: проверить, соответствует ли top списка цели последующих наблюдений.
- Результат: review note по качеству ranking.
- Зона ответственности: scientific review.
- Владельцы файлов: `analysis/notebooks/`, `docs/methodology/`, `docs/decisions/`.
- Зависимости: `MTZ-M19`.
- Проверки:
  - top списка объясним;
  - видно, где ranking ведет себя неадекватно.
- Критерий готовности: понятен список корректировок для следующей итерации.

### MTZ-M21. Обсудить ensemble или consensus layer

- Цель: отдельно решить, нужен ли новой волне `ensemble`-слой поверх нескольких моделей.
- Результат: инженерное решение по одному из вариантов:
  - `hard/soft voting`;
  - `stacking`;
  - comparative multi-model analysis без объединения предсказаний.
- Зона ответственности: models/evaluation/reporting.
- Владельцы файлов: `docs/architecture/`, `docs/methodology/`, позже `src/exohost/models/`, `src/exohost/evaluation/`, `src/exohost/reporting/`.
- Зависимости: `MTZ-M15`, `MTZ-M19`, `MTZ-M20`.
- Проверки:
  - решение опирается на реальные benchmark и real-run результаты;
  - ensemble не добавляется "для галочки" без прироста качества или explainability;
- если ensemble не нужен, это тоже фиксируется явно.
- Критерий готовности: понятно, остается ли новая волна single-model benchmark-first или получает отдельный consensus layer.

## Этап 8. Current DE Sprint

### MTZ-M22. Зафиксировать execution-plan текущего DE-спринта

- Цель: явно развести старый `V2`-контур, Gaia raw pools и новую MK ingestion-ветку.
- Результат: понятный рабочий план без догадок, что читаем из БД, что берем из внешнего `B/mk`, а что появится только после crossmatch.
- Зона ответственности: только планирование и границы источников.
- Владельцы файлов: `docs/architecture/`, `docs/methodology/`.
- Зависимости: `MTZ-M04`, `MTZ-M05`, `MTZ-M09`.
- Проверки:
  - зафиксировано, что `public.gaia_dr3_training` и `public.lab_gaia_mk_candidate_pool_raw_result` не подменяют MK label-source;
  - зафиксировано, что новая MK-ветка начинается с `B/mk -> raw/filtered/rejected`.
- Критерий готовности: следующий кодовый шаг не требует повторно спорить о роли источников.

### MTZ-M23. Перевести B/mk transform на one-pass builder

- Цель: убрать повторные проходы по полной CDS-таблице при построении `raw`, `filtered`, `rejected` и summary.
- Результат: единый transform builder для parser/pipeline слоя.
- Зона ответственности: `ingestion/bmk/filtering.py`, `ingestion/bmk/pipeline.py`.
- Владельцы файлов: `src/exohost/ingestion/bmk/`, `tests/unit/`.
- Зависимости: `MTZ-M22`.
- Проверки:
  - публичные контракты parser-а не ломаются;
  - `raw`, `filtered`, `rejected` и summary совпадают с текущим поведением;
  - pipeline реально использует один проход вместо нескольких.
- Критерий готовности: полный `B/mk` transform быстрее и остается воспроизводимым.

### MTZ-M24. Упростить export-контур без потери NULL-semantics

- Цель: ускорить запись staging CSV, не ломая порядок колонок и поведение `NULL`.
- Результат: более легкий export path для `raw`, `filtered`, `rejected`.
- Зона ответственности: `ingestion/bmk/export.py`, при необходимости `ingestion/bmk/normalization.py`.
- Владельцы файлов: `src/exohost/ingestion/bmk/`, `tests/unit/`.
- Зависимости: `MTZ-M23`.
- Проверки:
  - `None` и пустые значения не превращаются в строковый `nan`;
  - CSV-контракт совместим с DB-load.
- Критерий готовности: export быстрее и не меняет смысл данных.

### MTZ-M25. Прогнать полный B/mk в реальные raw/filtered/rejected relation

- Цель: заменить sample-проверку реальными таблицами новой MK ingestion-ветки.
- Результат: заполнены `lab.gaia_mk_external_raw`, `lab.gaia_mk_external_filtered`, `lab.gaia_mk_external_rejected`.
- Зона ответственности: `db`, `cli ingest`, controlled DB-write.
- Владельцы файлов: `src/exohost/db/`, `src/exohost/cli/ingest/`, `tests/unit/`.
- Зависимости: `MTZ-M23`, `MTZ-M24`.
- Проверки:
  - DDL соответствует docs;
  - DB-load проходит без ручного ad hoc SQL;
  - row counts в БД совпадают с parser summary.
- Критерий готовности: sample-этап закрыт, есть полноценные relation новой волны.

### MTZ-M26. Провести post-ingest audit новой MK staging-ветки

- Цель: снять реальные counts и reject breakdown до похода в `Gaia`.
- Результат: audit note по `raw`, `filtered`, `rejected` и числу строк, готовых к crossmatch.
- Зона ответственности: audit/reporting.
- Владельцы файлов: `docs/methodology/`, при необходимости `analysis/notebooks/`.
- Зависимости: `MTZ-M25`.
- Проверки:
  - есть count по каждому слою;
  - причины отбраковки видны явно;
  - решение о следующем Gaia-шаге основано на цифрах.
- Критерий готовности: понятно, какой объем реально идет в `Gaia Archive`.

### MTZ-M27. Зафиксировать upload-contract для Gaia Archive

- Цель: определить минимальный upload-friendly слой для crossmatch после локального filter-step.
- Результат: понятный contract для upload table из `lab.gaia_mk_external_filtered`.
- Зона ответственности: workflow/docs и следующий crossmatch-step.
- Владельцы файлов: `docs/methodology/`, позже `src/exohost/db/` или SQL templates.
- Зависимости: `MTZ-M26`.
- Проверки:
  - в upload layer остаются только поля, нужные для Gaia user table;
  - crossmatch не смешивается с label normalization и quality gate.
- Критерий готовности: к Gaia идем уже с зафиксированным и минимальным upload contract.

### MTZ-M28. Собрать локальный upload artifact из БД

- Цель: получить reproducible CSV для будущего `Gaia Archive` шага прямо из локальной БД.
- Результат: отдельный local export path для upload table поверх `lab.gaia_mk_external_filtered`.
- Зона ответственности: `db` read/export и отдельная CLI-команда pre-Gaia шага.
- Владельцы файлов: `src/exohost/db/`, `src/exohost/cli/`, `tests/unit/`.
- Зависимости: `MTZ-M27`.
- Проверки:
  - export читает именно upload-contract, а не ad hoc набор колонок;
  - output воспроизводим и не зависит от ручной сборки CSV;
  - шаг не подключается к `Gaia` и не смешивает upload с crossmatch.
- Критерий готовности: upload table можно собрать локально одной командой и передать в следующий Gaia-step без ручной подготовки.

### MTZ-M29. Materialize Узкий Crossmatch Layer После Gaia

- Цель: построить canonical relation `lab.gaia_mk_external_crossmatch` из wide Gaia export, не таща downstream-логику на `160` колонок raw landing table.
- Результат: deterministic crossmatch layer с `xmatch_rank`, `xmatch_selected` и `xmatch_batch_id`.
- Зона ответственности: `db` write/materialization, отдельная CLI-команда post-Gaia шага и связанные docs.
- Владельцы файлов: `src/exohost/db/`, `src/exohost/cli/`, `tests/unit/`, `docs/methodology/`.
- Зависимости: `MTZ-M28` и фактически завершенный Gaia xmatch/export.
- Проверки:
  - wide raw landing relation не используется как рабочий training-layer;
  - canonical crossmatch layer сохраняет все match-кандидаты и не теряет ambiguity;
  - выбранный рабочий match определяется детерминированно по минимальной separation.
- Критерий готовности: из post-Gaia raw export одной командой строится `lab.gaia_mk_external_crossmatch`, который уже можно использовать как следующий слой пайплайна.

### MTZ-M30. Собрать External Labeled Layer Локально В БД

- Цель: выполнить label normalization после `Gaia` без повторного похода в `Gaia Archive`.
- Результат: `lab.gaia_mk_external_labeled`, собранный из `lab.gaia_mk_external_filtered` и выбранных строк `lab.gaia_mk_external_crossmatch`.
- Зона ответственности: локальный DB-step и docs по post-Gaia normalization.
- Владельцы файлов: `src/exohost/db/`, `src/exohost/cli/`, `tests/unit/`, `docs/methodology/`.
- Зависимости: `MTZ-M29`.
- Проверки:
  - нормализация не читается напрямую из `public.raw_landing_table`;
  - в labeled слой попадают только `xmatch_selected = TRUE`;
  - `raw_sptype` и разложенные MK-поля сохраняют traceability к `external_row_id` и `source_id`;
  - дубли по `source_id` не исчезают молча и фиксируются как conflict-audit.
- Критерий готовности: следующий шаг пайплайна выполняется полностью локально в БД и не требует нового upload/download из `Gaia`.

### MTZ-M31. Зафиксировать Singleton Policy И Compatibility Stage Mapping

- Цель: определить первый безопасный training-grade срез после `external_labeled`.
- Результат: правило, что `training_reference` первой волны строится только на conflict-free `source_id`, а coarse `evolution_stage` получается из `luminosity_class` по явной схеме.
- Зона ответственности: `docs/methodology/` и отдельный label-модуль в `src/exohost/labels/`.
- Владельцы файлов: `docs/methodology/`, `src/exohost/labels/`, `tests/unit/`.
- Зависимости: `MTZ-M30`.
- Проверки:
  - mapping `luminosity_class -> evolution_stage` зафиксирован явно;
  - `external_labeled` не делает silent dedup по `source_id`;
  - coarse stage не подменяет canonical `luminosity_class`.
- Критерий готовности: compatibility policy не размазана по SQL и может переиспользоваться в training/view-коде.

### MTZ-M32. Зафиксировать Official Gaia Physical Contract И Data Gap

- Цель: синхронизировать новый `MK`-слой с официальным Gaia DR3 datamodel и не строить `training_reference` на неверных именах полей.
- Результат: в docs зафиксировано, что официальный радиусный field для Gaia DR3 — `radius_flame`, а текущий `public.raw_landing_table` не содержит FLAME-полей.
- Зона ответственности: `docs/methodology/`, `docs/architecture/`.
- Владельцы файлов: `docs/methodology/`, `docs/architecture/`.
- Зависимости: `MTZ-M30`.
- Проверки:
  - `radius_gspphot` не используется как canonical Gaia DR3 field в новой `MK`-ветке;
  - gap между live landing relation и целевым `training_reference` зафиксирован явно;
  - legacy compatibility alias описан отдельно и не смешан с canonical storage.
- Критерий готовности: команда не пытается строить `training_reference` до появления нужного Gaia enrichment.

### MTZ-M33. Выполнить Chunked Gaia Enrichment Для Missing FLAME Fields

- Цель: дообогатить `source_id`-срезы недостающими official Gaia DR3 полями без giant single-pass query.
- Результат: chunk-wise raw enrichment artifacts и локальные relation-ы с минимумом `source_id`, `radius_flame`, при необходимости `lum_flame`, `evolstage_flame` и связанными provenance-полями.
- Зона ответственности: Gaia batch export/import, затем `db` landing и audit.
- Владельцы файлов: `docs/methodology/`, позже `src/exohost/db/`, `src/exohost/cli/`, `tests/unit/`.
- Зависимости: `MTZ-M31`, `MTZ-M32`.
- Проверки:
  - повторный поход в `Gaia` идет только за недостающими official полями;
  - chunking выполняется по `source_id`-батчам и не ломает traceability;
  - wide raw landing и enrichment landing остаются отдельными relation-ами.
- Критерий готовности: после enrichment можно строить `lab.gaia_mk_training_reference` без хрупких заглушек и silent field substitution.

### MTZ-M34. Зафиксировать Hierarchical And OOD Strategy

- Цель: формально разделить проект на coarse classification, refinement и OOD/reject слой.
- Результат: отдельный docs-contract для `ID-space`, `refinement-space`, `OOD-space` и итогового decision layer.
- Зона ответственности: `docs/methodology/`, `docs/architecture/`.
- Владельцы файлов: `docs/methodology/`, `docs/architecture/`.
- Зависимости: `MTZ-M33`.
- Проверки:
  - `OBAFGKM` зафиксированы как normal ID-space первой волны;
  - OOD не определяется "по остаточному принципу";
  - disagreement Gaia-модулей описан как reliability signal, а не как независимая истина.
- Критерий готовности: дальнейшее дополнение проекта идет по явной исследовательской схеме, а не по неформальной идее в чате.

### MTZ-M35. Собрать Official Gaia OOD Candidate Pools

- Цель: получить отдельные OOD-source relation-ы из официальных Gaia DR3 таблиц и flags.
- Результат: зафиксированные candidate pools для white dwarfs, binaries/non-single stars, emission-line stars, carbon stars и outlier-like объектов.
- Зона ответственности: Gaia query/export, DB landing, docs с критериями отбора.
- Владельцы файлов: `docs/methodology/`, позже `src/exohost/db/`, `src/exohost/cli/`, `tests/unit/`.
- Зависимости: `MTZ-M34`.
- Проверки:
  - OOD-source не смешивается с ID training-source;
  - все критерии отбора идут из официального Gaia DR3 datamodel;
  - каждая OOD-группа сохраняет traceability к source relation и Gaia fields.
- Критерий готовности: у проекта есть отдельный и воспроизводимый OOD-pool, а не ad hoc набор "непонятных" объектов.

### MTZ-M36. Дополнить Project Decision Layer

- Цель: встроить в проект итоговый decision layer поверх coarse class, refinement и host/priority logic.
- Результат: формализованы выходные состояния `classified`, `low_priority`, `unknown/review`, `ood`.
- Зона ответственности: `docs/methodology/`, позже `src/exohost/labels/`, `src/exohost/ranking/`, `src/exohost/reporting/`.
- Владельцы файлов: `docs/methodology/`, позже `src/exohost/`, `tests/unit/`.
- Зависимости: `MTZ-M34`, `MTZ-M35`.
- Проверки:
  - итоговый статус не смешивает обычную классификацию с OOD;
  - low-priority logic и OOD logic разделены;
  - uncertain objects не теряются, а идут в отдельный review/OOD слой.
- Критерий готовности: архитектура проекта покрывает не только "предсказать класс", но и "что делать с объектом дальше".

## Финальная Напоминалка

Текущий `Gaia`-step закрыт, дальше идем в локальную DB-нормализацию.

## Этап 9. DB Layer Closure Before Code

### MTZ-M37. Зафиксировать Schema Layout `public` И `lab`

- Цель: убрать двусмысленность между raw/clean source assets и working/training layers.
- Результат: в docs и live-БД согласованно зафиксировано, что `public` хранит raw landing и clean reusable sources, а `lab` — normalized working, training, gate и decision layers.
- Зона ответственности: `docs/methodology/`, live DB layout.
- Владельцы файлов: `docs/methodology/`, при необходимости `docs/architecture/`.
- Зависимости: `MTZ-M35`, `MTZ-M36`.
- Проверки:
  - reusable clean source asset не остается только в `lab`;
  - training/gate relation не уходит в `public`;
  - текущие live relation и docs не противоречат друг другу.
- Критерий готовности: команда понимает, где хранить raw, clean source asset и derived training-layer без споров "по памяти".

### MTZ-M38. Закрыть `public` Reusable Enrichment Assets

- Цель: собрать и разложить по `public` все clean reusable relation, которые нужны нескольким downstream-слоям.
- Результат:
  - `public.gaia_id_flame_enrichment_clean`
  - `public.gaia_mk_core_enrichment_clean`
  - `public.gaia_mk_flame_enrichment_raw`
  - `public.gaia_mk_flame_enrichment_clean`
- Зона ответственности: live DB materialization и audit.
- Владельцы файлов: live DB, при необходимости `docs/methodology/`.
- Зависимости: `MTZ-M37`.
- Проверки:
  - canonical Gaia physics доступны без повторного giant join;
  - `radius_flame` хранится как canonical field;
  - compatibility `radius_gspphot` не подменяет official contract.
- Критерий готовности: downstream layers могут читать reusable enrichment-источники напрямую из `public`.

### MTZ-M39. Собрать `lab.gaia_id_coarse_reference`

- Цель: закрыть первый `coarse / ID` слой отдельным train-grade relation.
- Результат: `lab.gaia_id_coarse_reference` с явной политикой уникальности по `source_id` и target `spec_class`.
- Зона ответственности: live DB materialization и audit.
- Владельцы файлов: live DB, позже loaders/views.
- Зависимости: `MTZ-M38`.
- Проверки:
  - дубли между `class` и `evolved_class` обработаны явно;
  - `O/B/A/F/G/K/M` представлены как единый coarse-layer;
  - relation пригоден для первого классификатора без ad hoc cleanup.
- Критерий готовности: первый слой проекта читается из отдельного relation, а не из россыпи source-table.

### MTZ-M40. Собрать `lab.gaia_mk_training_reference`

- Цель: закрыть второй `refinement / subclass` слой как train-grade relation.
- Результат: `lab.gaia_mk_training_reference`, собранный из `lab.gaia_mk_external_labeled`, `public.gaia_mk_core_enrichment_clean` и `public.gaia_mk_flame_enrichment_clean`.
- Зона ответственности: live DB materialization и audit.
- Владельцы файлов: live DB, позже loaders/views.
- Зависимости: `MTZ-M38`.
- Проверки:
  - первая волна использует только `has_source_conflict = FALSE`;
  - canonical label fields сохраняются без silent rewrite;
  - relation уникален по `source_id`.
- Критерий готовности: второй слой готов к обучению subclass/luminosity задач без giant raw joins.

### MTZ-M41. Собрать `lab.gaia_ood_training_reference`

- Цель: закрыть отдельный OOD relation без смешения с normal `ID`.
- Результат: `lab.gaia_ood_training_reference` поверх `public.gaia_ood_candidate_pool_clean`.
- Зона ответственности: live DB materialization и audit.
- Владельцы файлов: live DB, позже loaders/views.
- Зависимости: `MTZ-M37`.
- Проверки:
  - `ood_group` и selector/provenance fields сохраняются;
  - overlap signals сохраняются;
  - OOD relation не смешан с обычным coarse/refinement source.
- Критерий готовности: проект имеет отдельный DB-source для OOD/unknown контура.

### MTZ-M42. Собрать `lab.gaia_mk_quality_gated` И `lab.gaia_mk_unknown_review`

- Цель: закрыть gate/review слой до перехода в code-side реализацию.
- Результат:
  - `lab.gaia_mk_quality_gated`
  - `lab.gaia_mk_unknown_review`
- Зона ответственности: live DB materialization, docs и audit.
- Владельцы файлов: live DB, позже loaders/views.
- Зависимости: `MTZ-M40`, `MTZ-M41`.
- Проверки:
  - `unknown`, `ood`, `reject` не теряются;
  - review-слой хранится отдельной relation;
  - обычный training/scoring не читает сырые uncertain rows как normal case.
- Критерий готовности: DB foundation новой иерархической схемы закрыт полностью, и только после этого команда идет в loaders, views и model-код.

## Этап 10. View Contracts And Loader Bridge

### Общий Инвариант Этапа

Для `MTZ-M43 ... MTZ-M48` обязательно:

- `1 файл = 1 ответственность`
- без монолитных loader/view-модулей
- `PEP 8`
- явная типизация
- простая Python-логика без лишней абстракции
- без лишних зависимостей
- после каждого небольшого куска:
  - micro-QA
  - `ruff`
  - точечный `mypy/pyright`
  - целевые тесты
- после завершения микро-ТЗ:
  - scoped big-QA только по написанному слою

### MTZ-M43. Зафиксировать Training View Contract Для Coarse Layer

- Цель: определить source и feature/target contract для первого coarse classifier.
- Результат: зафиксирован contract для `lab.v_gaia_id_coarse_training`.
- Зона ответственности: docs-contract и затем отдельный DB/view модуль.
- Владельцы файлов: `docs/methodology/`, позже `src/exohost/db/` или SQL/view модуль.
- Зависимости: `MTZ-M39`, `MTZ-M42`.
- Правило декомпозиции:
  - contract фиксируется отдельно от loader-кода;
  - view-логика не смешивается с refinement/OOD в одном файле.
- Micro-QA:
  - сверка полей с live relation;
  - проверка, что target один и не размазан;
  - проверка, что required features описаны явно.
- Критерий готовности: coarse training view можно materialize-ить без импровизации в loaders.

### MTZ-M44. Зафиксировать Training View Contract Для Refinement Layer

- Цель: определить source и feature/target contract для subclass/luminosity scenario.
- Результат: зафиксирован contract для `lab.v_gaia_mk_refinement_training`.
- Зона ответственности: docs-contract и затем отдельный DB/view модуль.
- Владельцы файлов: `docs/methodology/`, позже `src/exohost/db/` или SQL/view модуль.
- Зависимости: `MTZ-M40`, `MTZ-M42`.
- Правило декомпозиции:
  - refinement view проектируется отдельно от coarse и OOD;
  - gate-логика не размазывается по loader-коду.
- Micro-QA:
  - row-count sanity check;
  - null coverage по subclass/luminosity;
  - проверка, что `quality_state/ood_state` используются как явные filters.
- Критерий готовности: refinement source формально определен и не требует giant ad hoc SQL.

### MTZ-M45. Зафиксировать `ID vs OOD` Training Contract

- Цель: определить бинарный source для `ID`/`OOD` задачи.
- Результат: зафиксирован contract для `lab.v_gaia_id_ood_training`.
- Зона ответственности: docs-contract и затем отдельный DB/view модуль.
- Владельцы файлов: `docs/methodology/`, позже `src/exohost/db/` или SQL/view модуль.
- Зависимости: `MTZ-M41`, `MTZ-M42`.
- Правило декомпозиции:
  - OOD-view живет отдельно от normal refinement view;
  - `candidate_ood` и `unknown` не смешиваются молча с clean ID rows.
- Micro-QA:
  - counts по `id`/`ood`;
  - проверка feature overlap между источниками;
  - проверка, что OOD provenance не потерян.
- Критерий готовности: OOD-task можно materialize-ить и читать без ручной склейки relation.

### MTZ-M46. Собрать DB Views Для Новых Tasks

- Цель: materialize/create views поверх уже закрытых reference relation.
- Результат:
  - `lab.v_gaia_id_coarse_training`
  - `lab.v_gaia_mk_refinement_training`
  - `lab.v_gaia_id_ood_training`
- Зона ответственности: DB/views.
- Владельцы файлов: `src/exohost/db/` или SQL/view modules, `tests/unit/`.
- Зависимости: `MTZ-M43`, `MTZ-M44`, `MTZ-M45`.
- Правило декомпозиции:
  - coarse views, refinement views и OOD views не живут в одном giant модуле;
  - `1 view family = 1 файл`.
- Micro-QA после каждого view:
  - `SELECT count(*)`
  - `SELECT count(distinct source_id)`
  - profile по `NULL`
  - точечный `ruff` / `mypy` / `pyright` / tests по новому DB-модулю
- Scoped big-QA:
  - только view/DB-слой этого шага
- Критерий готовности: loaders получают устойчивые relation names.

### MTZ-M47. Реализовать Loaders Для Новых Views

- Цель: научить код читать новые training views без ручных SQL.
- Результат:
  - отдельный loader для coarse
  - отдельный loader для refinement
  - отдельный loader для OOD
- Реализация первой волны:
  - `src/exohost/contracts/hierarchical_dataset_contracts.py`
  - `src/exohost/datasets/load_gaia_id_coarse_training_dataset.py`
  - `src/exohost/datasets/load_gaia_mk_refinement_training_dataset.py`
  - `src/exohost/datasets/load_gaia_id_ood_training_dataset.py`
  - `src/exohost/features/hierarchical_training_frame.py`
- Зона ответственности: datasets/loaders.
- Владельцы файлов: `src/exohost/datasets/`, `src/exohost/contracts/`, `src/exohost/features/`, `tests/unit/`.
- Зависимости: `MTZ-M46`.
- Правило декомпозиции:
  - `1 loader = 1 файл`;
  - relation-contracts и frame-normalization не живут внутри loader-файлов;
  - общие типы и мелкие helper-функции — отдельно;
  - без giant shared loader на все задачи.
- Micro-QA после каждого loader:
  - `ruff`
  - точечный `mypy/pyright`
  - целевые тесты на колонки, target и shape
- Дополнительная policy первой волны:
  - coarse frame явно строит `evolution_stage` из `is_evolved`;
  - refinement frame явно маппит `luminosity_class -> evolution_stage`;
  - OOD frame явно схлопывает multi-membership rows до unique `source_id`,
    сохраняя `ood_membership_count`, `has_multi_ood_membership` и `ood_group_members`.
- Scoped big-QA:
  - только новые loaders и их тесты
- Критерий готовности: baseline run можно запускать поверх новых relation names.

### MTZ-M48. Первые Scoped Train/Benchmark Runs

- Цель: сделать первые baseline run по новым слоям.
- Статус: `closed` (`2026-03-28`).
- Результат:
  - coarse baseline
  - refinement baseline
  - OOD baseline
- Зона ответственности: training/evaluation/reporting.
- Владельцы файлов: `src/exohost/training/`, `src/exohost/evaluation/`, `src/exohost/reporting/`, `tests/unit/`.
- Зависимости: `MTZ-M47`.
- Правило декомпозиции:
  - сначала baseline;
  - без premature ensemble/consensus;
  - отдельные run-config или runner entrypoints по задачам.
- Micro-QA:
  - проверка artifacts;
  - проверка metrics и target alignment;
  - точечный `ruff` / `mypy` / `pyright` / tests по затронутому training/evaluation слою
- Scoped big-QA:
  - только новые training/evaluation pieces
- Критерий готовности: иерархическая схема впервые проходит end-to-end на реальных relation.

Фактический итог первой волны:

- `gaia_id_coarse_classification`
  - test accuracy: `0.992926`
  - test balanced_accuracy: `0.992379`
  - test macro_f1: `0.992573`
- `gaia_mk_refinement_classification`
  - test accuracy: `0.320336`
  - test balanced_accuracy: `0.187861`
  - test macro_f1: `0.189683`
- `gaia_id_ood_classification`
  - test accuracy: `0.995734`
  - test balanced_accuracy: `0.926215`
  - test macro_f1: `0.944521`

Зафиксированная first-wave policy:

- refinement-task использует support cutoff `>= 15` по full subclass
  `spec_class + spectral_subclass`;
- это убирает warning-ы о слишком редких классах при `30%` test split и `10-fold CV`;
- coarse и OOD baseline считаются уже устойчивыми baseline-ориентирами;
- refinement baseline считается рабочим, но еще не финальным исследовательским решением
  для второго слоя.

## Этап 11. Second-Wave Refinement And Decision Design

### Общий Инвариант Этапа

Для `MTZ-M49 ... MTZ-M53` обязательно:

- `1 файл = 1 ответственность`
- без монолитных training/inference/calibration модулей
- `PEP 8`
- явная типизация
- простая Python-логика без лишней абстракции
- без лишних зависимостей
- сначала design-contract, потом code-side реализация
- после каждого небольшого куска:
  - micro-QA
  - `ruff`
  - точечный `mypy/pyright`
  - целевые тесты
- после завершения микро-ТЗ:
  - scoped big-QA только по написанному слою

Official опора этапа:

- scikit-learn `HistGradientBoostingClassifier`
- scikit-learn multiclass docs
- scikit-learn cross-validation docs
- scikit-learn probability calibration docs
- scikit-learn `TunedThresholdClassifierCV`
- scikit-learn metrics docs (`balanced_accuracy`, `classification_report`)

### MTZ-M49. Зафиксировать Per-Class Support Audit Contract Для Second-Wave Refinement

- Цель: формально определить, какие coarse classes реально идут во второй слой
  refinement.
- Статус: `closed` (`2026-03-28`).
- Результат:
  - audit contract по `A/B/F/G/K/M/O`
  - explicit policy для rare-tail support
  - refinement-enabled class list первой second-wave
- Зона ответственности: docs, DB audit, reporting.
- Владельцы файлов: `docs/methodology/`, `src/exohost/db/` или DB audit modules,
  `tests/unit/`.
- Зависимости: `MTZ-M48`.
- Правило декомпозиции:
  - audit relation/view отдельно;
  - policy document отдельно;
  - без смешения с training-runner.
- Micro-QA:
  - counts по coarse class;
  - support profile по full subclass;
  - sanity-check по cutoff `>= 15`, `>= 20`, `>= 30`.
- Критерий готовности: refinement-enabled classes выбраны не "на глаз", а по audit.

Фактический итог:

- audit materialized в:
  - `lab.gaia_mk_refinement_support_audit`
  - `lab.gaia_mk_refinement_support_audit_summary`
- current refinement source:
  - `155373` rows
  - `155373` distinct `source_id`
  - `66` full subclasses
- refinement-enabled classes:
  - `A`
  - `B`
  - `F`
  - `G`
  - `K`
  - `M`
- coarse-only:
  - `O`
- explicit rare-tail exclusions на старте second-wave:
  - `K9`
  - весь `O` tail
- borderline subclass:
  - `M9` проходит на cutoff `15/20`, но выпадает на `30`

### MTZ-M50. Зафиксировать Contracts Для Per-Class Refinement Families

- Цель: заменить flat refinement-task на family of coarse-conditioned tasks.
- Статус: `closed` (`2026-03-28`).
- Результат:
  - contracts для `lab.v_gaia_mk_refinement_training_<class>`
  - единая policy для feature/target contract внутри каждого family-view
- Зона ответственности: docs-contract, DB/view design.
- Владельцы файлов: `docs/methodology/`, позже `src/exohost/db/`, `tests/unit/`.
- Зависимости: `MTZ-M49`.
- Правило декомпозиции:
  - `1 refinement family = 1 view contract`;
  - feature policy отдельно от calibration policy;
  - без giant universal refinement view.
- Micro-QA:
  - row-count sanity check по каждой family-view;
  - null coverage по обязательным features;
  - проверка, что unsupported classes не попали в family by accident.
- Критерий готовности: second-wave refinement имеет stable task decomposition.

Фактический итог:

- family contracts зафиксированы в:
  - [refinement_family_contracts_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/contracts/refinement_family_contracts_ru.md)
- second-wave family views:
  - `lab.v_gaia_mk_refinement_training_a`
  - `lab.v_gaia_mk_refinement_training_b`
  - `lab.v_gaia_mk_refinement_training_f`
  - `lab.v_gaia_mk_refinement_training_g`
  - `lab.v_gaia_mk_refinement_training_k`
  - `lab.v_gaia_mk_refinement_training_m`
- `O` family не создается и остается coarse-only
- explicit exclusion:
  - `K9`
- borderline subclass:
  - `M9` остается включенным на default cutoff `>= 15`

### MTZ-M51. Зафиксировать Calibration And Threshold Policy

- Цель: спроектировать confidence/reject layer на calibrated probabilities, а не на
  захардкоженных ad hoc порогах.
- Статус: `closed` (`2026-03-28`).
- Результат:
  - contract для calibration stage
  - threshold policy для `OOD`, `unknown/review` и refinement handoff
- Зона ответственности: docs, evaluation, later training/inference.
- Владельцы файлов: `docs/methodology/`, позже `src/exohost/evaluation/`,
  `src/exohost/training/`, `tests/unit/`.
- Зависимости: `MTZ-M48`.
- Правило декомпозиции:
  - calibration modules отдельно от base model runners;
  - threshold tuning отдельно от final decision mapping;
  - без giant inference policy file.
- Official policy:
  - binary `ID/OOD` threshold проектируем через `TunedThresholdClassifierCV`;
  - probability calibration проектируем через `CalibratedClassifierCV`;
  - multiclass thresholds подбираются только на validation.
- Micro-QA:
  - calibration metric/report contract;
  - threshold metric contract;
  - проверка, что tuning не использует test split.
- Критерий готовности: decision confidence policy формально определена.

Фактический итог:

- policy зафиксирована в:
  - [calibration_threshold_policy_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/contracts/calibration_threshold_policy_ru.md)
- official-backed решения:
  - `CalibratedClassifierCV` как calibration stage
  - `TunedThresholdClassifierCV` для binary `ID/OOD`
  - `balanced_accuracy` как default threshold-tuning metric
- project policy:
  - `ID/OOD` threshold тюним first
  - `coarse` пока оставляем как strong probability-output stage
  - `refinement` calibration включаем только после family baselines
  - numeric thresholds не фиксируем в docs до отдельного validation step

### MTZ-M52. Зафиксировать Final Decision Layer Contract

- Цель: определить, что именно возвращает система после coarse/OOD/refinement.
- Статус: `closed` (`2026-03-28`).
- Результат:
  - contract для final decision relation/view
  - states для `id`, `candidate_ood`, `ood`, `unknown`
  - states для refinement outcome
- Зона ответственности: docs, DB/view design, later inference layer.
- Владельцы файлов: `docs/methodology/`, позже `src/exohost/db/`,
  `src/exohost/models/`, `tests/unit/`.
- Зависимости: `MTZ-M50`, `MTZ-M51`.
- Правило декомпозиции:
  - final decision contract отдельно от scoring/priority;
  - unknown/review logic не живет inside subclass model code.
- Micro-QA:
  - state enumeration review;
  - traceability review;
  - проверка, что ни один outcome не теряется молча.
- Критерий готовности: end-to-end decision flow объясним и воспроизводим.

Фактический итог:

- final contract зафиксирован в:
  - [final_decision_contract_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/contracts/final_decision_contract_ru.md)
- явные final states:
  - `id`
  - `candidate_ood`
  - `ood`
  - `unknown`
- явные refinement states:
  - `not_attempted`
  - `accepted`
  - `rejected_to_unknown`
- зафиксирован routing order:
  - quality gate
  - `ID/OOD` gate
  - coarse
  - refinement handoff
  - refinement decision
  - priority integration
- зафиксированы traceability fields и правило,
  что `priority_state` не заполняется для `ood/unknown`.

### MTZ-M53. Реализовать Second-Wave Bridge В Коде

- Цель: только после фиксации contracts перевести second-wave design в code-side
  modules.
- Результат:
  - per-class refinement loaders/views
  - calibration modules
  - threshold-tuning modules
  - decision-layer modules
- Зона ответственности: training/evaluation/inference.
- Владельцы файлов: `src/exohost/datasets/`, `src/exohost/features/`,
  `src/exohost/evaluation/`, `src/exohost/training/`, `src/exohost/models/`,
  `tests/unit/`.
- Зависимости: `MTZ-M49`, `MTZ-M50`, `MTZ-M51`, `MTZ-M52`.
- Правило декомпозиции:
  - `1 module family = 1 responsibility`;
  - base model runners, calibration and decision orchestration не живут в одном
    файле;
  - сначала узкий кусок, потом следующий;
  - без giant hierarchical runner "на все".
- Micro-QA:
  - после каждого нового файла;
  - `ruff`
  - точечный `mypy/pyright`
  - целевые unit tests
- Scoped big-QA:
  - только second-wave code-slice
- Критерий готовности: second-wave design реализован без монолитов и ad hoc logic.

Текущий статус на `2026-03-28`:

- first code-slice закрыт:
  - family DB views helper
  - family dataset/feature contracts
  - family loaders
  - family training-frame
  - family task registry
  - family benchmark/training runners
  - CLI dispatch
- micro/scoped QA по этому срезу:
  - `ruff` ok
  - `mypy` ok
  - `pyright` ok
  - targeted `pytest` ok
- second code-slice тоже закрыт:
  - post-hoc calibration modules
  - threshold-tuning modules
  - explicit `ID/OOD` gate contract
  - calibrated+tuned `ID/OOD` runner
  - sklearn classifier-semantics fix для custom wrappers
  - loader ordering fix для limited `ID/OOD` slices
- live smoke-run по second code-slice:
  - `gaia_id_ood_classification`
  - `HistGradientBoosting`
  - `limit=5000`
  - test `accuracy=0.969272`
  - test `balanced_accuracy=0.968762`
  - test `macro_f1=0.961488`
  - tuned threshold `0.041898`
- third code-slice тоже закрыт:
  - refinement handoff module
  - final decision routing module
  - targeted unit-tests for final routing cases
  - separate exports in post-hoc package without giant decision file
- micro/scoped QA по third code-slice:
  - `ruff` ok
  - `mypy` ok
  - `pyright` ok
  - targeted `pytest` ok
- fourth code-slice тоже закрыт:
  - probability summary helper
  - compact coarse scoring helper
  - compact refinement family scoring helper
  - final decision bridge for real stage outputs
  - targeted unit-tests for stage scoring and merge contract
- micro/scoped QA по fourth code-slice:
  - `ruff` ok
  - `mypy` ok
  - `pyright` ok
  - targeted `pytest` ok
- fifth code-slice тоже закрыт:
  - explicit `candidate_ood` secondary policy module
  - frame-level final decision runner
  - targeted unit-tests for candidate policy and decision orchestration
- micro/scoped QA по fifth code-slice:
  - `ruff` ok
  - `mypy` ok
  - `pyright` ok
  - targeted `pytest` ok
- sixth code-slice тоже закрыт:
  - separate priority input adapter
  - separate priority integration module over explainable ranking
  - stale upstream `priority_state` removed from final routing layer
  - targeted unit-tests for priority integration contract
- micro/scoped QA по sixth code-slice:
  - `ruff` ok
  - `mypy` ok
  - `pyright` ok
  - targeted `pytest` ok
- seventh code-slice тоже закрыт:
  - separate threshold-policy artifact layer for `ID/OOD`
  - separate saved-artifact bundle loader for `OOD/coarse/refinement/host`
  - separate higher-level runner over saved artifacts
  - separate final-decision artifact persistence layer
  - separate `decide` CLI command
  - targeted unit-tests for artifacts, bundle, runner and CLI
- micro/scoped QA по seventh code-slice:
  - `ruff` ok
  - `mypy` ok
  - `pyright` ok
  - targeted `pytest` ok
- `MTZ-M53` закрыт полностью
- следующий пакет работ:
  - observability notebook for data/model state
  - end-to-end dry run over saved artifacts

## Следующий Пакет После Закрытия Основного Плана

После закрытия `MTZ-M01 ... MTZ-M53` проект переходит в stabilization и
scientific review phase.

Детальный исполнимый план этого этапа живет в:

- [post_run_stabilization_tz_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/plans/post_run_stabilization_tz_ru.md)

Коротко следующий пакет выглядит так:

- `MTZ-S01` baseline run registry
- `MTZ-S02` issue ledger
- `MTZ-S03` star-level review
- `MTZ-S04` traceability audit
- `MTZ-S05` performance profiling
- `MTZ-S06` precision bugfix cycle
- `MTZ-S07` scientific findings pack
- `MTZ-S08` next iteration boundary
