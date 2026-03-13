# QA Backlog and Decision Map

Дата фиксации: 13 марта 2026 года

Основание:
- [qa_full_audit_log_2026-03-13_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/experiments/QA/qa_full_audit_log_2026-03-13_ru.md)
- промежуточные ledgers по коду, тестам, docs, notebooks/SQL и артефактам

Сводка findings:
- `OK`: 37
- `TOLERABLE`: 35
- `FIX`: 18

Главный принцип этого backlog:
- не полировать проект до абстрактного идеала;
- исправлять только то, что реально даёт выигрыш в воспроизводимости, понятности, архитектурной чистоте и надёжности.

## P0

### P0-1. Зафиксировать version-control policy для текущего состояния проекта

Связанные findings:
- `QA-004`
- `QA-005`
- `QA-006`
- `QA-007`
- `QA-008`
- `QA-081`
- `QA-092`

Проблема:
- значимая часть текущего состояния проекта живёт как `untracked`, включая comparison-layer, новые тесты, notebooks, SQL/ADQL и docs.

Почему это важно:
- это главный риск воспроизводимости и “потери истины” о текущем состоянии repo.

Решение:
- либо трекать эти файлы как каноническое текущее состояние;
- либо явно формализовать, что из этого осознанно не входит в git.

### P0-2. Зафиксировать canonical policy для comparison artifacts

Связанные findings:
- `QA-085`
- `QA-086`
- `QA-092`

Проблема:
- в `experiments/model_comparison/` лежат несколько волн прогонов, но не зафиксировано, какая из них текущая каноническая.

Почему это важно:
- без этого findings, notebooks, slides и README могут ссылаться на разные поколения артефактов.

Решение:
- явным образом выделить canonical run;
- старые волны либо архивировать, либо пометить как historical.

## P1

### P1-1. Развязать `priority_pipeline.decision` и `priority_pipeline.input_data`

Связанные findings:
- `QA-028`
- `QA-051`
- `QA-056`
- `QA-061`
- `QA-093`

Проблема:
- decision-layer тянет helper из input-слоя.

Почему это важно:
- это ухудшает границы пакета и усложняет локальную эволюцию decision logic.

### P1-2. Развязать `host_model.db` и `host_model.fit`

Связанные findings:
- `QA-029`
- `QA-051`
- `QA-093`

Проблема:
- DB-layer зависит от training preparation helper-а.

Почему это важно:
- это неидеальная архитектурная зависимость между слоями доступа к данным и fit-логикой.

### P1-3. Убрать import-time `sys.path` side effects из EDA packages

Связанные findings:
- `QA-045`
- `QA-051`
- `QA-093`

Проблема:
- `analysis/host_eda/__init__.py` и `analysis/router_eda/__init__.py` модифицируют `sys.path` при импорте.

Почему это важно:
- это хрупко и создаёт неявные побочные эффекты на уровне package import.

### P1-4. Усилить targeted tests для production boundary layers

Связанные findings:
- `QA-056`
- `QA-057`
- `QA-058`
- `QA-061`

Проблема:
- внутренние слои `priority_pipeline`, `input_layer`, `infra/logbooks` и численные edge cases decision/calibration покрыты слабо.

Почему это важно:
- сейчас часть регрессов будет ловиться поздно и не очень диагностично.

### P1-5. Обновить устаревшие docs, которые выглядят как current state

Связанные findings:
- `QA-070`
- `QA-071`
- `QA-072`

Проблема:
- `vkr_requirements_traceability` и `ood_unknown_baselines_tz` уже частично расходятся с repo-state.

Почему это важно:
- документация начинает проигрывать коду именно там, где должна быть самой надёжной.

### P1-6. Очистить явный файловый мусор и неиспользуемые визуальные residue

Связанные findings:
- `QA-084`
- `QA-087`

Проблема:
- `.DS_Store` и generic screenshots не несут полезной роли.

Почему это важно:
- это дешёвый, но очевидный cleanup.

## P2

### P2-1. Привести `requirements.txt` к более осмысленной роли

Связанные findings:
- `QA-020`

Проблема:
- текущий файл больше похож на полный freeze локального окружения, чем на минимальный манифест проекта.

### P2-2. Решить вопрос с `pyrightconfig.json`

Связанные findings:
- `QA-013`
- `QA-022`

Проблема:
- pyright config в repo есть, а сам инструмент в окружении не установлен.

### P2-3. Определить судьбу planning docs

Связанные findings:
- `QA-072`

Проблема:
- в `docs/` рядом живут canonical docs и historical planning.

Решение:
- либо разнести их по разным каталогам;
- либо хотя бы явно помечать статус документа.

### P2-4. Подумать о future refactor `analysis/model_comparison/snapshot.py`

Связанные findings:
- `QA-047`
- `QA-051`

Проблема:
- snapshot-module уже самый тяжёлый узел research-layer.

Важно:
- это не urgent bugfix, а технический долг.

### P2-5. Сделать README чуть резче по фокусу

Связанные findings:
- `QA-069`
- `QA-074`

Проблема:
- README одновременно и overview, и user guide, и ВКР-карта, и roadmap.

Важно:
- это полезное улучшение, но не срочная правка.

### P2-6. Определить data/artifact policy в `data/`

Связанные findings:
- `QA-090`

Проблема:
- не до конца понятно, что в `data/` считать production artifact, sample data и EDA residue.

### P2-7. При желании подчистить notebook output noise

Связанные findings:
- `QA-077`
- `QA-082`

Проблема:
- notebooks сохраняют локальные пути и warning-noise в outputs.

Важно:
- это косметика, а не блокер.

## P3

### P3-1. Мониторить большие, но пока приемлемые модули

Связанные findings:
- `QA-016`
- `QA-017`
- `QA-052`
- `QA-094`

Кандидаты:
- `src/input_layer.py`
- `src/decision_calibration/reporting.py`
- `analysis/model_comparison/mlp_baseline.py`
- `analysis/model_comparison/presentation_assets.py`

Решение:
- пока не трогать без нового pain-point.

### P3-2. Не раздувать и без того широкие export surfaces

Связанные findings:
- `QA-027`
- `QA-046`

Решение:
- не обязательно срочно сужать фасады, но точно не расширять их дальше без причины.

## Осознанно оставляем как есть

### Ядро математики и production scoring

Почему оставляем:
- аудит не нашёл фундаментального математического дефекта.

Связанные findings:
- `QA-034`
- `QA-035`
- `QA-037`
- `QA-041`
- `QA-042`
- `QA-095`

### Большинство Python-модулей

Почему оставляем:
- project-wide file review показал, что хаоса нет, а сильных кандидатов на правку немного.

Связанные findings:
- `QA-050`

### SQL/ADQL слой

Почему оставляем:
- он уже разложен по этапам и читается очень хорошо.

Связанные findings:
- `QA-079`
- `QA-080`

### Основные живые docs

Почему оставляем:
- это сильный narrative layer проекта.

Связанные findings:
- `QA-067`
- `QA-068`
- `QA-073`

### Notebook set целиком

Почему оставляем:
- набор компактный, роли notebooks различимы, summary notebook актуален.

Связанные findings:
- `QA-075`
- `QA-076`
- `QA-078`

## Что точно не делать следующей волной

1. Не дробить большие, но пока рабочие модули ради эстетики.
2. Не переписывать математическое ядро без нового сильного сигнала.
3. Не устраивать “генеральную уборку” с удалением всех артефактов подряд.
4. Не пытаться довести покрытие до формального максимума без учёта риска и пользы.
5. Не смешивать cleanup, архитектурные правки и методические изменения в один большой рефакторинг.

## Рекомендуемый порядок следующей волны

1. Version-control и canonical artifacts.
2. Самые явные архитектурные сцепки.
3. Тестовые пробелы по production boundary layers.
4. Устаревшие docs и статус historical planning docs.
5. Лёгкий cleanup мусора.
6. Только потом — не срочные улучшения вроде README polish, notebook output cleanup и future refactor snapshot-layer.
