# Синтез текущей audit-wave

Дата фиксации: 19 марта 2026 года

## 1. Назначение документа

Этот документ подводит промежуточный итог текущей audit-wave.

Важно:

- здесь **нет** плана исправлений;
- здесь **нет** списка рефакторингов “на будущее”;
- здесь только итоговая картина того, что уже подтверждено на уровне:
  - кода;
  - артефактов;
  - notebooks;
  - прогонов;
  - quality-gates.

Подробный реестр observations и evidence находится в:

- [project_audit_findings_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/project_audit_findings_ru.md)

## 2. Краткий итог

На текущем этапе проект **не выглядит сломанным** и не выглядит
“фальшиво рабочим”.

Наоборот, у него уже есть сильное инженерное ядро:

- воспроизводимый preprocessing слой;
- production runtime с реальным разделением `router / OOD / host-model / decision layer`;
- comparison-layer с несколькими baseline-моделями;
- отдельный validation-layer;
- быстрый и зелёный полный test-suite;
- хороший docstring coverage;
- уже собранный summary-layer для защиты.

При этом audit-wave показала, что в проекте накопились **не катастрофы,
а semantic drift и presentation drift**:

- код, docs, snapshot-артефакты и notebooks местами уже говорят немного
  о разном;
- benchmark в целом воспроизводим;
- а вот production/runtime/snapshot narrative уже требует более строгого
  разведения смыслов.

## 3. Главный вывод по рискам

### 3.1 Чего audit не нашёл

В текущей волне **не найдено P0-level признаков**, что:

- математика полностью несостоятельна;
- pipeline в целом не соответствует теме ВКР;
- модели не работают вообще;
- тестовый слой фальшивый;
- результаты нельзя защищать в принципе.

### 3.2 Что audit нашёл

Найдены **12 findings уровня `P1`** и **22 findings уровня `P2`**.

По сути это означает:

- у проекта есть рабочее ядро;
- но есть несколько важных мест, где нельзя больше полагаться на
  “интуитивно вроде совпадает”;
- перед финальной стабилизацией нужно привести в порядок
  interpretation-layer, reproducibility-layer и часть code contracts.

## 4. Самые важные проблемы

Ниже не полный реестр, а именно то, что влияет на защиту, корректность
объяснения и доверие к результатам.

### 4.1 Production semantics уже расходится с частью docs

Это зафиксировано в:

- `A-LOGIC-001`
- `A-DOC-001`

Суть:

- current runtime уже живёт на `reliability_factor` и `followup_factor`;
- часть публичного explanation-layer всё ещё описывает старую формулу с
  `quality_factor`.

Почему это важно:

- это не cosmetic drift;
- это риск неправильно объяснить production ranking.

### 4.2 Persisted result contract отстаёт от runtime

Это зафиксировано в:

- `A-LOGIC-002`

Суть:

- runtime считает раздельные факторы;
- persisted output сохраняет только legacy `quality_factor`.

Почему это важно:

- часть explainability теряется уже на уровне result-table semantics.

### 4.3 Offline calibration не эквивалентен current production

Это зафиксировано в:

- `A-LOGIC-003`

Суть:

- calibration-layer и runtime больше нельзя трактовать как одну и ту же
  формулу, просто в разной среде.

Почему это важно:

- offline quality findings нельзя переносить на production без оговорок.

### 4.4 `UNKNOWN` по spec и `UNKNOWN` в runtime сейчас не одно и то же

Это зафиксировано в:

- `A-LOGIC-004`

Суть:

- structural missing-feature cases не доходят до canonical `UNKNOWN`,
  потому что input-layer фильтрует их раньше.

Почему это важно:

- open-set narrative сейчас уже, чем заявлено в spec.

### 4.5 Current benchmark стабилен, а runtime/snapshot уже не вполне

Это зафиксировано в:

- `A-ML-001`
- `A-ML-002`
- `A-ML-003`
- `A-ML-S002`

Суть:

- supervised benchmark текущей comparison-wave воспроизводится;
- а вот current snapshot и current production-like behavior уже заметно
  расходятся с versioned артефактами марта.

Почему это важно:

- это главный reproducibility-risk проекта на текущем этапе;
- проблема не в benchmark core, а в runtime/snapshot semantics и их
  versioned narrative.

### 4.6 Финальный порядок `K -> M -> G` сейчас частично ручной overlay

Это зафиксировано в:

- `A-PHYS-001`
- `A-NOTEBOOK-001`
- `A-NOTEBOOK-002`

Суть:

- `K` как устойчивый главный класс поддержан runtime хорошо;
- `M` как слой сильных top-кандидатов тоже поддержан;
- `G` как полноценная “третья очередь” поддержан слабее;
- при этом summary-layer дополнительно навешивает `priority_map`.

Почему это важно:

- вывод для ВКР должен быть аккуратным:
  не “данные сами жёстко доказали три очереди”,
  а “в рамках текущей V1 operational shortlist строится так-то”.

### 4.7 Type-gate больше не полностью зелёный

Это зафиксировано в:

- `A-CODE-001`

Суть:

- `ruff` зелёный;
- `pytest` зелёный;
- full `mypy` уже не зелёный.

Почему это важно:

- проект декларирует строгую типизацию как часть качества;
- значит это уже не мелочь, а реальный quality regression.

## 5. Что при этом уже хорошо

Это важно проговорить отдельно, чтобы не создать ложное впечатление,
будто audit-wave нашла только проблемы.

### 5.1 Архитектурное ядро проекта в целом хорошее

Подтверждено:

- `A-LOGIC-S001`
- `A-LOGIC-S002`
- `A-LOGIC-S003`
- `A-LOGIC-S004`

То есть:

- runtime chain реально разделён;
- OOD вынесен в отдельный слой;
- input normalization deterministic;
- production, comparison и heavy validation разведены.

### 5.2 Validation-layer не декоративный

Подтверждено:

- `A-ML-S001`

Это сильная сторона:

- validation реально ловит gap/instability,
  а не просто дублирует benchmark.

### 5.3 Test feedback loop хороший

Подтверждено:

- `A-TEST-S001`
- `A-TEST-S002`

То есть:

- полный suite быстрый;
- основная цена идёт от нескольких реалистичных fit-smoke тестов;
- дерево тестов пока не развалилось по времени.

### 5.4 Public docstring coverage хорошее

Подтверждено:

- `A-DOC-S001`

Это означает:

- проект в целом уже выглядит как maintainable codebase,
  а не как набор “скриптов без слов”.

### 5.5 High-level naming в целом читабельно

Подтверждено:

- `A-NAMING-S001`

То есть:

- на уровне notebooks и `analysis/*` стадии проекта читаются хорошо.

## 6. Что это означает для защиты

На текущем этапе проект **можно защищать как V1**, но только при
условии, что объяснение будет дисциплинированным.

Правильная защитная позиция сейчас такая:

1. `V1` уже даёт рабочий и воспроизводимый pipeline.
2. Главный устойчивый практический вывод — приоритетное внимание к
   `K dwarf`, затем `M dwarf`.
3. `G dwarf` допустимо оставлять как более осторожный третий слой, но
   не переутверждать этот вывод как “жёстко доказанный самим runtime”.
4. `RandomForest` и `MLP` сильны как benchmark baseline-модели.
5. `Contrastive V1` полезен как production ranking core, но его надо
   объяснять именно как retrieval/recall-heavy схему, а не как лучший
   строгий бинарный классификатор.

Чего не стоит делать на защите:

- говорить, что все слои проекта сейчас полностью синхронизированы;
- выдавать comparison snapshot за production run;
- говорить, что `UNKNOWN` уже покрывает все типы open-set случаев;
- говорить, что `K -> M -> G` извлечено безо всякого ручного
  operational overlay.

## 7. Приоритетная карта findings по смыслу

### 7.1 Самые важные для будущего fix-plan

Это зоны, которые потом почти наверняка войдут в план правок:

- docs/runtime semantic drift
- persist contract drift
- production vs calibration divergence
- `UNKNOWN` contract mismatch
- runtime/snapshot reproducibility drift
- current `mypy` regression
- notebook/slide narrative drift

### 7.2 Важные, но не срочные

- крупные modules в comparison-layer
- тяжёлые orchestration tests
- package/file naming drift
- notebook hygiene
- summary notebook portability

### 7.3 Что лучше не трогать без причины

- общий layering `production / comparison / validation`
- deterministic input normalization
- validation-layer как отдельный контур
- test feedback loop как таковой
- high-level notebook ordering and stage naming

## 8. Итог одной фразой

Текущая audit-wave показывает не “плохой проект”, а **хороший V1-проект
с накопившимся semantic drift между runtime, comparison, notebooks и
docs**.

Это значит:

- основу уже можно защищать;
- но перед финальной стабилизацией проекту нужен не новый feature rush,
  а точечное выравнивание смыслов, контрактов и presentation-layer.
