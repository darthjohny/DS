# Quality And OOD Contract V2

## Цель

Этот документ фиксирует ранний quality/OOD layer для новой волны `V2`.

Задача документа:

- не пускать плохие объекты в обычный pipeline молча;
- явно отделить `качество данных` от `астрофизической классификации`;
- заранее определить, где объект считается:
  - пригодным;
  - сомнительным;
  - выходящим за область доверия модели.

## Основной Принцип

`quality_state` и `ood_state` проверяются до основной тонкой классификации и до ranking.

То есть логика такая:

- сначала качество и доверие к объекту;
- потом labels;
- потом host-like и observability;
- потом итоговый priority.

## Обязательные Quality-Сигналы

На первой волне явно учитываем:

- `ruwe`
- `parallax_over_error`
- наличие критически важных признаков
- базовую физическую правдоподобность значений

К критическим признакам относятся минимум:

- `teff`
- `logg`
- `radius`

Если они отсутствуют или явно повреждены, объект не должен идти в обычную классификацию как будто с ним все нормально.

## RUWE

`RUWE` считаем ранним индикатором качества астрометрического решения.

Правило:

- пороговые значения не выдумываем;
- финальное правило для `RUWE` фиксируем по литературе и практике `Gaia`;
- конкретный threshold документируем явно, а не держим в голове.

На уровне архитектуры `RUWE` считается:

- quality feature;
- ранним gate signal;
- частью `observability` только вторично.

## Parallax Over Error

`parallax_over_error` считаем вторым ранним quality-сигналом.

Назначение:

- оценить надежность дистанционной информации;
- не подмешивать сомнительные distance-derived решения в обычный ranking без флага качества.

## Missing Critical Features

Если отсутствуют критические признаки:

- объект не должен автоматически считаться нормальным кандидатом;
- вводим отдельную ветку:
  - `unknown`
  - или `OOD`
  - или жесткий low-priority fallback

Точный путь потом выбираем в реализации, но не маскируем отсутствие данных под валидный normal case.

## Physical Plausibility Checks

До обычной классификации также проверяем:

- что числовые значения лежат в физически правдоподобных диапазонах;
- что в признаках нет грубых противоречий;
- что объект не выглядит как явный артефакт источника.

Это не заменяет полную OOD-логику, но защищает pipeline от грубого мусора.

## Разделение Quality State И OOD State

`quality_state` и `ood_state` не одно и то же.

`quality_state` отвечает на вопрос:

- хороши ли исходные данные и наблюдательные параметры.

`ood_state` отвечает на вопрос:

- находится ли объект в области, где модели вообще имеют право уверенно говорить.

Примеры:

- плохой `RUWE` может ухудшать `quality_state`, но не всегда означает чистый `OOD`;
- экзотический объект с необычным сочетанием признаков может быть `OOD`, даже если quality-поля хорошие.

## Допустимые Исходы Раннего Gate

На первой волне архитектурно допускаем:

- `pass`
- `unknown`
- `ood`
- `reject`

В коде конкретные значения могут называться иначе, но смысл должен остаться таким же.

## Что Не Делаем

- не считаем `RUWE` декоративным дополнительным признаком;
- не проверяем качество слишком поздно, когда объект уже прошел полpipeline;
- не используем `unknown` и `OOD` как синонимы;
- не объясняем плохой ranking тем, что "модель так решила", если объект изначально был низкого качества.

## Связь С Ranking

`quality/OOD gate` идет раньше ranking.

Итоговый priority layer должен видеть:

- объект прошел quality gate;
- объект сомнительный;
- объект `OOD`;
- объект отклонен.

Ranking не должен превращаться в место, где мы молча пытаемся исправить плохое качество данных.

## Связь С БД И Новыми Source

Для новой data engineering-волны должны быть явно доступны поля:

- `ruwe`
- `parallax_over_error`
- критические физические признаки

Если source этого не дает:

- это должно быть видно в audit;
- source не считается полноценным training/reference source без оговорок.

## First-Wave DB Gate Policy

Для первой DB-реализации gate используем только поля, которые уже
зафиксированы в official Gaia DR3 datamodel или в наших отдельных OOD-source.

### Official Gaia Signals, На Которые Опираемся

- `ruwe`
  - quality indicator из `gaia_source`
- `parallax_over_error`
  - отношение параллакса к его ошибке из `gaia_source`
- `non_single_star`
  - флаг наличия дополнительной информации в non-single star tables
- `classprob_dsc_combmod_star`
  - вероятность класса single star по DSC-Combmod
- `teff_gspphot`, `logg_gspphot`, `mh_gspphot`
  - параметры GSP-Phot, которые в Gaia DR3 описаны как оценки при предположении,
    что источник является single star
- `radius_flame`, `lum_flame`, `evolstage_flame`
  - official FLAME fields из `astrophysical_parameters`

### Минимальный Набор Critical Features

Для `pass` в normal refinement-поток на первой волне требуем минимум:

- `teff_gspphot`
- `logg_gspphot`
- `mh_gspphot`
- `bp_rp`
- `parallax`
- `parallax_over_error`
- `ruwe`

`radius_flame` считаем отдельным required refinement-signal:

- его отсутствие не делает строку автоматическим `reject`;
- но переводит ее как минимум в `unknown/review`, а не в чистый `pass`.

### Первая Практическая Threshold Policy

На первой волне фиксируем следующие project thresholds:

- `ruwe > 1.4` -> `unknown`
- `parallax_over_error < 5` -> `unknown`
- missing critical core features -> `reject`
- missing `radius_flame` -> `unknown`
- `non_single_star > 0` -> `candidate_ood`
- `classprob_dsc_combmod_star < 0.5` -> `candidate_ood`
- membership в отдельном OOD-pool -> `ood`

Важная оговорка:

- значения `1.4`, `5` и `0.5` здесь являются project policy первой волны;
- это не hard threshold из official Gaia docs;
- official docs задают смысл полей, а не один обязательный глобальный cut.

### First-Wave Выходы

На первой DB-волне используем:

- `quality_state`:
  - `pass`
  - `unknown`
  - `reject`
- `ood_state`:
  - `in_domain`
  - `candidate_ood`
  - `ood`

### Unknown / Review Table

Отдельный relation `lab.gaia_mk_unknown_review` должен содержать все строки, где:

- `quality_state <> 'pass'`
  или
- `ood_state <> 'in_domain'`

Это нужно, чтобы:

- не терять сомнительные строки;
- не пускать uncertain/OOD объекты в обычный training/scoring;
- иметь отдельный аналитический контур review.

## Критерий Готовности Контракта

Контракт считается зафиксированным, если:

- любой новый source можно проверить на quality/OOD readiness;
- loaders и pipeline понимают, что gate идет раньше классификации;
- ranking не маскирует проблемы качества данных.

## Post-Calibration Decision For First Wave

После первого calibration-study на live relation `lab.gaia_mk_quality_gated`
production policy первой волны оставляем без изменений.

Фактическая картина:

- baseline:
  - `pass`: `178439` (`44.36%`)
  - `unknown`: `63823` (`15.87%`)
  - `reject`: `159964` (`39.77%`)
- relaxed:
  - `pass`: `195815` (`48.68%`)
  - `unknown`: `46447` (`11.55%`)
  - `reject`: `159964`
- strict:
  - `pass`: `162741` (`40.46%`)
  - `unknown`: `79521` (`19.77%`)
  - `reject`: `159964`

Вывод:

- главный driver для `reject` это missing core features;
- смягчение threshold-политики дает ограниченный выигрыш по `pass`;
- `unknown` на первой волне оставляем как отдельный review-pool;
- baseline thresholds сохраняем до отдельной host/priority integration wave.

Это решение соответствует архитектурному принципу:

- не проталкивать сомнительные строки в normal `id` pipeline силой;
- не путать poor-quality objects с обычными in-domain объектами;
- держать `unknown/review` как отдельный аналитический контур.
