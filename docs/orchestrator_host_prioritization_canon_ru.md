# Канон Host-Prioritization Orchestrator

Дата: 14 марта 2026 года

## 1. Назначение документа

Этот документ фиксирует каноническую постановку orchestrator /
decision-layer для текущей итерации проекта.

Документ нужен, чтобы не смешивать:

- вероятность наличия экзопланетного хоста;
- наблюдательную пригодность объекта для follow-up;
- habitability и другие более поздние научные слои.

Главная цель: определить, какую именно научную величину ранжирует
текущий `final_score`, и из этого вывести единую каноническую формулу
для research- и production-контуров.

## 2. Научная цель текущей итерации

Текущая версия проекта решает задачу:

`host prioritization for exoplanet follow-up`

То есть проект ранжирует объекты по:

- похожести на звёзды-хосты уже известных экзопланет;
- физически интерпретируемым priors;
- practical follow-up feasibility.

Проект сейчас не оптимизируется под:

- habitability;
- "спокойные" и long-term stable системы;
- отдельную приоритизацию каменистых и газовых миров;
- полноценную теорию вероятности существования экзопланет во всей
  популяции Gaia.

Следовательно, `final_score` на текущем этапе должен интерпретироваться
как:

`host-prioritization score`

а не как строгая абсолютная вероятность наличия экзопланеты.

## 3. Канонические входные данные

### 3.1 Ядро распознавания и host-scoring

Эти поля являются основными физическими признаками:

- `teff_gspphot`
- `logg_gspphot`
- `radius_gspphot`

Они используются:

- `router`-слоем;
- `host_model`-слоем.

Именно эта тройка определяет:

- физический класс и стадию;
- host-like similarity / posterior.

### 3.2 Дополнительные научно-операционные признаки

Эти поля не являются ядром распознавания, но влияют на приоритет:

- `mh_gspphot`
- `parallax`
- `parallax_over_error`
- `ruwe`

Их роль:

- astrophysical prior;
- reliability;
- observability / follow-up feasibility.

### 3.3 Вторичные modifiers

Эти поля допускаются как мягкие modifiers:

- `bp_rp`
- `validation_factor`

Важно:

- отдельной raw-колонки `distance` во входной relation сейчас нет;
- `distance` выводится из `parallax` или учитывается через distance-like
  factor;
- `validation_factor` является техническим guard-модификатором, а не
  физической характеристикой звезды.

## 4. Канон по слоям

### 4.1 Router

Вход:

- `teff_gspphot`
- `logg_gspphot`
- `radius_gspphot`

Выход:

- `predicted_spec_class`
- `predicted_evolution_stage`
- `router_label`

Роль:

- определить физически допустимую ветку;
- не оценивать host-priority напрямую.

### 4.2 Host-model

Вход:

- `teff_gspphot`
- `logg_gspphot`
- `radius_gspphot`
- контекст router-ветки

Выход:

- `host_posterior`

Роль:

- быть главным модельным сигналом host-likeness.

### 4.3 Orchestrator / Decision-layer

Роль orchestrator-а:

- не угадывать host заново;
- не заменять router и host-модель;
- мягко вносить astrophysical priors;
- учитывать practical follow-up feasibility;
- превращать `host_posterior` в итоговый ranking score.

## 5. Канонические группы факторов

### 5.1 Главный фактор

- `host_posterior`

Это основной сигнал модели.
Именно он должен задавать базовый ranking.

### 5.2 Astrophysical prior

- `class_prior(predicted_spec_class)`
- `metallicity_factor(mh_gspphot)`

Роль:

- не доказывать наличие планеты;
- мягко сдвигать prior под задачу host-prioritization.

Требование:

- `class_prior` допускается умеренно сильным;
- `metallicity_factor` должен оставаться мягким.

### 5.3 Observability / Follow-up feasibility

- `distance_factor(parallax)`
- `parallax_precision_factor(parallax_over_error)`
- `ruwe_factor(ruwe)`

Роль:

- penalize объекты, которые трудно или ненадёжно проверять дальше;
- не подменять host-likelihood;
- быть отдельным feasibility-слоем.

Канонический вывод:

`distance_factor` должен быть отдельным фактором и не должен теряться
внутри усреднённого quality-блока.

### 5.4 Вторичные modifiers

- `color_factor(bp_rp)`
- `validation_factor`

Роль:

- `color_factor` допускается только как очень мягкий физический
  модификатор и не должен дублировать `class_prior` агрессивно;
- `validation_factor` должен оставаться техническим guard-слоем с
  нейтральным default.

## 6. Каноническая формула

Текущая production-canonical структура score выглядит так:

```text
host_score =
  host_posterior
  × class_prior
  × metallicity_factor

reliability_factor =
  ruwe_factor
  × parallax_precision_factor

followup_factor =
  distance_factor

final_score =
  clip_unit_interval(
    host_score
    × reliability_factor
    × followup_factor
    × color_factor
    × validation_factor
  )
```

Текущая `V1` operational mapping из `final_score` в `priority_tier`:

```text
HIGH   : final_score >= 0.50
MEDIUM : final_score >= 0.30
LOW    : final_score <  0.30
```

Где:

- `color_factor` является вторичным и может быть ослаблен или исключён
  после sensitivity-run;
- `validation_factor` по умолчанию должен быть близок к `1.0`.

Практическая интерпретация current `V1`:

- `HIGH` — рабочий shortlist для follow-up, который в текущей конфигурации
  почти целиком состоит из `K` и верхушки `M`;
- `G`-карлики в `V1` остаются в основном резервным `MEDIUM`-слоем и не
  должны искусственно проталкиваться в `HIGH` только ради “красивой”
  симметрии классов;
- это указывает скорее на консервативную operational настройку
  orchestrator-а, чем на поломку базовой физической логики.

## 7. Что считается каноничным, а что нет

### Канонично

- `host_posterior` как главный модельный сигнал;
- `class_prior` и `metallicity_factor` как astrophysical priors;
- `distance`, `parallax precision`, `ruwe` как observability / quality;
- ranking под задачу exoplanet host follow-up.

### Не входит в канон текущей итерации

- habitability;
- flare activity;
- "спокойность" звезды;
- brightness-based direct detectability как отдельный слой;
- planet-type physics model;
- отдельный priors-layer под gas giants vs rocky planets.

## 8. Текущее расхождение слоёв

Сейчас production и offline calibration не совпадают.

### Production

В [decision.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/priority_pipeline/decision.py)
используется:

```text
host_score =
  host_posterior
  × class_prior
  × metallicity_factor

reliability_factor =
  quality_factor
  = avg(ruwe_factor, parallax_precision_factor)

followup_factor =
  distance_factor(parallax)

final_score =
  clip_unit_interval(
    host_score
    × reliability_factor
    × followup_factor
    × color_factor
    × validation_factor
  )
```

То есть current production уже не прячет `distance_factor` внутри
`quality_factor`: расстояние живёт в отдельном `followup_factor`, а
`quality_factor` остаётся совместимым alias для `reliability_factor`.

### Offline calibration

В [scoring.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/decision_calibration/scoring.py)
используется:

```text
host_posterior
× class_prior
× distance_factor
× quality_factor
× metallicity_factor
```

Причём:

- `quality_factor = ruwe_factor × parallax_precision_factor`
- `color_factor` отсутствует
- `validation_factor` отсутствует

Вывод:

- current production и scientific canon по структуре score уже
  совпадают заметно лучше, чем раньше;
- основное расхождение сейчас осталось между production и offline
  calibration;
- это расхождение нужно рассматривать как текущий semantic drift, а не
  как две равноправные канонические формулы.

## 9. Микро ТЗ На Выравнивание Формулы

### Блок 1. Зафиксировать scientific objective

- использовать термин `host-prioritization score`;
- убрать двусмысленность "вероятность наличия экзопланеты" там, где это
  звучит как абсолютная вероятность;
- зафиксировать, что текущая цель — exoplanet host follow-up.

Definition of done:

- README и docs используют одну и ту же формулировку.

### Блок 2. Развести группы факторов в коде

- выделить отдельно:
  - `host_score`
  - `reliability_factor`
  - `followup_factor`
  - `secondary_modifiers`
- перестать прятать `distance_factor` внутри усреднённого
  `quality_factor`.

Definition of done:

- код decision-layer отражает научную структуру score.

### Блок 3. Синхронизировать production и offline semantics

- выбрать research-canonical формулу;
- привести offline calibrator и production orchestrator к одной и той же
  структуре;
- различия между ними допускаются только как явно задокументированные
  operational simplifications.

Definition of done:

- production и calibration больше не расходятся по базовой формуле.

### Блок 4. Провести orchestrator validation

- проверить baseline behavior;
- проверить class-wise compression / amplification;
- проверить sensitivity к изменению priors и observability factors.

Definition of done:

- понятна устойчивость orchestrator-а и вклад каждого слоя.

### Блок 5. После этого корректировать коэффициенты

- сначала validation;
- потом calibration;
- не наоборот.

Definition of done:

- коэффициенты меняются не по интуиции, а по sensitivity-результатам.

## 10. Требования К Реализации

- `PEP 8`
- `ruff`
- `mypy`
- `pytest`
- без новых runtime dependencies без сильной причины
- без скрытой магии и без смешения research / production слоёв
- с короткими и явными функциями
- с Pythonic разделением ответственности:
  - model evidence
  - scientific priors
  - observability
  - operational mapping

## 11. Статус

Статус документа: `accepted as target canon`

Этот документ считается каноническим ориентиром для следующей волны
работ по orchestrator validation, sensitivity-run и выравниванию
production / calibration formula.
