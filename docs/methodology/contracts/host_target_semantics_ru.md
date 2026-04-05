# Host Target Semantics

## Цель

Этот документ фиксирует, что именно должен предсказывать host-layer в новой
волне проекта и чего он предсказывать не должен.

Задача документа:

- убрать двусмысленность вокруг формулировки "вероятность наличия планеты";
- явно разделить:
  - source semantics из NASA Exoplanet Archive;
  - наш project target;
- не позволить `priority`-слою опираться на нестрогую интерпретацию host-score.

## Official Опора

- [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)
- [TAP User Guide](https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html)
- [About the PSCompPars Table](https://exoplanetarchive.ipac.caltech.edu/docs/pscp_about.html)
- [PS and PSCompPars Column Definitions](https://exoplanetarchive.ipac.caltech.edu/docs/API_PS_columns.html)
- [Stellar Hosts Column Definitions](https://exoplanetarchive.ipac.caltech.edu/docs/API_STELLARHOSTS_columns.html)
- [The Gaia-Kepler-TESS-Host Stellar Properties Catalog](https://arxiv.org/abs/2301.11338)

## Что Важно По Official NASA Semantics

### `ps`

`ps` хранит planetary-system solutions так, что каждая строка представляет
один planet+stellar+system solution для одной ссылки.

Это удобно как label-anchor для confirmed planets, потому что:

- таблица хранит подтвержденные planetary solutions;
- строка self-consistent на уровне одной reference solution;
- для одной звезды и одной планеты может быть несколько reference rows.

### `pscomppars`

`pscomppars` делает одну более полную строку на планету, но official docs
прямо предупреждают, что она может быть:

- составлена из нескольких references;
- не полностью self-consistent для одной системы.

Вывод для проекта:

- `pscomppars` годится для enrichment и демографического обзора;
- `pscomppars` не должен быть единственным source of truth для позитивного label.

### `stellarhosts`

Official docs для `stellarhosts` прямо указывают, что таблица включает:

- confirmed planet-hosting stars;
- stellar parameters, не привязанные к конкретному planetary solution;
- gravitationally bound non-planet-hosting stars в planetary systems.

Вывод для проекта:

- `stellarhosts` нельзя механически считать чистой таблицей positive host labels;
- она подходит как stellar-enrichment / alias / identifier source;
- positive label должен быть привязан к actual confirmed host anchor, а не ко всем
  строкам `stellarhosts`.

## Project Decision: Что Предсказывает Host-Layer

На новой волне host-layer предсказывает не
"вероятность существования планеты вообще".

Он предсказывает:

- `host-likeness`
  или
- `confirmed-host prior`

относительно нашей clean confirmed-host population и matched field population.

То есть смысл host-score такой:

- насколько объект похож на звезды из нашей confirmed-host training population;
- а не какова истинная астрофизическая вероятность того, что вокруг звезды
  есть планета вообще.

## Что Host-Layer Не Должен Обещать

Host-layer не должен выдавать это как:

- глобальную вероятность существования любой планеты;
- частоту встречаемости планет в популяции;
- вероятность обнаружения планеты конкретной survey-program;
- вероятность наличия именно каменистой планеты;
- финальный follow-up priority без учета class/quality/observability.

Все эти интерпретации слишком сильные для текущей задачи и источников.

## Clean Positive Label Semantics

Positive label первой clean host-wave должен означать:

- звезда относится к confirmed-host population,
  якоренной на confirmed planet records из NASA Exoplanet Archive.

Практический принцип:

- anchor-позитивы берем из confirmed-planet tables (`ps`-семантика);
- `stellarhosts` используем как дополнительный stellar source;
- `pscomppars` используем как enrichment/descriptive source, а не как primary label source.

## Clean Negative Label Semantics

Negative label первой clean host-wave не означает:

- доказанное отсутствие планет.

Он означает:

- matched field population, которая в текущем host source не отмечена как confirmed-host anchor.

То есть `field` здесь:

- operational negative class для classification task;
- а не физическое утверждение "у этой звезды точно нет планет".

## Связь С Текущим Кодом

Текущая first-wave задача в коде называется:

- `host_field_classification`

И по смыслу это уже близко к правильной постановке:

- positive = `host`
- negative = `field`

Но во всех docs, notebooks и downstream explainability дальше надо
использовать более аккуратную интерпретацию:

- `host_similarity_score`
- `confirmed-host prior`

а не "вероятность наличия планеты".

## Как Это Должно Выглядеть В Выводе

Корректные формулировки:

- "объект имеет высокий host-like score"
- "объект близок к confirmed-host population"
- "confirmed-host prior высокий/средний/низкий"

Некорректные формулировки:

- "вероятность того, что у звезды есть планета, равна X"
- "модель доказала, что звезда не имеет планет"

## Связь С Priority

`priority`-слой не должен напрямую трактовать host-score как конечную вероятность.

Правильная схема:

- `class priority`
- `host_similarity_score`
- `observability`

объединяются в explainable final priority.

То есть host-layer является одним из сигналов в priority integration,
а не самодостаточным итоговым ответом.

## First-Wave Decision

Для следующего implementation-пакета фиксируем:

- target semantics = `confirmed-host prior / host-likeness`;
- positive label source не берем "как есть" из `stellarhosts`;
- `pscomppars` не используем как primary positive truth;
- notebooks и final explainability больше не должны называть host-score
  "вероятностью наличия планеты".

## Критерий Готовности

Документ считается зафиксированным, если:

- известно, что именно предсказывает host-layer;
- известно, какие формулировки запрещены;
- дальше можно проектировать clean feature contract без semantic drift.
