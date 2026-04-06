# ТЗ На Донастройку Перед Боевым Прогоном

Дата фиксации: `2026-04-05`

Связанные документы:

- [post_run_stabilization_tz_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/plans/post_run_stabilization_tz_ru.md)
- [quality_gate_host_priority_tz_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/plans/quality_gate_host_priority_tz_ru.md)
- [priority_threshold_calibration_tz_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/plans/priority_threshold_calibration_tz_ru.md)

Внутренний инженерный стандарт проекта ведется вне публичного контура
репозитория.

## Зачем Нужен Этот Пакет

Свежий диагностический `decide` run показал, что базовый контур уже рабочий,
но перед следующим полноценным боевым прогоном нужно спокойно довести два
места:

- `quality_gate`;
- насыщение `priority`.

Сейчас нет смысла углубляться в интерпретацию итоговых объектов, пока эти два
слоя не приведены в устойчивое и понятное состояние.

## Текущее Диагностическое Состояние

Диагностический run:

- [hierarchical_final_decision_2026_04_05_090717_885503](/Users/evgeniikuznetsov/Desktop/dspro-vkr/artifacts/decisions/hierarchical_final_decision_2026_04_05_090717_885503)

Что он показал:

- `unknown = 223787` (`55.64%`);
- `id = 177674` (`44.17%`);
- `ood = 765` (`0.19%`);
- `quality_reject = 159964`;
- `quality_unknown = 63823`;
- `high priority = 100173` (`56.38%` от priority-пула).

Это означает:

- основной рабочий контур собран корректно;
- главная зона донастройки сейчас не в `coarse` и не в `host`-модели;
- главные operational вопросы сидят в:
  - политике quality-gate;
  - порогах и насыщении priority-слоя.

## Цель

До следующего боевого прогона:

- доказательно решить, оставляем ли текущий `quality_gate` без изменений или
  вносим точечную policy-правку;
- доказательно решить, достаточно ли ужать `priority` thresholds или нужен
  отдельный scaling-layer;
- ничего не менять вслепую в моделях;
- держать код и исследования в рамках инженерного стандарта.

## Инженерный Инвариант

Для этого пакета сохраняются те же правила:

- `1 файл = 1 ответственность`;
- без монолитов и без случайных фасадов;
- `PEP 8`;
- явная типизация;
- сначала сверяемся с официальной документацией;
- не лечим симптомы без подтвержденной причины;
- каждую policy-правку сначала проверяем на review-слое;
- после каждого небольшого шага:
  - `ruff`;
  - точечный `mypy/pyright`;
  - targeted `pytest`;
- только после этого двигаемся дальше.

## Official Опора

### Scikit-Learn

- [Probability calibration](https://scikit-learn.org/stable/modules/calibration.html)
- [Tuning the decision threshold for class prediction](https://scikit-learn.org/stable/modules/classification_threshold.html)
- [precision_recall_curve](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html)

### Gaia DR3

- [Gaia DR3 gaia_source datamodel](https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/chap_datamodel/sec_dm_main_source_catalogue/ssec_dm_gaia_source.html)
- [Gaia DR3 astrophysical_parameters datamodel](https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/chap_datamodel/sec_dm_astrophysical_parameter_tables/ssec_dm_astrophysical_parameters.html)
- [Gaia DR3 astrometric validation](https://gea.esac.esa.int/archive/documentation/GDR3/Catalogue_consolidation/chap_cu9val/sec_cu9val_942/ssec_cu9val_942_astrometry.html)

### Python / pandas

- [typing](https://docs.python.org/3/library/typing.html)
- [collections.abc](https://docs.python.org/3/library/collections.abc.html)
- [pandas missing data](https://pandas.pydata.org/docs/user_guide/missing_data.html)

## Что Принципиально Не Делаем В Этом Пакете

Не делаем до подтверждения причины:

- ретрейн `coarse`;
- ретрейн `host`;
- новые внешние кроссматчи;
- ручные астрофизические override-правила в mainline;
- подмену scientific uncertainty жесткими эвристиками.

## Рабочая Гипотеза По `quality_gate`

На текущем диагностическом run:

- большая доля `unknown/reject` все еще driven gate-слоем;
- главный вклад в `reject` дает `missing_core_features`;
- review buckets уже выглядят объяснимо:
  - `review_high_ruwe`
  - `review_missing_radius_flame`
  - `review_non_single_star`
  - `review_low_single_star_probability`
  - `review_low_parallax_snr`

Поэтому задача не “ослабить gate любой ценой”, а понять:

- где у нас честная осторожность;
- а где слишком жесткая project-policy.

Повторная проверка на свежем диагностическом baseline показала, что
`quality_gate` не drift-нул относительно старого stable baseline:

- distributions по `quality_state`;
- `final_decision_reason`;
- `review_bucket`

совпадают полностью.

Связанный review:

- [quality_gate_refresh_review_2026_04_05_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/run_reviews/quality_gate_refresh_review_2026_04_05_ru.md)
- [quality_gate_variant_review_2026_04_05_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/run_reviews/quality_gate_variant_review_2026_04_05_ru.md)
- [pre_battle_policy_candidate_run_2026_04_05_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/run_reviews/pre_battle_policy_candidate_run_2026_04_05_ru.md)

## Рабочая Гипотеза По `priority`

На текущем диагностическом run:

- `priority_score p50 = 0.806156`;
- `host_similarity_score p50 = 0.956772`;
- `high`-зона слишком широкая;
- исторический threshold-candidate `0.85 / 0.55` уже выглядел лучше.

Поэтому первый правильный ход:

- не переписывать ranking-формулу;
- сначала повторно проверить threshold-policy на текущем live baseline;
- только потом решать, нужен ли scaling-layer.

## Пакеты Работ

### Блок QG. Донастройка `quality_gate`

#### QG-T01. Зафиксировать актуальный baseline gate review

Нужно:

- привязать review к свежему диагностическому run;
- повторно снять distributions и review buckets;
- убедиться, что baseline для сравнения один и тот же в коде, docs и notebook.

#### QG-T02. Разделить rule families на три группы

Нужно:

- hard reject;
- review;
- informational only.

Цель:

- перестать смешивать “объект плохой” и “объект спорный”.

#### QG-T03. Собрать компактный variant review

Нужно:

- current policy;
- более мягкий вариант только для review-правил;
- более строгий вариант для чувствительных сигналов;
- сравнение coverage и причин переходов.

Важно:

- не менять сразу production код;
- сначала review-слой и таблица переходов.

Статус:

- закрыто review-слоем на диагностическом baseline:
  - [quality_gate_variant_review_2026_04_05_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/run_reviews/quality_gate_variant_review_2026_04_05_ru.md)

Главный вывод:

- review-variants двигают только `unknown <-> pass`;
- `reject_missing_core_features` не затрагивается вообще;
- следующее решение должно обсуждать только review-правила:
  - `high_ruwe`
  - `low_parallax_snr`
  - `missing_radius_flame`

#### QG-T04. Принять policy decision

Возможные исходы:

- baseline сохраняем;
- часть review-правил ослабляем;
- часть сигналов переводим из reject в review.

Критерий:

- решение должно объясняться данными, а не желанием “сделать pass побольше”.

Решение на текущей итерации:

- `RUWE` сохраняем на `1.4`;
- `parallax_over_error` сохраняем на `5.0`;
- для следующего active run убираем только требование `radius_flame` для
  `pass`.

### Блок PR. Донастройка `priority`

#### PR-T01. Зафиксировать baseline по насыщению

Нужно:

- снять текущее распределение `high / medium / low`;
- снять распределение по `spec_class`;
- снять квантильный профиль компонент score.

#### PR-T02. Повторно проверить threshold-only strategy

Нужно:

- current thresholds;
- historical tighter thresholds `0.85 / 0.55`;
- возможно один промежуточный вариант.

Цель:

- подтвердить, что насыщение решается порогами, а не формулой score.

Решение на текущей итерации:

- candidate thresholds для следующего run:
  - `high_min = 0.85`
  - `medium_min = 0.55`

Это дает:

- более узкую `high`-зону;
- содержательный `medium` bucket;
- без переписывания ranking-формулы.

#### PR-T03. Открывать scaling только если threshold review не помогает

Scaling-пакет открываем только если:

- tighter thresholds не дают читаемой верхней зоны;
- либо высокая зона все еще operationally слишком широкая.

#### PR-T04. Принять final priority policy

Возможные исходы:

- оставить только tighter thresholds;
- ввести отдельный scaling-layer;
- оставить score как есть, но менять label-policy.

## Финальный Шаг

После закрытия `QG` и `PR`:

- делаем новый боевой прогон;
- перепривязываем technical notebook к новому run;
- только после этого переходим к интерпретации системы в целом.

Candidate run с выбранной policy уже зафиксирован в:

- [pre_battle_policy_candidate_run_2026_04_05_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/run_reviews/pre_battle_policy_candidate_run_2026_04_05_ru.md)

После перепривязки technical notebook и повторной сверки этот run принят как
новый active baseline технического слоя.

## Критерий Готовности

Пакет считается закрытым, когда:

- для `quality_gate` есть формально принятая policy;
- для `priority` есть формально принятая policy;
- новый боевой run проходит на обновленных настройках;
- технические notebook показывают уже новый baseline;
- мы можем объяснить:
  - почему объект уходит в `unknown`;
  - почему объект получает `high / medium / low`;
  - и где заканчивается уверенное решение, а где начинается review-зона.
