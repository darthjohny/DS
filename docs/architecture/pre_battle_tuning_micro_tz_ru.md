# Микро-ТЗ На Донастройку Перед Боевым Прогоном

Дата фиксации: `2026-04-05`

Связанные документы:

- [pre_battle_tuning_tz_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/plans/pre_battle_tuning_tz_ru.md)
- [quality_gate_host_priority_tz_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/plans/quality_gate_host_priority_tz_ru.md)
- [priority_threshold_calibration_tz_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/plans/priority_threshold_calibration_tz_ru.md)
- [coding_standard_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/architecture/coding_standard_ru.md)

## Общий Подход

Работаем в два больших контура:

1. `quality_gate`;
2. `priority`.

На каждом шаге:

- сначала воспроизводим текущее состояние;
- сверяемся с официальной документацией;
- правим только подтвержденную причину;
- после каждого шага делаем micro-QA.

## Шаги

### MTZ-TUNE01. Зафиксировать диагностический baseline

- Цель: не спорить, какой run считаем точкой отсчета.
- Что делаем:
  - фиксируем `hierarchical_final_decision_2026_04_05_090717_885503`;
  - фиксируем связанные benchmark/model artifacts;
  - фиксируем текущие distributions по `quality_gate` и `priority`.
- Статус:
  - закрыто.

### MTZ-TUNE02. Перепроверить `quality_gate` на свежем baseline

- Цель: убедиться, что старые выводы по gate совпадают с текущим run.
- Что делаем:
  - снимаем `quality_state`;
  - снимаем `quality_reason`;
  - снимаем `review_bucket`;
  - проверяем переходы в `unknown / reject / pass`.
- QA:
  - `ruff`
  - `mypy/pyright`
  - targeted `pytest`
- Статус:
  - закрыто.

### MTZ-TUNE03. Разделить правила gate по ролям

- Цель: не смешивать hard reject и review.
- Что делаем:
  - помечаем правила как:
    - `reject`
    - `review`
    - `info`
  - фиксируем это в contracts/docs.
- Важно:
  - пока не меняем боевую policy.
- Статус:
  - закрыто.

### MTZ-TUNE04. Собрать review для candidate gate policies

- Цель: сравнить несколько аккуратных вариантов без правки mainline.
- Что делаем:
  - baseline;
  - мягкий вариант для review-правил;
  - строгий вариант;
  - таблицу переходов между состояниями.
- QA:
  - helper-слой
  - notebook/review smoke
- Статус:
  - закрыто.

### MTZ-TUNE05. Принять решение по `quality_gate`

- Цель: выбрать, что реально идет в следующий боевой run.
- Возможные исходы:
  - baseline без изменений;
  - перевод части правил в review;
  - ослабление отдельных project thresholds.
- Критерий:
  - решение объясняется данными и docs.
- Статус:
  - закрыто.
- Решение:
  - `RUWE` оставляем `1.4`;
  - `parallax_over_error` оставляем `5.0`;
  - для следующего active run снимаем требование `radius_flame` для `pass`.

### MTZ-TUNE06. Перепроверить насыщение `priority` на свежем baseline

- Цель: убедиться, что saturation не изменился после нового диагностического run.
- Что делаем:
  - снимаем `high / medium / low`;
  - снимаем class-level impact;
  - снимаем квантильный профиль компонент.
- QA:
  - `ruff`
  - `mypy/pyright`
  - targeted `pytest`
- Статус:
  - закрыто.

### MTZ-TUNE07. Сравнить threshold variants для `priority`

- Цель: сначала доказать или опровергнуть threshold-only fix.
- Что делаем:
  - current thresholds;
  - `0.85 / 0.55`;
  - при необходимости один промежуточный вариант.
- Что смотрим:
  - долю `high`;
  - переходы `high -> medium`;
  - class-level impact;
  - top-candidate readability.
- Статус:
  - закрыто.
- Решение:
  - candidate threshold policy для следующего run:
    - `high_min = 0.85`
    - `medium_min = 0.55`

### MTZ-TUNE08. Решить, нужен ли scaling-layer

- Цель: не открывать лишний пакет без необходимости.
- Scaling нужен только если:
  - thresholds не решают saturation;
  - top-zone все еще слишком широкая;
  - explainability ухудшается.

### MTZ-TUNE09. Внести выбранные policy-изменения в код

- Цель: применить только подтвержденные решения.
- Что делаем:
  - точечно меняем policy-конфиг;
  - не переписываем модели;
  - не трогаем unrelated слои.
- QA:
  - `ruff`
  - `mypy`
  - `pyright`
  - targeted `pytest`

### MTZ-TUNE10. Сделать новый боевой прогон

- Цель: проверить, как система работает уже на обновленных настройках.
- Что делаем:
  - запускаем `decide`;
  - снимаем distributions;
  - проверяем артефакты;
  - сравниваем новый run с диагностическим baseline.
- Статус:
  - закрыт candidate run:
    - [pre_battle_policy_candidate_run_2026_04_05_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/run_reviews/pre_battle_policy_candidate_run_2026_04_05_ru.md)

### MTZ-TUNE11. Перепривязать и проверить technical notebook

- Цель: чтобы весь технический слой смотрел на новый baseline.
- Что делаем:
  - обновляем run dir там, где он фиксированный;
  - прогоняем active technical notebook;
  - проверяем русский вывод и читаемость.
- Статус:
  - закрыто.
- Результат:
  - `final_decision_review.ipynb` перепривязан на
    `hierarchical_final_decision_2026_04_05_123111_055017`.

### MTZ-TUNE12. Финальный post-run обзор

- Цель: только после донастройки переходить к интерпретации.
- Что делаем:
  - коротко фиксируем:
    - что изменили;
    - что улучшилось;
    - что осталось ограничением;
  - обновляем docs baseline run registry.
- Статус:
  - закрыто.
- Результат:
  - candidate run принят как новый active baseline;
  - baseline registry обновлен;
  - дальше можно переходить к содержательной интерпретации.

## Порядок Выполнения

1. `MTZ-TUNE01`
2. `MTZ-TUNE02`
3. `MTZ-TUNE03`
4. `MTZ-TUNE04`
5. `MTZ-TUNE05`
6. `MTZ-TUNE06`
7. `MTZ-TUNE07`
8. `MTZ-TUNE08`
9. `MTZ-TUNE09`
10. `MTZ-TUNE10`
11. `MTZ-TUNE11`
12. `MTZ-TUNE12`

## Критерий Готовности

- `quality_gate` донастроен или осознанно оставлен без изменений;
- `priority` донастроен или осознанно оставлен на threshold-only policy;
- новый боевой run проходит;
- notebook и docs знают новый baseline;
- дальше можно переходить к интерпретации системы уже без долга по настройке.
