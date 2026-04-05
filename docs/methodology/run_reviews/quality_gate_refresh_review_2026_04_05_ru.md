# Повторная Проверка `quality_gate` На Свежем Baseline

Дата фиксации: `2026-04-05`

Сравниваем:

- старый stable baseline:
  - [hierarchical_final_decision_2026_03_29_075935_878508](/Users/evgeniikuznetsov/Desktop/dspro-vkr/artifacts/decisions/hierarchical_final_decision_2026_03_29_075935_878508)
- новый диагностический baseline:
  - [hierarchical_final_decision_2026_04_05_090717_885503](/Users/evgeniikuznetsov/Desktop/dspro-vkr/artifacts/decisions/hierarchical_final_decision_2026_04_05_090717_885503)

## Зачем Эта Проверка Нужна

Перед policy-донастройкой `quality_gate` нужно было подтвердить, что новый
диагностический run не внес скрытого drift в сам gate-слой.

Иначе мы бы сравнивали candidate policies уже на другом baseline.

## Результат

Поведение `quality_gate` на двух run совпадает полностью.

### Quality-State

На обоих run:

- `pass = 178439` (`44.36%`)
- `reject = 159964` (`39.77%`)
- `unknown = 63823` (`15.87%`)

### Причины Финального Решения

На обоих run:

- `refinement_accepted = 177674`
- `quality_reject = 159964`
- `quality_unknown = 63823`
- `hard_ood = 765`

### Review Buckets

На обоих run:

- `pass = 166847`
- `reject_missing_core_features = 159873`
- `review_high_ruwe = 28388`
- `review_missing_radius_flame = 16762`
- `review_non_single_star = 14402`
- `review_low_single_star_probability = 10474`
- `review_low_parallax_snr = 5292`

## Вывод

Это означает:

- `quality_gate` между этими run не drift-нул;
- старые выводы по gate-layer остаются актуальными;
- пакет донастройки можно строить на новом диагностическом baseline без
  пересмотра старых calibration findings;
- главный текущий вопрос остается тем же:
  - baseline policy сохраняем;
  - или аккуратно переразмечаем часть review-правил.

## Следующий Шаг

- переход к variant review для `quality_gate` поверх диагностического baseline.
