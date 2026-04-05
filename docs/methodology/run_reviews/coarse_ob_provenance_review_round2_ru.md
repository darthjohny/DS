# Coarse O/B Provenance Review Round 2

## Цель

Проверить, что происходит с downstream `O/B`-пулом после фикса parser-policy
для ambiguous `OB...` labels и после проталкивания этого фикса в downstream
relations.

## Что Было Сделано

1. Parser-policy для ambiguous hot-boundary labels изменена так, что:
   - `OB...`
   - `O/B...`
   - `O9.5/B0...`
   materialize-ятся как:
   - `spectral_class = 'OB'`
   - `label_parse_status = 'partial'`
   - `label_parse_notes = 'ambiguous_ob_boundary_label'`
2. Выполнен downstream sync:
   - `lab.gaia_mk_training_reference`
   - `lab.gaia_mk_quality_gated`
   - `lab.gaia_mk_unknown_review`
3. Пересобран local provenance audit слой:
   - `lab.gaia_ob_hot_provenance_audit_source`
   - `lab.gaia_ob_hot_provenance_audit_summary`
   - `lab.gaia_ob_hot_provenance_crosswalk_summary`

## Источники

- parser-fixed labeled source:
  - `lab.gaia_mk_external_labeled`
- synced downstream relations:
  - `lab.gaia_mk_training_reference`
  - `lab.gaia_mk_quality_gated`
  - `lab.gaia_mk_unknown_review`
- refreshed provenance audit:
  - `lab.gaia_ob_hot_provenance_audit_source`
  - `lab.gaia_ob_hot_provenance_audit_summary`
  - `lab.gaia_ob_hot_provenance_crosswalk_summary`
- Gaia hot-star result:
  - `public.gaia_ob_hot_provenance_audit_clean`

## Ключевые Результаты

### 1. Ambiguous `OB...` больше не сидят в downstream `O`

После sync:

- `lab.gaia_mk_training_reference`
  - `spectral_class='O'`: `1207`
  - `spectral_class='OB'`: `5165`
  - `ambiguous OB -> O`: `0`
- `lab.gaia_mk_quality_gated`
  - `spectral_class='O'`: `1207`
  - `spectral_class='OB'`: `5165`
  - `ambiguous OB -> O`: `0`
- `lab.gaia_mk_unknown_review`
  - `spectral_class='O'`: `1139`
  - `spectral_class='OB'`: `3684`
  - `ambiguous OB -> O`: `0`

То есть parser-fix реально дошел до downstream слоя, а не остался только в
`lab.gaia_mk_external_labeled`.

### 2. Широкий problematic `O`-пул почти целиком превратился в explicit `OB` boundary

На refreshed hot provenance source:

- `B`: `7112`
- `OB`: `1163`
- `O`: `25`

До sync-а downstream problematic `O`-пул был `1188` строк. После sync-а он
схлопнулся до `25` строк, а почти весь спорный хвост переехал в явный
`OB boundary` pool.

### 3. Новый `OB` boundary pool по Gaia hot-star semantics все еще почти целиком B-like

На refreshed crosswalk:

- `OB -> B`: `1115 / 1163 = 95.87%`
- `OB -> O`: `16 / 1163 = 1.38%`

Median:

- `OB median teff_esphs ≈ 20000.95 K`

Это хорошо согласуется с предыдущим выводом: широкий problematic downstream
пул не был clean `O`, а был hot boundary / B-like population.

### 4. Оставшийся explicit `O`-хвост уже не выглядит как полный provenance-failure

На refreshed crosswalk:

- `O -> O`: `11 / 25 = 44.00%`
- `O -> B`: `12 / 25 = 48.00%`
- `O -> U`: `2 / 25 = 8.00%`

Top raw labels в оставшемся downstream `O`:

- `O9V`: `92`
- `O8V`: `65`
- `O9.5V`: `65`
- `O9.5III`: `34`
- `O8.5V`: `26`

То есть оставшийся downstream `O` уже состоит в основном из explicit `O`-like
labels, а не из `OB...`.

## Интерпретация

Этот раунд сильно подтверждает root cause:

- основной mass-issue сидел не в coarse plumbing и не в train split;
- основной mass-issue сидел в parser/provenance semantics для `OB...`;
- после фикса этой semantics широкий ложный `O`-хвост исчез как класс.

Новый статус проблемы:

- broad issue `O -> B` для ambiguous `OB...` по сути подтвержден и локализован;
- остается уже узкий scientific вопрос для explicit `O`-tail:
  - являются ли эти `25` строк реально boundary-case;
  - или для них still нужен отдельный source-alignment / model-side review.

## Практический Вывод

Blind retrain coarse-модели на текущем этапе по-прежнему не выглядит первым
шагом.

Сначала честнее:

1. считать `OB` отдельным hot-boundary pool, а не чистым `O`;
2. проверить, как эту новую semantics лучше встроить в downstream routing;
3. только после этого решать, нужен ли narrow retrain для explicit `O`.
