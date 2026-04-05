# Coarse O/B Boundary Policy

## Цель

После parser-fix для ambiguous `OB...` labels не смешивать:

- `secure O-like` hot stars
- и широкий `O/B boundary` pool

в один flat local `O` класс.

## Почему Это Нужно

Round 2 provenance review показал:

- downstream wide problematic `O` pool почти целиком состоял из ambiguous
  `OB...` labels;
- после sync этот пул распался на:
  - `secure explicit O-like` хвост
  - `OB boundary` pool

Текущие live числа:

- `B = 7112`
- `OB = 1163`
- `O = 25`

То есть теперь broad issue уже не “модель не умеет O”, а “boundary semantics
для hot O/B pool надо держать отдельно”.

## Official Опора

Опираемся на official Gaia DR3 semantics:

- `spectraltype_esphs`
- `flags_esphs`

из `gaiadr3.astrophysical_parameters`.

Ключевая идея:

- `ESP-HS` — Gaia hot-star модуль для `O/B/A`;
- `spectraltype_esphs` дает Gaia-side hot-star spectral tag;
- `flags_esphs = 999` не считаем reliable secure signal.

Источники:

- [Gaia DR3 Apsis overview](https://gea.esac.esa.int/archive/documentation/GDR3/Data_analysis/chap_cu8par/sec_cu8par_intro/ssec_cu8par_intro_apsis.html)
- [Gaia DR3 astrophysical_parameters datamodel](https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/chap_datamodel/sec_dm_astrophysical_parameter_tables/ssec_dm_astrophysical_parameters.html)

## Policy V1

### Secure O-like

Источник:

- `public.gaia_ob_hot_provenance_audit_clean`

Критерий:

- `spectral_class IN ('O', 'OB')`
- `esphs_class_letter = 'O'`
- `flags_esphs != 999`

Почему без hard `teff_esphs` cut:

- live review показал, что `teff_esphs >= 25000` сожмет subset до `6` строк;
- это уже слишком агрессивно для first operational policy;
- `spectraltype_esphs='O'` itself already работает как primary hot-star
  classifier signal.

### O/B Boundary

Источник:

- тот же `public.gaia_ob_hot_provenance_audit_clean`

Критерий:

- `spectral_class IN ('O', 'OB')`
- и строка не попадает в `secure O-like`

Практически сюда попадают:

- все ambiguous `OB` labels;
- explicit local `O`, которые Gaia `ESP-HS` не подтверждает как secure `O`.

## Materialized Relations

Working relations в `lab`:

- `lab.gaia_ob_secure_o_like_subset`
- `lab.gaia_ob_boundary_subset`
- `lab.gaia_ob_boundary_policy_summary`
- `lab.gaia_ob_boundary_review`
- `lab.gaia_ob_boundary_review_summary`

Это именно working policy layer, а не final public source asset.

На текущем этапе `lab.gaia_ob_boundary_review` трактуется как explicit review-pool:

- не форсим automatic `O/B` split;
- не считаем этот пул clean truth для automatic model evaluation;
- не пытаемся “угадать” `O` vs `B` без spectroscopy-grade external support.
- operationally выносим этот пул в отдельный review-контур до появления
  stronger external spectroscopy support.

## Live Baseline Counts

После materialization `lab.gaia_ob_secure_o_like_subset` и
`lab.gaia_ob_boundary_subset`:

- `secure O-like`: `25`
- `O/B boundary`: `1163`

Breakdown:

- local `O`: `11 secure`, `14 boundary`
- local `OB`: `14 secure`, `1149 boundary`

Materialized summary:

- `secure_o_like / O`
  - `n_rows = 11`
  - `median_teff_esphs ≈ 33647.50 K`
- `secure_o_like / OB`
  - `n_rows = 14`
  - `median_teff_esphs ≈ 30657.71 K`
- `ob_boundary / O`
  - `n_rows = 14`
  - `median_teff_esphs ≈ 16496.00 K`
- `ob_boundary / OB`
  - `n_rows = 1149`
  - `median_teff_esphs ≈ 20000.25 K`

Boundary reasons:

- `ambiguous_ob_label`: `1149`
- `explicit_o_not_confirmed_by_esphs`: `14`

После отдельной materialization `lab.gaia_ob_boundary_review`:

- `review` rows: `1163`
- breakdown:
  - local `O = 14`
  - local `OB = 1149`

Review summary:

- `O / explicit_o_not_confirmed_by_esphs / B`
  - `n_rows = 12`
  - `median_teff_esphs ≈ 16496.00 K`
- `O / explicit_o_not_confirmed_by_esphs / U`
  - `n_rows = 2`
- `OB / ambiguous_ob_label / B`
  - `n_rows = 1115`
  - `median_teff_esphs ≈ 20000.25 K`

Это подтверждает, что boundary-пул на текущем этапе лучше трактовать как
review, а не как noisy training truth для forced `O/B` split.

Проверка на более жесткий temperature cut:

- внутри уже materialized `secure O-like` only `6` строк имеют
  `teff_esphs >= 25000`

Это подтверждает, что для first policy `teff_esphs` лучше оставлять review
signal, а не превращать в основной hard gate.

## Что Дальше

Current first-wave decision уже принят:

1. `OB boundary` остается в explicit review-pool
2. automatic `O/B` split для boundary-пула не делаем
3. `secure O-like` используем только как узкий hot-tail для дальнейшей
   scientific проверки

Следующий шаг уже не в parser и не в broad `OB` policy:

1. отдельно проверить explicit `O` tail на current coarse behaviour
2. решить, нужен ли для этого tail отдельный retrain / weighting / external
   spectroscopy alignment
