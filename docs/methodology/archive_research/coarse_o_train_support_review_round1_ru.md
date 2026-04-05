# Coarse O Train Support Review Round 1

## Контекст

Этот review закрывает следующий вопрос после
[coarse_ob_boundary_review_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/archive_research/coarse_ob_boundary_review_round1_ru.md):

- есть ли у coarse-модели вообще достаточная train-time поддержка для класса `O`;
- или проблема `O -> B` возникает уже при нормальном support.

Срез повторяет текущий benchmark policy:

- source: `lab.v_gaia_id_coarse_training`
- prepared frame: `prepare_gaia_id_coarse_training_frame(...)`
- split:
  - `test_size = 0.30`
  - `random_state = 42`
  - `stratify_columns = ('spec_class', 'evolution_stage')`

## Главный Результат

Train-time support для класса `O` в coarse source достаточный.

Проблема `O -> B` не объясняется тем, что:

- `O` мало в source;
- `O` почти не попадает в train/test;
- hottest `O` tail теряется на split;
- или benchmark split не совпадает с reconstructed review split.

Иными словами:

- support issue здесь не подтвердился;
- проблема сужается до model-side boundary issue `O vs B`.

## Что Показал Review

### 1. Reconstructed Split Совпадает С Benchmark Один В Один

- `full`: `32986`
- `train`: `23090`
- `test`: `9896`

Все три числа совпали с artifact:

- `artifacts/benchmarks/gaia_id_coarse_classification_2026_03_28_171258_103400`

Значит дальше мы анализируем именно тот же split, который использовался в coarse benchmark.

### 2. True `O` В Source Не Недопредставлен

- `full`: `3000`
- `train`: `2100`
- `test`: `900`

Доля `O` стабильна:

- `full`: `9.0948%`
- `train`: `9.0948%`
- `test`: `9.0946%`

Это не выглядит как train/test starvation.

### 3. Все True `O` В Coarse Source Идут Как `evolved`

На reconstructed source:

- `full`: `evolved = 3000`
- `train`: `evolved = 2100`
- `test`: `evolved = 900`

То есть внутри текущего coarse source:

- `O` не смешан с `dwarf`;
- stratify по `('spec_class', 'evolution_stage')` не режет `O` на несколько редких подметок;
- наоборот, `O` идет как один стабильный bucket.

### 4. Hottest `O` Tail Не Теряется

Для true `O` temperature-band review дал:

- `full`: `>= 25000 K = 3000`
- `train`: `>= 25000 K = 2100`
- `test`: `>= 25000 K = 900`

Это самый сильный вывод этого review:

- в coarse training source true `O` уже является полностью hot tail;
- `cool contamination`, которое мы видели на `quality_gate` review, в coarse training source не живет;
- значит full coarse source для `O` уже очищен сильнее, чем downstream pass-slice.

### 5. Narrow Hot `O/B` Boundary Имеет Симметричный Support До Inference

Hot boundary support в reconstructed split:

- `full`:
  - `B = 3000`
  - `O = 3000`
- `train`:
  - `B = 2100`
  - `O = 2100`
- `test`:
  - `B = 900`
  - `O = 900`

Это означает:

- до inference граница `O/B` по support идеальна и симметрична;
- asymmetry `O -> B` не объясняется class-count imbalance на train/test уровне.

### 6. Физика True `O` В Train И Test Стабильна

Медианная физика true `O`:

- `full`:
  - `teff_gspphot ≈ 34604.6 K`
  - `logg_gspphot ≈ 3.896`
  - `bp_rp ≈ 3.385`
  - `radius_feature ≈ 8.797`
- `train`:
  - `teff_gspphot ≈ 34673.9 K`
  - `radius_feature ≈ 8.794`
- `test`:
  - `teff_gspphot ≈ 34334.6 K`
  - `radius_feature ≈ 8.810`

Train/test drift по базовой физике здесь не виден.

## Интерпретация

После этого review можно достаточно уверенно утверждать:

- coarse source не starving для `O`;
- split policy не starving для `O`;
- hottest `O` tail полноценно присутствует и в train, и в test;
- narrow `O/B` boundary по support симметрична.

Следовательно:

- текущая проблема не в количестве `O`;
- и не в том, что `O` “теряется до модели”.

Текущая рабочая гипотеза:

- coarse-модель фактически учит boundary, в которой true `O` и true `B`
  схлопываются в один `B`-кластер по текущему feature contract.

## Что Это Значит Для Следующего Шага

Следующий корректный narrow step:

- не rebalance всего coarse source;
- не blind oversampling `O`;
- не ослабление `quality_gate`.

Следующий правильный пакет:

- review feature separability `O vs B` внутри train-time source;
- сравнение physical overlap и feature overlap для true `O` и true `B`;
- и только потом решение, нужен ли narrow retrain / class weighting / feature-policy change.

## Related

- [coarse_o_tail_review_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/archive_research/coarse_o_tail_review_round1_ru.md)
- [coarse_o_hot_subset_review_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/archive_research/coarse_o_hot_subset_review_round1_ru.md)
- [coarse_ob_boundary_review_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/archive_research/coarse_ob_boundary_review_round1_ru.md)
- [coarse_o_train_support_review_tz_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/archive_research/coarse_o_train_support_review_tz_ru.md)
- [stabilization_issue_ledger_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/stabilization/stabilization_issue_ledger_ru.md)
