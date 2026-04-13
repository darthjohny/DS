# Разбор обучающей поддержки O-класса: первый архивный обзор

## Зачем проводился разбор

Этот разбор закрывал следующий вопрос после
[coarse_ob_boundary_review_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/archive_research/coarse_ob_boundary_review_round1_ru.md):

- есть ли у coarse-модели вообще достаточная обучающая поддержка для класса `O`;
- или проблема `O -> B` возникает уже при нормальной поддержке класса.

Срез повторял текущую benchmark policy:

- source: `lab.v_gaia_id_coarse_training`
- prepared frame: `prepare_gaia_id_coarse_training_frame(...)`
- split:
  - `test_size = 0.30`
  - `random_state = 42`
  - `stratify_columns = ('spec_class', 'evolution_stage')`

## Главный результат

Обучающая поддержка класса `O` в coarse-источнике оказалась достаточной.

Проблема `O -> B` не объясняется тем, что:

- `O` мало в источнике;
- `O` почти не попадает в train/test;
- самый горячий хвост `O` теряется на split;
- или benchmark split не совпадает с восстановленным разбиением.

Иными словами:

- версия о нехватке поддержки здесь не подтвердилась;
- проблема сужается до границы `O vs B` на уровне модели.

## Что показал разбор

### 1. Восстановленный split совпадает с benchmark

- `full`: `32986`
- `train`: `23090`
- `test`: `9896`

Все три числа совпали с artifact:

- `artifacts/benchmarks/gaia_id_coarse_classification_2026_03_28_171258_103400`

Значит дальше анализировался именно тот же split, который использовался в
coarse benchmark.

### 2. True `O` в источнике не был недопредставлен

- `full`: `3000`
- `train`: `2100`
- `test`: `900`

Доля `O` стабильна:

- `full`: `9.0948%`
- `train`: `9.0948%`
- `test`: `9.0946%`

Это не выглядит как нехватка примеров в `train/test`.

### 3. Все true `O` в coarse-источнике шли как `evolved`

На восстановленном источнике:

- `full`: `evolved = 3000`
- `train`: `evolved = 2100`
- `test`: `evolved = 900`

То есть внутри текущего coarse-источника:

- `O` не смешан с `dwarf`;
- stratify по `('spec_class', 'evolution_stage')` не режет `O` на несколько редких подметок;
- наоборот, `O` идет как один стабильный bucket.

### 4. Самый горячий хвост `O` не терялся

Для true `O` temperature-band review дал:

- `full`: `>= 25000 K = 3000`
- `train`: `>= 25000 K = 2100`
- `test`: `>= 25000 K = 900`

Это был один из самых сильных выводов этого разбора:

- в coarse-источнике обучения true `O` уже представляет собой полностью
  горячий хвост;
- холодный шум, который был виден при разборе `quality_gate`, в этом источнике
  отсутствует;
- значит полный coarse-источник для `O` уже очищен сильнее, чем проходящий
  срез в downstream-контуре.

### 5. Горячая граница `O/B` имела симметричную поддержку до inference

Поддержка горячей границы в восстановленном split:

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

- до inference граница `O/B` по числу строк идеальна и симметрична;
- асимметрия `O -> B` не объясняется дисбалансом классов на уровне
  `train/test`.

### 6. Физика true `O` в train и test была стабильной

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

Сдвиг между `train` и `test` по базовой физике здесь не виден.

## Вывод на момент этого шага

После этого разбора можно было достаточно уверенно утверждать:

- coarse-источник не испытывал нехватки `O`;
- политика split не создавала нехватку `O`;
- самый горячий хвост `O` полноценно присутствовал и в `train`, и в `test`;
- узкая граница `O/B` была симметрична по числу строк.

Следовательно:

- текущая проблема не в количестве `O`;
- и не в том, что `O` “теряется до модели”.

Текущая рабочая гипотеза:

- coarse-модель фактически учит границу, в которой true `O` и true `B`
  схлопываются в один `B`-кластер при текущем наборе признаков.

## Почему документ сохранен в архиве

Позднее этот сюжет был продолжен через разбор разделимости признаков на
границе `O/B`. Поэтому документ хранится как промежуточный исследовательский
вывод.

На момент этого обзора следующий шаг виделся так:

- разбор разделимости признаков `O` и `B` внутри обучающего источника;
- сравнение физического перекрытия и перекрытия признаков для true `O` и
  true `B`;
- и только потом решение, нужен ли узкий повторный запуск обучения, веса
  классов или изменение набора признаков.

## Связанные документы

- [coarse_o_tail_review_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/archive_research/coarse_o_tail_review_round1_ru.md)
- [coarse_o_hot_subset_review_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/archive_research/coarse_o_hot_subset_review_round1_ru.md)
- [coarse_ob_boundary_review_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/archive_research/coarse_ob_boundary_review_round1_ru.md)
- [coarse_o_train_support_review_tz_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/archive_research/coarse_o_train_support_review_tz_ru.md)
- [stabilization_issue_ledger_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/stabilization/stabilization_issue_ledger_ru.md)
