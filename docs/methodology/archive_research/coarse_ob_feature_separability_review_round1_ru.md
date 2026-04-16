# Разбор разделимости признаков на границе O/B: первый архивный обзор

## Зачем проводился разбор

Этот разбор закрывал следующий вопрос после
[coarse_o_train_support_review_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/archive_research/coarse_o_train_support_review_round1_ru.md):

- различимы ли true `O` и true `B` по текущему набору coarse-признаков;
- и способен ли текущий coarse artifact различать их на собственном
  обучающем источнике.

Срез:

- source: prepared coarse training frame
- boundary:
  - `spec_class IN ('O', 'B')`
  - `teff_gspphot >= 10000 K`

Artifact:

- `artifacts/models/gaia_id_coarse_classification__hist_gradient_boosting__2026_03_28_215003_509969`

## Главный результат

На обучающей границе `O/B` текущий coarse artifact различал `O` и `B` почти
идеально.

Это значит:

- проблема `O -> B`, которую мы видели на горячем проходящем срезе рабочего контура,
  не воспроизводится на собственном обучающем источнике coarse-модели;
- текущий набор coarse-признаков уже содержит сильный различающий сигнал;
- основной риск связан не с отсутствием разделимости, а со сдвигом между
  обучающим и проходящим доменами.

## Что показал разбор

### 1. Обучающий граничный источник был симметричен

- `n_rows_boundary = 6000`
- `B = 3000`
- `O = 3000`

То есть это не несбалансированная граница.

### 2. Текущий coarse artifact на этом источнике работал нормально

Предсказания:

- `O = 3001`
- `B = 2995`
- `A = 4`

То есть на обучающем источнике coarse artifact не схлопывает `O` в `B`.

Это уже резко отличается от разбора рабочего контура, где на горячем проходящем
срезе было:

- true `O -> B = 1188`
- true `O -> O = 0`

### 3. Вероятности на обучающей границе были почти идеальными

Для true `B`:

- `median P(B) ≈ 0.999842`
- `median P(O) ≈ 0.000017`

Для true `O`:

- `median P(O) ≈ 0.999845`
- `median P(B) ≈ 0.000018`

Это был еще один сильный аргумент:

- coarse artifact умеет различать `O/B` на собственном источнике;
- схлопывание возникает уже после смены домена.

### 4. Разделимость по отдельным признакам уже была сильной

Top признаки по `separability_auc`:

- `teff_gspphot = 1.000`
- `radius_feature ≈ 0.959`
- `bp_rp ≈ 0.855`
- `parallax ≈ 0.731` в сторону `B`
- `mh_gspphot ≈ 0.710`

Слабые признаки:

- `ruwe ≈ 0.503`

То есть различающий сигнал действительно был и не выглядел тонким.

### 5. Permutation importance подтверждала тот же вывод

Главные признаки текущего coarse artifact на обучающей границе `O/B`:

- `teff_gspphot ≈ 0.494`
- `bp_rp ≈ 0.0022`
- `radius_feature ≈ 0.0019`
- `logg_gspphot ≈ 0.0016`
- `mh_gspphot ≈ 0.0015`

Остальное почти не влияет.

Иными словами:

- модель в основном держит границу через `teff`;
- и этого на обучающем источнике достаточно.

## Вывод на момент этого шага

На этом шаге гипотеза о том, что coarse-модель не умеет различать `O/B` даже
на собственном обучающем источнике, не подтвердилась.

Наоборот, разбор говорил следующее:

- поддержка класса `O` достаточная;
- разделимость признаков для `O/B` на обучающем источнике сильная;
- coarse artifact на этом источнике ведет себя почти идеально.

Значит текущая рабочая гипотеза теперь такая:

- проблема `O -> B` возникает не в coarse artifact как таковом;
- она возникает из-за несовпадения доменов между:
  - обучающим coarse-источником;
  - и горячим проходящим срезом из `quality_gated/final_decision`.

## Почему документ сохранен в архиве

Позднее этот результат был переосмыслен уже в активном рабочем контуре, где
основной акцент сместился с редкого хвоста `O` на прикладную задачу
приоритизации наблюдений. Поэтому документ сохранен как исторический
исследовательский вывод.

На момент этого обзора следующий шаг виделся так:

- отдельный разбор обучающей границы `O/B` и горячего проходящего среза `O/B`;
- сравнение физики, распределений признаков и пропусков между этими двумя
  доменами;
- и только после этого решение, нужен ли новый цикл обучения или выравнивание
  источников.

## Связанные документы

- [coarse_o_train_support_review_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/archive_research/coarse_o_train_support_review_round1_ru.md)
- [coarse_ob_boundary_review_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/archive_research/coarse_ob_boundary_review_round1_ru.md)
- [coarse_ob_feature_separability_review_tz_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/archive_research/coarse_ob_feature_separability_review_tz_ru.md)
- [stabilization_issue_ledger_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/stabilization/stabilization_issue_ledger_ru.md)
