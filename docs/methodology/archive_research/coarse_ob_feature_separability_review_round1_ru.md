# Coarse O/B Feature Separability Review Round 1

## Контекст

Этот review закрывает следующий вопрос после
[coarse_o_train_support_review_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/archive_research/coarse_o_train_support_review_round1_ru.md):

- separable ли true `O` и true `B` по текущему coarse feature contract;
- и способен ли текущий coarse artifact различать их на собственном train-time source.

Срез:

- source: prepared coarse training frame
- boundary:
  - `spec_class IN ('O', 'B')`
  - `teff_gspphot >= 10000 K`

Artifact:

- `artifacts/models/gaia_id_coarse_classification__hist_gradient_boosting__2026_03_28_215003_509969`

## Главный Результат

На train-time `O/B` boundary current coarse artifact различает `O` и `B` почти идеально.

Это значит:

- проблема `O -> B`, которую мы видели на downstream hot pass-slice,
  не воспроизводится на собственном train-time source coarse-модели;
- текущий coarse feature contract уже содержит сильный discriminative signal;
- следующий основной риск — не отсутствие separability,
  а domain shift / source mismatch между coarse reference source и downstream pass-source.

## Что Показал Review

### 1. Train-Time Boundary Источник Симметричен

- `n_rows_boundary = 6000`
- `B = 3000`
- `O = 3000`

То есть это не несбалансированный boundary.

### 2. Current Coarse Artifact На Этом Source Работает Нормально

Предсказания:

- `O = 3001`
- `B = 2995`
- `A = 4`

То есть на train-time source coarse artifact не схлопывает `O` в `B`.

Это уже резко отличается от downstream review, где на hot pass-slice было:

- true `O -> B = 1188`
- true `O -> O = 0`

### 3. Вероятности На Train-Time Boundary Почти Идеальны

Для true `B`:

- `median P(B) ≈ 0.999842`
- `median P(O) ≈ 0.000017`

Для true `O`:

- `median P(O) ≈ 0.999845`
- `median P(B) ≈ 0.000018`

Это еще один сильный аргумент:

- coarse artifact умеет различать `O/B` на своем source;
- collapse возникает где-то после смены source-domain.

### 4. Single-Feature Separability Уже Очень Сильна

Top признаки по `separability_auc`:

- `teff_gspphot = 1.000`
- `radius_feature ≈ 0.959`
- `bp_rp ≈ 0.855`
- `parallax ≈ 0.731` в сторону `B`
- `mh_gspphot ≈ 0.710`

Слабые признаки:

- `ruwe ≈ 0.503`

То есть signal реально есть и он не тонкий.

### 5. Permutation Importance Подтверждает То Же

Top признаки current coarse artifact на train-time `O/B` boundary:

- `teff_gspphot ≈ 0.494`
- `bp_rp ≈ 0.0022`
- `radius_feature ≈ 0.0019`
- `logg_gspphot ≈ 0.0016`
- `mh_gspphot ≈ 0.0015`

Остальное почти не влияет.

Иными словами:

- модель в основном держит границу через `teff`;
- и этого на train-time source достаточно.

## Интерпретация

На этом шаге hypothesis “coarse model не умеет `O/B` даже на своем source” не подтверждается.

Наоборот, review говорит следующее:

- support для `O` достаточный;
- feature separability для `O/B` на train-time source сильная;
- coarse artifact на этом source ведет себя почти идеально.

Значит текущая рабочая гипотеза теперь такая:

- проблема `O -> B` возникает не в coarse artifact как таковом;
- она возникает из-за domain mismatch между:
  - coarse reference / train-time source
  - и downstream hot pass-source из `quality_gated/final_decision`.

## Что Это Значит Для Следующего Шага

Следующий корректный narrow step:

- не retrain coarse model;
- не class-weighting;
- не oversampling `O`.

Следующий правильный пакет:

- отдельный review `train-time O/B boundary` vs `downstream hot pass O/B boundary`;
- сравнение физики, feature distributions и missingness между этими двумя domains;
- только после этого решать, нужен ли новый retrain или source-alignment step.

## Related

- [coarse_o_train_support_review_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/archive_research/coarse_o_train_support_review_round1_ru.md)
- [coarse_ob_boundary_review_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/archive_research/coarse_ob_boundary_review_round1_ru.md)
- [coarse_ob_feature_separability_review_tz_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/archive_research/coarse_ob_feature_separability_review_tz_ru.md)
- [stabilization_issue_ledger_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/stabilization/stabilization_issue_ledger_ru.md)
