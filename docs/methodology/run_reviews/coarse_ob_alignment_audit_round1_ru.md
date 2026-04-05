# Round 1: Coarse `O/B` Alignment Audit

## Цель

Проверить, нет ли у текущего `O -> B` кейса более простого инженерного объяснения:

- train-time coarse source и saved coarse artifact собраны по одному feature contract;
- в `decide` current coarse artifact получает те же признаки в том же порядке;
- alias/compatibility слой не подменяет coarse-модели другой тип данных.

Этот аудит нужен, чтобы не лечить scientific/domain проблему как будто это просто баг
в plumbing train/inference.

## Проверенные Источники

- train source:
  - `lab.v_gaia_id_coarse_training`
- downstream inference source:
  - `lab.gaia_mk_quality_gated`
- coarse artifact:
  - `/Users/evgeniikuznetsov/Desktop/dspro-vkr/artifacts/models/gaia_id_coarse_classification__hist_gradient_boosting__2026_03_28_215003_509969`

Проверенный код:

- [hierarchical_feature_contract.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/contracts/hierarchical_feature_contract.py)
- [load_gaia_id_coarse_training_dataset.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/datasets/load_gaia_id_coarse_training_dataset.py)
- [hierarchical_training_frame.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/features/hierarchical_training_frame.py)
- [load_final_decision_input_dataset.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/datasets/load_final_decision_input_dataset.py)

## Проверка Feature Contract

Feature contract coarse-модели:

- `teff_gspphot`
- `logg_gspphot`
- `mh_gspphot`
- `bp_rp`
- `parallax`
- `parallax_over_error`
- `ruwe`
- `radius_feature`

Saved artifact metadata содержит ровно тот же список признаков:

- `['teff_gspphot', 'logg_gspphot', 'mh_gspphot', 'bp_rp', 'parallax', 'parallax_over_error', 'ruwe', 'radius_feature']`

Вывод:

- сохраненный coarse artifact и train-time task contract согласованы.

## Проверка Inference Input

`load_final_decision_input_dataset(...)` для coarse feature set возвращает:

- те же coarse feature columns;
- `radius_feature` materialize-ится через compatibility policy.

На relation-уровне:

- `lab.v_gaia_id_coarse_training` содержит `radius_feature` физически;
- `lab.gaia_mk_quality_gated` не содержит `radius_feature`, но содержит:
  - `radius_flame`
  - `radius_gspphot`

Текущий `decide` alias-layer делает:

- `radius_feature <- radius_flame`
- fallback:
  - `radius_feature <- radius_gspphot`

Вывод:

- gross feature mismatch по именам и порядку признаков не обнаружен;
- current inference path действительно подает coarse-модели все требуемые feature columns.

## Важная Находка: Семантика `radius_feature`

Train-time relation `lab.v_gaia_id_coarse_training` использует гибридный `radius_feature`:

- overall:
  - `n_rows = 32986`
  - `n_equal_flame = 23177`
  - `n_equal_gspphot = 9809`

По классам `O/B`:

- `B`:
  - `n_radius_feature = 3000`
  - `n_radius_flame = 2734`
  - `n_radius_gspphot = 3000`
  - `radius_feature = radius_flame` для `2734`
  - `radius_feature = radius_gspphot` для `266`
- `O`:
  - `n_radius_feature = 3000`
  - `n_radius_flame = 0`
  - `n_radius_gspphot = 3000`
  - `radius_feature = radius_gspphot` для всех `3000`

Это значит:

- весь train-time coarse `O` учился на `radius_gspphot`, а не на `radius_flame`.

## Узкий Эксперимент На Downstream `O/B`

На downstream hot pass-boundary были проверены три варианта:

1. `flame_only`
   - `radius_feature = radius_flame`
2. `gspphot_only`
   - `radius_feature = radius_gspphot`
3. `train_like_hybrid`
   - `radius_feature = radius_flame`, fallback в `radius_gspphot`

Результат для true `O`:

- во всех трех вариантах:
  - `O -> B = 1188`
  - `O -> O = 0`

То есть:

- радиусный semantic mismatch существует;
- но именно он текущий `O -> B` провал не объясняет.

## Итог

На текущем шаге не найдено evidence, что coarse pipeline просто “кормит модель не тем”.

Что подтверждено:

- feature contract train/inference согласован;
- coarse artifact обучен и применяется по одному и тому же feature order;
- compatibility alias для `radius_feature` не ломает результат сам по себе;
- подмена `radius_feature` на `radius_gspphot` downstream true `O` не возвращает.

Следовательно:

- нынешний `O -> B` кейс по-прежнему выглядит как scientific/domain-side проблема,
  а не как engineering bug в train/inference plumbing.

## Следующий Шаг

Следующий правильный пакет:

- provenance/source-alignment review downstream true `O` pass-pool;
- проверить происхождение этих меток и их согласованность с hot-star physics;
- только потом решать, нужен ли retrain, relabeling или source-alignment.
