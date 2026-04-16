# Первый аудит согласованности `O/B` для coarse-модели

## Цель

Проверить, нет ли у текущего случая `O -> B` более простого инженерного объяснения:

- источник обучения coarse-модели и сохраненный артефакт собраны по одному контракту признаков;
- в `decide` текущий артефакт coarse-модели получает те же признаки в том же порядке;
- слой совместимости не подменяет coarse-модели другой тип данных.

Этот аудит нужен, чтобы не лечить предметную проблему домена так, как будто
это простой баг в связке обучения и применения модели.

## Проверенные Источники

- источник обучения:
  - `lab.v_gaia_id_coarse_training`
- источник для рабочего применения:
  - `lab.gaia_mk_quality_gated`
- артефакт coarse-модели:
  - `/Users/evgeniikuznetsov/Desktop/dspro-vkr/artifacts/models/gaia_id_coarse_classification__hist_gradient_boosting__2026_03_28_215003_509969`

Проверенный код:

- [hierarchical_feature_contract.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/contracts/hierarchical_feature_contract.py)
- [load_gaia_id_coarse_training_dataset.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/datasets/load_gaia_id_coarse_training_dataset.py)
- [hierarchical_training_frame.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/features/hierarchical_training_frame.py)
- [load_final_decision_input_dataset.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/datasets/load_final_decision_input_dataset.py)

## Проверка контракта признаков

Контракт признаков coarse-модели:

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

- сохраненный артефакт coarse-модели и контракт задачи на этапе обучения согласованы.

## Проверка Inference Input

`load_final_decision_input_dataset(...)` для coarse feature set возвращает:

- те же столбцы признаков для coarse-модели;
- `radius_feature` материализуется через политику совместимости.

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

- грубого расхождения по именам и порядку признаков не обнаружено;
- текущий путь применения действительно подает coarse-модели все требуемые признаки.

## Важная находка: семантика `radius_feature`

Таблица `lab.v_gaia_id_coarse_training` на этапе обучения использует гибридный `radius_feature`:

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

- весь класс `O` при обучении coarse-модели учился на `radius_gspphot`, а не на `radius_flame`.

## Узкий эксперимент на рабочем пуле `O/B`

На рабочем горячем проходном пуле были проверены три варианта:

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

- расхождение в семантике радиуса действительно существует;
- но именно оно текущий провал `O -> B` не объясняет.

## Итог

На текущем шаге не найдено подтверждения, что coarse-контур просто “кормит модель не тем”.

Что подтверждено:

- контракт признаков между обучением и применением согласован;
- артефакт coarse-модели обучен и применяется при одном и том же порядке признаков;
- compatibility alias для `radius_feature` не ломает результат сам по себе;
- подмена `radius_feature` на `radius_gspphot` не возвращает рабочие true `O`.

Следовательно:

- нынешний случай `O -> B` по-прежнему выглядит как предметная проблема домена,
  а не как инженерная ошибка в связке обучения и применения модели.

## Следующий Шаг

Следующий правильный пакет:

- обзор происхождения меток и согласования источников для рабочего проходного пула true `O`;
- проверить происхождение этих меток и их согласованность с физикой горячих звезд;
- только потом решать, нужен ли повторный запуск обучения, пересмотр меток или дополнительное согласование источников.
