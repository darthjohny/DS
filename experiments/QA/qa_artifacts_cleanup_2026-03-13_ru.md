# Artifacts and Cleanup Ledger

Дата фиксации: 13 марта 2026 года

Категории:
- `KEEP` — артефакт явно нужен
- `ARCHIVE?` — артефакт не мусорный, но, возможно, его лучше архивировать/перенести
- `REMOVE?` — явный кандидат на удаление/вынос из git
- `POLICY` — проблема не в файле, а в отсутствии правила хранения

## Явно полезные артефакты

- [data/router_gaussian_params.json](/Users/evgeniikuznetsov/Desktop/dspro-vkr/data/router_gaussian_params.json): `KEEP`
- [data/model_gaussian_params.json](/Users/evgeniikuznetsov/Desktop/dspro-vkr/data/model_gaussian_params.json): `KEEP`
- [experiments/QA](/Users/evgeniikuznetsov/Desktop/dspro-vkr/experiments/QA): `KEEP`
- [experiments/Логи работы программы](/Users/evgeniikuznetsov/Desktop/dspro-vkr/experiments/%D0%9B%D0%BE%D0%B3%D0%B8%20%D1%80%D0%B0%D0%B1%D0%BE%D1%82%D1%8B%20%D0%BF%D1%80%D0%BE%D0%B3%D1%80%D0%B0%D0%BC%D0%BC%D1%8B): `KEEP`
- [experiments/Логи калибровки decision_layer](/Users/evgeniikuznetsov/Desktop/dspro-vkr/experiments/%D0%9B%D0%BE%D0%B3%D0%B8%20%D0%BA%D0%B0%D0%BB%D0%B8%D0%B1%D1%80%D0%BE%D0%B2%D0%BA%D0%B8%20decision_layer): `KEEP`
- [docs/presentation/assets](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/presentation/assets): `KEEP`
- [docs/assets/gaia_archive_crossmatch_ui.png](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/assets/gaia_archive_crossmatch_ui.png): `KEEP`
- [docs/assets/gaia_archive_validation_ui.png](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/assets/gaia_archive_validation_ui.png): `KEEP`

## Нужна policy, а не слепая чистка

- [experiments/model_comparison](/Users/evgeniikuznetsov/Desktop/dspro-vkr/experiments/model_comparison): `POLICY`
  Здесь есть как минимум четыре волны одного дня:
  - `baseline_comparison_2026-03-13*`
  - `baseline_comparison_2026-03-13_mlp*`
  - `baseline_comparison_2026-03-13_snapshot*`
  - `baseline_comparison_2026-03-13_vkr30_cv10*`
  Нужно решить, что считается каноническим поколением, а что становится историческим архивом.

- [data/raw](/Users/evgeniikuznetsov/Desktop/dspro-vkr/data/raw): `POLICY`
  Похоже на sample/raw snapshots, но это не зафиксировано явно.

- [data/processed](/Users/evgeniikuznetsov/Desktop/dspro-vkr/data/processed): `POLICY`
  Слой выглядит полезным, но не отделён формально от runtime artifacts и EDA outputs.

- [data/eda](/Users/evgeniikuznetsov/Desktop/dspro-vkr/data/eda): `POLICY`
  Полезные EDA-артефакты, но нужен критерий, какие из них canonical, а какие локальные.

- [data/plots](/Users/evgeniikuznetsov/Desktop/dspro-vkr/data/plots): `POLICY`
  Небольшой объём, но без явной политики это будет расти как dump графиков.

## Кандидаты на архивирование

- старые comparison-волны до `vkr30_cv10` в [experiments/model_comparison](/Users/evgeniikuznetsov/Desktop/dspro-vkr/experiments/model_comparison): `ARCHIVE?`
  Не выглядят мусором, но рядом с канонической волной создают путаницу.

## Явные cleanup-кандидаты

- [data/.DS_Store](/Users/evgeniikuznetsov/Desktop/dspro-vkr/data/.DS_Store): `REMOVE?`
- [sql/.DS_Store](/Users/evgeniikuznetsov/Desktop/dspro-vkr/sql/.DS_Store): `REMOVE?`
- [experiments/.DS_Store](/Users/evgeniikuznetsov/Desktop/dspro-vkr/experiments/.DS_Store): `REMOVE?`
- [docs/assets/.DS_Store](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/assets/.DS_Store): `REMOVE?`
- [docs/assets/Снимок экрана 2026-03-12 в 16.17.24.png](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/assets/%D0%A1%D0%BD%D0%B8%D0%BC%D0%BE%D0%BA%20%D1%8D%D0%BA%D1%80%D0%B0%D0%BD%D0%B0%202026-03-12%20%D0%B2%2016.17.24.png): `REMOVE?`
- [docs/assets/Снимок экрана 2026-03-12 в 16.28.37.png](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/assets/%D0%A1%D0%BD%D0%B8%D0%BC%D0%BE%D0%BA%20%D1%8D%D0%BA%D1%80%D0%B0%D0%BD%D0%B0%202026-03-12%20%D0%B2%2016.28.37.png): `REMOVE?`
- [docs/assets/Снимок экрана 2026-03-12 в 16.29.44.png](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/assets/%D0%A1%D0%BD%D0%B8%D0%BC%D0%BE%D0%BA%20%D1%8D%D0%BA%D1%80%D0%B0%D0%BD%D0%B0%202026-03-12%20%D0%B2%2016.29.44.png): `REMOVE?`

## Краткий вывод

- Самая важная cleanup-задача здесь не “удалить много файлов”, а ввести политику хранения сравнительных артефактов и data outputs.
- Самые безопасные прямые кандидаты на удаление: `.DS_Store` и неиспользуемые generic screenshots.
- Самый важный кандидат на архивирование, а не на удаление: старые волны `experiments/model_comparison`.
