# Local Source Audit For MK Wave

## Цель

Этот документ фиксирует формальный audit локальных relation перед MK-волной.

Задача audit:

- понять, что уже есть локально;
- не строить новый source на догадках;
- явно зафиксировать, где именно локальных labels и quality-полей не хватает.

## Проверенные Relation

- `lab.v_gaia_router_training`
- `lab.v_nasa_gaia_train_classified`
- `lab.v_nasa_gaia_train_dwarfs`
- `lab.v_nasa_gaia_train_evolved`
- `lab.v_gaia_ref_mkgf_dwarfs`
- `lab.v_gaia_ref_mkgf_evolved`
- `lab.v_gaia_ref_abo_dwarfs`
- `lab.v_gaia_ref_abo_evolved`

## Проверенные Поля

- `source_id`
- `spec_class`
- `spec_subclass`
- `evolution_stage`
- `luminosity_class`
- `teff_gspphot`
- `logg_gspphot`
- `radius_gspphot`
- `mh_gspphot`
- `bp_rp`
- `parallax`
- `parallax_over_error`
- `phot_g_mean_mag`
- `ruwe`

## Главные Выводы

### 1. Полного локального MK-source сейчас нет

- ни один проверенный relation не содержит `luminosity_class`;
- полноценного локального source для `spectral_class + spectral_subclass + luminosity_class` сейчас нет.

### 2. Router source годится для coarse class, но не для subclass

`lab.v_gaia_router_training`

- `39413` строк;
- есть `spec_class`, `evolution_stage`, физические признаки и quality-поля;
- `spec_subclass` формально есть как колонка, но:
  - `n_labeled = 0`
  - `n_distinct = 0`

Вывод:

- подходит для coarse-class и stage логики;
- не подходит для реальной subclass-волны.

### 3. Gaia reference views по классам плотные, но без subclass labels

`lab.v_gaia_ref_mkgf_dwarfs`

- `12000` строк;
- по `3000` на `F/G/K/M`;
- `spec_subclass` полностью пустой.

`lab.v_gaia_ref_mkgf_evolved`

- `12000` строк;
- по `3000` на `F/G/K/M`;
- `spec_subclass` полностью пустой.

`lab.v_gaia_ref_abo_dwarfs`

- `6413` строк;
- `A/B/O` покрыты;
- `spec_subclass` полностью пустой.

`lab.v_gaia_ref_abo_evolved`

- `9000` строк;
- по `3000` на `A/B/O`;
- `spec_subclass` полностью пустой.

Вывод:

- это хорошие reference source для coarse training;
- это не source для новой MK subclass-волны.

### 4. NASA/Gaia host source дает только узкую subclass-разметку по M

`lab.v_nasa_gaia_train_classified`

- `3741` строк;
- `spec_subclass` размечен только для `M`;
- всего `232` размеченных строк;
- `n_distinct = 3`

`lab.v_nasa_gaia_train_dwarfs`

- `3394` строк;
- `224` размеченных subclass-строк;
- только `M`

`lab.v_nasa_gaia_train_evolved`

- `319` строк;
- `8` размеченных subclass-строк;
- только `M`

Вывод:

- локально уже можно построить очень узкий `M_early/M_mid/M_late` контур;
- это не покрывает полноценную задачу `G0..G9`, `K0..K9`, `F0..F9`, `M0..M9`.

### 5. Quality-поля локально в целом хорошие

Во всех проверенных relation есть:

- `ruwe`
- `parallax`
- `parallax_over_error`
- `teff_gspphot`
- `logg_gspphot`
- `radius_gspphot`
- `mh_gspphot`
- `bp_rp`

Вывод:

- quality/OOD gate уже можно строить на локальных полях;
- проблема локального слоя не в quality-features, а в label coverage.

### 6. Яркость доступна не везде

`phot_g_mean_mag` есть в:

- `lab.v_nasa_gaia_train_classified`
- `lab.v_nasa_gaia_train_dwarfs`
- `lab.v_nasa_gaia_train_evolved`

И отсутствует в:

- `lab.v_gaia_router_training`
- `lab.v_gaia_ref_mkgf_dwarfs`
- `lab.v_gaia_ref_mkgf_evolved`
- `lab.v_gaia_ref_abo_dwarfs`
- `lab.v_gaia_ref_abo_evolved`

Вывод:

- brightness-aware observability локально пока поддержан не всеми source;
- для новой candidate-scoring ветки это нужно учитывать отдельно.

## Итог Audit

На локальной БД уже есть хороший фундамент для:

- coarse-class tasks;
- stage/evolution tasks;
- quality/OOD gate;
- host-like tasks;
- observability без полного brightness coverage.

Но локальной БД сейчас недостаточно для:

- полного `spectral_subclass` learning;
- `luminosity_class` learning;
- полноценной MK-волны без внешнего label-source.

## Практический Вывод

Перед новой MK-волной:

- старые relation сохраняем;
- новый coarse контур не ломаем;
- идем за внешним spectral source;
- потом делаем crossmatch с `Gaia`;
- потом добавляем новые таблицы и views в БД;
- только после этого обучаем новые MK tasks.

## Критерий Готовности Audit

Audit считается закрытым, если:

- по каждой локальной relation понятно, для чего она годится;
- по каждой дыре в labels видно, нужен ли внешний source;
- дальнейший data engineering не строится на предположениях.
