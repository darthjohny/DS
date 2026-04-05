# Hierarchical Classification And OOD Strategy

## Цель

Этот документ фиксирует следующую исследовательскую идею для проекта:

- не строить один монолитный классификатор;
- разделить задачу на coarse classification, refinement и OOD/reject слой;
- использовать disagreement между связанными Gaia-параметрами как сигнал надежности,
  а не как независимый источник истины.

## Главная Постановка

Проект должен уметь:

1. определить крупный спектральный класс звезды;
2. по возможности уточнить подкласс;
3. оценить пригодность объекта для host/priority-контура;
4. вынести объекты вне целевого stellar-domain в `unknown` / `OOD` слой;
5. не терять такие объекты, а складывать их в отдельный аналитический контур.

## ID-Space Для Первой Волны

Целевое нормальное пространство первой волны:

- `O`
- `B`
- `A`
- `F`
- `G`
- `K`
- `M`

Локальный DB-фундамент под coarse stage уже есть:

- `public.gaia_ref_class_o` — `3000`
- `public.gaia_ref_class_b` — `3000`
- `public.gaia_ref_class_a` — `3000`
- `public.gaia_ref_class_f` — `3000`
- `public.gaia_ref_class_g` — `3000`
- `public.gaia_ref_class_k` — `3000`
- `public.gaia_ref_class_m` — `3000`
- `public.gaia_ref_evolved_class_o` — `3000`
- `public.gaia_ref_evolved_class_b` — `3000`
- `public.gaia_ref_evolved_class_a` — `3000`
- `public.gaia_ref_evolved_class_f` — `3000`
- `public.gaia_ref_evolved_class_g` — `3000`
- `public.gaia_ref_evolved_class_k` — `3000`
- `public.gaia_ref_evolved_class_m` — `3000`

Итог:

- coarse classifier можно учить не только на старых `view`, но и на отдельных
  Gaia class-table, уже лежащих в БД.

## Refinement-Space

Уточнение внутри coarse class строим отдельно.

Источники для refinement:

- новая `MK`-ветка через `B/mk -> Gaia -> external_labeled -> training_reference`;
- существующие локальные relation, если они дают узкий subclass signal.

Правило:

- refinement не должен ломать coarse classifier;
- refinement запускается только после coarse stage и только там, где coverage и
  качество label достаточны.

## OOD-Space

OOD в рамках проекта определяется не как "все неудобное", а как заранее
зафиксированные группы объектов вне нашего normal stellar training-domain.

Первая волна OOD-candidate групп:

- white dwarfs;
- binaries / unresolved multiple systems;
- non-single stars;
- emission-line stars;
- carbon stars;
- peculiar/outlier-like sources;
- объекты с плохой астрофизической согласованностью.

## Official Gaia Sources Для OOD

Для OOD-source используем только официальные Gaia DR3 tables/flags.

Первая очередь:

- `gaiadr3.astrophysical_parameters`
  - `classprob_dsc_combmod_whitedwarf`
  - `classprob_dsc_combmod_binarystar`
  - `classprob_dsc_allosmod_whitedwarf`
  - `classprob_dsc_allosmod_binarystar`
  - `classlabel_espels`
  - `classlabel_espels_flag`
  - `neuron_oa_id`
  - `flags_oa`
- `gaiadr3.gaia_source`
  - `non_single_star`
- Gaia DR3 performance verification
  - `gold_sample_carbon_stars`

Правило:

- OOD-pool строим отдельно от ID training-source;
- OOD labels не смешиваем с обычными `OBAFGKM` в один flat target.

## Decision Layer

Итоговое решение проекта должно быть многоступенчатым.

Нормальный случай:

- coarse class определен;
- subclass определен;
- host/priority score посчитан.

Low-priority случай:

- coarse class определен;
- объект попадает в low-priority stellar group;
- наблюдательный приоритет снижается.

Unknown / Review:

- coarse class неустойчив;
- refinement неуверен;
- объект отправляется в `unknown/review`.

OOD:

- объект попадает во внешний по отношению к ID-domain класс;
- в основной priority pipeline он не идет;
- параметры и координаты сохраняются для отдельного анализа.

## Как Использовать Gaia Disagreement

Расхождение между связанными Gaia-оценками используем как reliability signal.

Примеры:

- расхождение между `radius_gspphot` и `radius_flame`, если доступны оба поля;
- расхождения между связанными atmospheric/evolutionary estimates;
- сочетание disagreement с `ruwe`, `parallax_over_error` и quality flags.

Важное правило:

- disagreement не является независимым ground truth;
- это сигнал риска, неуверенности или выхода за training-domain.

## Научная Корректность

Идея считается научно корректной, если:

- coarse, refinement и OOD задачи разделены;
- OOD определяется заранее, а не после просмотра результатов;
- reject/unknown пороги подбираются на validation, а не на test;
- disagreement между Gaia-модулями используется как uncertainty signal,
  а не как "второе мнение независимого эксперта".

## Что Это Дает Проекту

В такой постановке проект умеет:

- определить класс звезды;
- при наличии достаточной уверенности определить подкласс;
- соотнести объект с host/priority-контуром;
- обнаружить объекты вне целевого stellar-domain;
- не выбрасывать их, а отправлять в отдельный аналитический слой.
