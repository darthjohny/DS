# Разбор горячего подмножества O-класса: первый архивный обзор

## Зачем проводился разбор

Этот документ фиксирует первый разбор горячего подмножества `O/B` внутри
true `O`.

Разбор был сделан на связке:

- coarse model artifact:
  `artifacts/models/gaia_id_coarse_classification__hist_gradient_boosting__2026_03_28_215003_509969`
- final decision run:
  `artifacts/decisions/hierarchical_final_decision_2026_03_29_111132_270743`
- baseline subset-config:
  - `teff_gspphot >= 10000 K`
- notebook:
  [12_coarse_o_hot_subset_review.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/notebooks/archive_research/12_coarse_o_hot_subset_review.ipynb)

## Почему понадобился этот шаг

Первый разбор по всему `O`-пулу уже показал:

- в true `O` есть заметный холодный шум;
- часть сохранившихся объектов `O` ведет себя как `F/G/K/M` по физическим
  признакам;
- поэтому весь пул `O` был слишком шумным для немедленного переобучения.

Этот шаг должен был ответить:

- остается ли broad contamination после простого physics cut;
- или проблема сужается до более чистой границы `O -> B`.

## Сводка по горячему подмножеству

При baseline `teff_gspphot >= 10000 K`:

- `n_rows_true_o_source = 6372`
- `n_rows_hot_subset = 3149`
- `share_hot_subset_in_source = 49.42%`
- `n_rows_hot_pass = 1188`
- `share_hot_pass_in_hot_subset = 37.73%`

Вывод:

- почти половина true `O` проходит минимальный порог горячей физики `O/B`;
- значит узкий разбор горячего подмножества статистически осмыслен и не
  упирается в
  слишком маленький sample.

## Что делает `quality_gate` на горячем подмножестве

Распределение `quality_state`:

- `unknown = 1961` (`62.27%`)
- `pass = 1188` (`37.73%`)
- `reject = 0`

Top `quality_reason`:

- `missing_radius_flame = 1754`
- `pass = 1188`
- `high_ruwe = 153`
- `low_parallax_snr = 54`

Практический вывод:

- после отсечения холодного хвоста главным ограничителем уже становится не
  `missing_core_features`, а
  `missing_radius_flame`;
- значит потери на `quality_gate` внутри этого горячего подмножества стали
  чище и лучше интерпретируются.

## Что делает coarse-модель на горячем подмножестве со статусом `pass`

Распределение `coarse_predicted_label`:

- `B = 1188` (`100%`)
- `O = 0`
- `A/F/G/K/M = 0`

Это был главный результат первого разбора:

- после удаления холодного шума coarse-модель больше не разбрасывает `O`
  в `F/G/K/M`;
- широкий холодный шум исчезает;
- проблема сужается до устойчивого routing `O -> B`.

## Физический профиль предсказанного горячего подмножества

Для predicted `B`:

- `median_teff_gspphot ≈ 15614 K`
- `median_logg_gspphot ≈ 3.67`
- `median_bp_rp ≈ 0.48`
- `median_radius_flame ≈ 4.78`

Интерпретация:

- горячее подмножество уже выглядит физически согласованнее, чем весь пул `O`;
- значит это подмножество лучше подходит для следующего узкого разбора, чем
  весь исходный `O`.

## Что происходило дальше в `final_decision`

Распределение `final_domain_state`:

- `unknown = 1961` (`62.27%`)
- `id = 1100` (`34.93%`)
- `ood = 88` (`2.79%`)

Распределение `final_coarse_class`:

- `<NA> = 2049` (`65.07%`)
- `B = 1100` (`34.93%`)
- `O = 0`

Распределение `final_decision_reason`:

- `quality_unknown = 1961`
- `refinement_accepted = 1100`
- `hard_ood = 88`

Вывод:

- если горячее подмножество проходит в `id`, оно почти всегда попадает туда
  как `B`, а не
  как `O`;
- значит проблема теперь уже не в широкой ошибке всего coarse-слоя, а в узкой
  границе между `O` и `B`.

## Что этот шаг изменил по сравнению с первым обзором

На полном пуле `O` мы видели:

- большой увод в `F/G/K/M`
- признаки холодного шума

На горячем подмножестве видно:

- `F/G/K/M` исчезают полностью;
- предварительный просмотр уверенных `non-O/B` ошибок становится пустым;
- остается чистая системная граница `O -> B`.

Это был важный шаг к сужению гипотезы.

## Почему документ сохранен в архиве

Позднее эта линия анализа была развита через отдельный разбор границы `O/B`.
Поэтому документ хранится как промежуточный исследовательский результат, а не
как текущая рабочая интерпретация.

На момент этого обзора следующий шаг виделся так:

1. открыть отдельный узкий разбор границы `O` и `B`;
2. проверить, насколько горячие true `O` пересекаются по физике и меткам с
   обучающим `B`;
3. только после этого решать:
   - нужна ли ребалансировка;
   - нужны ли веса классов;
   - нужна ли очистка источника для самого горячего хвоста.

## Связанные документы

- [coarse_o_hot_subset_review_tz_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/archive_research/coarse_o_hot_subset_review_tz_ru.md)
- [coarse_o_tail_review_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/archive_research/coarse_o_tail_review_round1_ru.md)
- [stabilization_issue_ledger_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/stabilization/stabilization_issue_ledger_ru.md)
