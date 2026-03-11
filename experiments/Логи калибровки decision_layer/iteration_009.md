# iteration_009

Дата: 2026-03-08 03:17:41 MSK
Идентификатор запуска: decision_calibration_20260308T001741Z
Статус: выполнено

## Что меняем
- class_prior
- metallicity_factor
- distance_factor
- quality_factor

## Формула
`final_score = similarity × class_prior × distance_factor × quality_factor × metallicity_factor`

## Параметры итерации
### class_prior
- K: 1.080
- G: 1.050
- M: 1.020
- F: 0.970

### metallicity_factor
- mh <= -0.30: 0.96
- mh < 0.20: 1.00
- mh < 0.50: 1.03
- mh >= 0.50: 1.05

### distance_factor
- distance <= 50 pc: 1.00
- distance <= 100 pc: 0.96
- distance <= 200 pc: 0.90
- distance <= 500 pc: 0.80
- distance > 500 pc: 0.65
- invalid distance: 0.70

### quality_factor
- quality_factor = ruwe_factor × parallax_precision_factor
- ruwe <= 1.1: 1.00
- ruwe <= 1.2: 0.98
- ruwe <= 1.4: 0.93
- ruwe <= 1.8: 0.80
- ruwe > 1.8: 0.60
- parallax_over_error >= 20: 1.00
- parallax_over_error >= 10: 0.97
- parallax_over_error >= 5: 0.90
- parallax_over_error >= 3: 0.75
- parallax_over_error < 3: 0.55

## Фактический результат
- relation: `public.gaia_dr3_training`
- source_name: `Gaia DR3 random 20k sample`
- input_rows: 20
- router_rows: 20
- host_rows: 18
- low_rows: 2
- final_score_min: 0.0
- final_score_mean: 0.2511113633017386
- final_score_max: 0.40320060170697625

## Итог
- smoke-проверка типизации

## Сводка по top-5
```text
predicted_spec_class  count
                   M      5
```

## Кандидаты top-5
```text
        source_id predicted_spec_class predicted_evolution_stage gauss_label  router_similarity similarity class_prior distance_factor quality_factor metallicity_factor  final_score priority_tier  reason_code        ra      dec  teff_gspphot  logg_gspphot  radius_gspphot  mh_gspphot  parallax  parallax_over_error     ruwe
 2914878340241536                    M                     dwarf       M_MID           0.294797   0.439216        1.02             0.9            1.0                1.0     0.403201        MEDIUM HOST_SCORING 48.899395 4.319973     3336.3276        4.9195          0.3432     -0.0945  9.057717           212.990550 0.978546
 4661035948722944                    M                     dwarf       M_MID           0.316742   0.455788        1.02             0.9            1.0               0.96     0.401677        MEDIUM HOST_SCORING 43.032627 3.351530     3371.7078        4.8989          0.3257     -0.3061  5.200450            53.667700 1.050651
22754771094131200                    M                     dwarf     M_EARLY           0.667824   0.484093        1.02             0.8            1.0               0.96     0.379219        MEDIUM HOST_SCORING 36.674683 8.477222     3649.5427        4.6669          0.4708     -0.6870  2.786661            36.221977 1.062366
20599968821903872                    M                     dwarf       M_MID           0.284509   0.410946        1.02             0.9            1.0                1.0     0.377248        MEDIUM HOST_SCORING 41.992140 8.477943     3314.6174        4.9332          0.3300     -0.1044  7.925336           106.501470 1.020755
 7092232251988480                    M                     dwarf       M_MID           0.294268   0.454606        1.02             0.8            1.0                1.0     0.370959        MEDIUM HOST_SCORING 44.542683 5.956296     3321.5322        4.8619          0.3903      0.0791  4.275059            40.916237 0.931582
```
