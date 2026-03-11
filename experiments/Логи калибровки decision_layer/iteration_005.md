# iteration_005

Дата: 2026-03-08 01:21:21 MSK
Идентификатор запуска: decision_calibration_20260307T222121Z
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
- input_rows: 50
- router_rows: 50
- host_rows: 40
- low_rows: 10
- final_score_min: 0.0
- final_score_mean: 0.2319760758011357
- final_score_max: 0.7624654256187785

## Итог
- повторная проверка типизации

## Сводка по top-5
```text
predicted_spec_class  count
                   M      3
                   G      1
                   K      1
```

## Кандидаты top-5
```text
        source_id predicted_spec_class predicted_evolution_stage gauss_label  router_similarity similarity class_prior distance_factor quality_factor metallicity_factor  final_score priority_tier  reason_code        ra       dec  teff_gspphot  logg_gspphot  radius_gspphot  mh_gspphot  parallax  parallax_over_error     ruwe
48883844590606720                    G                     dwarf           G           0.475317   0.756414        1.05            0.96            1.0                1.0     0.762465          HIGH HOST_SCORING 63.892180 20.133188     5566.4590        4.3652          1.0532      0.1828 11.011924           562.762630 1.048185
54884979056532608                    K                     dwarf           K           0.498496   0.759656        1.08             0.8            1.0                1.0     0.656343          HIGH HOST_SCORING 52.642540 17.457626     4848.5320        4.5430          0.7867     -0.0226  2.071356            82.686760 1.022895
 2914878340241536                    M                     dwarf       M_MID           0.294797   0.439216        1.02             0.9            1.0                1.0     0.403201        MEDIUM HOST_SCORING 48.899395  4.319973     3336.3276        4.9195          0.3432     -0.0945  9.057717           212.990550 0.978546
 4661035948722944                    M                     dwarf       M_MID           0.316742   0.455788        1.02             0.9            1.0               0.96     0.401677        MEDIUM HOST_SCORING 43.032627  3.351530     3371.7078        4.8989          0.3257     -0.3061  5.200450            53.667700 1.050651
37099863080963840                    M                     dwarf       M_MID           0.344057    0.51428        1.02             0.8           0.97               0.96     0.390781        MEDIUM HOST_SCORING 57.494404 12.662119     3394.8743        4.8202          0.3648     -0.3467  2.863103            14.560399 1.000001
```
