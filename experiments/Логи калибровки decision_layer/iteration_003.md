# iteration_003

Дата: 2026-03-08 01:09:06 MSK
Идентификатор запуска: decision_calibration_20260307T220905Z
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
- input_rows: 200
- router_rows: 200
- host_rows: 168
- low_rows: 32
- final_score_min: 0.0
- final_score_mean: 0.24025388352488114
- final_score_max: 0.7624654256187785

## Итог
- базовый preview-прогон

## Сводка по top-10
```text
predicted_spec_class  count
                   K      4
                   G      3
                   M      3
```

## Кандидаты top-10
```text
         source_id predicted_spec_class predicted_evolution_stage gauss_label  router_similarity similarity class_prior distance_factor quality_factor metallicity_factor  final_score priority_tier  reason_code        ra       dec  teff_gspphot  logg_gspphot  radius_gspphot  mh_gspphot  parallax  parallax_over_error     ruwe
 48883844590606720                    G                     dwarf           G           0.475317   0.756414        1.05            0.96            1.0                1.0     0.762465          HIGH HOST_SCORING 63.892180 20.133188     5566.4590        4.3652          1.0532      0.1828 11.011924           562.762630 1.048185
 54884979056532608                    K                     dwarf           K           0.498496   0.759656        1.08             0.8            1.0                1.0     0.656343          HIGH HOST_SCORING 52.642540 17.457626     4848.5320        4.5430          0.7867     -0.0226  2.071356            82.686760 1.022895
130957057548896384                    M                     dwarf       M_MID           0.331580   0.801747        1.02             0.8            1.0               0.96     0.628056          HIGH HOST_SCORING 35.527603 28.367382     3351.0880        4.4899          0.4522     -0.6104  4.035429            23.662992 0.977205
 59318278657519232                    K                     dwarf           K           0.562867   0.713712        1.08             0.8            1.0                1.0     0.616648          HIGH HOST_SCORING 50.223810 19.671267     4754.8115        4.5633          0.7496     -0.0741  2.307232            70.324250 1.072140
123265458316531584                    M                     dwarf       M_MID           0.350676   0.769061        1.02             0.8            1.0               0.96     0.602452          HIGH HOST_SCORING 46.441082 31.046682     3379.2422        4.5432          0.4874     -0.4614  3.557301            25.681345 1.071792
107818355033716224                    G                     dwarf           G           0.366599   0.667984        1.05             0.8           0.98                1.0     0.549884        MEDIUM HOST_SCORING 33.162290 28.258879     5690.4610        4.2922          1.2365     -0.2971  2.153379           140.290900 1.118119
109597536646000768                    K                     dwarf           K           0.504338   0.629077        1.08             0.8            1.0                1.0     0.543523        MEDIUM HOST_SCORING 42.538930 22.229538     4663.7217        4.5253          0.8108      0.0155  2.600858            73.562340 1.065383
153597341995289984                    M                     dwarf       M_MID           0.349761   0.692283        1.02             0.8            1.0               0.96     0.542307        MEDIUM HOST_SCORING 73.659000 26.767036     3369.6445        4.6395          0.4083     -0.4454  4.266183            29.340260 0.958028
117070814102205952                    G                     dwarf           G           0.421500   0.777581        1.05            0.65            1.0               0.96     0.509471        MEDIUM HOST_SCORING 45.208080 29.655820     5567.8250        4.3256          1.1381     -0.5622  0.720365            20.588390 0.977519
 63047547221876352                    K                     dwarf           K           0.540106   0.589233        1.08             0.8            1.0                1.0     0.509097        MEDIUM HOST_SCORING 50.512820 24.481660     4619.4863        4.5432          0.7657     -0.1404  3.211838           143.793260 1.045954
```
