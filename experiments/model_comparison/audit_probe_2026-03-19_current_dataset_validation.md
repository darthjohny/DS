# Benchmark Dataset Validation

Дата: 2026-03-19 10:38:54 MSK
Protocol: `baseline_host_vs_field_v1`

## Источники benchmark
- host view: `lab.v_nasa_gaia_train_dwarfs`
- field view: `lab.v_gaia_ref_mkgf_dwarfs`
- test_size: `0.30`
- cv_folds: `10`

## Summary
 full_rows  train_rows  test_rows  n_classes  n_stratify_labels  full_host_share  train_host_share  test_host_share  max_abs_label_share_gap  max_abs_feature_smd  error_count  warning_count
     15394       10775       4619          4                  8         0.220476          0.220418         0.220611                 0.000187             0.016651            0              0

## Errors
- Пусто

## Warnings
- Пусто

## Stratify Balance
scope_name stratify_label spec_class  is_host  n_rows    share
      full            F|0          F    False    3000 0.194881
      test            F|0          F    False     900 0.194847
     train            F|0          F    False    2100 0.194896
      full            F|1          F     True     412 0.026764
      test            F|1          F     True     124 0.026846
     train            F|1          F     True     288 0.026729
      full            G|0          G    False    3000 0.194881
      test            G|0          G    False     900 0.194847
     train            G|0          G    False    2100 0.194896
      full            G|1          G     True    1831 0.118942
      test            G|1          G     True     550 0.119073
     train            G|1          G     True    1281 0.118886
      full            K|0          K    False    3000 0.194881
      test            K|0          K    False     900 0.194847
     train            K|0          K    False    2100 0.194896
      full            K|1          K     True     927 0.060218
      test            K|1          K     True     278 0.060186
     train            K|1          K     True     649 0.060232
      full            M|0          M    False    3000 0.194881
      test            M|0          M    False     900 0.194847
     train            M|0          M    False    2100 0.194896
      full            M|1          M     True     224 0.014551
      test            M|1          M     True      67 0.014505
     train            M|1          M     True     157 0.014571

## Feature Drift
  feature_name  train_mean   test_mean   train_std   test_std  pooled_std  abs_standardized_mean_diff
  logg_gspphot    4.525049    4.520805    0.254950   0.254825    0.254887                    0.016651
radius_gspphot    0.813791    0.817778    0.395109   0.395229    0.395169                    0.010088
  teff_gspphot 5141.631172 5136.943636 1009.091467 996.995181 1003.061558                    0.004673

## Примечание
Audit probe run for current benchmark reproducibility
