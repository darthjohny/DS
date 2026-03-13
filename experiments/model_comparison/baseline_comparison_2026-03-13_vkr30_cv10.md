# Model Comparison Report

Дата: 2026-03-13 12:32:25 MSK
Protocol: `baseline_host_vs_field_v1`

## Что сравниваем
- model: `baseline_legacy_gaussian`
- model: `baseline_mlp_small`
- model: `baseline_random_forest`
- model: `main_contrastive_v1`

## Источники benchmark
- host view: `lab.v_nasa_gaia_train_dwarfs`
- field view: `lab.v_gaia_ref_mkgf_dwarfs`
- features: `teff_gspphot, logg_gspphot, radius_gspphot`
- split: `train/test`
- random_state: `42`
- test_size: `0.30`
- cv_folds: `10`
- cv_random_state: `42`
- search_refit_metric: `roc_auc`
- precision@k: `50`

## Итоговые метрики
              model_name split_name  n_rows  n_host  n_field  effective_k  roc_auc   pr_auc    brier  precision_at_k
baseline_legacy_gaussian       test    4619    1019     3600           50 0.846410 0.571297 0.148316            0.72
      baseline_mlp_small       test    4619    1019     3600           50 0.922728 0.755534 0.090139            0.92
  baseline_random_forest       test    4619    1019     3600           50 0.932604 0.761796 0.090081            0.92
     main_contrastive_v1       test    4619    1019     3600           50 0.867370 0.590399 0.154324            0.72
baseline_legacy_gaussian      train   10775    2375     8400           50 0.845976 0.572772 0.147836            0.74
      baseline_mlp_small      train   10775    2375     8400           50 0.928655 0.778179 0.086043            0.98
  baseline_random_forest      train   10775    2375     8400           50 0.995977 0.985460 0.033557            1.00
     main_contrastive_v1      train   10775    2375     8400           50 0.876787 0.620876 0.146627            0.78

## Class-wise метрики
              model_name split_name spec_class  n_rows  n_host  n_field  effective_k  roc_auc   pr_auc    brier  precision_at_k
baseline_legacy_gaussian       test          F    1024     124      900           50 0.841496 0.340285 0.137059            0.30
baseline_legacy_gaussian       test          G    1450     550      900           50 0.822220 0.701825 0.189914            0.80
baseline_legacy_gaussian       test          K    1178     278      900           50 0.927274 0.790200 0.116924            0.86
baseline_legacy_gaussian       test          M     967      67      900           50 0.802786 0.244128 0.136105            0.34
      baseline_mlp_small       test          F    1024     124      900           50 0.776407 0.297167 0.101084            0.34
      baseline_mlp_small       test          G    1450     550      900           50 0.894840 0.773902 0.123029            0.80
      baseline_mlp_small       test          K    1178     278      900           50 0.955396 0.836176 0.072117            0.92
      baseline_mlp_small       test          M     967      67      900           50 0.886318 0.420101 0.051183            0.34
  baseline_random_forest       test          F    1024     124      900           50 0.887124 0.454630 0.099396            0.54
  baseline_random_forest       test          G    1450     550      900           50 0.896158 0.774714 0.122172            0.78
  baseline_random_forest       test          K    1178     278      900           50 0.955122 0.839429 0.075538            0.88
  baseline_random_forest       test          M     967      67      900           50 0.917065 0.575353 0.049813            0.58
     main_contrastive_v1       test          F    1024     124      900           50 0.863557 0.393108 0.202614            0.34
     main_contrastive_v1       test          G    1450     550      900           50 0.849048 0.696463 0.155538            0.70
     main_contrastive_v1       test          K    1178     278      900           50 0.933809 0.787895 0.105472            0.84
     main_contrastive_v1       test          M     967      67      900           50 0.811526 0.339432 0.160879            0.38
baseline_legacy_gaussian      train          F    2388     288     2100           50 0.819721 0.342384 0.134374            0.40
baseline_legacy_gaussian      train          G    3381    1281     2100           50 0.828521 0.693903 0.189911            0.72
baseline_legacy_gaussian      train          K    2749     649     2100           50 0.938389 0.811461 0.115567            0.88
baseline_legacy_gaussian      train          M    2257     157     2100           50 0.769803 0.209724 0.138356            0.28
      baseline_mlp_small      train          F    2388     288     2100           50 0.749272 0.317612 0.101502            0.56
      baseline_mlp_small      train          G    3381    1281     2100           50 0.899357 0.783487 0.118169            0.82
      baseline_mlp_small      train          K    2749     649     2100           50 0.960692 0.870184 0.066635            0.98
      baseline_mlp_small      train          M    2257     157     2100           50 0.915199 0.548024 0.045200            0.88
  baseline_random_forest      train          F    2388     288     2100           50 0.986538 0.915396 0.055147            1.00
  baseline_random_forest      train          G    3381    1281     2100           50 1.000000 1.000000 0.016316            1.00
  baseline_random_forest      train          K    2749     649     2100           50 0.990396 0.970869 0.042552            1.00
  baseline_random_forest      train          M    2257     157     2100           50 0.996078 0.942083 0.025587            0.98
     main_contrastive_v1      train          F    2388     288     2100           50 0.843940 0.359860 0.191720            0.30
     main_contrastive_v1      train          G    3381    1281     2100           50 0.849202 0.693853 0.153185            0.70
     main_contrastive_v1      train          K    2749     649     2100           50 0.949773 0.846641 0.098894            0.96
     main_contrastive_v1      train          M    2257     157     2100           50 0.865253 0.450870 0.147229            0.80

## Hyperparameter Search
              model_name search_scope spec_class refit_metric  precision_k  cv_folds  n_train_rows  n_host  n_field  candidate_count  best_cv_score                                                            best_params_json
baseline_legacy_gaussian        model        NaN      roc_auc           50        10         10775    2375     8400                6       0.843709                            {"shrink_alpha": 0.15, "use_m_subclasses": true}
      baseline_mlp_small        class          F      roc_auc           50        10          2388     288     2100                6       0.737762                              {"alpha": 0.01, "hidden_layer_sizes": [16, 8]}
      baseline_mlp_small        class          G      roc_auc           50        10          3381    1281     2100                6       0.904066                            {"alpha": 0.0001, "hidden_layer_sizes": [16, 8]}
      baseline_mlp_small        class          K      roc_auc           50        10          2749     649     2100                6       0.960198                             {"alpha": 0.0001, "hidden_layer_sizes": [8, 4]}
      baseline_mlp_small        class          M      roc_auc           50        10          2257     157     2100                6       0.904849                            {"alpha": 0.0001, "hidden_layer_sizes": [16, 8]}
  baseline_random_forest        class          F      roc_auc           50        10          2388     288     2100                6       0.861399                                {"min_samples_leaf": 4, "n_estimators": 300}
  baseline_random_forest        class          G      roc_auc           50        10          3381    1281     2100                6       0.902537                                {"min_samples_leaf": 1, "n_estimators": 300}
  baseline_random_forest        class          K      roc_auc           50        10          2749     649     2100                6       0.961211                                {"min_samples_leaf": 4, "n_estimators": 100}
  baseline_random_forest        class          M      roc_auc           50        10          2257     157     2100                6       0.923160                                {"min_samples_leaf": 4, "n_estimators": 300}
     main_contrastive_v1        model        NaN      roc_auc           50        10         10775    2375     8400                8       0.875410 {"min_population_size": 2, "shrink_alpha": 0.05, "use_m_subclasses": false}

## Примечание
-
