# Model Comparison Report

Дата: 2026-03-19 09:06:59 MSK
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

## Threshold Selection
              model_name threshold_metric threshold_source_split  threshold_value  threshold_score  n_rows  n_host  n_field
baseline_legacy_gaussian               f1                  train         0.383947         0.601956   10775    2375     8400
      baseline_mlp_small               f1                  train         0.356746         0.741814   10775    2375     8400
  baseline_random_forest               f1                  train         0.613333         0.936877   10775    2375     8400
     main_contrastive_v1               f1                  train         0.498734         0.630831   10775    2375     8400

## Threshold-based Quality
              model_name split_name quality_scope spec_class threshold_metric  threshold_value  n_rows  n_host  n_field   tp   fp   tn  fn  precision   recall       f1  specificity  balanced_accuracy  accuracy
baseline_legacy_gaussian       test       overall       None               f1         0.383947    4619    1019     3600  717  679 2921 302   0.513610 0.703631 0.593789     0.811389           0.757510  0.787616
      baseline_mlp_small       test       overall       None               f1         0.356746    4619    1019     3600  799  379 3221 220   0.678268 0.784102 0.727355     0.894722           0.839412  0.870318
  baseline_random_forest       test       overall       None               f1         0.613333    4619    1019     3600  733  289 3311 286   0.717221 0.719333 0.718275     0.919722           0.819527  0.875514
     main_contrastive_v1       test       overall       None               f1         0.498734    4619    1019     3600  883  948 2652 136   0.482250 0.866536 0.619649     0.736667           0.801601  0.765317
baseline_legacy_gaussian      train       overall       None               f1         0.383947   10775    2375     8400 1662 1485 6915 713   0.528122 0.699789 0.601956     0.823214           0.761502  0.796009
      baseline_mlp_small      train       overall       None               f1         0.356746   10775    2375     8400 1869  795 7605 506   0.701577 0.786947 0.741814     0.905357           0.846152  0.879258
  baseline_random_forest      train       overall       None               f1         0.613333   10775    2375     8400 2256  185 8215 119   0.924211 0.949895 0.936877     0.977976           0.963935  0.971787
     main_contrastive_v1      train       overall       None               f1         0.498734   10775    2375     8400 2042 2057 6343 333   0.498170 0.859789 0.630831     0.755119           0.807454  0.778190

## Class-wise Quality
              model_name split_name quality_scope spec_class threshold_metric  threshold_value  n_rows  n_host  n_field   tp  fp   tn  fn  precision   recall       f1  specificity  balanced_accuracy  accuracy
baseline_legacy_gaussian       test     classwise          F               f1         0.383947    1024     124      900   99 213  687  25   0.317308 0.798387 0.454128     0.763333           0.780860  0.767578
baseline_legacy_gaussian       test     classwise          G               f1         0.383947    1450     550      900  353 170  730 197   0.674952 0.641818 0.657968     0.811111           0.726465  0.746897
baseline_legacy_gaussian       test     classwise          K               f1         0.383947    1178     278      900  216  76  824  62   0.739726 0.776978 0.757895     0.915556           0.846267  0.882852
baseline_legacy_gaussian       test     classwise          M               f1         0.383947     967      67      900   49 220  680  18   0.182156 0.731343 0.291667     0.755556           0.743449  0.753878
      baseline_mlp_small       test     classwise          F               f1         0.356746    1024     124      900    0   0  900 124   0.000000 0.000000 0.000000     1.000000           0.500000  0.878906
      baseline_mlp_small       test     classwise          G               f1         0.356746    1450     550      900  532 267  633  18   0.665832 0.967273 0.788732     0.703333           0.835303  0.803448
      baseline_mlp_small       test     classwise          K               f1         0.356746    1178     278      900  252 104  796  26   0.707865 0.906475 0.794953     0.884444           0.895460  0.889643
      baseline_mlp_small       test     classwise          M               f1         0.356746     967      67      900   15   8  892  52   0.652174 0.223881 0.333333     0.991111           0.607496  0.937952
  baseline_random_forest       test     classwise          F               f1         0.613333    1024     124      900   61  73  827  63   0.455224 0.491935 0.472868     0.918889           0.705412  0.867188
  baseline_random_forest       test     classwise          G               f1         0.613333    1450     550      900  408 128  772 142   0.761194 0.741818 0.751381     0.857778           0.799798  0.813793
  baseline_random_forest       test     classwise          K               f1         0.613333    1178     278      900  239  75  825  39   0.761146 0.859712 0.807432     0.916667           0.888189  0.903226
  baseline_random_forest       test     classwise          M               f1         0.613333     967      67      900   25  13  887  42   0.657895 0.373134 0.476190     0.985556           0.679345  0.943123
     main_contrastive_v1       test     classwise          F               f1         0.498734    1024     124      900  121 317  583   3   0.276256 0.975806 0.430605     0.647778           0.811792  0.687500
     main_contrastive_v1       test     classwise          G               f1         0.498734    1450     550      900  453 237  663  97   0.656522 0.823636 0.730645     0.736667           0.780152  0.769655
     main_contrastive_v1       test     classwise          K               f1         0.498734    1178     278      900  266 165  735  12   0.617169 0.956835 0.750353     0.816667           0.886751  0.849745
     main_contrastive_v1       test     classwise          M               f1         0.498734     967      67      900   43 229  671  24   0.158088 0.641791 0.253687     0.745556           0.693673  0.738366
baseline_legacy_gaussian      train     classwise          F               f1         0.383947    2388     288     2100  212 466 1634  76   0.312684 0.736111 0.438923     0.778095           0.757103  0.773032
baseline_legacy_gaussian      train     classwise          G               f1         0.383947    3381    1281     2100  835 370 1730 446   0.692946 0.651835 0.671762     0.823810           0.737822  0.758651
baseline_legacy_gaussian      train     classwise          K               f1         0.383947    2749     649     2100  515 155 1945 134   0.768657 0.793529 0.780895     0.926190           0.859859  0.894871
baseline_legacy_gaussian      train     classwise          M               f1         0.383947    2257     157     2100  100 494 1606  57   0.168350 0.636943 0.266312     0.764762           0.700852  0.755871
      baseline_mlp_small      train     classwise          F               f1         0.356746    2388     288     2100    0   0 2100 288   0.000000 0.000000 0.000000     1.000000           0.500000  0.879397
      baseline_mlp_small      train     classwise          G               f1         0.356746    3381    1281     2100 1231 568 1532  50   0.684269 0.960968 0.799351     0.729524           0.845246  0.817214
      baseline_mlp_small      train     classwise          K               f1         0.356746    2749     649     2100  585 206 1894  64   0.739570 0.901387 0.812500     0.901905           0.901646  0.901782
      baseline_mlp_small      train     classwise          M               f1         0.356746    2257     157     2100   53  21 2079 104   0.716216 0.337580 0.458874     0.990000           0.663790  0.944617
  baseline_random_forest      train     classwise          F               f1         0.613333    2388     288     2100  240  62 2038  48   0.794702 0.833333 0.813559     0.970476           0.901905  0.953936
  baseline_random_forest      train     classwise          G               f1         0.613333    3381    1281     2100 1278   0 2100   3   1.000000 0.997658 0.998828     1.000000           0.998829  0.999113
  baseline_random_forest      train     classwise          K               f1         0.613333    2749     649     2100  602 102 1998  47   0.855114 0.927581 0.889874     0.951429           0.939505  0.945798
  baseline_random_forest      train     classwise          M               f1         0.613333    2257     157     2100  136  21 2079  21   0.866242 0.866242 0.866242     0.990000           0.928121  0.981391
     main_contrastive_v1      train     classwise          F               f1         0.498734    2388     288     2100  259 686 1414  29   0.274074 0.899306 0.420114     0.673333           0.786319  0.700586
     main_contrastive_v1      train     classwise          G               f1         0.498734    3381    1281     2100 1038 508 1592 243   0.671410 0.810304 0.734347     0.758095           0.784200  0.777876
     main_contrastive_v1      train     classwise          K               f1         0.498734    2749     649     2100  630 379 1721  19   0.624381 0.970724 0.759952     0.819524           0.895124  0.855220
     main_contrastive_v1      train     classwise          M               f1         0.498734    2257     157     2100  115 484 1616  42   0.191987 0.732484 0.304233     0.769524           0.751004  0.766947

## Confusion Matrices
              model_name split_name quality_scope spec_class threshold_metric  threshold_value  actual_label  predicted_label  n_rows
baseline_legacy_gaussian       test       overall       None               f1         0.383947         False            False    2921
baseline_legacy_gaussian       test       overall       None               f1         0.383947         False             True     679
baseline_legacy_gaussian       test       overall       None               f1         0.383947          True            False     302
baseline_legacy_gaussian       test       overall       None               f1         0.383947          True             True     717
      baseline_mlp_small       test       overall       None               f1         0.356746         False            False    3221
      baseline_mlp_small       test       overall       None               f1         0.356746         False             True     379
      baseline_mlp_small       test       overall       None               f1         0.356746          True            False     220
      baseline_mlp_small       test       overall       None               f1         0.356746          True             True     799
  baseline_random_forest       test       overall       None               f1         0.613333         False            False    3311
  baseline_random_forest       test       overall       None               f1         0.613333         False             True     289
  baseline_random_forest       test       overall       None               f1         0.613333          True            False     286
  baseline_random_forest       test       overall       None               f1         0.613333          True             True     733
     main_contrastive_v1       test       overall       None               f1         0.498734         False            False    2652
     main_contrastive_v1       test       overall       None               f1         0.498734         False             True     948
     main_contrastive_v1       test       overall       None               f1         0.498734          True            False     136
     main_contrastive_v1       test       overall       None               f1         0.498734          True             True     883
baseline_legacy_gaussian      train       overall       None               f1         0.383947         False            False    6915
baseline_legacy_gaussian      train       overall       None               f1         0.383947         False             True    1485
baseline_legacy_gaussian      train       overall       None               f1         0.383947          True            False     713
baseline_legacy_gaussian      train       overall       None               f1         0.383947          True             True    1662
      baseline_mlp_small      train       overall       None               f1         0.356746         False            False    7605
      baseline_mlp_small      train       overall       None               f1         0.356746         False             True     795
      baseline_mlp_small      train       overall       None               f1         0.356746          True            False     506
      baseline_mlp_small      train       overall       None               f1         0.356746          True             True    1869
  baseline_random_forest      train       overall       None               f1         0.613333         False            False    8215
  baseline_random_forest      train       overall       None               f1         0.613333         False             True     185
  baseline_random_forest      train       overall       None               f1         0.613333          True            False     119
  baseline_random_forest      train       overall       None               f1         0.613333          True             True    2256
     main_contrastive_v1      train       overall       None               f1         0.498734         False            False    6343
     main_contrastive_v1      train       overall       None               f1         0.498734         False             True    2057
     main_contrastive_v1      train       overall       None               f1         0.498734          True            False     333
     main_contrastive_v1      train       overall       None               f1         0.498734          True             True    2042

## Hyperparameter Search
              model_name search_scope spec_class refit_metric  precision_k  cv_folds  n_train_rows  n_host  n_field  candidate_count  best_cv_score  cv_score_std  cv_score_min  cv_score_max                                                            best_params_json
baseline_legacy_gaussian        model        NaN      roc_auc           50        10         10775    2375     8400                6       0.843709           0.0           0.0           0.0                            {"shrink_alpha": 0.15, "use_m_subclasses": true}
      baseline_mlp_small        class          F      roc_auc           50        10          2388     288     2100                6       0.737762           0.0           0.0           0.0                              {"alpha": 0.01, "hidden_layer_sizes": [16, 8]}
      baseline_mlp_small        class          G      roc_auc           50        10          3381    1281     2100                6       0.904066           0.0           0.0           0.0                            {"alpha": 0.0001, "hidden_layer_sizes": [16, 8]}
      baseline_mlp_small        class          K      roc_auc           50        10          2749     649     2100                6       0.960198           0.0           0.0           0.0                             {"alpha": 0.0001, "hidden_layer_sizes": [8, 4]}
      baseline_mlp_small        class          M      roc_auc           50        10          2257     157     2100                6       0.904849           0.0           0.0           0.0                            {"alpha": 0.0001, "hidden_layer_sizes": [16, 8]}
  baseline_random_forest        class          F      roc_auc           50        10          2388     288     2100                6       0.861399           0.0           0.0           0.0                                {"min_samples_leaf": 4, "n_estimators": 300}
  baseline_random_forest        class          G      roc_auc           50        10          3381    1281     2100                6       0.902537           0.0           0.0           0.0                                {"min_samples_leaf": 1, "n_estimators": 300}
  baseline_random_forest        class          K      roc_auc           50        10          2749     649     2100                6       0.961211           0.0           0.0           0.0                                {"min_samples_leaf": 4, "n_estimators": 100}
  baseline_random_forest        class          M      roc_auc           50        10          2257     157     2100                6       0.923160           0.0           0.0           0.0                                {"min_samples_leaf": 4, "n_estimators": 300}
     main_contrastive_v1        model        NaN      roc_auc           50        10         10775    2375     8400                8       0.875410           0.0           0.0           0.0 {"min_population_size": 2, "shrink_alpha": 0.05, "use_m_subclasses": false}

## Generalization Diagnostics
              model_name    metric_name     train_scope   test_scope  train_value  test_value  train_minus_test  abs_train_test_gap  is_refit_metric    cv_summary_scope  cv_score_mean  cv_score_std  cv_score_min  cv_score_max  cv_minus_test
baseline_legacy_gaussian          brier in_sample_refit holdout_test     0.147836    0.148316         -0.000480            0.000480            False                 NaN            NaN           NaN           NaN           NaN            NaN
baseline_legacy_gaussian         pr_auc in_sample_refit holdout_test     0.572772    0.571297          0.001474            0.001474            False                 NaN            NaN           NaN           NaN           NaN            NaN
baseline_legacy_gaussian precision_at_k in_sample_refit holdout_test     0.740000    0.720000          0.020000            0.020000            False                 NaN            NaN           NaN           NaN           NaN            NaN
baseline_legacy_gaussian        roc_auc in_sample_refit holdout_test     0.845976    0.846410         -0.000433            0.000433             True               model       0.843709           0.0           0.0           0.0      -0.002701
      baseline_mlp_small          brier in_sample_refit holdout_test     0.086043    0.090139         -0.004096            0.004096            False                 NaN            NaN           NaN           NaN           NaN            NaN
      baseline_mlp_small         pr_auc in_sample_refit holdout_test     0.778179    0.755534          0.022645            0.022645            False                 NaN            NaN           NaN           NaN           NaN            NaN
      baseline_mlp_small precision_at_k in_sample_refit holdout_test     0.980000    0.920000          0.060000            0.060000            False                 NaN            NaN           NaN           NaN           NaN            NaN
      baseline_mlp_small        roc_auc in_sample_refit holdout_test     0.928655    0.922728          0.005927            0.005927             True class_weighted_mean       0.881694           0.0           0.0           0.0      -0.041034
  baseline_random_forest          brier in_sample_refit holdout_test     0.033557    0.090081         -0.056523            0.056523            False                 NaN            NaN           NaN           NaN           NaN            NaN
  baseline_random_forest         pr_auc in_sample_refit holdout_test     0.985460    0.761796          0.223663            0.223663            False                 NaN            NaN           NaN           NaN           NaN            NaN
  baseline_random_forest precision_at_k in_sample_refit holdout_test     1.000000    0.920000          0.080000            0.080000            False                 NaN            NaN           NaN           NaN           NaN            NaN
  baseline_random_forest        roc_auc in_sample_refit holdout_test     0.995977    0.932604          0.063373            0.063373             True class_weighted_mean       0.912709           0.0           0.0           0.0      -0.019895
     main_contrastive_v1          brier in_sample_refit holdout_test     0.146627    0.154324         -0.007698            0.007698            False                 NaN            NaN           NaN           NaN           NaN            NaN
     main_contrastive_v1         pr_auc in_sample_refit holdout_test     0.620876    0.590399          0.030477            0.030477            False                 NaN            NaN           NaN           NaN           NaN            NaN
     main_contrastive_v1 precision_at_k in_sample_refit holdout_test     0.780000    0.720000          0.060000            0.060000            False                 NaN            NaN           NaN           NaN           NaN            NaN
     main_contrastive_v1        roc_auc in_sample_refit holdout_test     0.876787    0.867370          0.009417            0.009417             True               model       0.875410           0.0           0.0           0.0       0.008040

## Per-model Generalization Audit
              model_name  audit_scope refit_metric  train_refit_score  test_refit_score  train_minus_test  abs_train_test_gap  cv_score_mean  cv_score_std  cv_score_min  cv_score_max  cv_minus_test  abs_cv_test_gap  test_classwise_min  test_classwise_max  test_classwise_range  test_classwise_std  test_brier calibration_status  risk_score risk_level                              risk_reasons
baseline_legacy_gaussian refit_metric      roc_auc           0.845976          0.846410         -0.000433            0.000433       0.843709           0.0           0.0           0.0      -0.002701         0.002701            0.802786            0.927274              0.124488            0.047526    0.148316              watch           1      WATCH class-wise test range for roc_auc = 0.124
      baseline_mlp_small refit_metric      roc_auc           0.928655          0.922728          0.005927            0.005927       0.881694           0.0           0.0           0.0      -0.041034         0.041034            0.776407            0.955396              0.178989            0.064544    0.090139               good           1      WATCH class-wise test range for roc_auc = 0.179
  baseline_random_forest refit_metric      roc_auc           0.995977          0.932604          0.063373            0.063373       0.912709           0.0           0.0           0.0      -0.019895         0.019895            0.887124            0.955122              0.067998            0.026177    0.090081               good           1      WATCH        train/test gap for roc_auc = 0.063
     main_contrastive_v1 refit_metric      roc_auc           0.876787          0.867370          0.009417            0.009417       0.875410           0.0           0.0           0.0       0.008040         0.008040            0.811526            0.933809              0.122283            0.044299    0.154324              watch           1      WATCH class-wise test range for roc_auc = 0.122

## Примечание
Quality layer refresh from existing saved score artifacts.
