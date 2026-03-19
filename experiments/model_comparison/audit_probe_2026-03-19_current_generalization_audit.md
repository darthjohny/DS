# Generalization Audit Report

Дата: 2026-03-19 10:41:09 MSK
Protocol: `baseline_host_vs_field_v1`

## Per-model verdict
              model_name  audit_scope refit_metric  train_refit_score  test_refit_score  train_minus_test  abs_train_test_gap  cv_score_mean  cv_score_std  cv_score_min  cv_score_max  cv_minus_test  abs_cv_test_gap  test_classwise_min  test_classwise_max  test_classwise_range  test_classwise_std  test_brier calibration_status  risk_score risk_level                              risk_reasons
baseline_legacy_gaussian refit_metric      roc_auc           0.845976          0.846410         -0.000433            0.000433       0.843709      0.012374      0.823319      0.864746      -0.002701         0.002701            0.802786            0.927274              0.124488            0.047526    0.148316              watch           1      WATCH class-wise test range for roc_auc = 0.124
      baseline_mlp_small refit_metric      roc_auc           0.928655          0.922728          0.005927            0.005927       0.881694      0.024581      0.677997      0.975751      -0.041034         0.041034            0.776407            0.955396              0.178989            0.064544    0.090139               good           1      WATCH class-wise test range for roc_auc = 0.179
  baseline_random_forest refit_metric      roc_auc           0.995977          0.932604          0.063373            0.063373       0.912709      0.022604      0.784565      0.983736      -0.019895         0.019895            0.887124            0.955122              0.067998            0.026177    0.090081               good           1      WATCH        train/test gap for roc_auc = 0.063
     main_contrastive_v1 refit_metric      roc_auc           0.876787          0.867370          0.009417            0.009417       0.875410      0.010147      0.864146      0.897333       0.008040         0.008040            0.811526            0.933809              0.122283            0.044299    0.154324              watch           1      WATCH class-wise test range for roc_auc = 0.122

## Примечание
Audit probe run for current benchmark reproducibility
