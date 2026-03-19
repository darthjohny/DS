# production_priority_2026-03-19_v1_calibrated_limit5000

Дата: 2026-03-19 14:55:40 MSK
Источник: `public.gaia_dr3_training`
limit: `5000`
run_id: `gaia_pipeline_20260319T115539Z_f5dc32c6`

## Operational semantics
- Это production-like result `run_pipeline()`.
- Этот артефакт не является comparison snapshot.
- Shortlist строится только из runtime `priority_results`.

## Tier summary
priority_tier  n_rows  top_final_score
         HIGH     172         0.777932
       MEDIUM    1266         0.499204
          LOW    3562         0.299845

## Class summary
predicted_spec_class  n_rows  high_rows  medium_rows  low_rows  top_final_score
                   K    1647        144          836       667         0.737914
                   M     799         28          275       496         0.777932
                   G    1512          0          128      1384         0.451426
                   F     859          0           27       832         0.365770
                   A     143          0            0       143         0.000000
                   B      11          0            0        11         0.000000
             UNKNOWN      29          0            0        29         0.000000

## Shortlist summary
 observation_priority  n_rows
                    1     144
                    2      28

## Note
Wave 5 canonical operational export after orchestrator calibration.