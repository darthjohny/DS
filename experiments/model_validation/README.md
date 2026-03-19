# Model Validation Artifacts

Каталог для heavy validation артефактов generalization-layer.

Канонический источник кода:
- `analysis.model_validation`

Текущий heavy validation слой уже пишет:
- markdown report по repeated split run;
- `repeated_splits.csv` с long-form generalization diagnostics;
- `model_summary.csv` с агрегированной устойчивостью моделей;
- `generalization_summary.csv` с явным разделением `train/cv/test`;
- `gap_diagnostics.csv` с aggregated train-test и cv-test gap сигналами;
- `risk_audit.csv` с per-model stability verdict и `risk_level`;
- каталог под optional plots и следующую волну diagnostics.

Канонические артефакты одного run:
- `*_validation_report.md`
- `*_repeated_splits.csv`
- `*_model_summary.csv`
- `*_generalization_summary.csv`
- `*_gap_diagnostics.csv`
- `*_risk_audit.csv`
- каталог `*_plots/`

Этот каталог не смешивается с:
- `experiments/model_comparison/` — benchmark и snapshot;
- `experiments/QA/` — audit и backlog.
