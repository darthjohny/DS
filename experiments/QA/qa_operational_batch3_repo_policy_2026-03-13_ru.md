# Operational Batch 3 Status

Дата фиксации: 13 марта 2026 года

## Что проверялось

Батч 3 из
[qa_operational_consolidation_plan_2026-03-13_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/experiments/QA/qa_operational_consolidation_plan_2026-03-13_ru.md):
repo policy и shared hygiene.

Проверяемый слой:

- `.gitignore`
- `README.md`
- `requirements.txt`
- `analysis/host_eda/__init__.py`
- `analysis/router_eda/__init__.py`
- `data/README.md`
- `docs/repository_state_policy_ru.md`
- `experiments/Логи работы программы/README.md`
- `experiments/Логи калибровки decision_layer/README.md`

## Acceptance-check

### Ignore policy

Проверено через `git check-ignore -v`:

- `.mypy_cache/*`
- `.ruff_cache/*`
- `.pyright/*`
- `docs/assets/Снимок экрана *.png`
- `experiments/Логи калибровки decision_layer/iteration_*.md`
- historical comparison-wave `baseline_comparison_2026-03-13_*`

Результат:

- все перечисленные пути действительно матчятся актуальными правилами
  `.gitignore`.

### Existence check

Проверено существование всех файлов батча.

Результат:

- missing paths не обнаружены.

### README linkage

Проверены ключевые ссылки из `README.md` на:

- `docs/repository_state_policy_ru.md`
- `experiments/QA/README.md`
- `experiments/model_comparison/README.md`
- historical planning docs:
  - `docs/documentation_audit_tz_ru.md`
  - `docs/preprocessing_and_comparison_tz_ru.md`
  - `docs/ood_unknown_tz_ru.md`
  - `docs/ood_unknown_baselines_tz_ru.md`

Результат:

- current-state ссылки валидны;
- ссылки на historical planning docs всё ещё присутствуют.

## Итог

Батч 3 считается operationally ready с важной оговоркой:

- shared hygiene и ignore-policy сейчас консистентны;
- README и repo-policy документы находятся в рабочем состоянии;
- но наличие historical links в `README.md` означает, что Батч 7 не
  является опциональной косметикой: его нужно либо осознанно продвинуть,
  либо убрать эти ссылки из current-state narrative.

## Следующий шаг

Следующий практический шаг по plan-order:

- Батч 4: current-state docs и presentation.
