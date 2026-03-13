# Promotion Wave 2: current-state docs and presentation

Дата фиксации: 13 марта 2026 года

## Назначение

Этот документ фиксирует вторую promotion wave для versioned current
state проекта.

Wave 2 отделена от code/test wave сознательно:

- code и docs не смешиваются в один большой promotion шаг;
- current-state narrative должен проверяться отдельно от runtime-кода;
- presentation assets и repo-policy удобнее валидировать как единый
  documentation block.

## Состав wave

### Current-state docs

- `docs/model_comparison_protocol_ru.md`
- `docs/model_comparison_findings_ru.md`
- `docs/preprocessing_pipeline_ru.md`
- `docs/notebook_review_2026-03-13_ru.md`
- `docs/vkr_requirements_traceability_ru.md`
- `docs/repository_state_policy_ru.md`
- `docs/assets/README.md`
- `data/README.md`

### Presentation materials

- `docs/presentation/vkr_slides_draft_ru.md`
- `docs/presentation/assets/baseline_comparison_2026-03-13_vkr30_cv10/*`

### Canonical doc-assets

- `docs/assets/gaia_archive_crossmatch_ui.png`
- `docs/assets/gaia_archive_validation_ui.png`

## Что проверялось

Для этой wave проверялись не lint/typing-метрики, а narrative- и
artifact-consistency:

1. отсутствие устаревших current-state ссылок на старые comparison waves;
2. корректность run-name для benchmark и snapshot-family;
3. наличие всех локальных markdown targets;
4. наличие presentation-assets, на которые ссылается slide-source;
5. согласованность repo-policy, traceability и notebook-review.

## Зафиксированные правки внутри wave

В рамках подготовки wave были выровнены два точечных current-state
рассинхрона:

- `docs/notebook_review_2026-03-13_ru.md`:
  snapshot-family уточнена до
  `baseline_comparison_2026-03-13_vkr30_cv10_limit5000_snapshot`;
- `docs/presentation/vkr_slides_draft_ru.md`:
  тот же snapshot-family приведён к точному current-state имени.

## Проверки

Выполнены следующие проверки:

```bash
./venv/bin/python - <<'PY'
# link-check для current-state docs wave
PY

rg -n --pcre2 \
  "baseline_comparison_2026-03-13_vkr30_cv10_limit5000(?!_snapshot)" \
  docs/model_comparison_protocol_ru.md \
  docs/model_comparison_findings_ru.md \
  docs/preprocessing_pipeline_ru.md \
  docs/notebook_review_2026-03-13_ru.md \
  docs/vkr_requirements_traceability_ru.md \
  docs/repository_state_policy_ru.md \
  docs/assets/README.md \
  docs/presentation/vkr_slides_draft_ru.md \
  data/README.md

find docs/presentation/assets -maxdepth 3 -type f | sort
find docs/assets -maxdepth 2 -type f | sort
```

Результат:

- локальные markdown-ссылки: `green`
- stale snapshot-prefix references: `not found`
- presentation asset set: `present`
- canonical Gaia doc-assets: `present`

## Что важно в этой wave

- docs теперь согласованы с канонической comparison-wave `vkr30_cv10`;
- current-state narrative не ссылается на устаревшую snapshot-family;
- notebook review, traceability, preprocessing narrative и slides не
  спорят друг с другом по главному current-state контуру;
- generic screenshots в `docs/assets/` не считаются частью этой wave и
  не должны трактоваться как canonical doc-assets.

## Статус

Статус wave: `promotion-ready`.

Это означает:

- docs/presentation block можно продвигать как осмысленный current-state
  слой;
- для него не требуется дополнительная волна содержательных правок перед
  консолидацией;
- следующий логичный шаг уже лежит вне docs: notebooks и SQL/ADQL.

## Что не входит в wave

В эту wave сознательно не включены:

- notebooks;
- SQL/ADQL;
- generated comparison CSV/markdown в `experiments/model_comparison/`;
- полный QA-архив;
- generic screenshots `docs/assets/Снимок экрана *.png`.

## Следующий кандидат на wave

Следующий естественный блок:

- `notebooks/eda/00*` и `04*`;
- `sql/preprocessing/*`;
- `sql/adql/*`;
- актуальные `sql/2026-03-13_*`.
