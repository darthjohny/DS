# Repository State Policy

Дата актуализации: 13 марта 2026 года

## Зачем нужен этот документ

В репозитории есть несколько разных типов содержимого:

- production-код и tests;
- research-контур и notebooks;
- SQL/ADQL для preprocessing;
- versioned experiment artifacts;
- historical planning docs;
- локальный generated мусор.

Этот документ фиксирует, что именно считается каноническим
versioned current state проекта, а что является historical или purely
local.

## Что должно считаться versioned current state

### 1. Production и совместимые фасады

- весь пакет `src/router_model/`;
- весь пакет `src/host_model/`;
- весь пакет `src/priority_pipeline/`;
- весь пакет `src/decision_calibration/`;
- верхнеуровневые фасады в `src/*.py`, если они ещё поддерживают старые
  CLI/import entrypoint-ы;
- `src/input_layer.py`.

### 2. Research-layer

- весь пакет `analysis/model_comparison/`;
- пакеты `analysis/host_eda/` и `analysis/router_eda/`.

### 3. Tests

К versioned current state относятся:

- все production-тесты в `tests/`;
- comparison-layer tests `tests/test_model_comparison_*.py`;
- unit-test на branch-логику `tests/test_input_layer.py`;
- integration-tests, если они отражают текущий канонический runtime.

### 4. Docs

К текущему состоянию относятся:

- `README.md`;
- `docs/documentation_style_ru.md`;
- `docs/preprocessing_pipeline_ru.md`;
- `docs/model_comparison_protocol_ru.md`;
- `docs/model_comparison_findings_ru.md`;
- `docs/vkr_requirements_traceability_ru.md`;
- `docs/notebook_review_2026-03-13_ru.md`;
- `docs/presentation/`;
- `docs/assets/README.md` и осмысленно названные doc-assets.

### 5. Notebooks

К текущему состоянию относятся:

- `notebooks/eda/00_data_extraction_and_preprocessing.ipynb`;
- `notebooks/eda/01_host_eda_overview.ipynb`;
- `notebooks/eda/02_router_readiness.ipynb`;
- `notebooks/eda/03_host_vs_field_contrastive.ipynb`;
- `notebooks/eda/04_model_comparison_summary.ipynb`.

### 6. SQL и ADQL

К текущему состоянию относятся:

- все файлы в `sql/preprocessing/`;
- все файлы в `sql/adql/`;
- актуальные schema/data-change SQL-файлы в корне `sql/`, если они
  соответствуют текущему runtime-контракту.

### 7. Versioned experiment artifacts

Не все generated artifacts являются мусором. В текущем проекте осознанно
versioned и нужны для воспроизводимости:

- каноническая волна `experiments/model_comparison/` для `vkr30_cv10`;
- текущая полная QA-волна в `experiments/QA/`;
- журналы в `experiments/Логи работы программы/` и
  `experiments/Логи калибровки decision_layer/`, если на них есть ссылки
  из README/docs или они поддерживают narrative ВКР.

## Что считается historical, но допустимо оставлять в git

- planning-документы с явной пометкой, что это historical/TZ;
- более ранние comparison-waves в `experiments/model_comparison/`, если
  они сохраняются как исследовательская история;
- ранние QA-срезы вроде `qa_mvp_report_2026-03-11.md`.

Такие файлы не должны маскироваться под current state. Для них нужна
либо явная пометка внутри документа, либо README-индекс каталога.

## Что считается local-only и не должно засорять git

- `.DS_Store`;
- `__pycache__/`, `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`,
  `.pyright/`;
- `.ipynb_checkpoints/`;
- одноразовые screenshots с generic-именами вроде
  `docs/assets/Снимок экрана *.png`;
- локальные временные notebooks вида `Untitled*.ipynb`;
- локальные duplicate copies проекта.

## Практическое правило

Если файл:

1. участвует в текущем production/research-контуре;
2. нужен для воспроизводимости результатов;
3. на него есть прямая ссылка из README, docs, notebooks или CLI;

то он должен считаться частью versioned current state.

Если файл:

1. generated автоматически;
2. не имеет осмысленного имени;
3. не используется в ссылках и narrative проекта;
4. легко регенерируется локально;

то он должен быть либо ignored, либо удалён, либо вынесен во внешний
архив вне репозитория.

## Связанный operational manifest

Для текущего рабочего дерева отдельная практическая раскладка
`must track / historical / local working outputs / noise` зафиксирована в:

- `experiments/QA/qa_tracking_manifest_2026-03-13_ru.md`
