# Микро-ТЗ на интерфейс Streamlit

Связанные документы:

- [streamlit_interface_tz_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/plans/streamlit_interface_tz_ru.md)
- [regression_testing_tz_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/plans/regression_testing_tz_ru.md)
- [project_polish_tz_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/plans/project_polish_tz_ru.md)

## Общий инвариант для всех шагов

Для каждого шага:

- интерфейс не дублирует логику `posthoc` и `reporting`;
- `1 файл = 1 ответственность`;
- все подписи, заголовки и пользовательский текст — на русском языке;
- состояние интерфейса хранится централизованно;
- тяжелые операции чтения и загрузки кэшируются только через стандартные
  механизмы `Streamlit`;
- после каждого локального шага:
  - `ruff`
  - targeted `mypy`
  - targeted `pyright`
  - targeted `pytest`

## Предлагаемое дерево

```text
streamlit_app.py
src/exohost/ui/
├── __init__.py
├── contracts.py
├── session_state.py
├── loaders.py
├── run_service.py
├── formatting.py
├── components/
│   ├── overview_metrics.py
│   ├── model_metrics.py
│   ├── run_summary.py
│   ├── priority_table.py
│   └── star_card.py
└── pages/
    ├── home_page.py
    ├── metrics_page.py
    ├── run_browser_page.py
    ├── csv_decide_page.py
    └── candidate_page.py

tests/unit/ui/
├── test_ui_loaders.py
├── test_ui_run_service.py
├── test_ui_session_state.py
├── test_ui_components.py
└── test_ui_metrics.py
```

## Порядок работы

### UI-01. Зафиксировать контракт интерфейса

- Цель: не строить UI поверх расплывчатых входов и выходов.
- Что делаем:
  - определяем, какие артефакты читает интерфейс;
  - определяем, какой `CSV` он принимает;
  - определяем минимальное состояние интерфейса.
- Статус:
  - completed.

### UI-02. Подготовить каркас `streamlit_app.py`

- Цель: сразу собрать правильную multipage-структуру.
- Что делаем:
  - создаем entrypoint;
  - используем `st.Page` и `st.navigation`;
  - задаем единый `page_config`.
- Статус:
  - completed.

### UI-03. Вынести contracts и session state

- Цель: не размазывать состояние по страницам.
- Что делаем:
  - создаем `contracts.py`;
  - создаем `session_state.py`;
  - фиксируем ключи и правила инициализации состояния.
- Статус:
  - completed.

### UI-04. Собрать слой загрузки артефактов

- Цель: открыть готовый `run_dir` без notebook и ручного парсинга.
- Что делаем:
  - читаем `metadata.json`;
  - читаем `decision_input.csv`, `final_decision.csv`, `priority_ranking.csv`;
  - валидируем обязательные файлы и колонки;
  - кэшируем чтение через `st.cache_data`.
- Статус:
  - completed.

### UI-05. Собрать главную страницу

- Цель: дать короткую и понятную витрину проекта.
- Что делаем:
  - краткое описание проекта;
  - схема системы;
  - главный прикладной результат;
  - короткие итоговые метрики по выбранному запуску.
- Статус:
  - completed.

### UI-06. Собрать страницу метрик моделей

- Цель: показать доверие к результату без notebook.
- Что делаем:
  - выводим метрики по слоям `ID/OOD`, `coarse`, `host`, `refinement`;
  - показываем размер тестового среза и названия артефактов;
  - отдельно показываем калибровочные метрики, где они уже есть;
  - формируем краткую поясняющую подпись для каждого слоя.
- Статус:
  - completed.

### UI-07. Собрать страницу просмотра готового запуска

- Цель: показать итоговые артефакты без CLI.
- Что делаем:
  - выбор `run_dir`;
  - summary по `final_decision`;
  - summary по `priority`;
  - верхние кандидаты и фильтры.
- Статус:
  - completed.

### UI-08. Собрать карточку объекта

- Цель: показать подробный результат по одной звезде.
- Что делаем:
  - поиск по `source_id`;
  - вывод класса, подкласса, причин решения;
  - ключевые физические параметры;
  - сигналы приоритета и качества.
- Статус:
  - completed.

### UI-09. Собрать страницу запуска по внешнему CSV

- Цель: дать простой демонстрационный вход для внешнего пользователя.
- Что делаем:
  - `file_uploader`;
  - валидация входного контракта;
  - временное сохранение файла;
  - запуск существующего `decide`;
  - показ пути к новому `run_dir`.
- Статус:
  - completed.

### UI-10. Вынести повторяющиеся блоки в `components`

- Цель: не плодить копипасту по страницам.
- Что делаем:
  - карточки summary;
  - таблицы top-кандидатов;
  - метрики и маленькие status-блоки;
  - отдельный блок сводки качества моделей.
- Статус:
  - completed.

### UI-11. Добавить тесты helper-слоя

- Цель: UI не должен быть черным ящиком.
- Что делаем:
  - тестируем loaders;
  - тестируем run_service;
  - тестируем форматирование, state helper и метрики.
- Статус:
  - completed.

### UI-12. Добавить smoke-сценарий

- Цель: убедиться, что entrypoint собирается и страницы связаны.
- Что делаем:
  - минимальный smoke на `streamlit_app.py`;
  - проверяем, что импорты и маршрутизация не ломаются.
- Статус:
  - completed.

### UI-13. Довести русский пользовательский слой

- Цель: интерфейс должен быть демонстрационным, а не техническим.
- Что делаем:
  - все заголовки, подписи и подсказки на русском;
  - без смешения русского и английского в обычной прозе;
  - технические идентификаторы только там, где они нужны.
- Статус:
  - completed.

### UI-14. Провести scoped QA

- Цель: не занести в проект новый хрупкий слой.
- Что делаем:
  - `ruff`
  - `mypy`
  - `pyright`
  - `pytest`
  - smoke по интерфейсу
- Статус:
  - completed.
- QA-фиксация от `2026-04-26`:
  - `venv/bin/python -m pytest tests/unit/ui tests/smoke/ui -q` — `79 passed`;
  - `venv/bin/python -m pytest tests/smoke/ui/test_streamlit_pages_smoke.py -q` — `7 passed`;
  - `venv/bin/python -m ruff check src/exohost/ui tests/unit/ui tests/smoke/ui` — passed;
  - `venv/bin/python -m mypy src/exohost/ui tests/unit/ui tests/smoke/ui` — no issues;
  - `venv/bin/python -m pyright src/exohost/ui tests/unit/ui tests/smoke/ui` — environment gap:
    пакет `pyright` не установлен в `venv`, отдельный executable также не найден.

## Блоки исполнения

```text
Блок A. Каркас
├── UI-01
├── UI-02
└── UI-03

Блок B. Read-only интерфейс
├── UI-04
├── UI-05
├── UI-06
├── UI-07
└── UI-08

Блок C. Запуск по внешнему CSV
├── UI-09
└── UI-10

Блок D. Качество и стабилизация
├── UI-11
├── UI-12
├── UI-13
└── UI-14
```

## Логика выполнения

1. Сначала делаем read-only интерфейс для готовых запусков.
2. Затем добавляем отдельную страницу качества моделей.
3. Потом добавляем запуск по внешнему `CSV`.
4. После этого выносим повторяющиеся элементы и закрепляем тесты.
5. Только в конце полируем визуальный и демонстрационный слой.
