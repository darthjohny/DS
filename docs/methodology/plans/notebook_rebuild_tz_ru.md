# ТЗ На Перестройку Notebook-Слоя

Дата фиксации: `2026-04-05`

Связанные документы:

- [analysis/notebooks/README.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/notebooks/README.md)
- [notebook_qa_policy_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/stabilization/notebook_qa_policy_ru.md)
- [project_cleanup_tz_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/plans/project_cleanup_tz_ru.md)

## Зачем Нужна Перестройка

Текущий слой notebook вырос органически:

- часть notebook посвящена исследованию данных и научным выводам;
- часть notebook посвящена техническому состоянию пайплайна и моделей;
- в текущем виде роли notebook уже в целом понятны, но не зафиксированы как
  отдельные контуры;
- пользователю сложно быстро понять:
  - где научное исследование;
  - где техническая диагностика;
  - где смотреть на качество моделей;
  - где смотреть на состояние данных и пайплайна.

Нужна чистая структура active notebook по рабочим каталогам:

1. `eda`;
2. `research`;
3. `technical`.

## Цель

Сделать notebook-слой:

- понятным;
- воспроизводимым;
- визуально аккуратным;
- полезным и для исследования, и для инженерной отладки;
- полностью русскоязычным в пользовательском выводе.

## Два Контура Notebook

### 1. Исследовательские Notebook

Это notebook для научной задачи и интерпретации данных.

В них живут:

- EDA;
- графики распределений;
- таблицы покрытий и пропусков;
- матрицы ошибок и сравнения классов;
- итоговые выводы по данным;
- выводы о том, насколько данным и результатам можно доверять.

Они должны отвечать на вопросы:

- что у нас за данные;
- как они распределены;
- где слабые места по классам и подклассам;
- как ведет себя исследовательская гипотеза;
- какие ограничения у результата.

### 2. Технические Notebook

Это notebook для инженерного состояния пайплайна и моделей.

В них живут:

- stage-level и end-to-end метрики;
- калибровки;
- пороги;
- разбиения train/test/cv;
- входы и выходы моделей;
- диагностика routing;
- speed/runtime и узкие места, если это важно;
- объяснение, почему модель приняла решение.

Они должны отвечать на вопросы:

- как работает пайплайн;
- какие данные приходят на этап;
- какие метрики у моделей;
- где именно ломается логика;
- какие настройки повлияли на результат.

## Базовые Правила Для Оба Контура

### Именование Notebook

Жесткой официальной схемы именования файлов Jupyter Notebook документация не
навязывает, поэтому в проекте фиксируется собственная конвенция, согласованная
с ролью notebook и структурой репозитория.

Нумерация в именах active notebook не используется.

Имена должны быть:

- короткими;
- предметными;
- читаемыми без знания внутренней истории проекта;
- одинаково устроенными внутри одного контура.

#### Исследовательский Контур

Для active notebook используем короткие предметные имена по роли файла и его
месту в каталоге.

Примеры:

- `router_training.ipynb`
- `host_training.ipynb`
- `label_coverage.ipynb`
- `quality_gate_calibration.ipynb`

#### Технический Контур

Для технических notebook фиксируем такой принцип:

- имя должно сразу говорить, какой слой пайплайна или настройки разбираются;
- в имени остается только полезная техническая роль;
- нумерация и переходные префиксы не используются.

Примеры:

- `model_pipeline_review.ipynb`
- `final_decision_review.ipynb`
- `host_priority_calibration_review.ipynb`
- `priority_threshold_review.ipynb`

#### Чего Не Делаем

Не используем:

- нумерацию ради порядка;
- имена, завязанные на случайную последовательность работ;
- неочевидные сокращения без контекста;
- смешение нескольких ролей в одном имени.

### Язык

Все пользовательские элементы notebook должны быть на русском:

- заголовки;
- markdown-описания;
- подписи графиков;
- названия сводных таблиц;
- комментарии в code-cell;
- выводы и next steps.

Английский допустим только там, где это нельзя переименовать без потери
смысла:

- имена столбцов;
- имена артефактов;
- API-контракты;
- имена файлов;
- названия метрик в сыром техническом виде, если рядом есть русское пояснение.

### Нагрузка На Notebook

Notebook не должен быть перегружен:

- не должно быть “свалки” из десятков графиков без цели;
- не должно быть длинных висячих ссылок вместо объяснения;
- не должно быть тяжелой бизнес-логики внутри notebook;
- повторяемая логика должна жить в `src/exohost/reporting`;
- notebook должен быть обзорным слоем, а не вторым приложением.

### Структура Каждого Notebook

Каждый активный notebook должен иметь понятную форму:

1. что он проверяет;
2. какие данные использует;
3. какой вопрос решает;
4. графики и таблицы;
5. короткие русские выводы;
6. следующий шаг.

### Техническая Честность

Если чего-то нет, notebook должен говорить об этом явно:

- какие поля пока недоступны;
- какие метрики отсутствуют;
- где данные неполные;
- где вывод ограничен текущим этапом проекта.

## Текущее Разложение По Будущим Контурам

### Исследовательский Контур

Сюда естественно относятся:

- `router_training.ipynb`
- `host_training.ipynb`
- `label_coverage.ipynb`
- `quality_gate_calibration.ipynb`
- `coarse_ob_domain_shift.ipynb`
- `secure_o_tail.ipynb`

### Технический Контур

Сюда естественно относятся:

- `scoring_review.ipynb`
- `model_pipeline_review.ipynb`
- `final_decision_review.ipynb`
- `host_priority_calibration_review.ipynb`
- `priority_threshold_review.ipynb`

### Архив Исследований

Глубокие разовые расследования остаются в архиве и не возвращаются в active
слой без отдельного решения:

- `analysis/notebooks/archive_research/*`

## Что Должно Появиться После Перестройки

## Текущий Статус

На текущем этапе закрыты шаги:

- `MTZ-NB01`
- `MTZ-NB02`
- `MTZ-NB03`
- `MTZ-NB04`
- `MTZ-NB05`
- `MTZ-NB06`
- `MTZ-NB07`
- `MTZ-NB08`
- `MTZ-NB09`
- `MTZ-NB10`

### Состояние Active-Слоя

Сейчас active notebook уже приведены к рабочей структуре:

- `analysis/notebooks/eda`
- `analysis/notebooks/research`
- `analysis/notebooks/technical`

Что уже получено:

- русские заголовки, пояснения и подписи выровнены;
- логика active notebook читается по ролям;
- явного смыслового перегруза не найдено;
- визуальный дефект с наложением длинных подписей на графике калибровки исправлен;
- smoke-QA и политика notebook QA уже знают новую структуру.

## Что Не Делаем В Этом Пакете

В этой волне не меняем:

- core train/inference логику без необходимости;
- формат артефактов только ради красоты notebook;
- исследовательский архив, если он не нужен активному контуру.

## Инженерный Стандарт Для Notebook-Слоя

- повторяемая логика сначала живет в `src/exohost/reporting`;
- один helper-модуль отвечает за одну роль;
- notebook остается тонким;
- комментарии в notebook и в helper-коде пишем на русском;
- все изменения проходят через:
  - `ruff`
  - `mypy`
  - `pyright`
  - `pytest`
  - адресный `nbclient`, если затронут active notebook.

## Критерий Завершения

Перестройку notebook-слоя считаем завершенной, когда:

- активные notebook разделены на исследовательский и технический контуры;
- человек без контекста понимает, куда идти за научным выводом, а куда за
  инженерной диагностикой;
- пользовательский вывод в активных notebook полностью русскоязычный;
- нет смешения ролей;
- нет висячих артефактных ссылок без интерпретации;
- notebook остаются обзорными и не превращаются в монолиты.

## Текущий Статус

- `MTZ-NB01` закрыт:
  - двухслойная структура notebook формально зафиксирована;
  - active notebook разложены на исследовательский и технический контуры;
  - конвенция именования без нумерации зафиксирована.
- `MTZ-NB02` закрыт:
  - [analysis/notebooks/README.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/notebooks/README.md)
    перестроен под два контура;
  - для каждого active notebook задано целевое имя и краткая роль;
  - архивный слой отделен явно.
- `MTZ-NB03` закрыт:
  - исследовательские notebook
    [router_training.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/notebooks/eda/router_training.ipynb),
    [host_training.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/notebooks/eda/host_training.ipynb)
    и
    [label_coverage.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/notebooks/eda/label_coverage.ipynb)
    приведены к русскому пользовательскому выводу;
  - убраны английские заголовки, подписи графиков, текстовые сообщения и
    лишний вывод пути;
  - для этих notebook добавлены корректные `cell id`, а сохраненные outputs
    обновлены через `nbclient`.
- `MTZ-NB04` закрыт:
  - исследовательские deep-dive notebook
    [quality_gate_calibration.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/notebooks/research/quality_gate_calibration.ipynb),
    [coarse_ob_domain_shift.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/notebooks/research/coarse_ob_domain_shift.ipynb)
    и
    [secure_o_tail.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/notebooks/research/secure_o_tail.ipynb)
    выровнены по русскому пользовательскому выводу и структуре;
  - в них добавлены понятные русские комментарии в code-cell, убрано смешение
    русского и английского в заголовках и пояснениях;
  - таблицы и подписи графиков приведены к более читаемому виду, а notebook
    заново исполнены через `nbclient`.
- `MTZ-NB05` закрыт:
  - проверен helper-слой исследовательских notebook;
  - общий display-helper вынесен в
    [notebook_display.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/reporting/notebook_display.py);
  - в
    [quality_gate_review.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/reporting/quality_gate_review.py)
    убран незакрытый `engine`, а типовая граница notebook-слоя приведена к
    устойчивому виду.
- `MTZ-NB06` закрыт:
  - технические notebook
    [scoring_review.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/notebooks/technical/scoring_review.ipynb),
    [model_pipeline_review.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/notebooks/technical/model_pipeline_review.ipynb),
    [final_decision_review.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/notebooks/technical/final_decision_review.ipynb),
    [host_priority_calibration_review.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/notebooks/technical/host_priority_calibration_review.ipynb)
    и
    [priority_threshold_review.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/notebooks/technical/priority_threshold_review.ipynb)
    приведены к русскому пользовательскому выводу и единой технической
    структуре;
  - из setup-слоя убраны бессмысленные выводы пути, а в `05` исправлено
    неверное значение `seaborn context`, затрагивавшее исполнение notebook;
  - весь технический контур заново исполнен через `nbclient`.
- `MTZ-NB07` закрыт:
  - helper-слой технических notebook проверен и адресно упрощен;
  - [scoring_review.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/reporting/scoring_review.py)
    и
    [priority_threshold_review.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/reporting/priority_threshold_review.py)
    переведены на тонкий публичный API;
  - внутренняя логика разнесена по модулям контрактов, загрузки и табличных
    review-функций без изменения внешних импортов notebook и тестов.
- `MTZ-NB08` закрыт:
  - введен единый словарь русских подписей для notebook-слоя;
  - общие названия метрик, состояний и сигналов перестали расходиться между active notebook;
  - notebook переведены на общий label/display helper.
- `MTZ-NB09` закрыт:
  - active notebook проверены по длине и смысловому наполнению;
  - явного смыслового перегруза не найдено;
  - содержательные блоки исследования и технического обзора сохранены;
  - точечно исправлен только визуальный перегруз на графике корзин вероятности в
    [host_priority_calibration_review.ipynb](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/notebooks/technical/host_priority_calibration_review.ipynb).
- `MTZ-NB10` закрыт:
  - active notebook разложены по каталогам `eda / research / technical`;
  - из имен active notebook убрана нумерация;
  - обновлены
    [analysis/notebooks/README.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/notebooks/README.md),
    [test_analysis_notebooks.py](/Users/evgeniikuznetsov/Desktop/dspro-vkr/tests/unit/notebooks/test_analysis_notebooks.py)
    и
    [notebook_qa_policy_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/stabilization/notebook_qa_policy_ru.md).
