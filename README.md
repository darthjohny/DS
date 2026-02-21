# DSPro VKR

Учебный проект ВКР по приоритизации наблюдений звезд для экономии времени телескопа.

Модель обучается на параметрах звезд-хостов экзопланет и оценивает,
насколько новая звезда похожа на «перспективные» объекты.

## Структура проекта

```text
src/
  eda.py                # EDA, очистка и базовая аналитика
  model_gaussian.py     # гауссова модель и скоринг
  db_test.py            # проверка подключения к PostgreSQL

data/
  raw/                  # сырые данные
  eda/                  # EDA-таблицы и снимки данных
  eda/plots/            # графики EDA
  plots/                # графики гауссовой модели
  model_gaussian_params.json
```

## Быстрый запуск

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Проверка БД:

```bash
python src/db_test.py
```

EDA:

```bash
python src/eda.py
```

Обучение/оценка модели:

```bash
python src/model_gaussian.py
```
