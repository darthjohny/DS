# Черновик слайдов ВКР

Дата фиксации: 19 марта 2026 года

Формат этого документа:
- это slide-source, а не готовый `.pptx`;
- графики и таблицы уже собраны в `docs/presentation/assets/baseline_comparison_2026-03-19_v1_calibrated_limit5000`;
- содержимое синхронизировано с каноническим run
  `baseline_comparison_2026-03-19_v1_calibrated_limit5000`;
- production shortlist дополнительно синхронизирован с
  `experiments/QA/production_runs/production_priority_2026-03-19_v1_calibrated_limit5000.md`.

## Слайд 1. Тема и цель

Заголовок:
`Physically Consistent Prioritization of Stellar Targets for Exoplanet Follow-up`

Тезисы на слайде:
- Цель проекта: не доказать наличие экзопланеты, а ранжировать звёзды для последующих наблюдений.
- Объект: звёзды `M/K/G/F dwarf` после router и OOD-контроля.
- Результат: reproducible pipeline, benchmark baseline-моделей и operational ranking.

Комментарий докладчика:
- Ключевая постановка задачи: при ограниченном времени телескопа нужен shortlist объектов с максимальной вероятностью полезного follow-up.

## Слайд 2. Постановка задачи

Тезисы на слайде:
- Вход: Gaia-подобные каталожные признаки и физические оценки звезды.
- Исследовательский вопрос: как оценивать `host vs field` так, чтобы ranking оставался физически интерпретируемым.
- Ограничение: сильная supervised-метрика ещё не гарантирует лучшую operational usefulness.

Комментарий докладчика:
- Отдельно проговорить, что project objective шире обычной бинарной классификации: в конце есть decision layer и приоритизация.

## Слайд 3. Архитектура решения

Тезисы на слайде:
- `router` определяет физический класс звезды и отсекает `UNKNOWN/OOD`
  внутри уже scoreable входного батча.
- `host-model` считает `host vs field` для релевантной dwarf-популяции.
- `decision layer` строит `final_score` и приоритет `HIGH / MEDIUM / LOW`.

Комментарий докладчика:
- Здесь лучше показать схему из записки или быстро собрать финальный рисунок в PowerPoint.
- Отдельно проговорить, что structurally incomplete строки input-layer
  фильтрует раньше и они не входят в `unknown_share`.
- На слайде достаточно трёх блоков: `router -> host-model -> decision layer`.

## Слайд 4. Данные и benchmark-контур

Тезисы на слайде:
- Benchmark строится на выделенном `host` и `field` датасете.
- Test split увеличен до `30%`, train остаётся внутри tuning-контура.
- Сравниваются четыре модели: `main_contrastive_v1`, `legacy_gaussian`, `random_forest`, `mlp_small`.

Ключевые числа:
- `N test = 4619`
- `Host = 1019`
- `Field = 3600`

Вставка:
- [benchmark_test_table.csv](./assets/baseline_comparison_2026-03-19_v1_calibrated_limit5000/benchmark_test_table.csv)

## Слайд 5. Методическая модернизация для ВКР

Тезисы на слайде:
- `test_size = 0.30`
- `10-fold Stratified CV`
- единый `refit_metric = roc_auc`
- search-контур для всех четырёх моделей

Комментарий докладчика:
- Подчеркнуть, что benchmark теперь методически строгий и воспроизводимый.
- Для `RF` и `MLP` используется `GridSearchCV`, для Gaussian-моделей эквивалентный manual CV search.

Вставка:
- [search_summary_table.csv](./assets/baseline_comparison_2026-03-19_v1_calibrated_limit5000/search_summary_table.csv)

## Слайд 6. Основные benchmark-результаты

Тезисы на слайде:
- Лучшие benchmark-метрики показали `RandomForest` и `MLP`.
- `RandomForest`: `ROC-AUC 0.9326`, `PR-AUC 0.7618`, `Precision@50 = 0.92`
- `MLP small`: `ROC-AUC 0.9227`, `PR-AUC 0.7555`, `Precision@50 = 0.92`
- `Contrastive V1`: `ROC-AUC 0.8674`, `PR-AUC 0.5904`, `Precision@50 = 0.72`

Вставка:
- ![Benchmark metrics](./assets/baseline_comparison_2026-03-19_v1_calibrated_limit5000/benchmark_metrics.png)

Комментарий докладчика:
- Здесь важно честно признать лидерство RF/MLP по supervised benchmark.

## Слайд 7. Поведение по спектральным классам

Тезисы на слайде:
- `K`-ветка остаётся самой стабильной для всех моделей.
- `M`-ветка чувствительнее к выбору модели и конфигурации.
- Classwise анализ подтверждает, что глобальная метрика не раскрывает весь профиль поведения.

Вставка:
- ![Classwise ROC-AUC](./assets/baseline_comparison_2026-03-19_v1_calibrated_limit5000/classwise_roc_auc_heatmap.png)

Комментарий докладчика:
- Отдельно отметить, что tuning был нужен не только формально, но и практически: class-specific поведение различается.

## Слайд 8. Search summary и выбранные конфигурации

Тезисы на слайде:
- `Contrastive V1`: `use_m_subclasses=false`, `shrink_alpha=0.05`, `min_population_size=2`
- `Legacy Gaussian`: `use_m_subclasses=true`, `shrink_alpha=0.15`
- `RF` и `MLP` подбирались отдельно по `F/G/K/M`.

Комментарий докладчика:
- Это сильный слайд для защиты формальной методики: параметры теперь выбираются не вручную, а по CV-контуру.

Вставка:
- [search_summary_table.csv](./assets/baseline_comparison_2026-03-19_v1_calibrated_limit5000/search_summary_table.csv)

## Слайд 9. Operational snapshot preview

Тезисы на слайде:
- Snapshot preview выполнен на `5000` входных строках production-like relation.
- После калибровки orchestrator самый широкий `HIGH`-tier формирует `RandomForest`: `744`.
- `MLP small` даёт второй по ширине shortlist: `402 HIGH`.
- `Contrastive V1` теперь ведёт себя заметно более сдержанно: `243 HIGH`.
- `Legacy Gaussian` остаётся самой слабой и консервативной схемой: `141 HIGH`.

Вставка:
- ![Snapshot priority mix](./assets/baseline_comparison_2026-03-19_v1_calibrated_limit5000/snapshot_priority_mix.png)

Комментарий докладчика:
- Это ключевой переход от benchmark-метрик к operational usefulness.
- Здесь важно честно сказать, что после настройки `V1` orchestrator-а
  baseline-модели формируют более широкий shortlist, чем contrastive.
- Но это всё ещё comparison snapshot, а не финальный production export.

## Слайд 10. Профиль top-ranked объектов

Тезисы на слайде:
- Наивысший `top_final_score` теперь у `RandomForest`: `0.9220`.
- `MLP` даёт второй по силе верхний хвост: `0.8987`.
- `Contrastive V1` остаётся сильной, но более сдержанной схемой: `0.7779`.
- Top shortlist основной contrastive-модели по-прежнему состоит в основном
  из компактных `M dwarf`, с вкраплением сильных `K dwarf`.

Вставка:
- ![Top score curves](./assets/baseline_comparison_2026-03-19_v1_calibrated_limit5000/top_score_curves.png)

Комментарий докладчика:
- Это удобно использовать как аргумент, что benchmark leader и
  operational leader в `V1` теперь ближе друг к другу, а contrastive
  схема осталась более интерпретируемой, а не самой широкой.

## Слайд 11. Почему production `V1` всё ещё может оставаться contrastive-first

Тезисы на слайде:
- Production-цель проекта: физически согласованная приоритизация, а не только максимум supervised-метрики.
- `Contrastive V1` лучше вписывается в цепочку `router + OOD + decision layer`.
- После калибровки orchestrator-а `RF` и `MLP` дают более широкий shortlist,
  но contrastive-модель остаётся более сдержанной и интерпретируемой.
- Для защиты важно не путать “лучшую численную модель” и “допустимую
  физически согласованную production-схему первой версии”.

Комментарий докладчика:
- Здесь важно развести два уровня:
- benchmark winner по метрикам;
- production `V1`, которую пока можно защищать за счёт интерпретируемости,
  но уже нельзя обосновывать тезисом про strongest snapshot ranking.

## Слайд 12. Примеры top-кандидатов и итог

Тезисы на слайде:
- Финальный production shortlist текущей `V1` публикуется отдельным export-слоем.
- В operational export `HIGH`-ядро формируют `K dwarf`, а верхушка `M dwarf` идёт вторым эшелоном.
- Практический физический вывод текущей `V1`: приоритет 1 — `K dwarf`, приоритет 2 — `M dwarf`, `G dwarf` — резервный follow-up слой.
- Проект получил воспроизводимый benchmark, обновлённые артефакты и формально корректный comparison-контур.
- Следующий шаг: либо оставить contrastive как консервативную `V1`, либо делать `V2` уже с учётом силы `RF/MLP`.

Вставка:
- [operational_shortlist_summary_table.csv](./assets/baseline_comparison_2026-03-19_v1_calibrated_limit5000/operational_shortlist_summary_table.csv)
- [operational_shortlist_top_table.csv](./assets/baseline_comparison_2026-03-19_v1_calibrated_limit5000/operational_shortlist_top_table.csv)

Итоговая фраза:
- `RandomForest` и `MLP` сильнее по benchmark и по ширине comparison shortlist, но `Contrastive V1` остаётся допустимой интерпретируемой production-схемой первой версии, а финальный follow-up shortlist публикуется отдельным calibrated export-слоем.
