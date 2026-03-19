# Выводы по сравнению моделей

Дата: 19 марта 2026 года

## 1. Назначение документа

Этот документ фиксирует интерпретацию результатов baseline-сравнения и
live snapshot-прогона для ВКР.

Он не заменяет сырые артефакты из `experiments/model_comparison/`, а
служит их кратким аналитическим слоем:

- что показал supervised benchmark;
- что показал operational snapshot;
- какую модель и как корректно обосновывать в пояснительной записке.

## 2. Использованные артефакты

Основные источники результатов:

- `experiments/model_comparison/baseline_comparison_2026-03-19_v1_calibrated_limit5000.md`
- `experiments/model_comparison/baseline_comparison_2026-03-19_v1_calibrated_limit5000_summary.csv`
- `experiments/model_comparison/baseline_comparison_2026-03-19_v1_calibrated_limit5000_classwise.csv`
- `experiments/model_comparison/baseline_comparison_2026-03-19_v1_calibrated_limit5000_thresholds.csv`
- `experiments/model_comparison/baseline_comparison_2026-03-19_v1_calibrated_limit5000_quality_summary.csv`
- `experiments/model_comparison/baseline_comparison_2026-03-19_v1_calibrated_limit5000_quality_classwise.csv`
- `experiments/model_comparison/baseline_comparison_2026-03-19_v1_calibrated_limit5000_confusion_matrices.csv`
- `experiments/model_comparison/baseline_comparison_2026-03-19_v1_calibrated_limit5000_search_summary.csv`
- `experiments/model_comparison/baseline_comparison_2026-03-19_v1_calibrated_limit5000_snapshot.md`
- `experiments/model_comparison/baseline_comparison_2026-03-19_v1_calibrated_limit5000_snapshot_summary.csv`
- `experiments/QA/production_runs/production_priority_2026-03-19_v1_calibrated_limit5000.md`
- `experiments/QA/production_runs/production_priority_2026-03-19_v1_calibrated_limit5000_shortlist.csv`
- `experiments/QA/production_runs/production_priority_2026-03-19_v1_calibrated_limit5000_shortlist_summary.csv`

Сравнивались четыре модели:

1. `main_contrastive_v1`
2. `baseline_legacy_gaussian`
3. `baseline_random_forest`
4. `baseline_mlp_small`

## 3. Краткий итог

На текущем benchmark с `test_size = 0.30`, `10-fold CV` и search summary
`baseline_random_forest` показывает лучшие численные метрики, а
`baseline_mlp_small` занимает устойчивое второе место.

При этом `main_contrastive_v1`:

- уверенно превосходит legacy baseline;
- лучше встроена в физически интерпретируемую архитектуру проекта;
- уже используется в production pipeline и согласована с `router + OOD`.

`baseline_legacy_gaussian` полезна как честная историческая точка
сравнения, но не как кандидат на лучшую итоговую модель.

## 4. Supervised benchmark

Основные test-метрики:

| Модель | ROC-AUC | PR-AUC | Brier | precision@50 |
| --- | ---: | ---: | ---: | ---: |
| `baseline_random_forest` | 0.9326 | 0.7618 | 0.0901 | 0.92 |
| `baseline_mlp_small` | 0.9227 | 0.7555 | 0.0901 | 0.92 |
| `main_contrastive_v1` | 0.8674 | 0.5904 | 0.1543 | 0.72 |
| `baseline_legacy_gaussian` | 0.8464 | 0.5713 | 0.1483 | 0.72 |

Главные наблюдения:

- `RandomForest` выигрывает у обеих Gaussian-схем по всем основным
  supervised метрикам.
- `MLP` тоже заметно сильнее обеих Gaussian-схем и по качеству близка к
  `RandomForest`.
- `main_contrastive_v1` устойчиво лучше legacy baseline на test split.
- `legacy Gaussian` остаётся рабочим baseline, но заметно уступает двум
  другим подходам.

## 5. Threshold-based quality

Дополнительно к ranking-метрикам был собран threshold-based quality-блок.
Порог классификации выбирался только на `train` split по `max F1`, а затем
фиксировался и применялся на `test`.

Основные test-метрики после выбора train-threshold:

| Модель | Train threshold | Precision | Recall | F1 | Specificity | Balanced accuracy | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline_random_forest` | 0.6133 | 0.7172 | 0.7193 | 0.7183 | 0.9197 | 0.8195 | 0.8755 |
| `baseline_mlp_small` | 0.3567 | 0.6783 | 0.7841 | 0.7274 | 0.8947 | 0.8394 | 0.8703 |
| `main_contrastive_v1` | 0.4987 | 0.4823 | 0.8665 | 0.6196 | 0.7367 | 0.8016 | 0.7653 |
| `baseline_legacy_gaussian` | 0.3839 | 0.5136 | 0.7036 | 0.5938 | 0.8114 | 0.7575 | 0.7876 |

Ключевые наблюдения:

- `RandomForest` даёт лучшую точность положительного решения и лучшую
  `specificity`, то есть ведёт себя как наиболее строгий и консервативный
  классификатор.
- `MLP` показывает лучший `F1` и лучшую `balanced_accuracy`, то есть даёт
  наиболее сильный компромисс между чувствительностью и ложноположительными
  срабатываниями.
- `main_contrastive_v1` выигрывает по `recall`: она пропускает меньше
  host-объектов, но платит за это заметно более низкой точностью и
  большим числом false positive.
- `legacy Gaussian` остаётся рабочим baseline, но уже по quality-блоку
  видно, что обе более современные модели устойчиво сильнее.

Именно поэтому quality-блок не заменяет ranking benchmark, а дополняет его:
он показывает, как одна и та же модель ведёт себя уже как бинарный
классификатор после выбора явного порога.

## 6. Class-wise наблюдения

На class-wise разрезе видно:

- класс `K` является наиболее стабильным и для contrastive, и для RF;
- класс `G` тоже выглядит уверенно, но gap между RF и contrastive там
  заметен;
- класс `M` остаётся самым трудным;
- именно на `M` разница между legacy и более современной схемой особенно
  заметна.

Это хорошо согласуется с архитектурным решением:

- в `V1` router работает грубо;
- для `M` уже введена дополнительная детализация;
- дальнейшее развитие именно в зоне `M` остаётся научно оправданным.

## 7. Snapshot на живом Gaia batch

Snapshot строился на `public.gaia_dr3_training` после общего
`router + OOD`.

В этой версии findings используется оперативный preview c
`snapshot_limit = 5000`, а не полный relation без ограничения.

Параметры потока:

- `input_rows = 5000`
- `router_rows = 5000`
- `host_candidates = 4202`
- `low_known_rows = 769`
- `unknown_rows = 29`

Важно:

- в current V1 `unknown_rows` считаются только внутри уже scoreable
  runtime batch;
- structurally incomplete строки отбрасываются input-layer ещё до
  router и не входят в это число.

Дополнительная operational оговорка:

- comparison snapshot остаётся исследовательским preview для сравнения
  моделей на одном и том же live batch;
- финальный production shortlist текущей `V1` теперь строится отдельным
  export-слоем на базе calibrated `main_contrastive_v1` и не смешивается
  с comparison snapshot CSV.

Распределение итоговых tier:

| Модель | HIGH | MEDIUM | LOW | top_final_score |
| --- | ---: | ---: | ---: | ---: |
| `baseline_random_forest` | 744 | 794 | 3462 | 0.9220 |
| `baseline_mlp_small` | 402 | 865 | 3733 | 0.8987 |
| `main_contrastive_v1` | 243 | 1199 | 3558 | 0.7779 |
| `baseline_legacy_gaussian` | 141 | 1109 | 3750 | 0.7621 |

Интерпретация:

- после калибровки orchestrator tier-thresholds именно `RandomForest`
  формирует самый широкий `HIGH`-tier;
- `MLP` остаётся вторым по силе operational baseline и тоже поднимает
  заметно более широкий shortlist, чем contrastive-модель;
- `main_contrastive_v1` теперь ведёт себя как более сдержанный,
  recall-oriented ranking внутри текущего `V1` decision layer;
- `legacy Gaussian` по-прежнему выглядит наиболее слабой и слишком
  консервативной схемой для боевого shortlist.

Это означает, что после выравнивания `V1` benchmark и snapshot больше не
поддерживают тезис о “самом широком ranking” у contrastive-модели.
Сильные baseline-модели теперь действительно формируют более широкий
операционный shortlist.

При этом этот вывод не равен автоматической замене production-слоя:
ширина comparison `HIGH`-tier и канонический production shortlist в
проекте теперь разведены как разные уровни результата.

## 8. Как корректно интерпретировать победу RandomForest

Важно не делать некорректный вывод вида:

> Раз `RandomForest` выиграл benchmark, значит production Gaussian нужно
> немедленно заменить.

Такой вывод был бы слишком сильным.

Что видно по факту:

- `RandomForest` лучше решает текущую supervised задачу `host vs field`;
- `MLP` даёт сопоставимо сильный нейросетевой baseline без тяжёлой
  инфраструктуры и сложных зависимостей;
- при этом он менее физически интерпретируем;
- его train-метрики почти идеальны, а значит нужно учитывать риск более
  сильного подгона под benchmark;
- production контур проекта строился как физически согласованный pipeline,
  а не как чисто “чёрный ящик” на максимальную метрику.

Поэтому корректная научная позиция для ВКР такая:

- `RandomForest` выступает сильным ML baseline;
- `baseline_mlp_small` закрывает нейросетевой блок и выступает
  компактным ИНС baseline;
- `main_contrastive_v1` выступает основной физически интерпретируемой
  моделью проекта;
- итоговый выбор production-схемы обосновывается не только метрикой, но и
  архитектурной согласованностью, интерпретируемостью и связью с физикой
  задачи.

## 9. Рекомендуемая формулировка для ВКР

Без претензии на окончательный текст, смысловой тезис должен быть таким:

1. В работе были сравнены четыре подхода:
   основная contrastive Gaussian-модель, legacy Gaussian baseline,
   классический ML baseline `RandomForest` и компактный нейросетевой
   baseline `MLP`.
2. По supervised метрикам лучшим оказался `RandomForest`.
3. `MLP` показал вторые по силе supervised метрики и подтвердил, что
   информативность базовых физических признаков воспроизводится и
   нейросетевым методом.
4. Основная contrastive-модель при этом существенно превзошла legacy
   Gaussian baseline и сохранила смысл как физически интерпретируемая
   `V1`-схема.
5. После калибровки orchestrator-а более широкий operational shortlist
   формируют `RandomForest` и `MLP`, а contrastive-модель остаётся более
   сдержанной.
6. В качестве production-ядра проекта contrastive Gaussian-модель всё
   ещё допустима как интерпретируемая `V1`, но уже не может
   обосновываться тезисом о самом сильном snapshot-ranking.
7. `RandomForest` и `MLP` зафиксированы как сильные внешние baseline,
   относительно которых можно оценивать дальнейшие версии основной
   Gaussian-схемы.
8. Финальный follow-up shortlist текущей `V1` публикуется отдельным
   production export-слоем, а не напрямую из comparison snapshot.

## 10. Практический вывод

На текущем этапе baseline-блок можно считать закрытым как исследовательский
контур ВКР:

- есть единый benchmark;
- есть четыре модели;
- есть train/test метрики;
- есть live snapshot;
- есть CLI и артефакты в репозитории.

## 11. Физический вывод по теме работы

Важно: текущая `V1` не доказывает наличие экзопланеты у конкретной
звезды. Она решает более аккуратную и научно корректную задачу:
физически согласованную приоритизацию объектов для последующих
наблюдений.

Если переводить результат проекта в прикладной ответ на вопрос
`за какими звёздами наблюдать в первую очередь`, то вывод сейчас такой.

### Первая очередь наблюдений

Главная целевая популяция — `K dwarf`.

Почему:

- именно `K`-карлики доминируют в `HIGH`-tier почти у всех моделей;
- по class-wise benchmark это самый стабильный класс;
- он хорошо поддерживается и физически интерпретируемой contrastive
  схемой, и внешними baseline-моделями;
- в operational snapshot именно `K`-карлики формируют основное ядро
  уверенных кандидатов.

Практическая формулировка:
в первую очередь нужно наблюдать звёзды главной последовательности класса
`K` с host-like физическими параметрами и хорошим качеством данных.

### Вторая очередь наблюдений

Следующий важный слой — `M dwarf`.

Почему:

- contrastive-модель часто поднимает именно `M`-карлики в самый верх
  ranking;
- среди top-кандидатов основной production-модели первые позиции часто
  занимают холодные компактные `M`-карлики;
- при этом класс `M` остаётся самым трудным и даёт наибольшее
  расхождение между моделями.

Практическая формулировка:
во вторую очередь нужно наблюдать физически согласованные `M`-карлики,
особенно если они попадают в верхние позиции contrastive-ranking и
поддерживаются хотя бы частью baseline-моделей.

### Третья очередь наблюдений

Третий слой — `G dwarf`.

Почему:

- `G`-карлики выглядят разумно на benchmark, но в current `V1`
  operational configuration в основном остаются в `MEDIUM`-tier;
- это не основной класс для текущего ranking-ядра, но и не полностью
  фоновые объекты;
- модели согласуются по `G` лучше, чем по `M`, но приоритет у них ниже,
  чем у `K`.

Практическая формулировка:
в третью очередь стоит наблюдать `G`-карлики с хорошими quality-факторами
и host-like положением в признаковом пространстве как резервный слой
после `K/M` shortlist.

### Что уходит в низкий приоритет

На текущем этапе не выглядят основной целью для follow-up:

- `F`-карлики как массовая приоритетная популяция;
- горячие `A/B/O` звёзды;
- эволюционировавшие звёзды;
- объекты, ушедшие в `unknown/OOD`.

### Короткий итог

В рамках текущей `V1` рабочий физический вывод можно сформулировать так:

1. Наиболее перспективная популяция для наблюдений — `K`-карлики.
2. Следующая по интересу зона — `M`-карлики, особенно верхушка
   contrastive-ranking.
3. Третья очередь — `G`-карлики как резервный follow-up слой.

То есть практический ответ проекта сейчас звучит так:

`наблюдать прежде всего K dwarf, затем M dwarf, затем G dwarf, при этом
внутри класса приоритет задаётся общей host-like близостью и quality
факторами pipeline`.
