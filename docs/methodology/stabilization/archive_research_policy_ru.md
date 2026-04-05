# Политика Архива Исследований

Дата фиксации: `2026-04-05`

Связанные документы:

- [project_cleanup_tz_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/plans/project_cleanup_tz_ru.md)
- [project_cleanup_micro_tz_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/architecture/project_cleanup_micro_tz_ru.md)
- [tests/archive_research/README.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/tests/archive_research/README.md)
- [analysis/notebooks/archive_research/README.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/notebooks/archive_research/README.md)
- [docs/methodology/archive_research/README.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/archive_research/README.md)

## Назначение

Архив исследований хранит вторичные deep-dive материалы, которые:

- были полезны для диагностики и научного разбора;
- уже не входят в текущий активный рабочий контур;
- сохраняются как воспроизводимый исследовательский след.

Архив нужен не для повседневной разработки, а для:

- доказательной базы в исследовании;
- истории гипотез и их проверки;
- возможного возврата к редкому кейсу в следующей итерации.

## Что Входит В Архив

В текущем проекте архивный слой включает:

- [src/exohost/reporting/archive_research](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/reporting/archive_research)
- [src/exohost/datasets/archive_research](/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/exohost/datasets/archive_research)
- [analysis/notebooks/archive_research](/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/notebooks/archive_research)
- [tests/archive_research](/Users/evgeniikuznetsov/Desktop/dspro-vkr/tests/archive_research)
- [docs/methodology/archive_research](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/archive_research)

## Что Не Делает Архив

Архивный слой:

- не считается частью активного production-like контура;
- не используется как аргумент активного тестового покрытия;
- не обязан немедленно приводиться к текущему стандарту распила и оформления;
- не должен импортироваться активным кодом без отдельного осознанного решения.

## Что Считается Активным Слоем

Активным считается только то, что лежит вне `archive_research` и участвует в:

- текущем pipeline;
- регулярном review;
- активном `pytest`-контуре;
- текущей документации рабочего процесса.

## Правило Возврата Из Архива В Активную Работу

Если архивный материал снова становится нужен в рабочем контуре, его нельзя
использовать напрямую как есть.

Сначала нужно:

1. вернуть материал в активный слой;
2. привести его к текущему инженерному стандарту;
3. добавить или восстановить активные тесты;
4. только после этого использовать его как часть рабочего контура.

## QA-Политика

Для архивного слоя действуют отдельные правила:

- `tests/archive_research` не входит в активный `pytest`-контур;
- архивные notebook не входят в обязательный scoped `nbclient` QA;
- активный код не должен зависеть от архивного слоя.

## Практический Критерий

Если материал нужен для текущего результата проекта, он должен быть в active
слое.

Если материал нужен только как история исследования или резерв для
потенциального возврата, он должен жить в `archive_research`.
