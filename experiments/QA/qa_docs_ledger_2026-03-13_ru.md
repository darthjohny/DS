# Docs Ledger

Дата фиксации: 13 марта 2026 года

Статусы:
- `OK` — документ живой и полезный
- `TOLERABLE` — документ нормальный, но есть ограничения или риск путаницы
- `FIX` — документ уже заметно отстаёт от repo-state или требует более явного статуса

## README

- [README.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/README.md): `TOLERABLE`
  Верхнеуровневый документ сильный и реально полезный, но слегка перегружен: совмещает архитектуру, запуск, ВКР-карту и future work.

## Канонические docs

- [docs/documentation_style_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/documentation_style_ru.md): `OK`
  Чёткий и рабочий стандарт документации.

- [docs/model_comparison_protocol_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/model_comparison_protocol_ru.md): `OK`
  Один из самых сильных документов repo: методический контракт benchmark-блока.

- [docs/model_comparison_findings_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/model_comparison_findings_ru.md): `OK`
  Хорошо связывает сырые benchmark-артефакты с narrative для ВКР.

- [docs/preprocessing_pipeline_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/preprocessing_pipeline_ru.md): `OK`
  Сильный документ по data lineage, источникам и месту preprocessing в проекте.

- [docs/notebook_review_2026-03-13_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/notebook_review_2026-03-13_ru.md): `OK`
  Короткий, актуальный и полезный техдок по состоянию notebook-слоя.

- [docs/presentation/vkr_slides_draft_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/presentation/vkr_slides_draft_ru.md): `OK`
  Хороший slide-source, уже связанный с актуальными ассетами и run-name.

## Документы с заметным риском устаревания

- [docs/vkr_requirements_traceability_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/vkr_requirements_traceability_ru.md): `FIX`
  Полезная карта соответствия, но частично отстаёт от текущего состояния repo; как минимум preprocessing notebook уже существует, а таблица этого не отражает.

- [docs/ood_unknown_baselines_tz_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/ood_unknown_baselines_tz_ru.md): `FIX`
  Ценный исторический design doc, но уже расходится с текущей реализацией comparison-layer и потому нуждается либо в sync, либо в явной archival-маркировке.

## Исторические/планировочные docs

- [docs/documentation_audit_tz_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/documentation_audit_tz_ru.md): `TOLERABLE`
  Большой и полезный след предыдущей волны унификации, но уже не quick operational doc.

- [docs/ood_unknown_tz_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/ood_unknown_tz_ru.md): `TOLERABLE`
  Похоже на design-contract, а не на ежедневную рабочую инструкцию; документ стоит держать, но воспринимать как зафиксированное решение/ТЗ.

- [docs/preprocessing_and_comparison_tz_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/preprocessing_and_comparison_tz_ru.md): `TOLERABLE`
  Хороший planning doc по уже largely completed волне работ; полезен как история решений, но не как главный “current state” документ.

## Краткий вывод

- Документация в repo в целом сильная.
- Основной риск не в broken links, а в смешении канонических документов и исторических ТЗ.
- Для будущей cleanup-волны имеет смысл явно разделить:
  - документы текущего состояния;
  - документы истории проектных решений.
