# Схема боевого pipeline

Ниже зафиксирована рабочая архитектура проекта после разделения
физического `router`-слоя, `host-vs-field` модели и боевого
`decision layer`.

```mermaid
flowchart LR
    REF["Референс-слой Gaia<br/>A/B/O/F/G/K/M + dwarf/evolved<br/>lab.v_gaia_router_training"]:::db
    ROUTER_EDA["EDA router-слоя<br/>/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/router_eda"]:::offline
    ROUTER["Gaussian router<br/>/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/router_model"]:::core

    NASA["NASA hosts + Gaia physics<br/>lab.v_nasa_gaia_train_dwarfs"]:::db
    HOST_EDA["EDA host-модели<br/>/Users/evgeniikuznetsov/Desktop/dspro-vkr/analysis/host_eda"]:::offline
    HOST_MODEL["Контрастивная host-модель<br/>/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/host_model"]:::core

    CAND["Входные кандидаты Gaia<br/>source_id + ra + dec + физические признаки"]:::db
    ORCH["Боевой pipeline<br/>/Users/evgeniikuznetsov/Desktop/dspro-vkr/src/priority_pipeline"]:::pipe

    RRES["Таблица распознавания<br/>lab.gaia_router_results"]:::db
    PRES["Таблица приоритизации<br/>lab.gaia_priority_results"]:::db

    LOW["Заглушка низкого приоритета<br/>A/B/O или evolved"]:::branch
    HOST_BRANCH["Ветка host-vs-field<br/>только M/K/G/F dwarf"]:::branch

    REF --> ROUTER_EDA
    REF --> ROUTER
    NASA --> HOST_EDA
    NASA --> HOST_MODEL

    CAND --> ORCH
    ORCH --> ROUTER
    ROUTER --> RRES

    ROUTER --> HOST_BRANCH
    ROUTER --> LOW

    HOST_BRANCH --> HOST_MODEL
    HOST_MODEL --> PRES
    LOW --> PRES

    classDef db fill:#fff3cd,stroke:#aa8b32,color:#111;
    classDef offline fill:#f3f0e8,stroke:#8b7e66,color:#222;
    classDef core fill:#ddebf7,stroke:#5b7ea3,color:#111;
    classDef pipe fill:#e8f5e9,stroke:#5d8a64,color:#111;
    classDef branch fill:#f8d7da,stroke:#a35d66,color:#111;
```

## Смысл схемы

1. `analysis/router_eda` проверяет и документирует физическую структуру
   reference-слоя для router-модели.
2. `src/router_model` распознаёт,
   какая звезда перед нами по физике:
   спектральный класс + эволюционная стадия.
3. Результат распознавания сохраняется в `lab.gaia_router_results`.
4. Только объекты класса `M/K/G/F dwarf`
   идут в `src/host_model`.
5. `src/host_model` считает,
   насколько объект относится к `host` или `field`
   внутри routed stellar class.
6. Если звезда распознана как `A/B/O` или `evolved`,
   применяется заглушка низкого приоритета.
7. Итог сохраняется в `lab.gaia_priority_results`.

## Почему это важно

- не смешивается физическое распознавание
  и `host-vs-field` scoring;
- карлики и evolved разделены;
- `A/B/O` не загрязняют host-модель;
- архитектура становится воспроизводимой
  и объяснимой для ВКР.
