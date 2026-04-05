# Stabilization Issue Ledger

## Цель

Этот реестр хранит реальные баги, сомнительные места и открытые
interpretation-риски текущей stabilization-фазы.

Формат записи:

- `id`
- `тип`
- `слой`
- `статус`
- `симптом`
- `корень`
- `следующий шаг`

## Текущий Реестр

### SI-001

- тип: `defect`
- слой: `final decision input loader`
- статус: `closed`
- симптом:
  - `decide` падал с ошибкой про отсутствующий `radius_feature`
- корень:
  - loader ожидал artifact-specific feature name прямо в DB relation
- решение:
  - compatibility alias перенесен в loader
  - canonical relation остался на `radius_flame`
- regression:
  - `tests/unit/test_load_final_decision_input_dataset.py`

### SI-002

- тип: `defect`
- слой: `final decision input loader`
- статус: `closed`
- симптом:
  - `decide` получал пустой input frame и падал в `SimpleImputer` на `0 samples`
- корень:
  - relation фильтровался по union-признакам всех stage-моделей сразу
- решение:
  - global non-null filter removed
  - missing values обрабатываются stage-level preprocessors
- regression:
  - `tests/unit/test_load_final_decision_input_dataset.py`

### SI-003

- тип: `defect`
- слой: `review notebooks`
- статус: `closed`
- симптом:
  - notebooks смотрели на устаревшие run dirs и путали роли `06` и `07`
- корень:
  - drift между живыми artifacts и notebook config
- решение:
  - `06` оставлен pipeline review notebook
  - `07` оставлен final decision notebook
  - оба привязаны к текущим baseline artifacts
  - notebook texts приведены к единому русскоязычному review-формату

### SI-004

- тип: `scientific_review`
- слой: `quality_gate`
- статус: `open-observed`
- симптом:
  - большая доля `unknown`
- корень:
  - главный драйвер не `RUWE`, а `missing_core_features`
- текущая трактовка:
  - это пока считается валидным selective outcome, а не bug
- следующий шаг:
  - policy-review на свежем диагностическом baseline
  - [pre_battle_diagnostic_run_2026_04_05_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/run_reviews/pre_battle_diagnostic_run_2026_04_05_ru.md)

### SI-005

- тип: `scientific_review`
- слой: `host / priority`
- статус: `open-observed`
- симптом:
  - нужно понять, не завышает ли новый host-like контур `high` priority
- корень:
  - новый clean host path уже подключен, но его поведение еще не разобрано
- следующий шаг:
  - star-level review топ-кандидатов и сомнительных кейсов

### SI-006

- тип: `performance`
- слой: `decide / notebooks / artifact IO`
- статус: `open`
- симптом:
  - пока нет measured bottleneck map
- корень:
  - profiling-фаза еще не проводилась инструментально
- следующий шаг:
  - `cProfile` / measured profiling для `decide` и notebook execution

### SI-007

- тип: `scientific_review`
- слой: `priority`
- статус: `open-observed`
- симптом:
  - top priority rows почти сразу насыщаются к `~1.0`
  - top reasons почти одинаковы
- корень:
  - не выглядит как arithmetic/code defect;
  - saturation рождается из сочетания:
    - `class_priority_score ~= 1.0` для `F/G/K`
    - очень концентрированного `host_similarity_score`
    - достаточно высокого `observability_score`
    - и широкого `high` threshold-zone
- evidence:
  - [priority_review_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/run_reviews/priority_review_round1_ru.md)
  - [priority_threshold_review_round2_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/run_reviews/priority_threshold_review_round2_ru.md)
  - [pre_battle_diagnostic_run_2026_04_05_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/run_reviews/pre_battle_diagnostic_run_2026_04_05_ru.md)
- следующий шаг:
  - повторный threshold-review на свежем диагностическом baseline
  - только потом решать, нужен ли дополнительный score scaling

### SI-008

- тип: `traceability`
- слой: `saved final decision artifacts`
- статус: `closed`
- симптом:
  - `quality_reason` и `review_bucket` в saved decision review сейчас пусты
- корень:
  - explainability columns не входили в `final decision` input contract
- решение:
  - `quality_reason`
  - `review_bucket`
  - `ood_state`
  - `ood_reason`
  - `quality_gate_version`
  - `quality_gated_at_utc`
  добавлены в optional contract decision input loader
- validation:
  - новый run
    `artifacts/decisions/hierarchical_final_decision_2026_03_29_075935_878508`
    содержит эти поля в `decision_input.csv`

### SI-009

- тип: `scientific_review`
- слой: `coarse classification`
- статус: `root_cause_confirmed`
- симптом:
  - класс `O` практически отсутствует в `final_coarse_class` на актуальном
    `final decision` run
- корень:
  - это не только `quality_gate`-эффект;
  - в source есть `6372` строк `spectral_class='O'`, из них `1725` проходят как
    `quality_state='pass'`;
  - дальше coarse stage системно не выдает `O` на pass-части:
    - `B`: `1189`
    - `F`: `353`
    - `G`: `79`
    - `A`: `75`
    - `K`: `28`
    - `M`: `1`
    - `O`: `0`
  - часть surviving `O` rows физически уже не похожа на hot stars, что указывает
    на вероятную source/label inconsistency
- текущая трактовка:
  - это не notebook и не priority issue;
  - после hot-subset review проблема сузилась:
    - broad cool contamination действительно есть в полном `O` source;
    - на узкой hot pass-boundary `O/B` coarse почти идеально схлопывает `O` в `B`;
    - обратного `B -> O` хвоста практически нет;
    - train-time support review показал, что причина не в starvation:
      - reconstructed split совпадает с benchmark;
      - true `O` в coarse source представлено как `3000 / 2100 / 900`
        для `full/train/test`;
      - весь coarse `O` source уже hot (`>= 25000 K`);
      - hot `O/B` boundary до inference симметрична `3000/3000`
        на `full`, `2100/2100` на `train`, `900/900` на `test`;
    - feature separability review показал, что и это не похоже на
      train-time incapacity модели:
      - на собственном train-time `O/B` boundary current coarse artifact
        различает `O` и `B` почти идеально;
      - true `O` и true `B` separable по coarse feature contract;
      - главный discriminative signal идет через `teff_gspphot`,
        и его на train-time source достаточно;
    - domain-shift review это подтвердил:
      - current coarse-artifact почти идеален на собственном train-time `O/B` boundary;
      - downstream true `O` имеют совсем другую физическую область;
      - strongest shift для true `O` идет через:
        - `teff_gspphot`
        - `bp_rp`
        - `radius_feature`
        - `parallax_over_error`
      - missingness при этом проблему не объясняет;
    - alignment audit показал, что gross train/inference plumbing bug тоже не наблюдается:
      - artifact feature contract совпадает с train-time task contract;
      - `decide` подает тот же coarse feature set;
      - у `radius_feature` действительно есть semantic nuance:
        - train-time `O` учились на `radius_gspphot`;
        - downstream `decide` по умолчанию использует `radius_flame`;
      - но отдельный `flame / gspphot / hybrid` experiment не меняет судьбу true `O`:
        они все равно полностью уходят в `B`;
    - Gaia hot-star provenance audit усилил эту трактовку:
      - downstream local `O` почти целиком Gaia `ESP-HS` относит к `B`;
      - `1127 / 1188` local `O` (`94.87%`) имеют `spectraltype_esphs='B'`;
      - только `27 / 1188` (`2.27%`) имеют `spectraltype_esphs='O'`;
      - membership в `gold_sample_oba_stars` здесь почти не различает `O` и `B`,
        то есть подтверждает OBA-like природу пула, но не саму границу `O/B`;
      - значит downstream issue сейчас выглядит уже не как model-only failure,
        а как provenance / label-semantics mismatch на hot-star boundary;
    - upstream provenance audit дал еще более конкретный root signal:
      - `1157 / 1188` downstream local `O` (`97.39%`) происходят из сырых
        `raw_sptype`, начинающихся с `OB...`;
      - `1172 / 1188` (`98.65%`) имеют parser-state
        `partial / missing_integer_subclass`;
      - значит широкий local `O` pool сейчас почти целиком состоит из
        ambiguous `OB` labels, а не из clean explicit `O` labels;
      - это делает parser/policy для ambiguous `OB` labels самым сильным
        кандидатом на root cause;
    - parser fix уже применен в коде и в live labeled relation:
      - ambiguous `OB...`, `O/B...`, `O9.5/B0...` теперь materialize-ятся как
        `spectral_class='OB'`;
      - `label_parse_status='partial'`;
      - `label_parse_notes='ambiguous_ob_boundary_label'`;
      - после rematerialization `lab.gaia_mk_external_labeled`:
        - `spectral_class='OB'`: `22701`
        - `spectral_class='O'`: `8860`
      - значит root cause не только локализован, но уже частично исправлен на
        source-layer;
    - downstream sync после parser-fix подтвердил это окончательно:
      - в `lab.gaia_mk_training_reference` и `lab.gaia_mk_quality_gated`
        ambiguous `OB...` больше не сидят в `spectral_class='O'`;
      - теперь:
        - `spectral_class='O'`: `1207`
        - `spectral_class='OB'`: `5165`
        - `ambiguous OB -> O`: `0`;
      - на refreshed hot provenance source:
        - `B = 7112`
        - `OB = 1163`
        - `O = 25`;
      - то есть широкий problematic `O`-пул почти целиком переехал в явный
        `OB boundary` pool;
      - оставшийся explicit `O`-tail уже узкий:
        - `O -> O = 11 / 25`
        - `O -> B = 12 / 25`
      - значит broad root cause сейчас подтвержден:
        проблема сидела в provenance/parser semantics для ambiguous `OB...`,
        а не в том, что coarse-модель “не умеет O” вообще;
    - O/B boundary policy first wave уже materialized:
      - `lab.gaia_ob_secure_o_like_subset = 25`
      - `lab.gaia_ob_boundary_subset = 1163`
      - внутри `secure O-like`:
        - local `O = 11`
        - local `OB = 14`
      - внутри `boundary`:
        - local `O = 14`
        - local `OB = 1149`
      - значит operationally мы уже можем не смешивать secure hot `O-like`
        и ambiguous `OB boundary` в один класс;
    - дальше policy уже materialized и operational:
      - `lab.gaia_ob_boundary_review = 1163`
      - весь boundary-пул вынесен в explicit review layer;
      - breakdown:
        - local `O = 14`
        - local `OB = 1149`
      - forced automatic `O/B` split для boundary-пула больше не используется;
    - secure `O` tail review показал:
      - после вынесения `OB` в review остается только `11` надежных local `O`;
      - в исходных метках там уже нет широкого ambiguous `OB...` шума;
      - для всех этих строк Gaia hot-star модуль все еще дает `spectraltype_esphs='O'`;
      - но численная `teff_esphs` есть только у `1` строки, поэтому строить
        жесткий temperature gate на этом хвосте пока нельзя;
    - значит текущий главный scientific risk сузился:
      - broad `OB boundary` semantics уже вынесены в review;
      - narrow explicit `O` tail действительно можно разбирать уже как отдельный hot-star case
- следующий шаг:
  - отдельно проверить поведение coarse-модели именно на secure `O` tail;
  - только потом решать, нужен ли narrow retrain / class weighting / source-alignment step
- related:
  - [coarse_o_tail_review_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/archive_research/coarse_o_tail_review_round1_ru.md)
  - [coarse_o_hot_subset_review_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/archive_research/coarse_o_hot_subset_review_round1_ru.md)
  - [coarse_ob_boundary_review_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/archive_research/coarse_ob_boundary_review_round1_ru.md)
  - [coarse_o_train_support_review_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/archive_research/coarse_o_train_support_review_round1_ru.md)
  - [coarse_ob_feature_separability_review_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/archive_research/coarse_ob_feature_separability_review_round1_ru.md)
  - [coarse_ob_domain_shift_review_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/run_reviews/coarse_ob_domain_shift_review_round1_ru.md)
  - [coarse_ob_alignment_audit_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/run_reviews/coarse_ob_alignment_audit_round1_ru.md)
  - [coarse_ob_provenance_review_round1_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/run_reviews/coarse_ob_provenance_review_round1_ru.md)
  - [coarse_ob_provenance_review_round2_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/run_reviews/coarse_ob_provenance_review_round2_ru.md)
  - [coarse_ob_boundary_policy_ru.md](/Users/evgeniikuznetsov/Desktop/dspro-vkr/docs/methodology/contracts/coarse_ob_boundary_policy_ru.md)

## Правило Работы С Реестром

- новый баг попадает сюда только после reproducible symptom;
- bugfix не считается закрытым без regression test или reproducible validation;
- scientific-risk записи не смешиваются с code defects.
