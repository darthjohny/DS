"""Калибровщик decision layer для итогового ranking.

Назначение файла:
    - прогонять готовые модели на проверенном входном наборе;
    - применять разные схемы `decision layer`;
    - сравнивать итоговый ranking без записи в боевые result-таблицы;
    - сохранять артефакты каждой итерации калибровки.

Почему это отдельный файл:
    - калибровка не является частью `star_orchestrator.py`;
    - подбор коэффициентов не должен смешиваться с боевым inference;
    - результаты калибровки должны сохраняться отдельно и воспроизводимо.

Что делает файл:
    1. Проверяет, что входной relation имеет статус `READY`.
    2. Загружает входные данные из БД.
    3. Загружает router-модель и host-модель.
    4. Выполняет базовый прогон:
       - router;
       - split на ветки;
       - host similarity только для MKGF dwarf.
    5. Применяет к host-ветке калибровочную формулу:
       final_score =
       similarity
       * class_prior
       * distance_factor
       * quality_factor
       * metallicity_factor
    6. Формирует summary и сохраняет артефакты итерации
       в каталог логов калибровки.

Что не делает:
    - не пишет в `lab.gaia_router_results`;
    - не пишет в `lab.gaia_priority_results`;
    - не обучает модели заново;
    - не заменяет `star_orchestrator.py`.
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, TypeAlias, cast

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from decision_layer_logbook import ensure_logbook_dir, next_iteration_number
from input_layer import DEFAULT_INPUT_RELATION, REGISTRY_TABLE, make_engine_from_env
from model_gaussian import score_df as score_host_df
from star_orchestrator import (
    clip_unit_interval,
    host_model_version,
    load_input_candidates,
    load_models,
    order_priority_results,
    priority_tier_from_score,
    run_router,
    split_branches,
    stub_reason_code,
)

CALIBRATOR_VERSION = "decision_layer_calibrator_v1"
DEFAULT_TOP_N = 50
JsonMapping: TypeAlias = Mapping[str, Any]
SummaryRecord: TypeAlias = tuple[str, object]


@dataclass(frozen=True)
class ReadyDatasetRecord:
    """Последняя валидированная запись о входном датасете."""

    relation_name: str
    source_name: str
    status: str
    row_count: int
    validated_at: datetime | None


@dataclass(frozen=True)
class ClassPriorConfig:
    """Мягкие приоритеты по спектральным классам."""

    k: float = 1.08
    g: float = 1.05
    m: float = 1.02
    f: float = 0.97

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any] | None,
    ) -> ClassPriorConfig:
        """Построить конфиг priors из mapping."""
        default = cls()
        if data is None:
            return default
        return cls(
            k=float(data.get("K", default.k)),
            g=float(data.get("G", default.g)),
            m=float(data.get("M", default.m)),
            f=float(data.get("F", default.f)),
        )


@dataclass(frozen=True)
class MetallicityConfig:
    """Пороговая схема мягкого влияния [M/H]."""

    low_threshold: float = -0.3
    solar_threshold: float = 0.2
    high_threshold: float = 0.5
    low_factor: float = 0.96
    neutral_factor: float = 1.00
    positive_factor: float = 1.03
    high_factor: float = 1.05

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any] | None,
    ) -> MetallicityConfig:
        """Построить конфиг металличности из mapping."""
        default = cls()
        if data is None:
            return default
        return cls(
            low_threshold=float(
                data.get("low_threshold", default.low_threshold)
            ),
            solar_threshold=float(
                data.get("solar_threshold", default.solar_threshold)
            ),
            high_threshold=float(
                data.get("high_threshold", default.high_threshold)
            ),
            low_factor=float(data.get("low_factor", default.low_factor)),
            neutral_factor=float(
                data.get("neutral_factor", default.neutral_factor)
            ),
            positive_factor=float(
                data.get("positive_factor", default.positive_factor)
            ),
            high_factor=float(data.get("high_factor", default.high_factor)),
        )


@dataclass(frozen=True)
class DistanceConfig:
    """Пороговая схема влияния расстояния в pc."""

    near_max_pc: float = 50.0
    moderate_max_pc: float = 100.0
    distant_max_pc: float = 200.0
    far_max_pc: float = 500.0
    near_factor: float = 1.00
    moderate_factor: float = 0.96
    distant_factor: float = 0.90
    far_factor: float = 0.80
    very_far_factor: float = 0.65
    invalid_factor: float = 0.70

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any] | None,
    ) -> DistanceConfig:
        """Построить конфиг расстояния из mapping."""
        default = cls()
        if data is None:
            return default
        return cls(
            near_max_pc=float(data.get("near_max_pc", default.near_max_pc)),
            moderate_max_pc=float(
                data.get("moderate_max_pc", default.moderate_max_pc)
            ),
            distant_max_pc=float(
                data.get("distant_max_pc", default.distant_max_pc)
            ),
            far_max_pc=float(data.get("far_max_pc", default.far_max_pc)),
            near_factor=float(data.get("near_factor", default.near_factor)),
            moderate_factor=float(
                data.get("moderate_factor", default.moderate_factor)
            ),
            distant_factor=float(
                data.get("distant_factor", default.distant_factor)
            ),
            far_factor=float(data.get("far_factor", default.far_factor)),
            very_far_factor=float(
                data.get("very_far_factor", default.very_far_factor)
            ),
            invalid_factor=float(
                data.get("invalid_factor", default.invalid_factor)
            ),
        )


@dataclass(frozen=True)
class RuweConfig:
    """Пороговая схема качества астрометрии по RUWE."""

    good_max: float = 1.1
    warning_max: float = 1.2
    alert_max: float = 1.4
    bad_max: float = 1.8
    good_factor: float = 1.00
    warning_factor: float = 0.98
    alert_factor: float = 0.93
    bad_factor: float = 0.80
    very_bad_factor: float = 0.60
    missing_factor: float = 0.90

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any] | None,
    ) -> RuweConfig:
        """Построить RUWE-конфиг из mapping."""
        default = cls()
        if data is None:
            return default
        return cls(
            good_max=float(data.get("good_max", default.good_max)),
            warning_max=float(data.get("warning_max", default.warning_max)),
            alert_max=float(data.get("alert_max", default.alert_max)),
            bad_max=float(data.get("bad_max", default.bad_max)),
            good_factor=float(data.get("good_factor", default.good_factor)),
            warning_factor=float(
                data.get("warning_factor", default.warning_factor)
            ),
            alert_factor=float(
                data.get("alert_factor", default.alert_factor)
            ),
            bad_factor=float(data.get("bad_factor", default.bad_factor)),
            very_bad_factor=float(
                data.get("very_bad_factor", default.very_bad_factor)
            ),
            missing_factor=float(
                data.get("missing_factor", default.missing_factor)
            ),
        )


@dataclass(frozen=True)
class ParallaxPrecisionConfig:
    """Пороговая схема качества расстояния по parallax_over_error."""

    excellent_min: float = 20.0
    good_min: float = 10.0
    acceptable_min: float = 5.0
    weak_min: float = 3.0
    excellent_factor: float = 1.00
    good_factor: float = 0.97
    acceptable_factor: float = 0.90
    weak_factor: float = 0.75
    poor_factor: float = 0.55
    missing_factor: float = 0.85

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any] | None,
    ) -> ParallaxPrecisionConfig:
        """Построить конфиг точности параллакса из mapping."""
        default = cls()
        if data is None:
            return default
        return cls(
            excellent_min=float(
                data.get("excellent_min", default.excellent_min)
            ),
            good_min=float(data.get("good_min", default.good_min)),
            acceptable_min=float(
                data.get("acceptable_min", default.acceptable_min)
            ),
            weak_min=float(data.get("weak_min", default.weak_min)),
            excellent_factor=float(
                data.get("excellent_factor", default.excellent_factor)
            ),
            good_factor=float(data.get("good_factor", default.good_factor)),
            acceptable_factor=float(
                data.get("acceptable_factor", default.acceptable_factor)
            ),
            weak_factor=float(data.get("weak_factor", default.weak_factor)),
            poor_factor=float(data.get("poor_factor", default.poor_factor)),
            missing_factor=float(
                data.get("missing_factor", default.missing_factor)
            ),
        )


@dataclass(frozen=True)
class QualityConfig:
    """Схема качества наблюдательных параметров."""

    ruwe: RuweConfig = RuweConfig()
    parallax_precision: ParallaxPrecisionConfig = (
        ParallaxPrecisionConfig()
    )

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any] | None,
    ) -> QualityConfig:
        """Построить quality-конфиг из mapping."""
        if data is None:
            return cls()
        return cls(
            ruwe=RuweConfig.from_mapping(data.get("ruwe")),
            parallax_precision=ParallaxPrecisionConfig.from_mapping(
                data.get("parallax_precision")
            ),
        )


@dataclass(frozen=True)
class CalibrationConfig:
    """Полный конфиг decision layer."""

    class_prior: ClassPriorConfig = ClassPriorConfig()
    metallicity: MetallicityConfig = MetallicityConfig()
    distance: DistanceConfig = DistanceConfig()
    quality: QualityConfig = QualityConfig()

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any] | None,
    ) -> CalibrationConfig:
        """Построить calibration-конфиг из mapping."""
        if data is None:
            return cls()
        return cls(
            class_prior=ClassPriorConfig.from_mapping(
                data.get("class_prior")
            ),
            metallicity=MetallicityConfig.from_mapping(
                data.get("metallicity")
            ),
            distance=DistanceConfig.from_mapping(data.get("distance")),
            quality=QualityConfig.from_mapping(data.get("quality")),
        )


@dataclass(frozen=True)
class BaseScoringResult:
    """Результат базового preview-прогона до decision layer."""

    input_df: pd.DataFrame
    router_df: pd.DataFrame
    host_input_df: pd.DataFrame
    low_input_df: pd.DataFrame
    host_scored_df: pd.DataFrame


@dataclass(frozen=True)
class IterationSummary:
    """Краткая сводка по одной итерации калибровки."""

    run_id: str
    relation_name: str
    source_name: str
    input_rows: int
    router_rows: int
    host_rows: int
    low_rows: int
    final_score_min: float | None
    final_score_mean: float | None
    final_score_max: float | None
    top_n: int
    calibrator_version: str = CALIBRATOR_VERSION


def parse_args() -> Namespace:
    """Разобрать аргументы CLI."""
    parser = ArgumentParser(
        description="Калибровка decision layer без записи в боевые таблицы."
    )
    parser.add_argument(
        "--relation",
        default=DEFAULT_INPUT_RELATION,
        help="Relation со статусом READY в registry-таблице.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Опциональный лимит строк для preview-прогона.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help="Сколько лучших кандидатов сохранять в summary.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Путь к JSON-файлу с переопределением коэффициентов.",
    )
    parser.add_argument(
        "--iteration-note",
        default="",
        help="Короткая заметка о цели текущей итерации.",
    )
    return parser.parse_args()


def normalize_json_object(raw_obj: object) -> dict[str, Any]:
    """Преобразовать сырой JSON-объект в mapping со строковыми ключами."""
    if not isinstance(raw_obj, dict):
        raise ValueError("Calibration config must be a JSON object.")

    # После runtime-проверки явно сужаем JSON-объект до mapping с объектными
    # ключами, а затем нормализуем его к словарю со строковыми ключами.
    raw_mapping_any = cast(Mapping[object, Any], raw_obj)
    normalized: dict[str, Any] = {}
    for raw_key, raw_value in raw_mapping_any.items():
        key_obj: object = raw_key
        value_obj: Any = raw_value
        if not isinstance(key_obj, str):
            raise ValueError("Calibration config keys must be strings.")
        key_str: str = key_obj
        normalized[key_str] = value_obj
    return normalized


def load_calibration_config(path: Path | None) -> CalibrationConfig:
    """Загрузить calibration config из JSON или вернуть baseline."""
    if path is None:
        return CalibrationConfig()
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    raw_any: Any = json.loads(path.read_text(encoding="utf-8"))
    raw_mapping = normalize_json_object(raw_any)
    return CalibrationConfig.from_mapping(raw_mapping)


def make_run_id(prefix: str = "decision_calibration") -> str:
    """Собрать уникальный run_id для оффлайн-итерации."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{timestamp}"


def fetch_ready_dataset_record(
    engine: Engine,
    relation_name: str,
) -> ReadyDatasetRecord:
    """Получить последнюю registry-запись и убедиться, что она READY."""
    query = text(
        f"""
        SELECT
            relation_name,
            source_name,
            status,
            row_count,
            validated_at
        FROM {REGISTRY_TABLE}
        WHERE relation_name = :relation_name
        ORDER BY validated_at DESC
        LIMIT 1;
        """
    )
    with engine.connect() as conn:
        row = conn.execute(
            query,
            {"relation_name": relation_name},
        ).mappings().first()

    if row is None:
        raise RuntimeError(
            "Dataset is missing in registry. "
            "Validate it with input_layer.py first."
        )

    record = ReadyDatasetRecord(
        relation_name=str(row["relation_name"]),
        source_name=str(row["source_name"]),
        status=str(row["status"]),
        row_count=int(row["row_count"]),
        validated_at=row["validated_at"],
    )
    if record.status != "READY":
        raise RuntimeError(
            f"Dataset status must be READY, got {record.status}."
        )
    return record


def load_ready_input_dataset(
    engine: Engine,
    relation_name: str,
    limit: int | None,
) -> tuple[ReadyDatasetRecord, pd.DataFrame]:
    """Загрузить только тот relation, который прошёл READY-валидацию."""
    record = fetch_ready_dataset_record(engine, relation_name)
    df = load_input_candidates(
        engine=engine,
        source_name=relation_name,
        limit=limit,
    )
    return record, df


def run_base_scoring(
    df_input: pd.DataFrame,
    router_model: Any,
    host_model: Mapping[str, Any],
) -> BaseScoringResult:
    """Выполнить router и host similarity до decision layer."""
    router_df = run_router(df_input, router_model)
    host_input_df, low_input_df = split_branches(router_df)

    if host_input_df.empty:
        host_scored_df = host_input_df.copy()
    else:
        host_scored_df = score_host_df(
            model=dict(host_model),
            df=host_input_df,
            spec_class_col="predicted_spec_class",
        )

    return BaseScoringResult(
        input_df=df_input,
        router_df=router_df,
        host_input_df=host_input_df,
        low_input_df=low_input_df,
        host_scored_df=host_scored_df,
    )


def class_prior(spec_class: Any, config: CalibrationConfig) -> float:
    """Вернуть мягкий prior по спектральному классу."""
    mapping = {
        "K": config.class_prior.k,
        "G": config.class_prior.g,
        "M": config.class_prior.m,
        "F": config.class_prior.f,
    }
    return float(mapping.get(str(spec_class), 0.90))


def metallicity_factor(value: Any, config: CalibrationConfig) -> float:
    """Вернуть мягкий множитель по металличности."""
    if pd.isna(value):
        return config.metallicity.neutral_factor
    mh = float(value)
    if mh <= config.metallicity.low_threshold:
        return config.metallicity.low_factor
    if mh < config.metallicity.solar_threshold:
        return config.metallicity.neutral_factor
    if mh < config.metallicity.high_threshold:
        return config.metallicity.positive_factor
    return config.metallicity.high_factor


def distance_pc_from_parallax(parallax: Any) -> float | None:
    """Перевести параллакс в расстояние в parsec."""
    if pd.isna(parallax):
        return None
    plx = float(parallax)
    if plx <= 0.0:
        return None
    return 1000.0 / plx


def distance_factor(parallax: Any, config: CalibrationConfig) -> float:
    """Вернуть мягкий множитель по расстоянию."""
    distance_pc = distance_pc_from_parallax(parallax)
    if distance_pc is None:
        return config.distance.invalid_factor
    if distance_pc <= config.distance.near_max_pc:
        return config.distance.near_factor
    if distance_pc <= config.distance.moderate_max_pc:
        return config.distance.moderate_factor
    if distance_pc <= config.distance.distant_max_pc:
        return config.distance.distant_factor
    if distance_pc <= config.distance.far_max_pc:
        return config.distance.far_factor
    return config.distance.very_far_factor


def ruwe_factor(value: Any, config: CalibrationConfig) -> float:
    """Вернуть множитель качества астрометрии по RUWE."""
    ruwe = config.quality.ruwe
    if pd.isna(value):
        return ruwe.missing_factor
    current = float(value)
    if current <= ruwe.good_max:
        return ruwe.good_factor
    if current <= ruwe.warning_max:
        return ruwe.warning_factor
    if current <= ruwe.alert_max:
        return ruwe.alert_factor
    if current <= ruwe.bad_max:
        return ruwe.bad_factor
    return ruwe.very_bad_factor


def parallax_precision_factor(
    value: Any,
    config: CalibrationConfig,
) -> float:
    """Вернуть множитель качества расстояния по parallax_over_error."""
    precision = config.quality.parallax_precision
    if pd.isna(value):
        return precision.missing_factor
    current = float(value)
    if current >= precision.excellent_min:
        return precision.excellent_factor
    if current >= precision.good_min:
        return precision.good_factor
    if current >= precision.acceptable_min:
        return precision.acceptable_factor
    if current >= precision.weak_min:
        return precision.weak_factor
    return precision.poor_factor


def quality_factor(
    ruwe_value: Any,
    parallax_over_error: Any,
    config: CalibrationConfig,
) -> float:
    """Объединить RUWE и точность параллакса в один фактор качества."""
    value = (
        ruwe_factor(ruwe_value, config)
        * parallax_precision_factor(parallax_over_error, config)
    )
    return clip_unit_interval(float(value))


def build_low_priority_preview(df_low: pd.DataFrame) -> pd.DataFrame:
    """Собрать stub-ветку для A/B/O и evolved без полного ranking."""
    if df_low.empty:
        return df_low.copy()

    result = df_low.copy()
    result["gauss_label"] = None
    result["d_mahal"] = None
    result["similarity"] = None
    result["class_prior"] = None
    result["distance_factor"] = None
    result["quality_factor"] = None
    result["metallicity_factor"] = None
    result["final_score"] = 0.0
    result["priority_tier"] = "LOW"
    result["reason_code"] = [
        stub_reason_code(spec_class, stage)
        for spec_class, stage in result[
            ["predicted_spec_class", "predicted_evolution_stage"]
        ].itertuples(index=False, name=None)
    ]
    result["host_model_version"] = None
    return result


def apply_calibration_config(
    df_scored: pd.DataFrame,
    config: CalibrationConfig,
    host_model_version_value: str,
) -> pd.DataFrame:
    """Применить к host-ветке формулу calibrated final_score."""
    if df_scored.empty:
        return df_scored.copy()

    result = df_scored.copy()
    result["class_prior"] = [
        class_prior(spec_class, config)
        for spec_class in result["predicted_spec_class"]
    ]
    result["distance_factor"] = [
        distance_factor(parallax, config)
        for parallax in result["parallax"]
    ]
    result["quality_factor"] = [
        quality_factor(ruwe_value, plx_error, config)
        for ruwe_value, plx_error in result[
            ["ruwe", "parallax_over_error"]
        ].itertuples(index=False, name=None)
    ]
    result["metallicity_factor"] = [
        metallicity_factor(value, config)
        for value in result["mh_gspphot"]
    ]

    score_rows = result[
        [
            "similarity",
            "class_prior",
            "distance_factor",
            "quality_factor",
            "metallicity_factor",
        ]
    ].itertuples(index=False, name=None)
    result["final_score"] = [
        clip_unit_interval(
            float(similarity)
            * float(prior_value)
            * float(distance_value)
            * float(quality_value)
            * float(metallicity_value)
        )
        for (
            similarity,
            prior_value,
            distance_value,
            quality_value,
            metallicity_value,
        ) in score_rows
    ]
    result["priority_tier"] = [
        priority_tier_from_score(float(score))
        for score in result["final_score"]
    ]
    result["reason_code"] = "HOST_SCORING"
    result["host_model_version"] = host_model_version_value
    return result


def build_iteration_summary(
    run_id: str,
    dataset: ReadyDatasetRecord,
    base_result: BaseScoringResult,
    ordered_results: pd.DataFrame,
    top_n: int,
) -> IterationSummary:
    """Собрать короткую агрегированную сводку по прогону."""
    if ordered_results.empty:
        min_score = None
        mean_score = None
        max_score = None
    else:
        scores = ordered_results["final_score"].astype(float)
        min_score = float(scores.min())
        mean_score = float(scores.mean())
        max_score = float(scores.max())

    return IterationSummary(
        run_id=run_id,
        relation_name=dataset.relation_name,
        source_name=dataset.source_name,
        input_rows=int(len(base_result.input_df)),
        router_rows=int(len(base_result.router_df)),
        host_rows=int(len(base_result.host_input_df)),
        low_rows=int(len(base_result.low_input_df)),
        final_score_min=min_score,
        final_score_mean=mean_score,
        final_score_max=max_score,
        top_n=int(top_n),
    )


def top_candidates_frame(
    ordered_results: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    """Вернуть top-N кандидатов с ключевыми полями."""
    columns = [
        "source_id",
        "predicted_spec_class",
        "predicted_evolution_stage",
        "gauss_label",
        "router_similarity",
        "similarity",
        "class_prior",
        "distance_factor",
        "quality_factor",
        "metallicity_factor",
        "final_score",
        "priority_tier",
        "reason_code",
        "ra",
        "dec",
        "teff_gspphot",
        "logg_gspphot",
        "radius_gspphot",
        "mh_gspphot",
        "parallax",
        "parallax_over_error",
        "ruwe",
    ]
    existing = [column for column in columns if column in ordered_results.columns]
    return ordered_results.loc[:, existing].head(top_n).copy()


def class_distribution_frame(
    top_candidates: pd.DataFrame,
) -> pd.DataFrame:
    """Посчитать распределение классов в top-N."""
    if top_candidates.empty:
        return pd.DataFrame(
            columns=["predicted_spec_class", "count"]
        )
    counts = (
        top_candidates["predicted_spec_class"]
        .astype(str)
        .value_counts(dropna=False)
        .rename_axis("predicted_spec_class")
        .reset_index(name="count")
    )
    return counts


def score_summary_frame(summary: IterationSummary) -> pd.DataFrame:
    """Собрать CSV-friendly summary key/value."""
    records: list[SummaryRecord] = [
        ("run_id", summary.run_id),
        ("relation_name", summary.relation_name),
        ("source_name", summary.source_name),
        ("input_rows", summary.input_rows),
        ("router_rows", summary.router_rows),
        ("host_rows", summary.host_rows),
        ("low_rows", summary.low_rows),
        ("final_score_min", summary.final_score_min),
        ("final_score_mean", summary.final_score_mean),
        ("final_score_max", summary.final_score_max),
        ("top_n", summary.top_n),
        ("calibrator_version", summary.calibrator_version),
    ]
    return pd.DataFrame.from_records(records, columns=["metric", "value"])


def frame_to_text(df: pd.DataFrame) -> str:
    """Преобразовать DataFrame в компактный текстовый блок."""
    if df.empty:
        return "Пусто"
    return df.to_string(index=False)


def build_iteration_markdown(
    iteration_id: str,
    config: CalibrationConfig,
    summary: IterationSummary,
    top_candidates: pd.DataFrame,
    class_distribution: pd.DataFrame,
    iteration_note: str,
) -> str:
    """Собрать markdown-отчёт по текущей калибровочной итерации."""
    created_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    note = iteration_note.strip() or "-"
    return f"""# {iteration_id}

Дата: {created_at}
Run ID: {summary.run_id}
Статус: выполнено

## Что меняли
- class_prior
- metallicity_factor
- distance_factor
- quality_factor

## Формула
`final_score = similarity × class_prior × distance_factor × quality_factor × metallicity_factor`

## Параметры итерации
### class_prior
- K: {config.class_prior.k:.3f}
- G: {config.class_prior.g:.3f}
- M: {config.class_prior.m:.3f}
- F: {config.class_prior.f:.3f}

### metallicity_factor
- mh <= {config.metallicity.low_threshold:.2f}: {config.metallicity.low_factor:.2f}
- mh < {config.metallicity.solar_threshold:.2f}: {config.metallicity.neutral_factor:.2f}
- mh < {config.metallicity.high_threshold:.2f}: {config.metallicity.positive_factor:.2f}
- mh >= {config.metallicity.high_threshold:.2f}: {config.metallicity.high_factor:.2f}

### distance_factor
- distance <= {config.distance.near_max_pc:.0f} pc: {config.distance.near_factor:.2f}
- distance <= {config.distance.moderate_max_pc:.0f} pc: {config.distance.moderate_factor:.2f}
- distance <= {config.distance.distant_max_pc:.0f} pc: {config.distance.distant_factor:.2f}
- distance <= {config.distance.far_max_pc:.0f} pc: {config.distance.far_factor:.2f}
- distance > {config.distance.far_max_pc:.0f} pc: {config.distance.very_far_factor:.2f}
- invalid distance: {config.distance.invalid_factor:.2f}

### quality_factor
- quality_factor = ruwe_factor × parallax_precision_factor
- ruwe <= {config.quality.ruwe.good_max:.1f}: {config.quality.ruwe.good_factor:.2f}
- ruwe <= {config.quality.ruwe.warning_max:.1f}: {config.quality.ruwe.warning_factor:.2f}
- ruwe <= {config.quality.ruwe.alert_max:.1f}: {config.quality.ruwe.alert_factor:.2f}
- ruwe <= {config.quality.ruwe.bad_max:.1f}: {config.quality.ruwe.bad_factor:.2f}
- ruwe > {config.quality.ruwe.bad_max:.1f}: {config.quality.ruwe.very_bad_factor:.2f}
- parallax_over_error >= {config.quality.parallax_precision.excellent_min:.0f}: {config.quality.parallax_precision.excellent_factor:.2f}
- parallax_over_error >= {config.quality.parallax_precision.good_min:.0f}: {config.quality.parallax_precision.good_factor:.2f}
- parallax_over_error >= {config.quality.parallax_precision.acceptable_min:.0f}: {config.quality.parallax_precision.acceptable_factor:.2f}
- parallax_over_error >= {config.quality.parallax_precision.weak_min:.0f}: {config.quality.parallax_precision.weak_factor:.2f}
- parallax_over_error < {config.quality.parallax_precision.weak_min:.0f}: {config.quality.parallax_precision.poor_factor:.2f}

## Короткая сводка
- relation: `{summary.relation_name}`
- source_name: `{summary.source_name}`
- input_rows: {summary.input_rows}
- router_rows: {summary.router_rows}
- host_rows: {summary.host_rows}
- low_rows: {summary.low_rows}
- final_score_min: {summary.final_score_min}
- final_score_mean: {summary.final_score_mean}
- final_score_max: {summary.final_score_max}

## Примечание к итерации
- {note}

## Распределение классов в top-{summary.top_n}
```text
{frame_to_text(class_distribution)}
```

## Top-{summary.top_n} кандидатов
```text
{frame_to_text(top_candidates)}
```
"""


def save_iteration_artifacts(
    logbook_dir: Path,
    config: CalibrationConfig,
    summary: IterationSummary,
    ordered_results: pd.DataFrame,
    top_n: int,
    iteration_note: str,
) -> Path:
    """Сохранить markdown и CSV-артефакты калибровочной итерации."""
    iteration_number = next_iteration_number(logbook_dir)
    iteration_id = f"iteration_{iteration_number:03d}"
    prefix = logbook_dir / iteration_id

    top_candidates = top_candidates_frame(ordered_results, top_n)
    class_distribution = class_distribution_frame(top_candidates)
    score_summary = score_summary_frame(summary)

    markdown_path = prefix.with_suffix(".md")
    config_path = prefix.parent / f"{iteration_id}_config.json"
    top_path = prefix.parent / f"{iteration_id}_top_candidates.csv"
    score_path = prefix.parent / f"{iteration_id}_score_summary.csv"
    class_path = prefix.parent / f"{iteration_id}_class_distribution.csv"

    markdown_path.write_text(
        build_iteration_markdown(
            iteration_id=iteration_id,
            config=config,
            summary=summary,
            top_candidates=top_candidates,
            class_distribution=class_distribution,
            iteration_note=iteration_note,
        ),
        encoding="utf-8",
    )
    config_path.write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    top_candidates.to_csv(top_path, index=False)
    score_summary.to_csv(score_path, index=False)
    class_distribution.to_csv(class_path, index=False)

    return markdown_path


def print_summary(
    summary: IterationSummary,
    markdown_path: Path,
    top_candidates: pd.DataFrame,
) -> None:
    """Вывести короткий итог в терминал."""
    print("=== DECISION LAYER CALIBRATION ===")
    print(f"Run ID: {summary.run_id}")
    print(f"Relation: {summary.relation_name}")
    print(f"Source name: {summary.source_name}")
    print(f"Input rows: {summary.input_rows}")
    print(f"Host rows: {summary.host_rows}")
    print(f"Low rows: {summary.low_rows}")
    print(f"final_score min/mean/max: "
          f"{summary.final_score_min} / "
          f"{summary.final_score_mean} / "
          f"{summary.final_score_max}")
    print(f"Markdown summary: {markdown_path}")
    print("Top preview:")
    print(frame_to_text(top_candidates.head(min(10, len(top_candidates)))))


def main() -> None:
    """Запустить одну offline-итерацию калибровки decision layer."""
    args = parse_args()
    engine = make_engine_from_env()

    dataset, df_input = load_ready_input_dataset(
        engine=engine,
        relation_name=args.relation,
        limit=args.limit,
    )
    router_model, host_model = load_models()
    config = load_calibration_config(args.config)
    run_id = make_run_id()

    base_result = run_base_scoring(
        df_input=df_input,
        router_model=router_model,
        host_model=host_model,
    )
    host_version = host_model_version(dict(host_model))
    scored = apply_calibration_config(
        df_scored=base_result.host_scored_df,
        config=config,
        host_model_version_value=host_version,
    )
    low_preview = build_low_priority_preview(base_result.low_input_df)
    combined = pd.concat([scored, low_preview], ignore_index=True, sort=False)
    ordered_results = order_priority_results(combined)

    summary = build_iteration_summary(
        run_id=run_id,
        dataset=dataset,
        base_result=base_result,
        ordered_results=ordered_results,
        top_n=args.top_n,
    )
    logbook_dir = ensure_logbook_dir()
    markdown_path = save_iteration_artifacts(
        logbook_dir=logbook_dir,
        config=config,
        summary=summary,
        ordered_results=ordered_results,
        top_n=args.top_n,
        iteration_note=args.iteration_note,
    )
    top_candidates = top_candidates_frame(ordered_results, args.top_n)
    print_summary(summary, markdown_path, top_candidates)


if __name__ == "__main__":
    main()
