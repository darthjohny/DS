# gaussian_router.py
# ============================================================
# Назначение
# ------------------------------------------------------------
# Физический Gaussian-router для проекта ВКР
# (Gaia DR3 + NASA hosts).
#
# Цель файла:
# обучить многомерную гауссову модель
# на reference-звёздах Gaia
# и затем использовать её
# для первичного распознавания
# физической природы звезды.
#
# Что делает этот файл:
# 1. Загружает reference-выборку из БД
#    (lab.v_gaia_router_training).
# 2. Выполняет глобальную
#    нормализацию признаков
#    teff_gspphot / logg_gspphot / radius_gspphot.
# 3. Строит гауссианы по router-классам:
#    A_dwarf, A_evolved, ..., M_dwarf, M_evolved.
# 4. Для новой звезды ищет
#    ближайшее гауссово облако
#    через Mahalanobis distance.
# 5. Возвращает распознанный
#    спектральный класс
#    и эволюционную стадию (dwarf / evolved).
# 6. Сохраняет и загружает
#    обученную router-модель с диска.
#
# Важно:
# - здесь нет final ranking;
# - здесь нет host-like similarity;
# - здесь нет записи результатов в БД;
# - здесь нет decision layer.
#
# То есть:
# gaussian_router.py отвечает только
# за физическое распознавание звезды,
# а дальнейший пайплайн будет выполняться
# в star_orchestrator.py.
# ============================================================

"""Гауссов роутер для первичного
физического распознавания звёзд."""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, Iterable, List, Tuple, TypedDict

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.engine import Engine

FEATURES: List[str] = ["teff_gspphot", "logg_gspphot", "radius_gspphot"]
SPEC_CLASSES: List[str] = ["A", "B", "F", "G", "K", "M", "O"]
EVOLUTION_STAGES: List[str] = ["dwarf", "evolved"]
ROUTER_VIEW = "lab.v_gaia_router_training"
ROUTER_MODEL_VERSION = "gaussian_router_v1"
EPS = 1e-12


class RouterClassParams(TypedDict):
    """Serialized Gaussian parameters for one router class."""

    n: int
    spec_class: str
    evolution_stage: str
    mu: List[float]
    cov: List[List[float]]
    inv_cov: List[List[float]]


class RouterMeta(TypedDict):
    """Metadata stored alongside the router model."""

    model_version: str
    source_view: str
    shrink_alpha: float
    min_class_size: int


class RouterModel(TypedDict):
    """Top-level serialized router model."""

    global_mu: List[float]
    global_sigma: List[float]
    classes: Dict[str, RouterClassParams]
    features: List[str]
    meta: RouterMeta


class RouterScoreResult(TypedDict):
    """Typed score payload returned by router scoring helpers."""

    predicted_spec_class: str
    predicted_evolution_stage: str
    router_label: str
    d_mahal_router: float
    router_similarity: float
    second_best_label: str
    margin: float
    model_version: str


def _load_dotenv_local(path: str = ".env") -> None:
    """Load .env values into the process environment without overwriting."""
    if not os.path.exists(path):
        return

    try:
        with open(path, "r", encoding="utf-8") as file:
            for raw in file:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except OSError:
        return


def make_engine_from_env() -> Engine:
    """Create SQLAlchemy engine from DATABASE_URL or PG* variables."""
    _load_dotenv_local(".env")
    url = os.getenv("DATABASE_URL")

    if url:
        bad_tokens = ["HOST", "USER", "PASSWORD", "DBNAME"]
        if any(token in url for token in bad_tokens):
            raise RuntimeError(
                "DATABASE_URL looks like a placeholder. Provide a real DSN."
            )
        return create_engine(url)

    host = os.getenv("PGHOST")
    port = os.getenv("PGPORT", "5432")
    dbname = os.getenv("PGDATABASE")
    user = os.getenv("PGUSER")
    password = os.getenv("PGPASSWORD")

    if all([host, dbname, user, password]):
        return create_engine(
            f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"
        )

    raise RuntimeError(
        "Database connection is missing. Set DATABASE_URL or PG* variables."
    )


def load_router_training_from_db(
    engine: Engine,
    view_name: str = ROUTER_VIEW,
) -> pd.DataFrame:
    """Load the Gaia router reference layer from DB."""
    if "." in view_name:
        schema, rel = view_name.split(".", 1)
    else:
        schema, rel = "public", view_name

    inspector = sa_inspect(engine)
    has_rel = (
        rel in inspector.get_table_names(schema=schema)
        or rel in inspector.get_view_names(schema=schema)
    )
    if not has_rel:
        raise RuntimeError(
            f"Router training source does not exist: {view_name}"
        )

    query = f"""
    SELECT
        source_id,
        spec_class,
        evolution_stage,
        {", ".join(FEATURES)}
    FROM {schema}.{rel}
    WHERE spec_class IN ('A','B','F','G','K','M','O')
      AND evolution_stage IN ('dwarf','evolved');
    """
    return pd.read_sql(query, engine)


def zscore_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate z-score parameters."""
    mu = X.mean(axis=0)
    sigma = X.std(axis=0, ddof=0)
    sigma = np.where(np.abs(sigma) < EPS, EPS, sigma)
    return mu, sigma


def zscore_apply(
    X: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    """Apply z-score normalization."""
    return (X - mu) / sigma


def cov_sample(X: np.ndarray) -> np.ndarray:
    """Sample covariance with ddof=1."""
    if X.shape[0] < 2:
        raise ValueError(
            "At least 2 rows are required to estimate covariance."
        )
    return np.cov(X, rowvar=False, ddof=1)


def shrink_covariance(cov_matrix: np.ndarray, alpha: float) -> np.ndarray:
    """Shrink covariance toward its diagonal."""
    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1].")
    diag = np.diag(np.diag(cov_matrix))
    return (1.0 - alpha) * cov_matrix + alpha * diag


def mahalanobis_distance(
    x: np.ndarray,
    mu: np.ndarray,
    inv_cov: np.ndarray,
) -> float:
    """Return Mahalanobis distance."""
    delta = x - mu
    value = float(delta.T @ inv_cov @ delta)
    return math.sqrt(max(value, 0.0))


def similarity_from_distance(distance: float) -> float:
    """Convert Mahalanobis distance into similarity in [0, 1]."""
    return 1.0 / (1.0 + float(distance))


def is_missing_scalar(value: Any) -> bool:
    """Return True for None / NaN / pd.NA."""
    if value is None or value is pd.NA:
        return True
    try:
        return bool(math.isnan(float(value)))
    except (TypeError, ValueError):
        return False


def has_missing_values(values: Iterable[Any]) -> bool:
    """Return True if any core feature is missing."""
    return any(is_missing_scalar(value) for value in values)


def normalize_spec_class(spec_class: Any) -> str:
    """Normalize spectrum label to the router contract."""
    value = str(spec_class).strip().upper()
    if value not in SPEC_CLASSES:
        raise ValueError(f"Unsupported spec_class: {spec_class}")
    return value


def normalize_evolution_stage(evolution_stage: Any) -> str:
    """Normalize evolution stage to the router contract."""
    value = str(evolution_stage).strip().lower()
    if value not in EVOLUTION_STAGES:
        raise ValueError(f"Unsupported evolution_stage: {evolution_stage}")
    return value


def make_router_label(spec_class: Any, evolution_stage: Any) -> str:
    """Build the combined router label."""
    return (
        f"{normalize_spec_class(spec_class)}_"
        f"{normalize_evolution_stage(evolution_stage)}"
    )


def split_router_label(router_label: str) -> Tuple[str, str]:
    """Split combined router label into class and stage."""
    try:
        spec_class, evolution_stage = router_label.split("_", 1)
    except ValueError as exc:
        raise ValueError(f"Invalid router label: {router_label}") from exc
    return spec_class, evolution_stage


def _empty_router_score(model_version: str) -> RouterScoreResult:
    """Return a neutral payload for unsupported or incomplete rows."""
    return {
        "predicted_spec_class": "UNKNOWN",
        "predicted_evolution_stage": "unknown",
        "router_label": "UNKNOWN",
        "d_mahal_router": float("nan"),
        "router_similarity": 0.0,
        "second_best_label": "UNKNOWN",
        "margin": float("nan"),
        "model_version": model_version,
    }


def fit_router_model(
    df_router: pd.DataFrame,
    shrink_alpha: float = 0.15,
    min_class_size: int = 3,
    source_view: str = ROUTER_VIEW,
) -> RouterModel:
    """Fit Gaussian router for spec_class + evolution_stage labels."""
    required = ["spec_class", "evolution_stage"] + FEATURES
    missing = [col for col in required if col not in df_router.columns]
    if missing:
        raise ValueError(f"Missing required columns in df_router: {missing}")

    df = df_router.dropna(subset=required).copy()
    if df.empty:
        raise ValueError("No training rows remain after dropping NULL values.")

    # Router labels фиксируются заранее,
    # чтобы одно и то же сочетание
    # spec_class + evolution_stage
    # всегда мапилось в один класс модели.
    df["router_label"] = [
        make_router_label(
            spec_class=spec_class,
            evolution_stage=evolution_stage,
        )
        for spec_class, evolution_stage in zip(
            df["spec_class"], df["evolution_stage"]
        )
    ]

    X_all = df[FEATURES].astype(float).to_numpy()
    z_mu, z_sigma = zscore_fit(X_all)

    classes: Dict[str, RouterClassParams] = {}
    for router_label, subset in df.groupby("router_label", sort=True):
        n = int(subset.shape[0])
        if n < min_class_size:
            continue

        X = subset[FEATURES].astype(float).to_numpy()
        Xz = zscore_apply(X, z_mu, z_sigma)
        mu_z = Xz.mean(axis=0)
        cov_matrix = cov_sample(Xz)
        cov_matrix = shrink_covariance(cov_matrix, alpha=shrink_alpha)

        try:
            inv_cov = np.linalg.inv(cov_matrix)
        except np.linalg.LinAlgError:
            # Малый jitter сохраняет
            # стабильность инверсии
            # на границе вырожденных ковариаций.
            jitter = 1e-6
            inv_cov = np.linalg.inv(
                cov_matrix + jitter * np.eye(cov_matrix.shape[0])
            )

        router_label_str = str(router_label)
        spec_class, evolution_stage = split_router_label(router_label_str)
        classes[router_label_str] = {
            "n": n,
            "spec_class": spec_class,
            "evolution_stage": evolution_stage,
            "mu": mu_z.tolist(),
            "cov": cov_matrix.tolist(),
            "inv_cov": inv_cov.tolist(),
        }

    if not classes:
        raise ValueError(
            "No router classes were fitted. "
            "Check class counts and training filters."
        )

    return {
        "global_mu": z_mu.tolist(),
        "global_sigma": z_sigma.tolist(),
        "classes": classes,
        "features": FEATURES,
        "meta": {
            "model_version": ROUTER_MODEL_VERSION,
            "source_view": source_view,
            "shrink_alpha": float(shrink_alpha),
            "min_class_size": int(min_class_size),
        },
    }


def score_router_one(
    model: RouterModel,
    teff: Any,
    logg: Any,
    radius: Any,
) -> RouterScoreResult:
    """Score one star against all router Gaussian classes."""
    model_version = model["meta"]["model_version"]
    if has_missing_values([teff, logg, radius]):
        return _empty_router_score(model_version=model_version)

    classes = model.get("classes", {})
    if not classes:
        return _empty_router_score(model_version=model_version)

    z_mu = np.array(model["global_mu"], dtype=float)
    z_sigma = np.array(model["global_sigma"], dtype=float)
    x = np.array([float(teff), float(logg), float(radius)], dtype=float)
    xz = zscore_apply(x, z_mu, z_sigma)

    ranked: List[Tuple[str, float]] = []
    for router_label, params in classes.items():
        mu = np.array(params["mu"], dtype=float)
        inv_cov = np.array(params["inv_cov"], dtype=float)
        distance = mahalanobis_distance(xz, mu, inv_cov)
        ranked.append((router_label, distance))

    ranked.sort(key=lambda item: item[1])
    best_label, best_distance = ranked[0]
    second_label = "UNKNOWN"
    margin = float("nan")
    if len(ranked) > 1:
        second_label = ranked[1][0]
        margin = float(ranked[1][1] - best_distance)

    spec_class, evolution_stage = split_router_label(best_label)
    return {
        "predicted_spec_class": spec_class,
        "predicted_evolution_stage": evolution_stage,
        "router_label": best_label,
        "d_mahal_router": float(best_distance),
        "router_similarity": float(similarity_from_distance(best_distance)),
        "second_best_label": second_label,
        "margin": margin,
        "model_version": model_version,
    }


def score_router_df(
    model: RouterModel,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Score a DataFrame against the router model."""
    missing = [col for col in FEATURES if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in df: {missing}")

    rows: List[RouterScoreResult] = []
    for teff, logg, radius in df[FEATURES].itertuples(index=False, name=None):
        rows.append(
            score_router_one(
                model=model,
                teff=teff,
                logg=logg,
                radius=radius,
            )
        )

    result = df.copy()
    result["predicted_spec_class"] = [
        item["predicted_spec_class"] for item in rows
    ]
    result["predicted_evolution_stage"] = [
        item["predicted_evolution_stage"] for item in rows
    ]
    result["router_label"] = [item["router_label"] for item in rows]
    result["d_mahal_router"] = [item["d_mahal_router"] for item in rows]
    result["router_similarity"] = [item["router_similarity"] for item in rows]
    result["second_best_label"] = [item["second_best_label"] for item in rows]
    result["margin"] = [item["margin"] for item in rows]
    result["model_version"] = [item["model_version"] for item in rows]
    return result


def save_router_model(model: RouterModel, path: str) -> None:
    """Save router model as JSON."""
    with open(path, "w", encoding="utf-8") as file:
        json.dump(model, file, ensure_ascii=False, indent=2)


def load_router_model(path: str) -> RouterModel:
    """Load router model from JSON."""
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


if __name__ == "__main__":
    engine = make_engine_from_env()
    df_router = load_router_training_from_db(engine)
    model = fit_router_model(df_router=df_router)

    os.makedirs("data", exist_ok=True)
    output_path = "data/router_gaussian_params.json"
    save_router_model(model, output_path)
    print(f"Saved router model to {output_path}")

    sample = score_router_df(model=model, df=df_router.head(10))
    preview = sample[
        [
            "spec_class",
            "evolution_stage",
            "predicted_spec_class",
            "predicted_evolution_stage",
            "router_label",
            "d_mahal_router",
            "router_similarity",
        ]
    ]
    print(preview.to_string(index=False))
