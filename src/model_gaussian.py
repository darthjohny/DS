# model_gaussian.py
# ============================================================
# Назначение
# ------------------------------------------------------------
# Математическое ядро для Gaussian similarity / Mahalanobis
# в проекте ВКР (Gaia DR3 + NASA hosts).
#
# Цель файла:
# обучить многомерную гауссову модель
# на звёздах главной последовательности
# классов M/K/G/F
# и затем использовать её
# для оценки физической похожести объектов.
#
# Что делает этот файл:
# 1. Загружает обучающую выборку DWARFS из БД
#    (только объекты с logg >= 4.0).
# 2. Выполняет глобальную нормализацию признаков
#    teff_gspphot / logg_gspphot / radius_gspphot.
# 3. Считает параметры гауссианы по классам:
#    mu, covariance, shrinkage-covariance.
# 4. Поддерживает подклассы M-карликов:
#    M_EARLY / M_MID / M_LATE.
# 5. Считает Mahalanobis distance
#    и similarity score для новых объектов.
# 6. Сохраняет и загружает обученную модель с диска.
#
# Важно:
# - здесь нет EDA и визуализаций;
# - здесь нет финального ranking layer;
# - здесь нет маршрутизации A/B/O / evolved;
# - здесь нет записи итоговых результатов в БД.
#
# То есть:
# model_gaussian.py отвечает только
# за физическое Gaussian-ядро,
# а orchestration / ranking
# будет вынесен в отдельный файл пайплайна.
# ============================================================

"""Гауссово ядро физической похожести для MKGF-карликов."""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Tuple, TypedDict

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.engine import Engine

FEATURES: List[str] = ["teff_gspphot", "logg_gspphot", "radius_gspphot"]
DWARF_CLASSES: List[str] = ["M", "K", "G", "F"]
LOGG_DWARF_MIN = 4.0
M_EARLY_MIN = 3500.0
M_EARLY_MAX = 4000.0
M_MID_MIN = 3200.0
M_MID_MAX = 3500.0
M_LATE_MAX = 3200.0
EPS = 1e-12


class ScoreResult(TypedDict):
    """Typed score payload returned by Gaussian scoring helpers."""

    label: str
    d_mahal: float
    similarity: float


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


def load_dwarfs_from_db(
    engine: Engine,
    view_name: str = "lab.v_nasa_gaia_train_dwarfs",
) -> pd.DataFrame:
    """Load MKGF dwarfs for Gaussian training."""
    if "." in view_name:
        schema, rel = view_name.split(".", 1)
    else:
        schema, rel = "public", view_name

    inspector = sa_inspect(engine)
    has_rel = (
        rel in inspector.get_table_names(schema=schema)
        or rel in inspector.get_view_names(schema=schema)
    )

    if has_rel:
        source = f"{schema}.{rel}"
        # Prefer the dedicated dwarfs view when it is available.
        query = f"""
        SELECT spec_class, {", ".join(FEATURES)}
        FROM {source}
        WHERE spec_class IN ('M','K','G','F');
        """
    else:
        source = "lab.v_nasa_gaia_train_classified"
        query = f"""
        SELECT spec_class, {", ".join(FEATURES)}
        FROM {source}
        WHERE spec_class IN ('M','K','G','F')
          AND logg_gspphot >= {LOGG_DWARF_MIN};
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


def similarity_from_distance(d: float) -> float:
    """Convert Mahalanobis distance into similarity in [0, 1]."""
    return 1.0 / (1.0 + float(d))


def split_m_subclasses(df_m: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Split M dwarfs into early/mid/late subclasses by Teff."""
    teff = df_m["teff_gspphot"].astype(float)
    return {
        "M_EARLY": df_m[(teff >= M_EARLY_MIN) & (teff < M_EARLY_MAX)].copy(),
        "M_MID": df_m[(teff >= M_MID_MIN) & (teff < M_MID_MAX)].copy(),
        "M_LATE": df_m[teff < M_LATE_MAX].copy(),
    }


def choose_m_subclass_label(teff: float) -> str:
    """Choose M subclass label from Teff."""
    if M_EARLY_MIN <= teff < M_EARLY_MAX:
        return "M_EARLY"
    if M_MID_MIN <= teff < M_MID_MAX:
        return "M_MID"
    if teff < M_LATE_MAX:
        return "M_LATE"
    return "M_UNKNOWN"


def is_missing_scalar(value: Any) -> bool:
    """Return True for None / NaN / pd.NA."""
    if value is None or value is pd.NA:
        return True
    try:
        return bool(math.isnan(float(value)))
    except (TypeError, ValueError):
        return False


def has_missing_values(teff: Any, logg: Any, radius: Any) -> bool:
    """Return True if any core feature is missing."""
    return (
        is_missing_scalar(teff)
        or is_missing_scalar(logg)
        or is_missing_scalar(radius)
    )


def _empty_score(label: str) -> ScoreResult:
    """Return a neutral score payload for missing or unsupported cases."""
    return {
        "label": label,
        "d_mahal": float("nan"),
        "similarity": 0.0,
    }


def fit_gaussian_model(
    df_dwarfs: pd.DataFrame,
    use_m_subclasses: bool = True,
    shrink_alpha: float = 0.15,
) -> Dict[str, Any]:
    """Fit Gaussian model for MKGF dwarfs."""
    required = ["spec_class"] + FEATURES
    missing = [col for col in required if col not in df_dwarfs.columns]
    if missing:
        raise ValueError(f"Missing required columns in df_dwarfs: {missing}")

    df = df_dwarfs.dropna(subset=required).copy()
    if df.empty:
        raise ValueError("No training rows remain after dropping NULL values.")

    X_all = df[FEATURES].astype(float).to_numpy()
    z_mu, z_sigma = zscore_fit(X_all)

    subsets: Dict[str, pd.DataFrame] = {
        cls: df[df["spec_class"] == cls].copy() for cls in DWARF_CLASSES
    }
    if use_m_subclasses and "M" in subsets:
        # M dwarfs are modeled with narrower subclasses to reduce covariance
        # smearing across the full M temperature range.
        subsets.pop("M", None)
        subsets.update(split_m_subclasses(df[df["spec_class"] == "M"].copy()))

    classes: Dict[str, Dict[str, Any]] = {}
    for label, subset in subsets.items():
        n = int(subset.shape[0])
        if n < 3:
            continue

        X = subset[FEATURES].astype(float).to_numpy()
        Xz = zscore_apply(X, z_mu, z_sigma)
        mu_z = Xz.mean(axis=0)
        cov_matrix = cov_sample(Xz)
        cov_matrix = shrink_covariance(cov_matrix, alpha=shrink_alpha)

        try:
            inv_cov = np.linalg.inv(cov_matrix)
        except np.linalg.LinAlgError:
            # Keep scoring robust when a covariance matrix is near-singular.
            jitter = 1e-6
            inv_cov = np.linalg.inv(
                cov_matrix + jitter * np.eye(cov_matrix.shape[0])
            )

        classes[label] = {
            "n": n,
            "mu": mu_z.tolist(),
            "cov": cov_matrix.tolist(),
            "inv_cov": inv_cov.tolist(),
        }

    if not classes:
        raise ValueError(
            "No Gaussian classes were fitted. "
            "Check input filters and class counts."
        )

    return {
        "global_mu": z_mu.tolist(),
        "global_sigma": z_sigma.tolist(),
        "classes": classes,
        "features": FEATURES,
        "meta": {
            "logg_dwarf_min": LOGG_DWARF_MIN,
            "use_m_subclasses": bool(use_m_subclasses),
            "shrink_alpha": float(shrink_alpha),
        },
    }


def score_one(
    model: Dict[str, Any],
    spec_class: str,
    teff: Any,
    logg: Any,
    radius: Any,
) -> ScoreResult:
    """Score one star against the class selected by spec_class."""
    if has_missing_values(teff, logg, radius):
        return _empty_score(str(spec_class))

    teff_val = float(teff)
    logg_val = float(logg)
    radius_val = float(radius)

    classes = model.get("classes", {})
    label = str(spec_class)
    if label == "M" and "M_EARLY" in classes:
        label = choose_m_subclass_label(teff_val)

    if label not in classes:
        return _empty_score(label)

    z_mu = np.array(model["global_mu"], dtype=float)
    z_sigma = np.array(model["global_sigma"], dtype=float)
    x = np.array([teff_val, logg_val, radius_val], dtype=float)
    xz = zscore_apply(x, z_mu, z_sigma)

    params = classes[label]
    mu = np.array(params["mu"], dtype=float)
    inv_cov = np.array(params["inv_cov"], dtype=float)
    d_mahal = mahalanobis_distance(xz, mu, inv_cov)

    return {
        "label": label,
        "d_mahal": float(d_mahal),
        "similarity": float(similarity_from_distance(d_mahal)),
    }


def score_one_all_classes(
    model: Dict[str, Any],
    teff: Any,
    logg: Any,
    radius: Any,
) -> ScoreResult:
    """Score one star against all Gaussian classes and take the best match."""
    if has_missing_values(teff, logg, radius):
        return _empty_score("UNKNOWN")

    classes = model.get("classes", {})
    if not classes:
        return _empty_score("UNKNOWN")

    z_mu = np.array(model["global_mu"], dtype=float)
    z_sigma = np.array(model["global_sigma"], dtype=float)
    x = np.array([float(teff), float(logg), float(radius)], dtype=float)
    xz = zscore_apply(x, z_mu, z_sigma)

    best_label = "UNKNOWN"
    best_d = float("inf")
    for label, params in classes.items():
        mu = np.array(params["mu"], dtype=float)
        inv_cov = np.array(params["inv_cov"], dtype=float)
        distance = mahalanobis_distance(xz, mu, inv_cov)
        if distance < best_d:
            best_d = distance
            best_label = label

    if best_label == "UNKNOWN":
        return _empty_score("UNKNOWN")

    return {
        "label": best_label,
        "d_mahal": float(best_d),
        "similarity": float(similarity_from_distance(best_d)),
    }


def score_df(
    model: Dict[str, Any],
    df: pd.DataFrame,
    spec_class_col: str = "spec_class",
) -> pd.DataFrame:
    """Score a DataFrame using the provided spec_class column."""
    required = [spec_class_col] + FEATURES
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column in df: {col}")

    rows: List[ScoreResult] = []
    for _, row in df.iterrows():
        rows.append(
            score_one(
                model=model,
                spec_class=str(row[spec_class_col]),
                teff=row["teff_gspphot"],
                logg=row["logg_gspphot"],
                radius=row["radius_gspphot"],
            )
        )

    result = df.copy()
    result["gauss_label"] = [item["label"] for item in rows]
    result["d_mahal"] = [item["d_mahal"] for item in rows]
    result["similarity"] = [item["similarity"] for item in rows]
    return result


def score_df_all_classes(
    model: Dict[str, Any],
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Score a DataFrame against all Gaussian classes."""
    for col in FEATURES:
        if col not in df.columns:
            raise ValueError(f"Missing required column in df: {col}")

    rows: List[ScoreResult] = []
    for _, row in df.iterrows():
        rows.append(
            score_one_all_classes(
                model=model,
                teff=row["teff_gspphot"],
                logg=row["logg_gspphot"],
                radius=row["radius_gspphot"],
            )
        )

    result = df.copy()
    result["gauss_label"] = [item["label"] for item in rows]
    result["d_mahal"] = [item["d_mahal"] for item in rows]
    result["similarity"] = [item["similarity"] for item in rows]
    return result


def save_model(model: Dict[str, Any], path: str) -> None:
    """Save model as JSON."""
    with open(path, "w", encoding="utf-8") as file:
        json.dump(model, file, ensure_ascii=False, indent=2)


def load_model(path: str) -> Dict[str, Any]:
    """Load model from JSON."""
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


if __name__ == "__main__":
    engine = make_engine_from_env()
    df_dwarfs = load_dwarfs_from_db(engine)
    model = fit_gaussian_model(
        df_dwarfs=df_dwarfs,
        use_m_subclasses=True,
        shrink_alpha=0.15,
    )

    os.makedirs("data", exist_ok=True)
    output_path = "data/model_gaussian_params.json"
    save_model(model, output_path)
    print(f"Saved Gaussian model to {output_path}")
