"""
model_gaussian.py

Упрощённый модуль для ВКР:
- Обучение многомерной гауссовой модели (3D)
  по звёздам-хостам.
- Оценка «похожести» через расстояние
  Махаланобиса.
- Работаем с 3 признаками Gaia DR3 (GSP-Phot):
  teff_gspphot, logg_gspphot, radius_gspphot

Важно:
- Обучаем модель только на карликах (logg >= 4.0).
- Эволюционировавшие (logg < 4.0) —
  отдельный слой,
  в модель не входят.
- Для M-класса используем подтипы
  (Early/Mid/Late) по Teff,
  чтобы модель была более «узкой»
  и физически однородной.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Tuple, cast

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.engine import Engine

# ------------------------------------------------------------
# 0) Константы и соглашения
# ------------------------------------------------------------

# Порядок признаков важен для mu, cov и Mahalanobis
FEATURES: List[str] = ["teff_gspphot", "logg_gspphot", "radius_gspphot"]

# Базовые классы карликов
DWARF_CLASSES: List[str] = ["M", "K", "G", "F"]

# Порог карликов
LOGG_DWARF_MIN = 4.0

# Подтипы M по Teff
M_EARLY_MIN = 3500.0
M_EARLY_MAX = 4000.0
M_MID_MIN = 3200.0
M_MID_MAX = 3500.0
M_LATE_MAX = 3200.0

# Защита от деления на ноль
EPS = 1e-12


# ------------------------------------------------------------
# 1) Мини-лоадер .env (без python-dotenv)
# ------------------------------------------------------------

def _load_dotenv_local(path: str = ".env") -> None:
    """Читает .env и кладёт переменные
    в окружение.

    Если переменная уже задана —
    не перетираем.
    """
    if not os.path.exists(path):
        return

    try:
        with open(path, "r", encoding="utf-8") as file:
            for raw in file:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip("\"").strip("'")
                if key and (key not in os.environ):
                    os.environ[key] = value
    except OSError:
        return


# ------------------------------------------------------------
# 2) Работа с БД
# ------------------------------------------------------------

def make_engine_from_env() -> Engine:
    """Создаёт SQLAlchemy engine
    из переменных окружения.

    Приоритет:
    1) DATABASE_URL
    2) PGHOST/PGPORT/PGDATABASE/PGUSER/PGPASSWORD
    """
    _load_dotenv_local(".env")
    url = os.getenv("DATABASE_URL")

    if url:
        bad_tokens = ["HOST", "USER", "PASSWORD", "DBNAME"]
        if any(tok in url for tok in bad_tokens):
            raise RuntimeError(
                "DATABASE_URL выглядит как пример с "
                "плейсхолдерами. Задай реальную "
                "строку подключения."
            )
        return create_engine(url)

    host = os.getenv("PGHOST")
    port = os.getenv("PGPORT", "5432")
    dbname = os.getenv("PGDATABASE")
    user = os.getenv("PGUSER")
    password = os.getenv("PGPASSWORD")

    if all([host, dbname, user, password]):
        url = (
            f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"
        )
        return create_engine(url)

    raise RuntimeError(
        "Не найдено подключение к БД. "
        "Задай DATABASE_URL или PG* переменные."
    )


def load_dwarfs_from_db(
    engine: Engine,
    view_name: str = "lab.v_nasa_gaia_train_dwarfs",
) -> pd.DataFrame:
    """Загружает карликов MKGF
    для обучения модели.

    Если отдельной view нет,
    берём lab.v_nasa_gaia_train_classified
    и фильтруем logg >= 4.0 прямо в SQL.
    """
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


# ------------------------------------------------------------
# 3) Математика: нормализация,
#    ковариация, расстояние
# ------------------------------------------------------------

def zscore_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Оценивает параметры z-score (mean, std)."""
    mu = X.mean(axis=0)
    sigma = X.std(axis=0, ddof=0)
    sigma = np.where(sigma < EPS, EPS, sigma)
    return mu, sigma


def zscore_apply(
    X: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    """Применяет z-score нормализацию."""
    return (X - mu) / sigma


def cov_sample(X: np.ndarray) -> np.ndarray:
    """Выборочная ковариация (ddof=1)."""
    if X.shape[0] < 2:
        raise ValueError(
            "Нужно минимум 2 наблюдения "
            "для ковариации."
        )
    return np.cov(X, rowvar=False, ddof=1)


def shrink_covariance(
    cov_matrix: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Простая shrinkage ковариации."""
    alpha = float(alpha)
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(
            "alpha должен быть в диапазоне [0, 1]."
        )
    diag = np.diag(np.diag(cov_matrix))
    return (1.0 - alpha) * cov_matrix + alpha * diag


def mahalanobis_distance(
    x: np.ndarray,
    mu: np.ndarray,
    inv_cov: np.ndarray,
) -> float:
    """d_M(x) = sqrt((x - mu)^T * inv_cov * (x - mu))"""
    d = x - mu
    v = float(d.T @ inv_cov @ d)
    v = max(v, 0.0)
    return math.sqrt(v)


def similarity_from_distance(d: float) -> float:
    """Перевод расстояния в «похожесть» (0..1)."""
    return 1.0 / (1.0 + float(d))


# ------------------------------------------------------------
# 4) Разбиение M-класса на подтипы
# ------------------------------------------------------------

def split_m_subclasses(df_m: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Делит M-карликов на подтипы."""
    teff = df_m["teff_gspphot"].astype(float)

    df_early = df_m[(teff >= M_EARLY_MIN) & (teff < M_EARLY_MAX)].copy()
    df_mid = df_m[(teff >= M_MID_MIN) & (teff < M_MID_MAX)].copy()
    df_late = df_m[(teff < M_LATE_MAX)].copy()

    return {
        "M_EARLY": df_early,
        "M_MID": df_mid,
        "M_LATE": df_late,
    }


def choose_m_subclass_label(teff: float) -> str:
    """Выбирает label подкласса M по Teff."""
    if M_EARLY_MIN <= teff < M_EARLY_MAX:
        return "M_EARLY"
    if M_MID_MIN <= teff < M_MID_MAX:
        return "M_MID"
    if teff < M_LATE_MAX:
        return "M_LATE"
    # За пределами M-диапазона не делаем
    # искусственного отнесения к M_LATE.
    return "M_UNKNOWN"


def has_missing_values(
    teff: Any,
    logg: Any,
    radius: Any,
) -> bool:
    """Проверяет, есть ли пропуски
    во входных признаках.
    """
    return bool(
        is_missing_scalar(teff)
        or is_missing_scalar(logg)
        or is_missing_scalar(radius)
    )


def is_missing_scalar(value: Any) -> bool:
    """Проверяет одно значение
    на пропуск (None / NaN / pd.NA).
    """
    if value is None or value is pd.NA:
        return True

    # Для чисел (включая numpy-скаляры)
    # проверяем NaN через math.isnan.
    try:
        return bool(math.isnan(cast(float, value)))
    except (TypeError, ValueError):
        return False


# ------------------------------------------------------------
# 5) Обучение модели
# ------------------------------------------------------------

def fit_gaussian_model(
    df_dwarfs: pd.DataFrame,
    use_m_subclasses: bool = True,
    shrink_alpha: float = 0.15,
) -> Dict[str, Any]:
    """Обучает модель по DWARFS MKGF.

    Возвращает словарь модели:
    - global_mu, global_sigma
    - classes: параметры для каждого
      класса/подкласса
    """
    required = ["spec_class"] + FEATURES
    missing = [c for c in required if c not in df_dwarfs.columns]
    if missing:
        raise ValueError(
            f"В df_dwarfs не хватает колонок: {missing}"
        )

    df = df_dwarfs.dropna(subset=required).copy()

    # Глобальная нормализация по всем DWARFS MKGF
    X_all = df[FEATURES].astype(float).to_numpy()
    z_mu, z_sigma = zscore_fit(X_all)

    # Сбор поднаборов по классам
    subsets: Dict[str, pd.DataFrame] = {}
    for cls in DWARF_CLASSES:
        subsets[cls] = df[df["spec_class"] == cls].copy()

    if use_m_subclasses and "M" in subsets:
        m_sub = split_m_subclasses(subsets["M"])
        subsets.pop("M", None)
        subsets.update(m_sub)

    classes: Dict[str, Dict[str, Any]] = {}

    for label, sdf in subsets.items():
        n = int(sdf.shape[0])
        if n < 3:
            # Слишком мало данных
            # для ковариации
            continue

        X = sdf[FEATURES].astype(float).to_numpy()
        Xz = zscore_apply(X, z_mu, z_sigma)

        mu_z = Xz.mean(axis=0)
        cov_matrix = cov_sample(Xz)
        cov_matrix = shrink_covariance(cov_matrix, alpha=shrink_alpha)

        try:
            inv_cov_matrix = np.linalg.inv(cov_matrix)
        except np.linalg.LinAlgError:
            jitter = 1e-6
            inv_cov_matrix = np.linalg.inv(
                cov_matrix + jitter * np.eye(cov_matrix.shape[0])
            )

        classes[label] = {
            "n": n,
            "mu": mu_z.tolist(),
            "cov": cov_matrix.tolist(),
            "inv_cov": inv_cov_matrix.tolist(),
        }

    model: Dict[str, Any] = {
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
    return model


# ------------------------------------------------------------
# 6) Скоринг
# ------------------------------------------------------------

def score_one(
    model: Dict[str, Any],
    spec_class: str,
    teff: Any,
    logg: Any,
    radius: Any,
) -> Dict[str, Any]:
    """Оценивает одну звезду по spec_class."""
    # Если есть пропуски,
    # корректно возвращаем NaN/0.0,
    # чтобы дальше не падать в расчётах.
    if has_missing_values(teff, logg, radius):
        return {
            "label": str(spec_class),
            "d_mahal": float("nan"),
            "similarity": 0.0,
        }

    teff_val = float(teff)
    logg_val = float(logg)
    radius_val = float(radius)

    label = spec_class
    if spec_class == "M" and "M_EARLY" in model["classes"]:
        label = choose_m_subclass_label(teff_val)

    if label not in model["classes"]:
        return {
            "label": label,
            "d_mahal": float("nan"),
            "similarity": 0.0,
        }

    z_mu = np.array(model["global_mu"], dtype=float)
    z_sigma = np.array(model["global_sigma"], dtype=float)

    x = np.array([teff_val, logg_val, radius_val], dtype=float)
    xz = zscore_apply(x, z_mu, z_sigma)

    params = model["classes"][label]
    mu = np.array(params["mu"], dtype=float)
    inv_cov = np.array(params["inv_cov"], dtype=float)

    d = mahalanobis_distance(xz, mu, inv_cov)
    s = similarity_from_distance(d)

    return {
        "label": label,
        "d_mahal": float(d),
        "similarity": float(s),
    }


def score_one_all_classes(
    model: Dict[str, Any],
    teff: Any,
    logg: Any,
    radius: Any,
) -> Dict[str, Any]:
    """Оценивает звезду по всем классам
    и берёт лучший вариант.
    """
    if has_missing_values(teff, logg, radius):
        return {
            "label": "UNKNOWN",
            "d_mahal": float("nan"),
            "similarity": 0.0,
        }

    teff_val = float(teff)
    logg_val = float(logg)
    radius_val = float(radius)

    classes = model.get("classes", {})
    if not classes:
        return {
            "label": "UNKNOWN",
            "d_mahal": float("nan"),
            "similarity": 0.0,
        }

    z_mu = np.array(model["global_mu"], dtype=float)
    z_sigma = np.array(model["global_sigma"], dtype=float)
    x = np.array([teff_val, logg_val, radius_val], dtype=float)
    xz = zscore_apply(x, z_mu, z_sigma)

    best_label = "UNKNOWN"
    best_d = float("inf")

    for label, params in classes.items():
        mu = np.array(params["mu"], dtype=float)
        inv_cov = np.array(params["inv_cov"], dtype=float)
        d = mahalanobis_distance(xz, mu, inv_cov)
        if d < best_d:
            best_d = d
            best_label = label

    if best_label == "UNKNOWN":
        return {
            "label": "UNKNOWN",
            "d_mahal": float("nan"),
            "similarity": 0.0,
        }

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
    """Применяет score_one ко всему DataFrame."""
    required = [spec_class_col] + FEATURES
    for col in required:
        if col not in df.columns:
            raise ValueError(
                f"В df не хватает колонки: {col}"
            )

    rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        out = score_one(
            model=model,
            spec_class=str(r[spec_class_col]),
            teff=r["teff_gspphot"],
            logg=r["logg_gspphot"],
            radius=r["radius_gspphot"],
        )
        rows.append(out)

    res = df.copy()
    res["gauss_label"] = [x["label"] for x in rows]
    res["d_mahal"] = [x["d_mahal"] for x in rows]
    res["similarity"] = [x["similarity"] for x in rows]
    return res


def score_df_all_classes(
    model: Dict[str, Any],
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Скоринг DataFrame без spec_class:
    выбор лучшего класса.
    """
    for col in FEATURES:
        if col not in df.columns:
            raise ValueError(
                f"В df не хватает колонки: {col}"
            )

    rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        out = score_one_all_classes(
            model=model,
            teff=r["teff_gspphot"],
            logg=r["logg_gspphot"],
            radius=r["radius_gspphot"],
        )
        rows.append(out)

    res = df.copy()
    res["gauss_label"] = [x["label"] for x in rows]
    res["d_mahal"] = [x["d_mahal"] for x in rows]
    res["similarity"] = [x["similarity"] for x in rows]
    return res


# ------------------------------------------------------------
# 7) Сохранение / загрузка модели
# ------------------------------------------------------------

def save_model(model: Dict[str, Any], path: str) -> None:
    """Сохраняет модель в JSON."""
    with open(path, "w", encoding="utf-8") as file:
        json.dump(model, file, ensure_ascii=False, indent=2)


def load_model(path: str) -> Dict[str, Any]:
    """Загружает модель из JSON."""
    with open(path, "r", encoding="utf-8") as file:
        data: Dict[str, Any] = json.load(file)
    return data


# ------------------------------------------------------------
# 8) Мини-точка входа для ручного прогона
# ------------------------------------------------------------

if __name__ == "__main__":
    print("=== FIT GaussianModel (DWARFS MKGF) ===")

    engine = make_engine_from_env()

    try:
        df_dwarfs = load_dwarfs_from_db(engine)
    except Exception as exc:
        print("\n[ОШИБКА] Не удалось")
        print("прочитать данные из БД.")
        print("Проверь DATABASE_URL или PG*")
        print("переменные окружения.")
        print("Текст ошибки:", repr(exc))
        raise

    print("Загружено DWARFS:", df_dwarfs.shape)

    model = fit_gaussian_model(
        df_dwarfs=df_dwarfs,
        use_m_subclasses=True,
        shrink_alpha=0.15,
    )

    print("Классы в модели:", sorted(model["classes"].keys()))
    for label, params in model["classes"].items():
        print(f"[{label}] n={params['n']}")

    os.makedirs("data", exist_ok=True)
    save_model(model, "data/model_gaussian_params.json")
    print("Сохранено: data/model_gaussian_params.json")
