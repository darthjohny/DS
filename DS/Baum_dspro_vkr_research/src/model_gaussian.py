

"""
model_gaussian.py
============================================================
Назначение
------------------------------------------------------------
Модуль «инструмента» для ВКР: многомерная гауссовая модель (3D)
и Mahalanobis similarity для приоритизации астрономических целей.

Мы работаем с 3 основными физическими признаками Gaia DR3 (GSP-Phot):
  - teff_gspphot   : эффективная температура (K)
  - logg_gspphot   : поверхностная гравитация (log10 cgs)
  - radius_gspphot : радиус (R_sun)

Ключевая идея:
- Для каждого спектрального класса строим 3D-распределение (μ, Σ)
- Оцениваем «похожесть» звезды на класс через расстояние Махаланобиса:
    d_M(x) = sqrt( (x-μ)^T Σ^{-1} (x-μ) )

ВАЖНО про физику/данные:
- Гауссовскую модель для «карликов» обучаем на слое DWARFS (logg >= 4.0),
  который фиксируем на стороне БД (view: lab.v_nasa_gaia_train_dwarfs).
- Эволюционировавшие (logg < 4.0) не считаются мусором — это отдельный слой
  (view: lab.v_nasa_gaia_train_evolved), но в базовую модель карликов не входят.
- A/B/O держим отдельным reference-слоем для sanity-check и OOD (не хосты).

Структура проекта (минимальная):
- fit()        : обучение μ и Σ по классам/подклассам (карлики MKGF)
- score_one()  : оценка одной звезды (класс, d_M, similarity)
- score_df()   : оценка таблицы (pandas DataFrame)
- save/load    : сохранение параметров модели в JSON (для воспроизводимости)

Примечание:
- В этом модуле мы НЕ делаем финальные “инженерные” поправки скоринга
  (ruwe, parallax, metallicity, observability). Это будет следующий слой.
============================================================
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple

import os
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy import inspect as sa_inspect
# ------------------------------------------------------------
# 4) Загрузка данных из Postgres (через SQLAlchemy)
# ------------------------------------------------------------

def _load_dotenv_local(path: str = ".env") -> None:
    """Мини-лоадер .env (без python-dotenv).

    VSCode часто запускает скрипт в новом процессе, где переменные окружения
    из твоего терминала НЕ видны. Поэтому читаем .env сами.

    Правило: если переменная уже задана в окружении — не перетираем.
    """
    if not os.path.exists(path):
        return

    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k and (k not in os.environ):
                    os.environ[k] = v
    except OSError:
        # если .env временно недоступен — просто идём дальше
        return


# ------------------------------------------------------------
# 0) Константы и соглашения
# ------------------------------------------------------------

# Единый порядок признаков (важно для μ, Σ и Mahalanobis)
FEATURES: List[str] = ["teff_gspphot", "logg_gspphot", "radius_gspphot"]

# Какие классы считаем в базовой модели карликов
DWARF_CLASSES: List[str] = ["M", "K", "G", "F"]

# Подклассы M по Teff (как в нашем EDA)
# ВАЖНО: верхняя граница M оставлена < 4000K, чтобы не залезать в K.
M_EARLY_MIN = 3500.0
M_EARLY_MAX = 4000.0
M_MID_MIN = 3200.0
M_MID_MAX = 3500.0
M_LATE_MAX = 3200.0

# Порог для главной последовательности (карлики)
LOGG_DWARF_MIN = 4.0

# Численная защита (когда матрица ковариации близка к вырожденной)
EPS = 1e-12


# ------------------------------------------------------------
# 1) Вспомогательные функции: нормализация, ковариация, shrinkage
# ------------------------------------------------------------

def zscore_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Оценивает параметры z-score нормализации (mean, std) по данным X.
    Возвращает (mu, sigma).
    """
    mu = X.mean(axis=0)
    sigma = X.std(axis=0, ddof=0)

    # Защита от деления на 0: если std очень маленькое, поднимаем до EPS
    sigma = np.where(sigma < EPS, EPS, sigma)
    return mu, sigma


def zscore_apply(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """
    Применяет z-score нормализацию.
    """
    return (X - mu) / sigma


def cov_samp(X: np.ndarray) -> np.ndarray:
    """
    Выборочная ковариация (как covar_samp/var_samp в SQL).
    X: shape (n, p)
    Возвращает Σ: shape (p, p)
    """
    if X.shape[0] < 2:
        raise ValueError("Нужно минимум 2 наблюдения для выборочной ковариации.")
    return np.cov(X, rowvar=False, ddof=1)


def shrink_covariance(S: np.ndarray, alpha: float) -> np.ndarray:
    """
    Простейший shrinkage ковариации:
      Σ_shrink = (1 - alpha) * S + alpha * diag(S)

    Где alpha ∈ [0, 1].
    - alpha=0 : без shrinkage
    - alpha=1 : только диагональная ковариация (считаем признаки независимыми)

    Почему это нужно:
    - Если Σ плохо обусловлена, Σ^{-1} становится нестабильной.
    - Shrinkage «поднимает» маленькие собственные значения и стабилизирует инверсию.

    В будущем можно заменить на Ledoit–Wolf / OAS (sklearn),
    но для ВКР достаточно этой устойчивой версии.
    """
    alpha = float(alpha)
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha должен быть в диапазоне [0, 1].")

    D = np.diag(np.diag(S))
    return (1.0 - alpha) * S + alpha * D


def is_positive_definite(S: np.ndarray) -> bool:
    """
    Проверка положительной определённости через собственные значения.
    """
    try:
        eig = np.linalg.eigvalsh(S)
        return bool(np.all(eig > 0))
    except np.linalg.LinAlgError:
        return False


def condition_number(S: np.ndarray) -> float:
    """
    Число обусловленности (2-норма).
    Чем больше — тем сложнее/нестабильнее инвертировать Σ.
    """
    return float(np.linalg.cond(S))


# ------------------------------------------------------------
# 2) Mahalanobis distance и similarity
# ------------------------------------------------------------

def mahalanobis_distance(x: np.ndarray, mu: np.ndarray, inv_cov: np.ndarray) -> float:
    """
    d_M(x) = sqrt( (x-μ)^T Σ^{-1} (x-μ) )
    """
    d = x - mu
    v = float(d.T @ inv_cov @ d)
    # Численная защита от отрицательного нуля из-за ошибок округления
    v = max(v, 0.0)
    return math.sqrt(v)


def similarity_from_distance(d: float) -> float:
    """
    Перевод расстояния в «похожесть» (0..1).
    Простая монотонная функция: чем меньше d, тем больше similarity.

    Здесь используем:
      similarity = 1 / (1 + d)

    Можно заменить на exp(-0.5 * d^2), если захочешь «гауссовский» профиль.
    """
    return 1.0 / (1.0 + float(d))


# ------------------------------------------------------------
# 3) Данные модели (что сохраняем/загружаем)
# ------------------------------------------------------------

@dataclass
class ClassGaussianParams:
    """
    Параметры гауссовой модели для одного класса/подкласса.
    Все параметры оцениваются в НОРМАЛИЗОВАННОМ пространстве (z-score).
    """
    label: str                      # например: "K" или "M_EARLY"
    n: int                          # размер обучающей выборки
    z_mu: List[float]               # μ в z-пространстве (обычно ~0)
    cov: List[List[float]]          # Σ (shrinked) в z-пространстве
    inv_cov: List[List[float]]      # Σ^{-1}
    det_cov: float                  # det(Σ)
    cond_cov: float                 # cond(Σ)
    pd: bool                        # положительно определена ли Σ


@dataclass
class GaussianModel:
    """
    Главная модель.
    - global_z_mu/global_z_sigma: глобальные параметры нормализации
      (по всем DWARFS MKGF вместе)
    - classes: параметры по каждому классу/подклассу
    """
    global_z_mu: List[float]
    global_z_sigma: List[float]
    classes: Dict[str, ClassGaussianParams]


# ------------------------------------------------------------
# 4) Загрузка данных из Postgres (через SQLAlchemy)
# ------------------------------------------------------------

def make_engine_from_env() -> Engine:
    """
    Создаёт SQLAlchemy engine из переменных окружения.

    Приоритет:
      1) DATABASE_URL
         Формат:
           postgresql+psycopg2://USER:PASSWORD@HOST:PORT/DBNAME
      2) Набор PG-переменных (как у psql):
         PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD

    Частая ошибка:
    - поставить плейсхолдеры HOST/USER/PASSWORD/DBNAME (как в примере)
      и забыть заменить на реальные значения.
    
    Примечание про VSCode/Pylance:
    - Pylance часто ругается на типы SQLAlchemy (unknown/any) — это статический анализ.
      На выполнение кода это не влияет.
    - Эти аннотации (Engine + sa_inspect) нужны, чтобы подсветка была спокойнее.
    """
    _load_dotenv_local(".env")
    url = os.getenv("DATABASE_URL")

    if url:
        # Если кто-то случайно оставил плейсхолдеры, лучше упасть с понятным сообщением.
        bad_tokens = ["HOST", "USER", "PASSWORD", "DBNAME"]
        if any(tok in url for tok in bad_tokens):
            raise RuntimeError(
                "DATABASE_URL выглядит как пример с плейсхолдерами (HOST/USER/PASSWORD/DBNAME). "
                "Задай реальную строку подключения. Например: "
                "postgresql+psycopg2://myuser:mypass@localhost:5432/mydb"
            )
        return create_engine(url)

    # Фоллбек: собираем URL из PG* переменных
    host = os.getenv("PGHOST")
    port = os.getenv("PGPORT", "5432")
    dbname = os.getenv("PGDATABASE")
    user = os.getenv("PGUSER")
    password = os.getenv("PGPASSWORD")

    if all([host, dbname, user, password]):
        # ВАЖНО: если в пароле есть спецсимволы, лучше URL-энкодить.
        # Но для типичных паролей это не требуется.
        url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"
        return create_engine(url)

    raise RuntimeError(
        "Не найдено подключение к БД.\n"
        "1) Задай DATABASE_URL, или\n"
        "2) Задай PGHOST/PGPORT/PGDATABASE/PGUSER/PGPASSWORD.\n"
        "Подсказка: в терминале у тебя был export DATABASE_URL=... но там стояли плейсхолдеры."
    )


def load_dwarfs_from_db(engine: Engine, view_name: str = "lab.v_nasa_gaia_train_dwarfs") -> pd.DataFrame:
    """
    Загружает карликов MKGF для обучения модели.

    ВАЖНО (почему сейчас упало):
    - В твоей БД может не существовать отдельной вьюхи lab.v_nasa_gaia_train_dwarfs.
      Тогда мы автоматически берём базовую вьюху lab.v_nasa_gaia_train_classified
      и режем карликов по условию logg_gspphot >= 4.0 прямо в SQL.

    Это делает код устойчивым: можно жить и с отдельными вьюхами, и без них.
    """
    # 1) проверяем, существует ли relation view_name (view или table)
    if "." in view_name:
        schema, rel = view_name.split(".", 1)
    else:
        schema, rel = "public", view_name

    insp = sa_inspect(engine)
    has_rel = rel in insp.get_table_names(schema=schema) or rel in insp.get_view_names(schema=schema)

    # 2) формируем запрос
    if has_rel:
        source = f"{schema}.{rel}"
        q = f"""
        SELECT spec_class, {", ".join(FEATURES)}
        FROM {source}
        WHERE spec_class IN ('M','K','G','F');
        """
    else:
        # Фоллбек: используем уже существующую классифицированную вьюху
        # и отбираем карликов по logg >= 4.0.
        source = "lab.v_nasa_gaia_train_classified"
        q = f"""
        SELECT spec_class, {", ".join(FEATURES)}
        FROM {source}
        WHERE spec_class IN ('M','K','G','F')
          AND logg_gspphot >= {LOGG_DWARF_MIN};
        """

    df = pd.read_sql(q, engine)
    return df


# ------------------------------------------------------------
# 5) Fit: обучение μ и Σ по классам (+ подклассы M)
# ------------------------------------------------------------

def split_m_subclasses(df_m: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Делит M-карликов на подклассы по Teff.
    Возвращает словарь:
      { "M_EARLY": df, "M_MID": df, "M_LATE": df }
    """
    teff = df_m["teff_gspphot"].astype(float)

    df_early = df_m[(teff >= M_EARLY_MIN) & (teff < M_EARLY_MAX)].copy()
    df_mid = df_m[(teff >= M_MID_MIN) & (teff < M_MID_MAX)].copy()
    df_late = df_m[(teff < M_LATE_MAX)].copy()

    return {
        "M_EARLY": df_early,
        "M_MID": df_mid,
        "M_LATE": df_late,
    }


def fit_gaussian_model(
    df_dwarfs: pd.DataFrame,
    use_m_subclasses: bool = True,
    shrink_alpha: float = 0.15,
) -> GaussianModel:
    """
    Обучает гауссовые параметры по DWARFS MKGF.

    Что делаем:
    1) Глобальная нормализация z-score (по всем DWARFS MKGF).
       Это стабилизирует масштаб и улучшает численную устойчивость.
    2) Для каждого класса (и подкласса M) считаем Σ в z-пространстве.
    3) Делаем shrinkage Σ (чтобы Σ^{-1} была устойчивой).
    4) Сохраняем μ, Σ, Σ^{-1}, det, cond, PD.

    Почему глобальная нормализация:
    - Признаки разномасштабные (Teff >> logg ~ 4..5 >> radius ~ 0..2)
    - Это одна из причин гигантских cond(Σ), которые мы увидели в EDA.
    """
    # --- 0) базовые проверки ---
    required_cols = ["spec_class"] + FEATURES
    missing = [c for c in required_cols if c not in df_dwarfs.columns]
    if missing:
        raise ValueError(f"В df_dwarfs не хватает колонок: {missing}")

    # --- 1) глобальная нормализация (по всем DWARFS MKGF) ---
    X_all = df_dwarfs[FEATURES].astype(float).to_numpy()
    z_mu, z_sigma = zscore_fit(X_all)

    # --- 2) сбор поднаборов по классам/подклассам ---
    subsets: Dict[str, pd.DataFrame] = {}
    for cls in DWARF_CLASSES:
        subsets[cls] = df_dwarfs[df_dwarfs["spec_class"] == cls].copy()

    # Если включены подклассы M — заменяем общий M на три подкласса
    if use_m_subclasses:
        df_m = subsets["M"]
        m_sub = split_m_subclasses(df_m)
        # Удаляем общий M и добавляем подклассы
        subsets.pop("M", None)
        subsets.update(m_sub)

    # --- 3) считаем параметры по каждому subset ---
    class_params: Dict[str, ClassGaussianParams] = {}

    for label, sdf in subsets.items():
        n = int(sdf.shape[0])
        if n < 3:
            # Для 3D ковариации нужно хотя бы несколько точек
            # (иначе Σ плохо оценивается).
            continue

        X = sdf[FEATURES].astype(float).to_numpy()
        Xz = zscore_apply(X, z_mu, z_sigma)

        # μ и Σ в z-пространстве
        mu_z = Xz.mean(axis=0)
        S = cov_samp(Xz)

        # Shrinkage для устойчивости
        S_shrink = shrink_covariance(S, alpha=shrink_alpha)

        # Численные характеристики
        det = float(np.linalg.det(S_shrink))
        cond = condition_number(S_shrink)
        pd_flag = is_positive_definite(S_shrink)

        # Инверсия (если Σ не PD или det ~ 0, пробуем добавить маленький jitter)
        S_inv = None
        try:
            S_inv = np.linalg.inv(S_shrink)
        except np.linalg.LinAlgError:
            # jitter на диагональ
            jitter = 1e-6
            S_j = S_shrink + jitter * np.eye(S_shrink.shape[0])
            S_inv = np.linalg.inv(S_j)

        class_params[label] = ClassGaussianParams(
            label=label,
            n=n,
            z_mu=mu_z.tolist(),
            cov=S_shrink.tolist(),
            inv_cov=S_inv.tolist(),
            det_cov=det,
            cond_cov=cond,
            pd=pd_flag,
        )

    model = GaussianModel(
        global_z_mu=z_mu.tolist(),
        global_z_sigma=z_sigma.tolist(),
        classes=class_params,
    )
    return model


# ------------------------------------------------------------
# 6) Score: оценка одной звезды / таблицы
# ------------------------------------------------------------

def choose_m_subclass_label(teff: float) -> str:
    """
    Возвращает label подкласса M по Teff.
    """
    if teff >= M_EARLY_MIN and teff < M_EARLY_MAX:
        return "M_EARLY"
    if teff >= M_MID_MIN and teff < M_MID_MAX:
        return "M_MID"
    return "M_LATE"


def score_one(
    model: GaussianModel,
    spec_class: str,
    teff: float,
    logg: float,
    radius: float,
) -> Dict[str, float | str]:
    """
    Оценивает одну звезду по заданному spec_class (предварительно определённому).
    Возвращает словарь с расстоянием Махаланобиса и similarity.

    ВАЖНО:
    - Здесь spec_class считается уже известным (классификация — отдельный шаг).
    - Для M при use_m_subclasses=True мы используем подкласс по Teff.
    """
    # 1) выбираем label
    label = spec_class
    if spec_class == "M" and ("M_EARLY" in model.classes):
        label = choose_m_subclass_label(teff)

    if label not in model.classes:
        return {
            "label": label,
            "d_mahal": float("nan"),
            "similarity": 0.0,
        }

    params = model.classes[label]

    # 2) нормализуем x -> z
    z_mu = np.array(model.global_z_mu, dtype=float)
    z_sigma = np.array(model.global_z_sigma, dtype=float)
    x = np.array([teff, logg, radius], dtype=float)
    xz = zscore_apply(x, z_mu, z_sigma)

    # 3) считаем d и similarity
    mu = np.array(params.z_mu, dtype=float)
    inv_cov = np.array(params.inv_cov, dtype=float)

    d = mahalanobis_distance(xz, mu, inv_cov)
    s = similarity_from_distance(d)

    return {
        "label": label,
        "d_mahal": float(d),
        "similarity": float(s),
    }


def score_df(
    model: GaussianModel,
    df: pd.DataFrame,
    spec_class_col: str = "spec_class",
) -> pd.DataFrame:
    """
    Применяет score_one ко всему DataFrame.

    Требования к df:
    - должен содержать spec_class_col и FEATURES.
    """
    required = [spec_class_col] + FEATURES
    for c in required:
        if c not in df.columns:
            raise ValueError(f"В df не хватает колонки: {c}")

    rows: List[Dict[str, float | str]] = []
    for _, r in df.iterrows():
        out = score_one(
            model=model,
            spec_class=str(r[spec_class_col]),
            teff=float(r["teff_gspphot"]),
            logg=float(r["logg_gspphot"]),
            radius=float(r["radius_gspphot"]),
        )
        rows.append(out)

    res = df.copy()
    res["gauss_label"] = [x["label"] for x in rows]
    res["d_mahal"] = [x["d_mahal"] for x in rows]
    res["similarity"] = [x["similarity"] for x in rows]
    return res


# ------------------------------------------------------------
# 7) Сохранение/загрузка модели (JSON)
# ------------------------------------------------------------

def save_model(model: GaussianModel, path: str) -> None:
    """
    Сохраняет модель в JSON.
    Это удобно для воспроизводимости (и для отчёта/приложения к ВКР).
    """
    payload: Dict[str, Any] = {
        "global_z_mu": model.global_z_mu,
        "global_z_sigma": model.global_z_sigma,
        "classes": {k: asdict(v) for k, v in model.classes.items()},
        "features": FEATURES,
        "meta": {
            "logg_dwarf_min": LOGG_DWARF_MIN,
            "use_features": FEATURES,
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_model(path: str) -> GaussianModel:
    """
    Загружает модель из JSON.
    """
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    classes: Dict[str, ClassGaussianParams] = {}
    for k, v in payload["classes"].items():
        classes[k] = ClassGaussianParams(**v)

    return GaussianModel(
        global_z_mu=payload["global_z_mu"],
        global_z_sigma=payload["global_z_sigma"],
        classes=classes,
    )


# ------------------------------------------------------------
# 8) Мини-точка входа для ручного прогона (по желанию)
# ------------------------------------------------------------

if __name__ == "__main__":
    # Этот блок нужен только для быстрого ручного теста.
    # В продовом пайплайне будем вызывать fit/score из отдельного скрипта.

    print("=== FIT GaussianModel (DWARFS MKGF) ===")

    engine = make_engine_from_env()


    try:
        df_dwarfs = load_dwarfs_from_db(engine)
    except Exception as e:
        print("\n[ОШИБКА] Не удалось прочитать данные из БД.")
        print("Проверь DATABASE_URL (host/port/db/user/password) или PG* переменные окружения.")
        print("Текст ошибки:", repr(e))
        raise

    print("Загружено DWARFS:", df_dwarfs.shape)

    model = fit_gaussian_model(
        df_dwarfs=df_dwarfs,
        use_m_subclasses=True,
        shrink_alpha=0.15,
    )

    print("Классы в модели:", sorted(model.classes.keys()))
    for k, p in model.classes.items():
        print(f"[{k}] n={p.n} det={p.det_cov:.6g} cond={p.cond_cov:.3g} PD={p.pd}")

    # Сохраняем параметры модели
    save_model(model, "data/model_gaussian_params.json")
    print("Сохранено: data/model_gaussian_params.json")