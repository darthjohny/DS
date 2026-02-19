# eda.py
# ============================================================
# Назначение
# ------------------------------------------------------------
# EDA для проекта ВКР (Gaia DR3 + NASA hosts).
# Цель файла:
# проверить данные
# и подготовить статистику
# для ML-этапа (Gaussian / Mahalanobis similarity).
#
# Важно: здесь разделяем 4 слоя данных:
# (A) ALL MKGF  : все объекты M/K/G/F (включая evolved)
# (B) DWARFS    : главная последовательность
#                 (logg >= 4.0)
# (C) EVOLVED   : субгиганты/гиганты (logg < 4.0)
# (D) A/B/O REF : референс-популяция A/B/O (не хосты)
# ============================================================

import os

import numpy as np
import pandas as pd
from sqlalchemy import create_engine


# ============================================================
# 1. ПАРАМЕТРЫ ПОДКЛЮЧЕНИЯ К БД
# ============================================================

USER = "postgres"
PASSWORD = "1234"  # пароль как в DBeaver
HOST = "127.0.0.1"
PORT = 5432
DB = "dspro_vkr_research"

engine = create_engine(
    f"postgresql+psycopg2://{USER}:{PASSWORD}@{HOST}:{PORT}/{DB}"
)


# ============================================================
# 2. ЗАГРУЗКА ДАННЫХ ДЛЯ EDA
# ============================================================

# Источники:
# - lab.v_nasa_gaia_train_classified : MKGF хосты с spec_class
# - lab.v_nasa_gaia_train_dwarfs     : MKGF, только logg >= 4
# - lab.v_nasa_gaia_train_evolved    : MKGF, только logg < 4

# Базовая обучающая выборка
# (все MKGF, включая evolved)
QUERY_ALL_MKGF = """
SELECT
    spec_class,
    teff_gspphot,
    logg_gspphot,
    radius_gspphot
FROM lab.v_nasa_gaia_train_classified
WHERE spec_class IN ('M','K','G','F');
"""

# Карлики для обучения гауссовской модели
QUERY_DWARFS_MKGF = """
SELECT
    spec_class,
    teff_gspphot,
    logg_gspphot,
    radius_gspphot
FROM lab.v_nasa_gaia_train_dwarfs
WHERE spec_class IN ('M','K','G','F');
"""

# Эволюционировавшие
# (отдельный аналитический слой)
QUERY_EVOLVED_MKGF = """
SELECT
    spec_class,
    teff_gspphot,
    logg_gspphot,
    radius_gspphot
FROM lab.v_nasa_gaia_train_evolved
WHERE spec_class IN ('M','K','G','F');
"""

# Загружаем данные
# ALL MKGF: общая картина
# DWARFS: основа для mu/cov
# EVOLVED: отдельный слой

df = pd.read_sql(QUERY_ALL_MKGF, engine)
df_dwarfs = pd.read_sql(QUERY_DWARFS_MKGF, engine)
df_evolved = pd.read_sql(QUERY_EVOLVED_MKGF, engine)

print("\n=== ДАННЫЕ ЗАГРУЖЕНЫ ===")
print("Размер ALL MKGF:", df.shape)
print("Размер DWARFS (logg>=4.0):", df_dwarfs.shape)
print("Размер EVOLVED (logg<4.0):", df_evolved.shape)


# ============================================================
# 3. ПРОВЕРКА NULL (ALL MKGF)
# ============================================================

print("\n=== ПРОВЕРКА NULL ===")
print("NULL по столбцам (ALL MKGF):\n", df.isnull().sum())


# ============================================================
# 4. ОБЩИЕ СТАТИСТИКИ (ALL MKGF)
# ============================================================

print("\n=== ОБЩАЯ СТАТИСТИКА (describe) ===")
print(df.describe())


# ============================================================
# 5. КОРРЕЛЯЦИИ (ALL MKGF)
# ============================================================

print("\n=== КОРРЕЛЯЦИИ МЕЖДУ ПРИЗНАКАМИ ===")
print(df[["teff_gspphot", "logg_gspphot", "radius_gspphot"]].corr())


# ============================================================
# 6. СТАТИСТИКА ПО КЛАССАМ (ALL MKGF: M/K/G/F)
# ============================================================

print(
    "\n=== СТАТИСТИКА "
    "ПО СПЕКТРАЛЬНЫМ КЛАССАМ ==="
)

group_stats = (
    df.groupby("spec_class")[
        ["teff_gspphot", "logg_gspphot", "radius_gspphot"]
    ].agg(["mean", "std", "min", "max"])
)

print(group_stats)


# ============================================================
# 7. БЫСТРАЯ ПРОВЕРКА EVOLVED НА ALL MKGF
# ============================================================

print(
    "\n=== ALL MKGF: ТОП-20 ПО РАДИУСУ "
    "(быстрая проверка EVOLVED/гигантов) ==="
)

top_radius = df.sort_values("radius_gspphot", ascending=False).head(20)
print(top_radius)


# ============================================================
# 8. СЛОЙ DWARFS (ГЛАВНАЯ ПОСЛЕДОВАТЕЛЬНОСТЬ)
# ============================================================

print(
    "\n=== ТОЛЬКО ГЛАВНАЯ ПОСЛЕДОВАТЕЛЬНОСТЬ "
    "(из view: v_nasa_gaia_train_dwarfs) ==="
)
print("Размер выборки:", df_dwarfs.shape)

print("\n=== СТАТИСТИКА ПОСЛЕ ФИЛЬТРА ===")
print(df_dwarfs.describe())

print("\n=== КОРРЕЛЯЦИИ ПОСЛЕ ФИЛЬТРА ===")
print(df_dwarfs[["teff_gspphot", "logg_gspphot", "radius_gspphot"]].corr())


# ============================================================
# 9. СЛОЙ EVOLVED (logg < 4.0)
# ============================================================

# Эти объекты не входят
# в гауссову модель карликов,
# но сохраняются для аналитики и отчёта.
df_anomalies = df_evolved.copy()

print(
    "\n=== EVOLVED (logg < 4.0): "
    "отдельный аналитический слой ==="
)
print("Количество:", df_anomalies.shape)

os.makedirs("data/eda", exist_ok=True)
df_anomalies.to_csv("data/eda/evolved_stars_snapshot.csv", index=False)

# Единый порядок признаков
FEATURES = ["teff_gspphot", "logg_gspphot", "radius_gspphot"]


# ============================================================
# 9.1. СЛОЙ A/B/O REF (не хосты)
# ============================================================

print(
    "\n=== A/B/O REF (не хосты): "
    "диапазоны v_gaia_ref_abo_training ==="
)

QUERY_ABO_REF = """
SELECT
    spec_class,
    teff_gspphot,
    logg_gspphot,
    radius_gspphot
FROM lab.v_gaia_ref_abo_training
WHERE spec_class IN ('A','B','O');
"""

df_abo = pd.read_sql(QUERY_ABO_REF, engine)
print("Размер ABO ref:", df_abo.shape)
print("\n=== ABO ref: describe ===")
print(df_abo.describe())

print("\n=== ABO ref: статистика по классам ===")
abo_stats = (
    df_abo.groupby("spec_class")[FEATURES]
    .agg(["count", "mean", "std", "min", "max"])
)
print(abo_stats)

df_abo.sort_values("teff_gspphot", ascending=False).head(20).to_csv(
    "data/eda/abo_top20_by_teff.csv",
    index=False,
)


# ============================================================
# 10. МНОГОМЕРНАЯ ГАУССОВА МОДЕЛЬ (Dwarfs only)
# ============================================================

print(
    "\n=== DWARFS: mu и cov по классам "
    "(M/K/G/F), только logg>=4.0 ==="
)


# Считает mu и cov (3D)
# для одного класса/подкласса.
def calc_gauss_stats(df_part: pd.DataFrame, label: str) -> None:
    x = df_part[FEATURES].to_numpy(dtype=float)
    n = x.shape[0]

    if n < 5:
        print(
            f"\n[{label}] Слишком мало объектов "
            f"для устойчивой cov: n={n}"
        )
        return

    mu = x.mean(axis=0)
    sigma = np.cov(x, rowvar=False, ddof=1)

    det_sigma = float(np.linalg.det(sigma))
    eigvals = np.linalg.eigvalsh(sigma)
    cond = float(np.linalg.cond(sigma))

    print(f"\n[{label}] n={n}")
    print("mu =", mu)
    print("cov =\n", sigma)
    print("det(cov) =", det_sigma)
    print("eigenvalues(cov) =", eigvals)
    print("cond(cov) =", cond)
    print("PD (все eigenvalues > 0):", bool(np.all(eigvals > 0)))


for cls in ["M", "K", "G", "F"]:
    part = df_dwarfs[df_dwarfs["spec_class"] == cls]
    calc_gauss_stats(part, f"CLASS {cls}")


# ============================================================
# 11. ПОДКЛАССЫ M-КАРЛИКОВ (Early / Mid / Late)
# ============================================================

print(
    "\n=== DWARFS: mu и cov "
    "для подклассов M (Early/Mid/Late) ==="
)

M_EARLY_MIN = 3500.0
M_MID_MIN = 3200.0
M_UPPER = 4000.0

df_m = df_dwarfs[df_dwarfs["spec_class"] == "M"].copy()

m_early = df_m[
    (df_m["teff_gspphot"] >= M_EARLY_MIN)
    & (df_m["teff_gspphot"] <= M_UPPER)
]
m_mid = df_m[
    (df_m["teff_gspphot"] >= M_MID_MIN)
    & (df_m["teff_gspphot"] < M_EARLY_MIN)
]
m_late = df_m[df_m["teff_gspphot"] < M_MID_MIN]

calc_gauss_stats(m_early, "M_EARLY (3500-4000K)")
calc_gauss_stats(m_mid, "M_MID (3200-3500K)")
calc_gauss_stats(m_late, "M_LATE (<3200K)")


# ============================================================
# 12. МИКРО-ИТОГ EDA
# ============================================================
# - EVOLVED (logg < 4.0)
#   искажают радиусы и cov для карликов.
# - Это не ошибка данных,
#   а отдельная физическая популяция.
# - Для модели карликов
#   используем только слой DWARFS.
# - Для классов M/K/G/F
#   рассчитаны mu, cov, det, eigenvalues, cond.
# - Следующий шаг:
#   нормализация + shrinkage
#   перед финальным скорингом.
# - A/B/O храним отдельно
#   как reference-слой для sanity-check и OOD.
# ============================================================
