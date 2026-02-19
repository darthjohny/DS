# eda.py
# ============================================================
# Назначение
# ------------------------------------------------------------
# Производственный EDA для проекта ВКР (Gaia DR3 + NASA hosts).
# Цель файла: диагностировать данные и подготовить статистику
# для дальнейшей ML-части (Gaussian/Mahalanobis similarity).
#
# ВАЖНО: в этом файле мы явно разделяем 4 слоя данных:
#   (A) ALL MKGF   : все объекты классов M/K/G/F (включая evolved)
#   (B) DWARFS     : только главная последовательность (logg >= 4.0)
#   (C) EVOLVED    : субгиганты/гиганты (logg < 4.0) — отдельный слой
#   (D) A/B/O REF  : референс-популяция классов A/B/O (не хосты)
#
# Почему это важно:
# - EVOLVED сильно раздувают радиусы и портят ковариации Σ для карликов.
# - A/B/O держим отдельно: это не «аномалии», а другие классы.
# ============================================================

import os
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


# ============================================================
# 1. ПАРАМЕТРЫ ПОДКЛЮЧЕНИЯ К БД
# ============================================================

USER = "postgres"
PASSWORD = "1234"          # пароль как в DBeaver
HOST = "127.0.0.1"
PORT = 5432
DB = "dspro_vkr_research"

engine = create_engine(
    f"postgresql+psycopg2://{USER}:{PASSWORD}@{HOST}:{PORT}/{DB}"
)


# ============================================================
# 2. ЗАГРУЗКА ДАННЫХ ДЛЯ EDA (4 слоя: ALL MKGF / DWARFS / EVOLVED / A/B/O REF)
# ============================================================

# Источник данных:
# - lab.v_nasa_gaia_train_classified : MKGF (hosts) с уже проставленным spec_class
# - lab.v_nasa_gaia_train_dwarfs     : MKGF, но только главная последовательность (logg>=4)
# - lab.v_nasa_gaia_train_evolved    : MKGF, но только эволюционировавшие (logg<4)
#
# Важно: DWARFS/EVOLVED берём из view в БД, чтобы критерий был единым
# для SQL и Python (и не расходился в разных местах проекта).

# Базовая обучающая выборка (все MKGF, включая эволюционировавшие)
QUERY_ALL_MKGF = """
SELECT
    spec_class,
    teff_gspphot,
    logg_gspphot,
    radius_gspphot
FROM lab.v_nasa_gaia_train_classified
WHERE spec_class IN ('M','K','G','F');
"""

# «Правильные» карлики для обучения гауссовской модели (физически: logg >= 4.0)
# ВАЖНО: это представление (view) создаём в БД, чтобы фильтр был единым для SQL и Python.
QUERY_DWARFS_MKGF = """
SELECT
    spec_class,
    teff_gspphot,
    logg_gspphot,
    radius_gspphot
FROM lab.v_nasa_gaia_train_dwarfs
WHERE spec_class IN ('M','K','G','F');
"""

# Эволюционировавшие (субгиганты/гиганты) — отдельный аналитический слой (logg < 4.0)
QUERY_EVOLVED_MKGF = """
SELECT
    spec_class,
    teff_gspphot,
    logg_gspphot,
    radius_gspphot
FROM lab.v_nasa_gaia_train_evolved
WHERE spec_class IN ('M','K','G','F');
"""

# Загружаем всё (для общей картины)
df = pd.read_sql(QUERY_ALL_MKGF, engine)

# Загружаем карликов (это будет основа для μ и Σ)
df_dwarfs = pd.read_sql(QUERY_DWARFS_MKGF, engine)

# Загружаем эволюционировавшие (это не «мусор», а отдельная категория)
df_evolved = pd.read_sql(QUERY_EVOLVED_MKGF, engine)

print("\n=== ДАННЫЕ ЗАГРУЖЕНЫ ===")
print("Размер ALL MKGF:", df.shape)
print("Размер DWARFS (logg>=4.0):", df_dwarfs.shape)
print("Размер EVOLVED (logg<4.0):", df_evolved.shape)


# ============================================================
# 3. ПРОВЕРКА NULL (на слое ALL MKGF)
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
print(
    df[["teff_gspphot", "logg_gspphot", "radius_gspphot"]]
    .corr()
)


# ============================================================
# 6. СТАТИСТИКА ПО КЛАССАМ (ALL MKGF: M/K/G/F)
# ============================================================

print("\n=== СТАТИСТИКА ПО СПЕКТРАЛЬНЫМ КЛАССАМ ===")

group_stats = (
    df
    .groupby("spec_class")[
        ["teff_gspphot", "logg_gspphot", "radius_gspphot"]
    ]
    .agg(["mean", "std", "min", "max"])
)

print(group_stats)

# ============================================================
# 7. БЫСТРАЯ ПРОВЕРКА EVOLVED НА ALL MKGF (топ-радиусы, logg обычно < 4.0)
# ============================================================

print("\n=== ALL MKGF: ТОП-20 ПО РАДИУСУ (быстрая проверка EVOLVED/гигантов) ===")

top_radius = df.sort_values("radius_gspphot", ascending=False).head(20)
print(top_radius)

# ============================================================
# 8. СЛОЙ DWARFS (ГЛАВНАЯ ПОСЛЕДОВАТЕЛЬНОСТЬ)
# ------------------------------------------------------------
# Физический критерий: logg >= 4.0
# Важно: здесь мы НЕ фильтруем вручную, а используем view из БД
# (lab.v_nasa_gaia_train_dwarfs), чтобы критерий был единым.
# ============================================================

print("\n=== ТОЛЬКО ГЛАВНАЯ ПОСЛЕДОВАТЕЛЬНОСТЬ (из view: v_nasa_gaia_train_dwarfs) ===")
print("Размер выборки:", df_dwarfs.shape)

print("\n=== СТАТИСТИКА ПОСЛЕ ФИЛЬТРА ===")
print(df_dwarfs.describe())

print("\n=== КОРРЕЛЯЦИИ ПОСЛЕ ФИЛЬТРА ===")
print(
    df_dwarfs[["teff_gspphot", "logg_gspphot", "radius_gspphot"]]
    .corr()
)

# ============================================================
# 9. СЛОЙ EVOLVED (logg < 4.0): сохранение отдельного аналитического набора
# ------------------------------------------------------------
#    Эти объекты (субгиганты/гиганты) НЕ используются для построения
#    гауссовской модели карликов, но сохраняются для анализа и отчёта.
# ============================================================

# Берём из view, чтобы критерий полностью совпадал с SQL-частью проекта
# (и чтобы не зависеть от того, как именно мы фильтруем в Python)
df_anomalies = df_evolved.copy()

print("\n=== EVOLVED (logg < 4.0): отдельный аналитический слой ===")
print("Количество:", df_anomalies.shape)

os.makedirs("data/eda", exist_ok=True)
df_anomalies.to_csv("data/eda/evolved_stars_snapshot.csv", index=False)

# Единый набор признаков для модели (везде одинаковый порядок признаков)
FEATURES = ["teff_gspphot", "logg_gspphot", "radius_gspphot"]

# ============================================================
# 9.1. СЛОЙ A/B/O REF (не хосты): референс-популяция для sanity-check и OOD
# ============================================================

print("\n=== A/B/O REF (не хосты): проверка диапазонов (v_gaia_ref_abo_training) ===")

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
    df_abo
    .groupby("spec_class")[FEATURES]
    .agg(["count", "mean", "std", "min", "max"])
)
print(abo_stats)

df_abo.sort_values("teff_gspphot", ascending=False).head(20).to_csv(
    "data/eda/abo_top20_by_teff.csv", index=False
)

# ============================================================
# 10. МНОГОМЕРНАЯ ГАУССОВА МОДЕЛЬ: μ и Σ ДЛЯ КАЖДОГО КЛАССА (Dwarfs only)
#    Здесь мы готовим математику для Mahalanobis distance:
#      - μ (вектор средних) по (Teff, logg, R)
#      - Σ (ковариационная матрица 3x3)
#      - det(Σ), собственные значения (eigenvalues), число обусловленности (cond)
#    Важно: считаем на df_dwarfs (из view v_nasa_gaia_train_dwarfs), чтобы не раздувать Σ гигантами.
# ============================================================

print("\n=== DWARFS: μ и Σ по классам (M/K/G/F), только logg>=4.0 ===")

# Вспомогательная функция: считает μ и Σ (3D) для одного поднабора (класс/подкласс)
def calc_gauss_stats(df_part: pd.DataFrame, label: str) -> None:
    x = df_part[FEATURES].to_numpy(dtype=float)
    n = x.shape[0]

    if n < 5:
        print(f"\n[{label}] Слишком мало объектов для устойчивой Σ: n={n}")
        return

    mu = x.mean(axis=0)
    sigma = np.cov(x, rowvar=False, ddof=1)

    det_sigma = float(np.linalg.det(sigma))
    eigvals = np.linalg.eigvalsh(sigma)
    cond = float(np.linalg.cond(sigma))

    print(f"\n[{label}] n={n}")
    print("μ =", mu)
    print("Σ =\n", sigma)
    print("det(Σ) =", det_sigma)
    print("eigenvalues(Σ) =", eigvals)
    print("cond(Σ) =", cond)
    print("PD (все eigenvalues > 0):", bool(np.all(eigvals > 0)))

for cls in ["M", "K", "G", "F"]:
    part = df_dwarfs[df_dwarfs["spec_class"] == cls]
    calc_gauss_stats(part, f"CLASS {cls}")

# ============================================================
# 11. ПОДКЛАССЫ M-КАРЛИКОВ (Early / Mid / Late) ПО Teff
#    Зачем:
#      - у M-карликов большой разброс по температуре и радиусу
#      - один общий M-гаусс может быть слишком "широким"
#      - подклассы дают физически более однородные эллипсоиды
#
#    Пороговые значения можно править (важно: верхняя граница M <= 4000K)
# ============================================================

print("\n=== DWARFS: μ и Σ для подклассов M (Early/Mid/Late по Teff) ===")

M_EARLY_MIN = 3500.0
M_MID_MIN = 3200.0
M_UPPER = 4000.0

df_m = df_dwarfs[df_dwarfs["spec_class"] == "M"].copy()

m_early = df_m[(df_m["teff_gspphot"] >= M_EARLY_MIN) & (df_m["teff_gspphot"] <= M_UPPER)]
m_mid   = df_m[(df_m["teff_gspphot"] >= M_MID_MIN)   & (df_m["teff_gspphot"] <  M_EARLY_MIN)]
m_late  = df_m[df_m["teff_gspphot"] < M_MID_MIN]

calc_gauss_stats(m_early, "M_EARLY (3500–4000K)")
calc_gauss_stats(m_mid,   "M_MID   (3200–3500K)")
calc_gauss_stats(m_late,  "M_LATE  (<3200K)")

# ============================================================
# 12. МИКРО-ИТОГ EDA (ВАЖНО НЕ ЗАБЫТЬ КОНТЕКСТ)
# ============================================================
#
# Что произошло:
# 1) В исходной выборке MKGF присутствовали эволюционировавшие звезды (logg < 4.0),
#    что приводило к сильно раздутым радиусам (до ~84 R☉) и искажало статистику.
# 2) Это не ошибка данных, а физически ожидаемые субгиганты/гиганты.
# 3) Для корректного обучения модели карликов мы отделили:
#       - Главную последовательность (logg >= 4.0)
#       - Эволюционировавшие объекты (logg < 4.0)
#
# Что мы получили:
# - Для каждого класса (M, K, G, F) рассчитаны:
#     μ (вектор средних)
#     Σ (ковариационная матрица 3x3)
#     det(Σ), собственные значения, cond(Σ)
# - Матрицы положительно определены (PD = True).
# - Однако обнаружена высокая обусловленность (cond большое),
#   особенно у некоторых классов → возможная численная нестабильность.
#
# Почему это важно:
# - Mahalanobis distance использует Σ⁻¹.
# - Если Σ плохо обусловлена, инверсия становится чувствительной к шуму.
# - Это может приводить к нестабильным оценкам "похожести".
#
# Что будем делать дальше (правильный и надежный путь):
# 1) Нормализация признаков (Teff, logg, radius) перед финальной моделью.
# 2) Применение shrinkage-регуляризации ковариации для устойчивости.
# 3) Сохранение μ и Σ (или Σ_shrink) как параметров модели.
# 4) Эволюционировавшие звезды не удаляются навсегда —
#    они сохраняются как отдельный аналитический слой.
#
# Вывод:
# Мы получили физически корректную обучающую базу карликов
# и подтвердили возможность построения многомерной гауссовой модели.
# Следующий этап — стабилизация и переход к ML-части.
#
# Дополнение про A/B/O:
# - A/B/O — это НЕ «аномалии» внутри MKGF, а отдельные классы, с которыми мы сравниваемся.
# - Их держим отдельным датасетом (view: v_gaia_ref_abo_training), чтобы:
#     (а) не портить гауссы карликов MKGF,
#     (б) уметь помечать звезды как "вне распределения" (OOD),
#         если они ближе к A/B/O, чем к MKGF-карликам.
#
# ============================================================