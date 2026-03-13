-- QA checks for host-train and router reference layers.
--
-- Что делает файл:
--   - проверяет объёмы классов в `lab.v_nasa_gaia_train_classified`;
--   - оценивает корреляции и ковариации признаков `Teff/logg/Radius`;
--   - проверяет невырожденность и положительную определённость ковариаций;
--   - даёт sanity-check по derived views router-слоя.
--
-- Файл не меняет схему и может выполняться как повторяемый read-only QA.

/* -------------------------------------------------------------------------
   1. Объёмы выборок по классам и стадиям
   ------------------------------------------------------------------------- */

SELECT
    spec_class,
    COUNT(*) AS n_objects
FROM lab.v_nasa_gaia_train_classified
WHERE spec_class IN ('M', 'K', 'G', 'F')
GROUP BY spec_class
ORDER BY spec_class;

SELECT
    COUNT(*) AS n_total,
    COUNT(*) FILTER (WHERE teff_gspphot IS NULL) AS n_null_teff,
    COUNT(*) FILTER (WHERE logg_gspphot IS NULL) AS n_null_logg,
    COUNT(*) FILTER (WHERE radius_gspphot IS NULL) AS n_null_radius
FROM lab.v_nasa_gaia_train_classified
WHERE spec_class IN ('M', 'K', 'G', 'F');

SELECT COUNT(*) AS n_dwarfs
FROM lab.v_nasa_gaia_train_dwarfs;

SELECT COUNT(*) AS n_evolved
FROM lab.v_nasa_gaia_train_evolved;

/* -------------------------------------------------------------------------
   2. Корреляции и ковариации
   ------------------------------------------------------------------------- */

SELECT
    spec_class,
    corr(teff_gspphot, logg_gspphot) AS corr_teff_logg,
    corr(teff_gspphot, radius_gspphot) AS corr_teff_radius,
    corr(logg_gspphot, radius_gspphot) AS corr_logg_radius
FROM lab.v_nasa_gaia_train_classified
WHERE spec_class IN ('M', 'K', 'G', 'F')
GROUP BY spec_class
ORDER BY spec_class;

SELECT
    spec_class,
    COUNT(*) AS n_objects,
    var_samp(teff_gspphot) AS a_var_teff,
    var_samp(logg_gspphot) AS d_var_logg,
    var_samp(radius_gspphot) AS f_var_radius
FROM lab.v_nasa_gaia_train_classified
WHERE spec_class IN ('M', 'K', 'G', 'F')
GROUP BY spec_class
ORDER BY spec_class;

SELECT
    spec_class,
    COUNT(*) AS n_objects,
    covar_samp(teff_gspphot, logg_gspphot) AS b_cov_teff_logg,
    covar_samp(teff_gspphot, radius_gspphot) AS c_cov_teff_radius,
    covar_samp(logg_gspphot, radius_gspphot) AS e_cov_logg_radius
FROM lab.v_nasa_gaia_train_classified
WHERE spec_class IN ('M', 'K', 'G', 'F')
GROUP BY spec_class
ORDER BY spec_class;

/* -------------------------------------------------------------------------
   3. Компоненты ковариационной матрицы
   ------------------------------------------------------------------------- */

WITH cov_by_class AS (
    SELECT
        spec_class,
        COUNT(*) AS n_objects,
        var_samp(teff_gspphot) AS a_var_teff,
        var_samp(logg_gspphot) AS d_var_logg,
        var_samp(radius_gspphot) AS f_var_radius,
        covar_samp(teff_gspphot, logg_gspphot) AS b_cov_teff_logg,
        covar_samp(teff_gspphot, radius_gspphot) AS c_cov_teff_radius,
        covar_samp(logg_gspphot, radius_gspphot) AS e_cov_logg_radius
    FROM lab.v_nasa_gaia_train_classified
    WHERE spec_class IN ('M', 'K', 'G', 'F')
    GROUP BY spec_class
)
SELECT *
FROM cov_by_class
ORDER BY spec_class;

/* -------------------------------------------------------------------------
   4. Проверка главных миноров и детерминанта
   ------------------------------------------------------------------------- */

WITH cov_by_class AS (
    SELECT
        spec_class,
        COUNT(*) AS n_objects,
        var_samp(teff_gspphot) AS a_var_teff,
        var_samp(logg_gspphot) AS d_var_logg,
        var_samp(radius_gspphot) AS f_var_radius,
        covar_samp(teff_gspphot, logg_gspphot) AS b_cov_teff_logg,
        covar_samp(teff_gspphot, radius_gspphot) AS c_cov_teff_radius,
        covar_samp(logg_gspphot, radius_gspphot) AS e_cov_logg_radius
    FROM lab.v_nasa_gaia_train_classified
    WHERE spec_class IN ('M', 'K', 'G', 'F')
    GROUP BY spec_class
)
SELECT
    spec_class,
    n_objects,
    a_var_teff AS m1,
    (a_var_teff * d_var_logg - (b_cov_teff_logg * b_cov_teff_logg)) AS m2
FROM cov_by_class
ORDER BY spec_class;

WITH cov_by_class AS (
    SELECT
        spec_class,
        COUNT(*) AS n_objects,
        var_samp(teff_gspphot) AS a_var_teff,
        var_samp(logg_gspphot) AS d_var_logg,
        var_samp(radius_gspphot) AS f_var_radius,
        covar_samp(teff_gspphot, logg_gspphot) AS b_cov_teff_logg,
        covar_samp(teff_gspphot, radius_gspphot) AS c_cov_teff_radius,
        covar_samp(logg_gspphot, radius_gspphot) AS e_cov_logg_radius
    FROM lab.v_nasa_gaia_train_classified
    WHERE spec_class IN ('M', 'K', 'G', 'F')
    GROUP BY spec_class
)
SELECT
    spec_class,
    n_objects,
    (
        a_var_teff * (d_var_logg * f_var_radius - (e_cov_logg_radius * e_cov_logg_radius))
        - b_cov_teff_logg * (b_cov_teff_logg * f_var_radius - c_cov_teff_radius * e_cov_logg_radius)
        + c_cov_teff_radius * (b_cov_teff_logg * e_cov_logg_radius - c_cov_teff_radius * d_var_logg)
    ) AS det_sigma
FROM cov_by_class
ORDER BY spec_class;

WITH cov_by_class AS (
    SELECT
        spec_class,
        COUNT(*) AS n_objects,
        var_samp(teff_gspphot) AS a_var_teff,
        var_samp(logg_gspphot) AS d_var_logg,
        var_samp(radius_gspphot) AS f_var_radius,
        covar_samp(teff_gspphot, logg_gspphot) AS b_cov_teff_logg,
        covar_samp(teff_gspphot, radius_gspphot) AS c_cov_teff_radius,
        covar_samp(logg_gspphot, radius_gspphot) AS e_cov_logg_radius
    FROM lab.v_nasa_gaia_train_classified
    WHERE spec_class IN ('M', 'K', 'G', 'F')
    GROUP BY spec_class
),
calc AS (
    SELECT
        spec_class,
        n_objects,
        a_var_teff AS m1,
        (a_var_teff * d_var_logg - (b_cov_teff_logg * b_cov_teff_logg)) AS m2,
        (
            a_var_teff * (d_var_logg * f_var_radius - (e_cov_logg_radius * e_cov_logg_radius))
            - b_cov_teff_logg * (b_cov_teff_logg * f_var_radius - c_cov_teff_radius * e_cov_logg_radius)
            + c_cov_teff_radius * (b_cov_teff_logg * e_cov_logg_radius - c_cov_teff_radius * d_var_logg)
        ) AS det_sigma
    FROM cov_by_class
)
SELECT
    spec_class,
    n_objects,
    det_sigma,
    (det_sigma <> 0) AS is_non_singular,
    (m1 > 0 AND m2 > 0 AND det_sigma > 0) AS is_positive_definite
FROM calc
ORDER BY spec_class;

/* -------------------------------------------------------------------------
   5. Sanity-check router reference layer
   ------------------------------------------------------------------------- */

SELECT
    spec_class,
    evolution_stage,
    COUNT(*) AS n_objects
FROM lab.v_gaia_router_training
GROUP BY spec_class, evolution_stage
ORDER BY spec_class, evolution_stage;

SELECT
    evolution_stage,
    MIN(logg_gspphot) AS min_logg,
    MAX(logg_gspphot) AS max_logg
FROM lab.v_gaia_router_training
GROUP BY evolution_stage
ORDER BY evolution_stage;

SELECT COUNT(*) AS n_router_training
FROM lab.v_gaia_router_training;
