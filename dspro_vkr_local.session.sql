/*
 dspro_vkr_local.session.sql
 Purpose:
 "Smoke test" for local Postgres connection + lab schema objects used in DSPro VKR.
 
 How to use (DBeaver / VSCode SQLTools / psql):
 1) Run blocks top-to-bottom.
 2) If every block returns results (and no ERROR), DB side is OK.
 
 Notes:
 - We do NOT embed comments inside queries in a way that can confuse DBeaver.
 Each block is: comment header -> query.
 - Replace nothing here unless you rename DB/schema/view/table.
 */
/* ------------------------------------------------------------
 0) Session identity & where we are
 Why: confirm we are in the correct database/user/schema context.
 ------------------------------------------------------------ */
SELECT current_database() AS current_database,
    current_user AS current_user,
    current_schema() AS current_schema;
/* ------------------------------------------------------------
 1) Schema existence
 Why: confirm that schema `lab` exists in this database.
 ------------------------------------------------------------ */
SELECT catalog_name,
    schema_name,
    schema_owner
FROM information_schema.schemata
WHERE schema_name = 'lab';
/* ------------------------------------------------------------
 2) Objects existence (tables/views we rely on)
 Why: confirm key objects are present before we "pythonize".
 ------------------------------------------------------------ */
SELECT table_schema,
    table_name,
    table_type
FROM information_schema.tables
WHERE table_schema = 'lab'
    AND table_name IN (
        'nasa_gaia_train',
        'v_nasa_gaia_train_classified',
        'v_gaia_ref_abo_training'
    )
ORDER BY table_name;
/* ------------------------------------------------------------
 3) Row counts (quick sanity)
 Why: confirm data is accessible and non-empty.
 ------------------------------------------------------------ */
SELECT COUNT(*) AS n_rows
FROM lab.nasa_gaia_train;
SELECT COUNT(*) AS n_rows
FROM lab.v_nasa_gaia_train_classified;
SELECT COUNT(*) AS n_rows
FROM lab.v_gaia_ref_abo_training;
/* ------------------------------------------------------------
 4) Sampling (peek at columns + values)
 Why: confirm column names/types match what we expect in Python.
 ------------------------------------------------------------ */
SELECT *
FROM lab.v_nasa_gaia_train_classified
LIMIT 5;
SELECT *
FROM lab.v_gaia_ref_abo_training
LIMIT 5;
/* ------------------------------------------------------------
 5) Basic distribution check for MKGF (training hosts dwarfs)
 Why: confirm we have enough objects per class.
 ------------------------------------------------------------ */
SELECT spec_class,
    COUNT(*) AS n_objects
FROM lab.v_nasa_gaia_train_classified
WHERE spec_class IN ('M', 'K', 'G', 'F')
GROUP BY spec_class
ORDER BY spec_class;
/* ------------------------------------------------------------
 6) Sanity check: NULLs in key fields
 Why: covariance/det(PD) tests require complete data.
 ------------------------------------------------------------ */
SELECT COUNT(*) AS n_total,
    COUNT(*) FILTER (
        WHERE teff_gspphot IS NULL
    ) AS n_null_teff,
    COUNT(*) FILTER (
        WHERE logg_gspphot IS NULL
    ) AS n_null_logg,
    COUNT(*) FILTER (
        WHERE radius_gspphot IS NULL
    ) AS n_null_radius
FROM lab.v_nasa_gaia_train_classified
WHERE spec_class IN ('M', 'K', 'G', 'F');
/* ------------------------------------------------------------
 7) Covariance matrix Σ per class + PD checks
 Why:
 - We model each spectral class with (Teff, logg, radius) ~ Gaussian.
 - That needs Σ to be non-singular and positive definite.
 Method:
 - Σ = covariances/variances (sample estimators)
 - Sylvester criterion for PD via leading principal minors:
 m1 > 0, m2 > 0, det(Σ) > 0
 ------------------------------------------------------------ */
WITH cov_by_class AS (
    SELECT spec_class,
        COUNT(*) AS n_objects,
        /* Variances (diagonal): a=Var(Teff), d=Var(logg), f=Var(radius) */
        var_samp(teff_gspphot) AS a_var_teff,
        var_samp(logg_gspphot) AS d_var_logg,
        var_samp(radius_gspphot) AS f_var_radius,
        /* Covariances (off-diagonal): b=Cov(Teff,logg), c=Cov(Teff,radius), e=Cov(logg,radius) */
        covar_samp(teff_gspphot, logg_gspphot) AS b_cov_teff_logg,
        covar_samp(teff_gspphot, radius_gspphot) AS c_cov_teff_radius,
        covar_samp(logg_gspphot, radius_gspphot) AS e_cov_logg_radius
    FROM lab.v_nasa_gaia_train_classified
    WHERE spec_class IN ('M', 'K', 'G', 'F')
    GROUP BY spec_class
),
calc AS (
    SELECT spec_class,
        n_objects,
        /* Leading principal minors (Sylvester) */
        a_var_teff AS m1,
        (
            a_var_teff * d_var_logg - (b_cov_teff_logg * b_cov_teff_logg)
        ) AS m2,
        /* det(Σ) for 3x3 matrix:
         | a  b  c |
         | b  d  e |
         | c  e  f |
         */
        (
            a_var_teff * (
                d_var_logg * f_var_radius - (e_cov_logg_radius * e_cov_logg_radius)
            ) - b_cov_teff_logg * (
                b_cov_teff_logg * f_var_radius - c_cov_teff_radius * e_cov_logg_radius
            ) + c_cov_teff_radius * (
                b_cov_teff_logg * e_cov_logg_radius - c_cov_teff_radius * d_var_logg
            )
        ) AS det_sigma
    FROM cov_by_class
)
SELECT spec_class,
    n_objects,
    det_sigma,
    (det_sigma <> 0) AS is_non_singular,
    (
        m1 > 0
        AND m2 > 0
        AND det_sigma > 0
    ) AS is_positive_definite
FROM calc
ORDER BY spec_class;
/* ------------------------------------------------------------
 8) Optional: show Σ components as a table (for debugging)
 Why: if PD fails later, these numbers explain why.
 ------------------------------------------------------------ */
WITH cov_by_class AS (
    SELECT spec_class,
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