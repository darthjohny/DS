-- Build host training tables from validated NASA x Gaia crossmatch.
--
-- Что делает файл:
--   - фиксирует QA входных relation перед сборкой train-слоя;
--   - строит `lab.nasa_gaia_crossmatch` из `lab.validation_unique` и
--     `lab.host_validated_gaia_physics_result`;
--   - добавляет `validation_factor` как инженерный вес качества match;
--   - строит `lab.nasa_gaia_train` с одной строкой на `source_id`;
--   - оставляет набор контрольных запросов для быстрой проверки результата.
--
-- Предпосылки:
--   - ADQL crossmatch уже выполнен и результат staged локально;
--   - `lab.validation_unique` содержит ближайшие или заранее отобранные match-строки;
--   - `lab.host_validated_gaia_physics_result` содержит физику Gaia DR3
--     для провалидированных `source_id`.
--
-- Важно про порядок выполнения:
--   - если в БД уже существуют derived views на базе `lab.nasa_gaia_train`,
--     этот файл сначала удалит их, а затем пересоберёт таблицы;
--   - после него нужно заново выполнить
--     `02_train_classification_views.sql`.

/* -------------------------------------------------------------------------
   1. Диагностика входных relation
   ------------------------------------------------------------------------- */

SELECT table_schema, table_name
FROM information_schema.tables
WHERE table_schema = 'lab'
  AND table_name IN (
      'nasa_hosts_for_gaia_xmatch',
      'nasa_hosts_for_gaia_xmatch_unique',
      'validation',
      'validation_unique',
      'host_validated_gaia_physics_result'
  )
ORDER BY table_name;

SELECT * FROM lab.nasa_hosts_for_gaia_xmatch LIMIT 10;
SELECT * FROM lab.nasa_hosts_for_gaia_xmatch_unique LIMIT 10;
SELECT * FROM lab.validation_unique LIMIT 10;
SELECT * FROM lab.host_validated_gaia_physics_result LIMIT 10;

SELECT
    COUNT(*) AS n_total,
    COUNT(*) FILTER (WHERE ra IS NULL OR dec IS NULL) AS n_missing_coords
FROM lab.nasa_hosts_for_gaia_xmatch;

SELECT
    COUNT(*) AS n_rows,
    MIN(dist_arcsec) AS min_dist_arcsec,
    MAX(dist_arcsec) AS max_dist_arcsec,
    AVG(dist_arcsec) AS avg_dist_arcsec,
    percentile_cont(0.50) WITHIN GROUP (ORDER BY dist_arcsec) AS p50_dist_arcsec,
    percentile_cont(0.95) WITHIN GROUP (ORDER BY dist_arcsec) AS p95_dist_arcsec,
    percentile_cont(0.99) WITHIN GROUP (ORDER BY dist_arcsec) AS p99_dist_arcsec
FROM lab.validation_unique;

SELECT
    COUNT(*) AS n_rows,
    COUNT(*) FILTER (WHERE teff_gspphot IS NULL) AS n_teff_null,
    COUNT(*) FILTER (WHERE logg_gspphot IS NULL) AS n_logg_null,
    COUNT(*) FILTER (WHERE radius_gspphot IS NULL) AS n_radius_null,
    COUNT(*) FILTER (WHERE mh_gspphot IS NULL) AS n_mh_null
FROM lab.host_validated_gaia_physics_result;

SELECT
    COUNT(*) AS n_rows,
    COUNT(*) FILTER (WHERE ruwe > 1.4) AS n_ruwe_gt_1_4,
    COUNT(*) FILTER (WHERE parallax_over_error < 5.0) AS n_plx_over_err_lt_5
FROM lab.host_validated_gaia_physics_result;

/* -------------------------------------------------------------------------
   2. Сборка `lab.nasa_gaia_crossmatch`
   ------------------------------------------------------------------------- */

DROP TABLE IF EXISTS lab.nasa_gaia_crossmatch;

CREATE TABLE lab.nasa_gaia_crossmatch AS
SELECT
    v.id,
    v.hostname,
    v.ra_nasa,
    v.dec_nasa,
    v.source_id,
    v.ra_gaia,
    v.dec_gaia,
    v.dist_arcsec,
    p.teff_gspphot,
    p.logg_gspphot,
    p.radius_gspphot,
    p.mh_gspphot,
    p.ruwe,
    p.parallax,
    p.parallax_over_error,
    p.phot_g_mean_mag,
    p.bp_rp
FROM lab.validation_unique AS v
LEFT JOIN lab.host_validated_gaia_physics_result AS p
    ON v.source_id = p.source_id;

ALTER TABLE lab.nasa_gaia_crossmatch
ADD COLUMN validation_factor numeric;

UPDATE lab.nasa_gaia_crossmatch
SET validation_factor = CASE
    WHEN dist_arcsec <= 0.8 THEN 1.0
    WHEN dist_arcsec <= 1.0 THEN 0.7
    ELSE 0.4
END;

/* -------------------------------------------------------------------------
   3. Контроль результата crossmatch
   ------------------------------------------------------------------------- */

SELECT
    source_id,
    COUNT(*) AS n_rows_for_source
FROM lab.nasa_gaia_crossmatch
GROUP BY source_id
HAVING COUNT(*) > 1
ORDER BY n_rows_for_source DESC, source_id;

SELECT * FROM lab.nasa_gaia_crossmatch LIMIT 20;

SELECT COUNT(*) AS n_crossmatch
FROM lab.nasa_gaia_crossmatch;

SELECT
    MIN(dist_arcsec) AS min_dist_arcsec,
    MAX(dist_arcsec) AS max_dist_arcsec,
    AVG(dist_arcsec) AS avg_dist_arcsec,
    percentile_cont(0.50) WITHIN GROUP (ORDER BY dist_arcsec) AS p50_dist_arcsec,
    percentile_cont(0.95) WITHIN GROUP (ORDER BY dist_arcsec) AS p95_dist_arcsec,
    percentile_cont(0.99) WITHIN GROUP (ORDER BY dist_arcsec) AS p99_dist_arcsec
FROM lab.nasa_gaia_crossmatch;

SELECT
    COUNT(*) AS n_total,
    COUNT(*) FILTER (
        WHERE teff_gspphot IS NOT NULL
          AND logg_gspphot IS NOT NULL
          AND radius_gspphot IS NOT NULL
    ) AS n_complete_physics
FROM lab.nasa_gaia_crossmatch;

/* -------------------------------------------------------------------------
   4. Сборка `lab.nasa_gaia_train`
   ------------------------------------------------------------------------- */

DROP VIEW IF EXISTS lab.v_nasa_gaia_train_dwarfs;
DROP VIEW IF EXISTS lab.v_nasa_gaia_train_evolved;
DROP VIEW IF EXISTS lab.v_nasa_gaia_train_classified;

DROP TABLE IF EXISTS lab.nasa_gaia_train;

CREATE TABLE lab.nasa_gaia_train AS
SELECT DISTINCT ON (source_id)
    *
FROM lab.nasa_gaia_crossmatch
WHERE teff_gspphot IS NOT NULL
  AND logg_gspphot IS NOT NULL
  AND radius_gspphot IS NOT NULL
ORDER BY source_id, dist_arcsec ASC;

/* -------------------------------------------------------------------------
   5. Контроль train-слоя
   ------------------------------------------------------------------------- */

SELECT
    id,
    hostname,
    source_id,
    dist_arcsec,
    teff_gspphot,
    logg_gspphot,
    radius_gspphot,
    mh_gspphot,
    ruwe,
    parallax,
    parallax_over_error,
    bp_rp,
    validation_factor
FROM lab.nasa_gaia_train
ORDER BY id
LIMIT 20;

SELECT COUNT(*) AS n_crossmatch
FROM lab.nasa_gaia_crossmatch;

SELECT COUNT(*) AS n_train
FROM lab.nasa_gaia_train;

SELECT
    MIN(dist_arcsec) AS min_dist_arcsec,
    MAX(dist_arcsec) AS max_dist_arcsec,
    AVG(dist_arcsec) AS avg_dist_arcsec,
    percentile_cont(0.50) WITHIN GROUP (ORDER BY dist_arcsec) AS p50_dist_arcsec,
    percentile_cont(0.95) WITHIN GROUP (ORDER BY dist_arcsec) AS p95_dist_arcsec,
    percentile_cont(0.99) WITHIN GROUP (ORDER BY dist_arcsec) AS p99_dist_arcsec
FROM lab.nasa_gaia_train;

SELECT
    COUNT(*) AS n_train,
    COUNT(*) FILTER (WHERE ruwe > 1.4) AS n_ruwe_gt_1_4,
    COUNT(*) FILTER (WHERE parallax_over_error < 5.0) AS n_plx_over_err_lt_5
FROM lab.nasa_gaia_train;

SELECT
    source_id,
    COUNT(*) AS n_rows
FROM lab.nasa_gaia_train
GROUP BY source_id
HAVING COUNT(*) > 1
ORDER BY n_rows DESC;
