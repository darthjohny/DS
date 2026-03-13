-- Build classification views on top of `lab.nasa_gaia_train`.
--
-- Что делает файл:
--   - присваивает host-train строкам спектральный класс и эволюционную стадию;
--   - выделяет подклассы внутри `M`;
--   - формирует отдельные views для `dwarf` и `evolved` веток `M/K/G/F`.
--
-- Эти views используются в:
--   - `src/host_model`;
--   - `analysis/host_eda`;
--   - части DB-smoke и diagnostic-сценариев проекта.

CREATE OR REPLACE VIEW lab.v_nasa_gaia_train_classified AS
SELECT
    t.*,
    CASE
        WHEN t.logg_gspphot IS NULL THEN 'unknown'
        WHEN t.logg_gspphot >= 4.0 THEN 'dwarf'
        WHEN t.logg_gspphot >= 3.5 THEN 'subgiant'
        ELSE 'giant'
    END AS evolution_stage,
    CASE
        WHEN t.teff_gspphot IS NULL THEN 'Unknown'
        WHEN t.teff_gspphot > 30000 THEN 'O'
        WHEN t.teff_gspphot >= 10000 THEN 'B'
        WHEN t.teff_gspphot >= 7500 THEN 'A'
        WHEN t.teff_gspphot >= 6000 THEN 'F'
        WHEN t.teff_gspphot >= 5200 THEN 'G'
        WHEN t.teff_gspphot >= 4000 THEN 'K'
        ELSE 'M'
    END AS spec_class,
    CASE
        WHEN t.teff_gspphot IS NULL THEN NULL
        WHEN t.teff_gspphot >= 4000 THEN NULL
        WHEN t.teff_gspphot >= 3400 THEN 'M_early'
        WHEN t.teff_gspphot >= 2900 THEN 'M_mid'
        ELSE 'M_late'
    END AS spec_subclass
FROM lab.nasa_gaia_train AS t;

DROP VIEW IF EXISTS lab.v_nasa_gaia_train_dwarfs;

CREATE VIEW lab.v_nasa_gaia_train_dwarfs AS
SELECT *
FROM lab.v_nasa_gaia_train_classified
WHERE spec_class IN ('M', 'K', 'G', 'F')
  AND teff_gspphot IS NOT NULL
  AND logg_gspphot IS NOT NULL
  AND radius_gspphot IS NOT NULL
  AND logg_gspphot >= 4.0;

DROP VIEW IF EXISTS lab.v_nasa_gaia_train_evolved;

CREATE VIEW lab.v_nasa_gaia_train_evolved AS
SELECT *
FROM lab.v_nasa_gaia_train_classified
WHERE spec_class IN ('M', 'K', 'G', 'F')
  AND teff_gspphot IS NOT NULL
  AND logg_gspphot IS NOT NULL
  AND radius_gspphot IS NOT NULL
  AND logg_gspphot < 4.0;

/* -------------------------------------------------------------------------
   Контрольные запросы
   ------------------------------------------------------------------------- */

SELECT spec_class, COUNT(*)
FROM lab.v_nasa_gaia_train_classified
GROUP BY spec_class
ORDER BY spec_class;

SELECT spec_subclass, COUNT(*)
FROM lab.v_nasa_gaia_train_classified
GROUP BY spec_subclass
ORDER BY spec_subclass;

SELECT evolution_stage, COUNT(*)
FROM lab.v_nasa_gaia_train_classified
GROUP BY evolution_stage
ORDER BY evolution_stage;

SELECT COUNT(*) AS n_classified
FROM lab.v_nasa_gaia_train_classified;

SELECT COUNT(*) AS n_dwarfs
FROM lab.v_nasa_gaia_train_dwarfs;

SELECT COUNT(*) AS n_evolved
FROM lab.v_nasa_gaia_train_evolved;
