-- Build normalized Gaia reference views for the router layer.
--
-- Что делает файл:
--   - собирает референсные Gaia-популяции по классам и стадиям;
--   - нормализует старые `A/B/O` таблицы к общей схеме router-слоя;
--   - строит единый источник `lab.v_gaia_router_training`.
--
-- Что НЕ делает:
--   - не строит host-training;
--   - не определяет финальный scoring;
--   - не дублирует вспомогательную legacy-view `lab.v_gaia_ref_abo_training`,
--     которая остаётся допустимым EDA-артефактом, но не нужна для
--     канонического router training layer.

CREATE OR REPLACE VIEW lab.v_gaia_ref_abo_dwarfs AS
SELECT
    r.source_id,
    r.ra,
    r.dec,
    r.parallax,
    NULL::real AS parallax_over_error,
    NULL::real AS ruwe,
    NULL::real AS bp_rp,
    r.teff_gspphot,
    r.logg_gspphot,
    r.radius_gspphot,
    NULL::real AS mh_gspphot,
    UPPER(r.spectral_class)::text AS spec_class,
    NULL::text AS spec_subclass,
    'gaia_ref'::text AS source_type,
    'dwarf'::text AS evolution_stage,
    NULL::integer AS random_index
FROM (
    SELECT * FROM public.gaia_ref_class_a
    UNION ALL
    SELECT * FROM public.gaia_ref_class_b
    UNION ALL
    SELECT * FROM public.gaia_ref_class_o
) AS r
WHERE r.logg_gspphot >= 4.0;

CREATE OR REPLACE VIEW lab.v_gaia_ref_abo_evolved AS
SELECT
    r.source_id,
    r.ra,
    r.dec,
    r.parallax,
    r.parallax_over_error,
    r.ruwe,
    r.bp_rp,
    r.teff_gspphot,
    r.logg_gspphot,
    r.radius_gspphot,
    r.mh_gspphot,
    r.spec_class::text AS spec_class,
    NULL::text AS spec_subclass,
    'gaia_ref'::text AS source_type,
    'evolved'::text AS evolution_stage,
    r.random_index
FROM (
    SELECT * FROM public.gaia_ref_evolved_class_a
    UNION ALL
    SELECT * FROM public.gaia_ref_evolved_class_b
    UNION ALL
    SELECT * FROM public.gaia_ref_evolved_class_o
) AS r
WHERE r.logg_gspphot < 4.0;

CREATE OR REPLACE VIEW lab.v_gaia_ref_mkgf_dwarfs AS
SELECT
    r.source_id,
    r.ra,
    r.dec,
    r.parallax,
    r.parallax_over_error,
    r.ruwe,
    r.bp_rp,
    r.teff_gspphot,
    r.logg_gspphot,
    r.radius_gspphot,
    r.mh_gspphot,
    r.spec_class::text AS spec_class,
    NULL::text AS spec_subclass,
    'gaia_ref'::text AS source_type,
    'dwarf'::text AS evolution_stage,
    r.random_index
FROM (
    SELECT * FROM public.gaia_ref_class_m
    UNION ALL
    SELECT * FROM public.gaia_ref_class_k
    UNION ALL
    SELECT * FROM public.gaia_ref_class_g
    UNION ALL
    SELECT * FROM public.gaia_ref_class_f
) AS r;

CREATE OR REPLACE VIEW lab.v_gaia_ref_mkgf_evolved AS
SELECT
    r.source_id,
    r.ra,
    r.dec,
    r.parallax,
    r.parallax_over_error,
    r.ruwe,
    r.bp_rp,
    r.teff_gspphot,
    r.logg_gspphot,
    r.radius_gspphot,
    r.mh_gspphot,
    r.spec_class::text AS spec_class,
    NULL::text AS spec_subclass,
    'gaia_ref'::text AS source_type,
    'evolved'::text AS evolution_stage,
    r.random_index
FROM (
    SELECT * FROM public.gaia_ref_evolved_class_m
    UNION ALL
    SELECT * FROM public.gaia_ref_evolved_class_k
    UNION ALL
    SELECT * FROM public.gaia_ref_evolved_class_g
    UNION ALL
    SELECT * FROM public.gaia_ref_evolved_class_f
) AS r;

CREATE OR REPLACE VIEW lab.v_gaia_router_training AS
SELECT * FROM lab.v_gaia_ref_mkgf_dwarfs
UNION ALL
SELECT * FROM lab.v_gaia_ref_mkgf_evolved
UNION ALL
SELECT * FROM lab.v_gaia_ref_abo_dwarfs
UNION ALL
SELECT * FROM lab.v_gaia_ref_abo_evolved;
