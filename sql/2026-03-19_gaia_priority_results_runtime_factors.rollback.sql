-- Rollback for sql/2026-03-19_gaia_priority_results_runtime_factors.sql
--
-- Use only if the application must return to the pre-runtime-factor schema.

BEGIN;

ALTER TABLE lab.gaia_priority_results
    DROP CONSTRAINT IF EXISTS chk_priority_followup_factor;

ALTER TABLE lab.gaia_priority_results
    DROP CONSTRAINT IF EXISTS chk_priority_reliability_factor;

ALTER TABLE lab.gaia_priority_results
    DROP COLUMN IF EXISTS followup_factor,
    DROP COLUMN IF EXISTS reliability_factor;

COMMENT ON COLUMN lab.gaia_priority_results.quality_factor
    IS NULL;

COMMIT;
