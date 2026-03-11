-- Rollback for sql/2026-03-11_gaia_results_posterior_host_fields.sql
--
-- Use only if the application must be returned to the pre-posterior schema.

BEGIN;

ALTER TABLE lab.gaia_priority_results
    DROP CONSTRAINT IF EXISTS chk_priority_host_posterior;

ALTER TABLE lab.gaia_priority_results
    DROP CONSTRAINT IF EXISTS chk_priority_posterior_margin;

ALTER TABLE lab.gaia_priority_results
    DROP COLUMN IF EXISTS host_posterior,
    DROP COLUMN IF EXISTS host_log_lr,
    DROP COLUMN IF EXISTS field_log_likelihood,
    DROP COLUMN IF EXISTS host_log_likelihood,
    DROP COLUMN IF EXISTS posterior_margin,
    DROP COLUMN IF EXISTS router_log_posterior,
    DROP COLUMN IF EXISTS router_log_likelihood;

ALTER TABLE lab.gaia_router_results
    DROP CONSTRAINT IF EXISTS chk_router_posterior_margin;

ALTER TABLE lab.gaia_router_results
    DROP COLUMN IF EXISTS posterior_margin,
    DROP COLUMN IF EXISTS router_log_posterior,
    DROP COLUMN IF EXISTS router_log_likelihood;

COMMIT;
