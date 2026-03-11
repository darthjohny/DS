-- Forward migration for result tables after router posterior +
-- contrastive host-field rollout.
--
-- Strategy:
--   - ALTER existing tables in place.
--   - Keep current table names because production code already writes to
--     lab.gaia_router_results and lab.gaia_priority_results.
--   - Keep all new columns nullable to preserve:
--       * existing historical rows;
--       * LOW-priority stub rows where host diagnostics are intentionally NULL.
--
-- Apply:
--   psql "$DATABASE_URL" -f sql/2026-03-11_gaia_results_posterior_host_fields.sql
--
-- Rollback:
--   psql "$DATABASE_URL" -f sql/2026-03-11_gaia_results_posterior_host_fields.rollback.sql

BEGIN;

ALTER TABLE lab.gaia_router_results
    ADD COLUMN IF NOT EXISTS router_log_likelihood double precision,
    ADD COLUMN IF NOT EXISTS router_log_posterior double precision,
    ADD COLUMN IF NOT EXISTS posterior_margin double precision;

ALTER TABLE lab.gaia_router_results
    DROP CONSTRAINT IF EXISTS chk_router_posterior_margin;

ALTER TABLE lab.gaia_router_results
    ADD CONSTRAINT chk_router_posterior_margin
    CHECK (
        posterior_margin IS NULL
        OR posterior_margin >= 0.0
    );

COMMENT ON COLUMN lab.gaia_router_results.router_log_likelihood
    IS 'Gaussian router log-likelihood for the selected class.';
COMMENT ON COLUMN lab.gaia_router_results.router_log_posterior
    IS 'Gaussian router log-posterior for the selected class.';
COMMENT ON COLUMN lab.gaia_router_results.posterior_margin
    IS 'Gap between best and runner-up router posterior scores.';

ALTER TABLE lab.gaia_priority_results
    ADD COLUMN IF NOT EXISTS router_log_likelihood double precision,
    ADD COLUMN IF NOT EXISTS router_log_posterior double precision,
    ADD COLUMN IF NOT EXISTS posterior_margin double precision,
    ADD COLUMN IF NOT EXISTS host_log_likelihood double precision,
    ADD COLUMN IF NOT EXISTS field_log_likelihood double precision,
    ADD COLUMN IF NOT EXISTS host_log_lr double precision,
    ADD COLUMN IF NOT EXISTS host_posterior double precision;

ALTER TABLE lab.gaia_priority_results
    DROP CONSTRAINT IF EXISTS chk_priority_posterior_margin;

ALTER TABLE lab.gaia_priority_results
    ADD CONSTRAINT chk_priority_posterior_margin
    CHECK (
        posterior_margin IS NULL
        OR posterior_margin >= 0.0
    );

ALTER TABLE lab.gaia_priority_results
    DROP CONSTRAINT IF EXISTS chk_priority_host_posterior;

ALTER TABLE lab.gaia_priority_results
    ADD CONSTRAINT chk_priority_host_posterior
    CHECK (
        host_posterior IS NULL
        OR (
            host_posterior >= 0.0
            AND host_posterior <= 1.0
        )
    );

COMMENT ON COLUMN lab.gaia_priority_results.router_log_likelihood
    IS 'Gaussian router log-likelihood for the selected class.';
COMMENT ON COLUMN lab.gaia_priority_results.router_log_posterior
    IS 'Gaussian router log-posterior for the selected class.';
COMMENT ON COLUMN lab.gaia_priority_results.posterior_margin
    IS 'Gap between best and runner-up router posterior scores.';
COMMENT ON COLUMN lab.gaia_priority_results.host_log_likelihood
    IS 'Contrastive host-population log-likelihood.';
COMMENT ON COLUMN lab.gaia_priority_results.field_log_likelihood
    IS 'Contrastive field-population log-likelihood.';
COMMENT ON COLUMN lab.gaia_priority_results.host_log_lr
    IS 'Host-vs-field log-likelihood ratio within the routed class.';
COMMENT ON COLUMN lab.gaia_priority_results.host_posterior
    IS 'Bounded host score used by the decision layer.';

COMMIT;
