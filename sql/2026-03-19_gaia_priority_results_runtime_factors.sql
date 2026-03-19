-- Forward migration for priority-result runtime factors.
--
-- Strategy:
--   - ALTER existing table in place.
--   - Keep legacy `quality_factor` for compatibility.
--   - Add explicit `reliability_factor` and `followup_factor` so DB schema
--     matches the current production runtime contract.
--
-- Apply:
--   psql "$DATABASE_URL" -f sql/2026-03-19_gaia_priority_results_runtime_factors.sql
--
-- Rollback:
--   psql "$DATABASE_URL" -f sql/2026-03-19_gaia_priority_results_runtime_factors.rollback.sql

BEGIN;

ALTER TABLE lab.gaia_priority_results
    ADD COLUMN IF NOT EXISTS reliability_factor double precision,
    ADD COLUMN IF NOT EXISTS followup_factor double precision;

ALTER TABLE lab.gaia_priority_results
    DROP CONSTRAINT IF EXISTS chk_priority_reliability_factor;

ALTER TABLE lab.gaia_priority_results
    ADD CONSTRAINT chk_priority_reliability_factor
    CHECK (
        reliability_factor IS NULL
        OR (
            reliability_factor >= 0.0
            AND reliability_factor <= 1.0
        )
    );

ALTER TABLE lab.gaia_priority_results
    DROP CONSTRAINT IF EXISTS chk_priority_followup_factor;

ALTER TABLE lab.gaia_priority_results
    ADD CONSTRAINT chk_priority_followup_factor
    CHECK (
        followup_factor IS NULL
        OR (
            followup_factor >= 0.0
            AND followup_factor <= 1.0
        )
    );

COMMENT ON COLUMN lab.gaia_priority_results.quality_factor
    IS 'Legacy compatibility alias for reliability_factor.';
COMMENT ON COLUMN lab.gaia_priority_results.reliability_factor
    IS 'Astrometric reliability factor derived from RUWE and parallax precision.';
COMMENT ON COLUMN lab.gaia_priority_results.followup_factor
    IS 'Observational follow-up factor derived from parallax as a distance proxy.';

COMMIT;
