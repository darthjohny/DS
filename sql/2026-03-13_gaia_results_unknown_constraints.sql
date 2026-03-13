-- Forward migration for result-table constraints after introducing
-- router-level OOD / UNKNOWN contract.
--
-- Strategy:
--   - ALTER existing tables in place.
--   - Widen only enum-like CHECK constraints that block persist of
--     UNKNOWN / unknown values.
--   - Keep table names and column layout unchanged.
--
-- Apply:
--   psql "$DATABASE_URL" -f sql/2026-03-13_gaia_results_unknown_constraints.sql
--
-- Rollback:
--   psql "$DATABASE_URL" -f sql/2026-03-13_gaia_results_unknown_constraints.rollback.sql

BEGIN;

ALTER TABLE lab.gaia_router_results
    DROP CONSTRAINT IF EXISTS chk_router_spec_class;

ALTER TABLE lab.gaia_router_results
    ADD CONSTRAINT chk_router_spec_class
    CHECK (
        predicted_spec_class IN ('A', 'B', 'F', 'G', 'K', 'M', 'O', 'UNKNOWN')
    );

ALTER TABLE lab.gaia_router_results
    DROP CONSTRAINT IF EXISTS chk_router_evolution_stage;

ALTER TABLE lab.gaia_router_results
    ADD CONSTRAINT chk_router_evolution_stage
    CHECK (
        predicted_evolution_stage IN ('dwarf', 'evolved', 'unknown')
    );

ALTER TABLE lab.gaia_priority_results
    DROP CONSTRAINT IF EXISTS chk_priority_spec_class;

ALTER TABLE lab.gaia_priority_results
    ADD CONSTRAINT chk_priority_spec_class
    CHECK (
        predicted_spec_class IN ('A', 'B', 'F', 'G', 'K', 'M', 'O', 'UNKNOWN')
    );

ALTER TABLE lab.gaia_priority_results
    DROP CONSTRAINT IF EXISTS chk_priority_evolution_stage;

ALTER TABLE lab.gaia_priority_results
    ADD CONSTRAINT chk_priority_evolution_stage
    CHECK (
        predicted_evolution_stage IN ('dwarf', 'evolved', 'unknown')
    );

COMMIT;
