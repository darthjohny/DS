-- Rollback for UNKNOWN constraint widening.
--
-- Important:
--   rollback will fail if result tables already contain rows with
--   predicted_spec_class='UNKNOWN' or predicted_evolution_stage='unknown'.
--   Remove or rewrite such rows before applying this rollback.

BEGIN;

ALTER TABLE lab.gaia_priority_results
    DROP CONSTRAINT IF EXISTS chk_priority_evolution_stage;

ALTER TABLE lab.gaia_priority_results
    ADD CONSTRAINT chk_priority_evolution_stage
    CHECK (
        predicted_evolution_stage IN ('dwarf', 'evolved')
    );

ALTER TABLE lab.gaia_priority_results
    DROP CONSTRAINT IF EXISTS chk_priority_spec_class;

ALTER TABLE lab.gaia_priority_results
    ADD CONSTRAINT chk_priority_spec_class
    CHECK (
        predicted_spec_class IN ('A', 'B', 'F', 'G', 'K', 'M', 'O')
    );

ALTER TABLE lab.gaia_router_results
    DROP CONSTRAINT IF EXISTS chk_router_evolution_stage;

ALTER TABLE lab.gaia_router_results
    ADD CONSTRAINT chk_router_evolution_stage
    CHECK (
        predicted_evolution_stage IN ('dwarf', 'evolved')
    );

ALTER TABLE lab.gaia_router_results
    DROP CONSTRAINT IF EXISTS chk_router_spec_class;

ALTER TABLE lab.gaia_router_results
    ADD CONSTRAINT chk_router_spec_class
    CHECK (
        predicted_spec_class IN ('A', 'B', 'F', 'G', 'K', 'M', 'O')
    );

COMMIT;
