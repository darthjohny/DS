# Общие fixtures и helpers для регресс-тестов.
#
# Этот файл отвечает только за:
# - общий каркас frozen fixtures для слоя `tests/regression`;
# - компактные helpers чтения и проверки маленьких тестовых входов.
#
# Следующий слой:
# - доменные регресс-тесты в `decision`, `posthoc` и `reporting`;
# - замороженные данные из каталога `fixtures`.

from pathlib import Path

REGRESSION_ROOT = Path(__file__).resolve().parent
REGRESSION_FIXTURES_ROOT = REGRESSION_ROOT / "fixtures"
QUALITY_GATE_SMALL_FIXTURE_PATH = REGRESSION_FIXTURES_ROOT / "quality_gate_small.csv"
PRIORITY_BASE_SMALL_FIXTURE_PATH = REGRESSION_FIXTURES_ROOT / "priority_base_small.csv"
PRIORITY_FINAL_DECISION_SMALL_FIXTURE_PATH = (
    REGRESSION_FIXTURES_ROOT / "priority_final_decision_small.csv"
)
DECIDE_INPUT_SMALL_FIXTURE_PATH = REGRESSION_FIXTURES_ROOT / "decide_input_small.csv"
DECISION_ARTIFACT_SCHEMA_FIXTURE_PATH = (
    REGRESSION_FIXTURES_ROOT / "decision_artifact_schema_small.json"
)
