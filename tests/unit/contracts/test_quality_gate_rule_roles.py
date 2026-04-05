# Тестовый файл `test_quality_gate_rule_roles.py` домена `contracts`.
#
# Этот файл проверяет только:
# - проверку логики домена: контракты датасетов, колонок и policy-слоев;
# - регрессионные сценарии и ожидаемые контракты целевого слоя.
#
# Следующий слой:
# - реализация домена `contracts` в `src/exohost`;
# - соседние тесты и testkit этого же пакета.

from __future__ import annotations

from exohost.contracts.quality_gate_rule_roles import QUALITY_GATE_RULE_SPECS


def test_quality_gate_rule_specs_have_unique_names() -> None:
    rule_names = [spec.rule_name for spec in QUALITY_GATE_RULE_SPECS]
    assert len(rule_names) == len(set(rule_names))


def test_quality_gate_rule_roles_keep_expected_first_wave_assignments() -> None:
    role_map = {spec.rule_name: spec for spec in QUALITY_GATE_RULE_SPECS}

    assert role_map["missing_core_features"].role == "reject"
    assert role_map["missing_core_features"].scope == "quality"

    assert role_map["high_ruwe"].role == "review"
    assert role_map["low_parallax_snr"].role == "review"
    assert role_map["missing_flame_features"].role == "review"

    assert role_map["non_single_star_flag"].role == "info"
    assert role_map["non_single_star_flag"].scope == "ood"
    assert role_map["low_single_star_probability"].role == "info"
    assert role_map["low_single_star_probability"].scope == "ood"


def test_quality_gate_rule_specs_reference_live_labels_consistently() -> None:
    role_map = {spec.rule_name: spec for spec in QUALITY_GATE_RULE_SPECS}

    assert role_map["missing_core_features"].live_review_buckets == (
        "reject_missing_core_features",
    )
    assert role_map["high_ruwe"].live_review_buckets == ("review_high_ruwe",)
    assert role_map["low_parallax_snr"].live_review_buckets == (
        "review_low_parallax_snr",
    )
    assert role_map["missing_flame_features"].live_review_buckets == (
        "review_missing_radius_flame",
    )
