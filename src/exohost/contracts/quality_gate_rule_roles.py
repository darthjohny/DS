# Файл `quality_gate_rule_roles.py` слоя `contracts`.
#
# Этот файл отвечает только за:
# - контракты колонок, датасетов и ролей правил;
# - локальную ответственность текущего модуля без смешения соседних ролей.
#
# Следующий слой:
# - соседние модули слоя `contracts` и orchestration поверх них;
# - unit-тесты и более верхние слои проекта.

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

type QualityGateRuleRole = Literal["reject", "review", "info"]
type QualityGateRuleScope = Literal["quality", "ood", "support"]


@dataclass(frozen=True, slots=True)
class QualityGateRuleSpec:
    # Описывает, какую роль сигнал играет в текущей первой волне gate-политики.
    rule_name: str
    role: QualityGateRuleRole
    scope: QualityGateRuleScope
    signal_columns: tuple[str, ...]
    live_reason_labels: tuple[str, ...]
    live_review_buckets: tuple[str, ...]
    rationale: str


QUALITY_GATE_RULE_SPECS: tuple[QualityGateRuleSpec, ...] = (
    # `reject`-правила отсекают строки, которые уже нельзя безопасно подавать
    # в основной прикладной контур без потери ключевых признаков.
    QualityGateRuleSpec(
        rule_name="missing_core_features",
        role="reject",
        scope="quality",
        signal_columns=("has_missing_core_features", "has_core_features"),
        live_reason_labels=("reject_missing_core_features",),
        live_review_buckets=("reject_missing_core_features",),
        rationale=(
            "Отсутствие критических core-признаков делает объект непригодным "
            "для normal pass в первую волну."
        ),
    ),
    # `review`-правила не удаляют объект навсегда, а переводят его в проверку.
    # Это отдельный слой аккуратности, который мы донастраивали перед боевым прогоном.
    QualityGateRuleSpec(
        rule_name="high_ruwe",
        role="review",
        scope="quality",
        signal_columns=("has_high_ruwe", "ruwe"),
        live_reason_labels=("review_high_ruwe",),
        live_review_buckets=("review_high_ruwe",),
        rationale=(
            "Повышенный RUWE трактуется как ранний quality-risk и переводит "
            "строку в review/unknown, а не в hard reject."
        ),
    ),
    QualityGateRuleSpec(
        rule_name="low_parallax_snr",
        role="review",
        scope="quality",
        signal_columns=("has_low_parallax_snr", "parallax_over_error"),
        live_reason_labels=("review_low_parallax_snr",),
        live_review_buckets=("review_low_parallax_snr",),
        rationale=(
            "Слабая дистанционная информация ухудшает надежность, но сама по "
            "себе не считается hard reject."
        ),
    ),
    QualityGateRuleSpec(
        rule_name="missing_flame_features",
        role="review",
        scope="quality",
        signal_columns=("has_missing_flame_features", "has_flame_features", "radius_flame"),
        live_reason_labels=("review_missing_radius_flame",),
        live_review_buckets=("review_missing_radius_flame",),
        rationale=(
            "Отсутствие FLAME-радиуса ограничивает normal pass, но остается "
            "review-сигналом, а не hard reject."
        ),
    ),
    # `info`-правила нужны для объяснения и диагностики. Они не должны превращаться
    # в скрытые hard reject, иначе контур доверия станет непрозрачным.
    QualityGateRuleSpec(
        rule_name="non_single_star_flag",
        role="info",
        scope="ood",
        signal_columns=("has_non_single_star_flag", "non_single_star"),
        live_reason_labels=(),
        live_review_buckets=("review_non_single_star",),
        rationale=(
            "Флаг non-single star не должен смешиваться с quality reject; это "
            "сигнал о possible domain/interpretation risk."
        ),
    ),
    QualityGateRuleSpec(
        rule_name="low_single_star_probability",
        role="info",
        scope="ood",
        signal_columns=(
            "has_low_single_star_probability",
            "classprob_dsc_combmod_star",
        ),
        live_reason_labels=(),
        live_review_buckets=("review_low_single_star_probability",),
        rationale=(
            "Низкая single-star probability — это OOD-подозрение и review "
            "сигнал, а не quality reject."
        ),
    ),
    QualityGateRuleSpec(
        rule_name="core_feature_presence",
        role="info",
        scope="support",
        signal_columns=("has_core_features",),
        live_reason_labels=(),
        live_review_buckets=(),
        rationale=(
            "Поддерживающий диагностический сигнал для explainability и audit, "
            "не самостоятельное policy-решение."
        ),
    ),
    QualityGateRuleSpec(
        rule_name="flame_feature_presence",
        role="info",
        scope="support",
        signal_columns=("has_flame_features",),
        live_reason_labels=(),
        live_review_buckets=(),
        rationale=(
            "Поддерживающий диагностический сигнал для explainability и audit, "
            "не самостоятельное policy-решение."
        ),
    ),
)


__all__ = [
    "QUALITY_GATE_RULE_SPECS",
    "QualityGateRuleRole",
    "QualityGateRuleScope",
    "QualityGateRuleSpec",
]
