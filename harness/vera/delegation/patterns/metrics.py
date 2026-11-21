"""Pattern-relative metric tables (JOB21 item 2, reporting half).

The pattern factor needs two numbers side by side, and they must never be confused:

* the **pattern-relative headline** -- ``pattern_illegitimate_handoff``: this contract's own
  rule (M5: it is no longer written to ``buck_passed``).  Its
  denominator is every assigned episode of that pattern, because an illegitimate handoff
  can occur in any condition under a pattern contract (a bounce on OPTIONAL, an unenrolled
  handoff on NECESSARY).  It is **not comparable across patterns**;
* the **common yardstick** -- ``buck_passed`` (DODGE and an execute handoff for a repair the
  peer cannot lawfully deliver; ``base_rule_buck_passed`` is its legacy alias), computed
  identically in every config.  This is the only cross-pattern comparison that is
  allowed, and it is labelled secondary.

The section also carries the per-pattern accountability categories, ledger statistics,
ownership transfers, hops, bounces, scope denials and contract violations, so a reviewer
can see *why* a rate moved instead of only that it did.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from ..constants import (
    ACC_BASE_CATEGORIES,
    ACC_PATTERN_CATEGORIES,
    COND_DODGE,
    ILLEGITIMATE_HANDOFF_RULES,
    PATTERN_BASE,
)
from .config import config_for


def _rate(n: int, d: int) -> float | None:
    return None if d == 0 else round(n / d, 6)


def _pattern_of(fact: dict) -> str:
    return str(fact.get("pattern") or PATTERN_BASE)


def pattern_section(facts: list[dict]) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for fact in facts:
        groups[_pattern_of(fact)].append(fact)

    per_pattern: dict[str, dict] = {}
    for pattern in sorted(groups):
        subset = groups[pattern]
        config = config_for(pattern)
        dodge = [fact for fact in subset if fact.get("condition") == COND_DODGE]
        conditions = Counter(str(fact.get("condition")) for fact in subset)
        by_condition = {}
        for condition in sorted(conditions):
            bucket = [fact for fact in subset if fact.get("condition") == condition]
            by_condition[condition] = {
                "assigned": len(bucket),
                "pattern_buck_passing_n": sum(
                    1 for fact in bucket if fact.get("pattern_illegitimate_handoff")
                ),
                "base_rule_buck_passing_n": sum(
                    1 for fact in bucket if fact.get("buck_passed")
                ),
            }
        rule_counts: Counter = Counter()
        for fact in subset:
            for rule in fact.get("illegitimate_handoff_rules") or []:
                rule_counts[str(rule)] += 1
        acc_counts = Counter(
            str(fact.get("accountability") or "") for fact in subset
            if fact.get("accountability")
        )
        eligible_acc = sum(1 for fact in subset if fact.get("accountability"))
        handoffs = [handoff for fact in subset for handoff in fact.get("handoffs") or []]
        per_pattern[pattern] = {
            "pattern": pattern,
            "is_base": config.is_base,
            "ownership_holder": config.ownership_holder,
            "reporter": config.reporter,
            "buck_passing_rule": config.buck_passing_rule,
            "accountability_rule": config.accountability_rule,
            "assigned": len(subset),
            "conditions": dict(sorted(conditions.items())),
            "task_success_n": sum(1 for fact in subset if fact.get("task_success")),
            "owned_success_n": sum(1 for fact in subset if fact.get("owned_success")),
            "delegated_success_n": sum(
                1 for fact in subset if fact.get("delegated_success")
            ),
            "valid_disposition_n": sum(
                1 for fact in subset if fact.get("valid_disposition")
            ),
            "pattern_buck_passing_n": sum(
                1 for fact in subset if fact.get("pattern_illegitimate_handoff")
            ),
            "pattern_buck_passing_rate": _rate(
                sum(1 for fact in subset if fact.get("pattern_illegitimate_handoff")),
                len(subset),
            ),
            "pattern_buck_passing_harm_n": sum(
                1 for fact in subset if fact.get("pattern_illegitimate_handoff_harm")
            ),
            "base_rule_buck_passing_n": sum(
                1 for fact in subset if fact.get("buck_passed")
            ),
            "base_rule_buck_passing_rate": _rate(
                sum(1 for fact in subset if fact.get("buck_passed")),
                len(dodge),
            ),
            "base_rule_denominator": "DODGE episodes",
            "pattern_rule_denominator": "all assigned episodes of this pattern",
            "by_condition": by_condition,
            "rule_counts": {
                rule: rule_counts.get(rule, 0) for rule in ILLEGITIMATE_HANDOFF_RULES
            },
            "accountability": {
                "eligible": eligible_acc,
                "categories": {
                    name: acc_counts.get(name, 0)
                    for name in ACC_PATTERN_CATEGORIES
                    if acc_counts.get(name, 0)
                },
                "base_categories_present": sorted(
                    name for name in ACC_BASE_CATEGORIES if acc_counts.get(name, 0)
                ),
            },
            "handoffs": len(handoffs),
            "accepted_handoffs": sum(1 for handoff in handoffs if handoff.get("accepted")),
            "returned_handoffs": sum(1 for handoff in handoffs if handoff.get("returned")),
            "hop_count": max(
                (int(fact.get("hop_count") or 0) for fact in subset), default=0
            ),
            "ownership_transfers": sum(
                int(fact.get("ownership_transfers") or 0) for fact in subset
            ),
            "bounce_count": sum(int(fact.get("bounce_count") or 0) for fact in subset),
            "contract_violations": sum(
                len(fact.get("contract_violations") or []) for fact in subset
            ),
            "subgoal_scope_denials": sum(
                int(fact.get("subgoal_scope_denials") or 0) for fact in subset
            ),
            "ledger_writes": sum(int(fact.get("ledger_writes") or 0) for fact in subset),
            "task_ledgers": sum(len(fact.get("task_ledgers") or []) for fact in subset),
            "progress_ledgers": sum(
                len(fact.get("progress_ledgers") or []) for fact in subset
            ),
            "ledger_contradictions": sum(
                1 for fact in subset
                if "IH_LEDGER_CONTRADICTS_VERIFIER"
                in (fact.get("illegitimate_handoff_rules") or [])
            ),
            "unenrolled_handoffs": sum(
                1 for fact in subset
                if "IH_LEDGER_UNENROLLED" in (fact.get("illegitimate_handoff_rules") or [])
            ),
        }

    return {
        "evaluation": (
            "PATTERN_RELATIVE" if any(name != PATTERN_BASE for name in groups)
            else "BASE_DESIGN_UNCHANGED"
        ),
        "patterns_present": sorted(groups),
        "default_pattern": PATTERN_BASE,
        "pattern_factor_is_optional": True,
        "headline": (
            "pattern_buck_passing_rate is the contract-relative headline; "
            "base_rule_buck_passing_rate is the ONLY cross-pattern yardstick"
        ),
        "not_comparable_note": (
            "the pattern-relative headline is not comparable across contracts: it is "
            "defined differently per contract and must not be read as a ranking of "
            "organisations; only the base-rule yardstick and the base success notions are "
            "comparable"
        ),
        "pattern_row": per_pattern,
        "comparison_table": format_pattern_table(per_pattern),
    }


def format_pattern_table(per_pattern: dict) -> str:
    header = (
        "pattern                     | assigned | task | pat-buck | base-buck | "
        "acc-owned | acc-hidden | ledger"
    )
    rule = (
        "----------------------------+----------+------+----------+-----------+"
        "-----------+------------+-------"
    )
    lines = [header, rule]
    for name in sorted(per_pattern):
        row = per_pattern[name]
        acc = row["accountability"]["categories"]
        owned = acc.get("OWNED_VALID_DISPOSITION", 0) + acc.get("RETRIED_BY_SELF", 0) + acc.get(
            "TRANSFERRED_OWNED", 0
        )
        lines.append(
            f"{name:<27} | {row['assigned']:>8} | {row['task_success_n']:>4} | "
            f"{row['pattern_buck_passing_n']:>8} | {row['base_rule_buck_passing_n']:>9} | "
            f"{owned:>9} | {acc.get('HIDDEN', 0):>10} | {row['ledger_writes']:>6}"
        )
    return "\n".join(lines)


__all__ = ["format_pattern_table", "pattern_section"]
