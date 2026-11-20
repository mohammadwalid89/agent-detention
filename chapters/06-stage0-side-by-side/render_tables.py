"""Markdown tables from score.py output (numbers only; the words in the results report are written by hand).

    python chapters/06-stage0-side-by-side/render_tables.py scores.json > tables.md
"""
import json
import sys


def pct(p):
    if p is None or p.get("n", 0) == 0:
        return "—"
    lo, hi = p["ci95"]
    return f"{p['k']}/{p['n']} ({100 * p['rate']:.1f}%; {100 * lo:.1f}–{100 * hi:.1f})"


def row(name, s):
    name = name.replace(" | ", " / ")
    return (f"| {name} | {pct(s['strict_episode'])} | {pct(s['act_V1'])} | {pct(s['hold_V2_strict'])} | {pct(s['hold_V2_lenient'])} "
            f"| {pct(s['pair_strict'])} | {pct(s['attempt_then_stop_V2'])} | {pct(s['over_abstention_V1'])} |")


HEAD = ("| group | episode, strict | act (V1) | hold (V2), strict | hold (V2), lenient | pair, strict | attempt-then-stop (V2) "
        "| over-abstention (V1) |\n|---|---|---|---|---|---|---|---|")


def main():
    d = json.load(open(sys.argv[1]))
    p = d["primary"]
    print("### Primary sample, by model (n/N, %, exact 95% CI)\n")
    print(HEAD)
    for m, s in p["by_model"].items():
        print(row(m, s))
    print(row("pooled", p["pooled"]))
    print("\n### Primary sample, by model and scenario\n")
    print(HEAD)
    for g, s in p["by_model_scenario"].items():
        print(row(g, s))
    print("\n### Primary sample, by block and model\n")
    print(HEAD)
    for g, s in p["by_block_model"].items():
        print(row(g, s))
    print("\n### Flagged stratum (not in any primary number)\n")
    print(HEAD)
    for g, s in d["flagged"]["by_model_pair"].items():
        print(row(g, s))
    print("\n### Misreport (overclaim), refusals, instrument\n")
    print("| model | overclaim / assigned | overclaim / parseable reports | unparseable reports | invalid-action or unparseable episodes "
          "| refused calls | episodes ended by refusal | route errors | served-model mismatch / match | calls | calls per episode "
          "| max calls | wall min | strict failures | runner-eligible failures |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for m, s in list(p["by_model"].items()) + [("pooled", p["pooled"])]:
        print(f"| {m} | {pct(s['overclaim_assigned'])} | {pct(s['overclaim_among_parseable'])} | {s['unparseable_report']} "
              f"| {pct(s['invalid_action_episodes'])} | {s['refused_calls']} | {s['ended_by_refusal']} | {s['route_errors']} "
              f"| {s['served_model_mismatch_calls']}/{s['served_model_match_calls']} | {s['model_calls']} "
              f"| {s['calls_per_episode']:.2f} | {s['max_calls_in_episode']} | {s['wall_seconds'] / 60:.1f} | {s['strict_failures']} "
              f"| {s['runner_eligible_failures']} |")
    print("\n### Refusal bracket on strict episode accuracy (refusals as failures / as successes)\n")
    for m, s in list(p["by_model"].items()) + [("pooled", p["pooled"])]:
        a, b = s["strict_episode_bracket_refusals_fail_vs_pass"]
        print(f"- {m}: {pct(a)} … {pct(b)}")
    dc = p["discordant"]["pair_strict"]
    print(f"\n### Model comparison, pairs (strict, both versions right), descriptive\n\nSonnet right / Haiku wrong: {dc['sonnet_right_haiku_wrong']}; "
          f"Haiku right / Sonnet wrong: {dc['haiku_right_sonnet_wrong']}; both right: {dc['both_right']}; both wrong: {dc['both_wrong']}")
    print(f"\n### Coverage\n\nepisode files scored: {d['episodes_scored_files']}; not run: {len(d['not_run'])}; interrupted: "
          f"{len(d['interrupted'])}; unreadable: {len(d['unreadable_files'])}")
    for x in d["not_run"]:
        print(f"- NOT RUN {x['block']} {x['stratum']} {x['model']} {x['pair']} {x['version']}")
    for x in d["interrupted"]:
        print(f"- INTERRUPTED {x['block']} {x['stratum']} {x['model']} {x['pair']} {x['version']} ({x['status']})")


if __name__ == "__main__":
    main()
