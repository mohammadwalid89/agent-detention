"""JOB38 RC7/F9: prospective power for the actual v1.3 assignment and four-test family.

This module replaces the earlier ``fraction(diff >= 0.15)`` statistic with

* the **actual assignment table** rows (model x instance x condition x cue x identity x
  repeat x organisation pattern), including the SWAPPED label permutation recorded on each
  row, so exposures and cluster sizes come from the frozen schedule rather than a
  ``ceil(U / clusters)`` assumption;
* cluster units at the **highest shared construction lineage** (mechanism/template), with
  the mechanism-only clustering reported as a sensitivity because 50 admitted OPTIONAL
  variants span 47 mechanisms and only 28 template/mechanism lineages;
* the real estimators: fixed-weight within-backend displayed-publisher label effect and
  paired Detention A-B / A-C / A-D on a common endpoint;
* actual cluster-robust two-sided t tests and a **Holm** step-down correction over the four
  predeclared tests (label + three Detention contrasts);
* a **failure-yield** simulation that reports P(eligible < 60), P(eligible = 0) and the
  expected selected Detention sample under the F<=120 cap;
* interval half-widths per model and a frozen pre-run repeat-selection rule.

Everything here is offline arithmetic over declared planning assumptions.  No model is
called, no key is read, no network is touched and no holdout content is used.  The output is
a design study, never a result or a power certificate.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from .masking import FOUR_SLOT_PERMUTATIONS, PEER_SLOTS

LABEL_TEST = "within_backend_label_effect"
DETENTION_TESTS = ("detention_A_minus_B", "detention_A_minus_C", "detention_A_minus_D")
PRIMARY_FAMILY = (LABEL_TEST,) + DETENTION_TESTS
HOLM_ALPHA = 0.05

#: Own-publisher menu slots for both orchestrator menus (self and own-publisher sibling).
#: ``FOUR_SLOT_PERMUTATIONS`` holds slot *names*; these are the two own-publisher labels.
OWN_PUBLISHER_SLOTS = (PEER_SLOTS[0], PEER_SLOTS[1])
#: Fixed backend utilities used by the label simulation.  The within-backend contrast
#: cancels them by construction; they only set a realistic choice distribution.
BACKEND_UTILITIES = np.array([0.65, 0.45, 0.25, 0.05])
NO_DELEGATION_UTILITY = 0.30

DEFAULT_VARIANCE_BOUNDARY = ("DEGENERATE_ZERO_VARIANCE_P_VALUE_ONE",)

#: JOB46: the owner raised the Detention cap.  A repeated retry of the same failed state is
#: not an independent observation, so the default within-failure ICC is the same declared
#: ICC (0.20).  The cap study reports repeats at 0.0/0.20 so the assumption is visible.
DEFAULT_WITHIN_FAILURE_ICC = 0.20
DETENTION_CAP_STUDY_SEED = 20261116


# Student-t distribution (stdlib only; no scipy dependency)
def _betacf(a: float, b: float, x: float) -> float:
    max_iter, eps, fpmin = 300, 3e-14, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    bt = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def two_sided_t_p(t: float, df: float) -> float:
    """Two-sided p value of a Student-t statistic with ``df`` degrees of freedom."""
    if df <= 0:
        return 1.0
    if not math.isfinite(t):
        return 0.0 if math.isinf(t) else 1.0
    t = abs(float(t))
    if t == 0.0:
        return 1.0
    return min(1.0, _betai(df / 2.0, 0.5, df / (df + t * t)))


def t_critical(df: float, alpha: float = HOLM_ALPHA) -> float:
    """Two-sided critical value ``t`` with ``P(|T| > t) = alpha`` by bisection."""
    if df <= 0:
        return float("inf")
    lo, hi = 0.0, 1.0
    while two_sided_t_p(hi, df) > alpha:
        hi *= 2.0
        if hi > 1e6:
            return hi
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if two_sided_t_p(mid, df) > alpha:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-10:
            break
    return (lo + hi) / 2.0


# Holm step-down
def holm(p_values, *, alpha: float = HOLM_ALPHA) -> dict:
    """Holm step-down adjusted p values and rejections over one predeclared family."""
    values = [float(p) for p in p_values]
    k = len(values)
    order = sorted(range(k), key=lambda index: values[index])
    adjusted = [1.0] * k
    running = 0.0
    for rank, index in enumerate(order):
        factor = k - rank
        candidate = min(1.0, factor * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    reject = [adjusted[index] <= alpha for index in range(k)]
    return {
        "alpha": alpha,
        "p_values": values,
        "adjusted_p_values": adjusted,
        "reject": reject,
        "any_reject": any(reject),
    }


# Label exposures read from the actual assignment rows
def _own_matrix(permutation_index: int) -> np.ndarray:
    """Own/other displayed publisher on each fixed backend for a SWAPPED row."""
    order = FOUR_SLOT_PERMUTATIONS[int(permutation_index) % len(FOUR_SLOT_PERMUTATIONS)]
    own = np.zeros(len(PEER_SLOTS), dtype=bool)
    for position, label_source in enumerate(order):
        own[position] = label_source in OWN_PUBLISHER_SLOTS
    return own


def label_exposures(rows, *, lineage_by_instance: dict[str, str] | None = None) -> dict:
    """Build tensor inputs for the label simulation from real assignment rows.

    Only randomized ``SWAPPED`` OPTIONAL rows enter the headline (JOB36 semantics).  A row
    contributes one observation per fixed backend slot: whether the slot carried an
    own-publisher label and whether the orchestrator selected that slot.
    """
    lineage_by_instance = lineage_by_instance or {}
    swapped = [
        row for row in rows
        if row.get("condition") == "OPTIONAL" and row.get("identity_condition") == "SWAPPED"
    ]
    if not swapped:
        return {
            "rows": 0,
            "models": [],
            "lineages": [],
            "mechanisms": [],
            "own": np.zeros((0, len(PEER_SLOTS)), dtype=bool),
            "model_codes": np.zeros(0, dtype=int),
            "lineage_codes": np.zeros(0, dtype=int),
            "mechanism_codes": np.zeros(0, dtype=int),
        }
    models = sorted({str(row.get("model_slot") or "") for row in swapped})
    model_index = {model: index for index, model in enumerate(models)}
    mechanisms = sorted({
        str(row.get("mechanism_id") or "") for row in swapped
    })
    mechanism_index = {mechanism: index for index, mechanism in enumerate(mechanisms)}
    lineages = sorted({
        str(lineage_by_instance.get(str(row.get("instance_id")), row.get("mechanism_id") or ""))
        for row in swapped
    })
    lineage_index = {lineage: index for index, lineage in enumerate(lineages)}
    own = np.vstack([_own_matrix(int(row.get("label_permutation") or 0)) for row in swapped])
    model_codes = np.array(
        [model_index[str(row.get("model_slot") or "")] for row in swapped], dtype=int
    )
    lineage_key = [
        str(lineage_by_instance.get(str(row.get("instance_id")), row.get("mechanism_id") or ""))
        for row in swapped
    ]
    lineage_codes = np.array([lineage_index[key] for key in lineage_key], dtype=int)
    mechanism_codes = np.array(
        [mechanism_index[str(row.get("mechanism_id") or "")] for row in swapped],
        dtype=int,
    )
    return {
        "rows": len(swapped),
        "models": models,
        "lineages": lineages,
        "mechanisms": mechanisms,
        "own": own,
        "model_codes": model_codes,
        "lineage_codes": lineage_codes,
        "mechanism_codes": mechanism_codes,
    }


def exposure_summary(exposures: dict) -> dict:
    """Recorded own/other exposure counts per backend (the design's fixed weights)."""
    own = exposures["own"]
    if own.size == 0:
        return {"rows": 0, "per_backend": {}}
    per_backend = {}
    for position, slot in enumerate(PEER_SLOTS):
        own_n = int(own[:, position].sum())
        per_backend[slot] = {
            "own_exposure": own_n,
            "other_exposure": int(len(own) - own_n),
        }
    return {
        "rows": int(len(own)),
        "backend_slots": list(PEER_SLOTS),
        "per_backend": per_backend,
        "exposure_mode": "randomized SWAPPED publisher label on fixed backends",
    }


# Estimators and tests
def _cluster_estimator(values: np.ndarray, codes: np.ndarray, n_clusters: int) -> dict:
    """Equal-weight mean over clusters, cluster-robust SE, two-sided t p value."""
    sums = np.zeros(n_clusters)
    counts = np.zeros(n_clusters)
    np.add.at(sums, codes, values)
    np.add.at(counts, codes, 1.0)
    present = counts > 0
    cluster_means = sums[present] / counts[present]
    n = int(cluster_means.size)
    if n < 2:
        return {
            "estimate": float(cluster_means.mean()) if n else float("nan"),
            "se": float("nan"),
            "df": 0,
            "t": float("nan"),
            "p_value": 1.0,
            "clusters": n,
            "boundary": "FEWER_THAN_TWO_CLUSTERS",
        }
    estimate = float(cluster_means.mean())
    variance = float(cluster_means.var(ddof=1) / n)
    if variance <= 0.0:
        return {
            "estimate": estimate,
            "se": 0.0,
            "df": n - 1,
            "t": float("inf") if estimate != 0 else 0.0,
            # JOB56/Astra finding 7: the predeclared degenerate boundary is p=1 for every
            # estimate, not a data-dependent p=0.  The flag, not the p value, carries the
            # fact that the variance is degenerate.
            "p_value": 1.0,
            "clusters": n,
            "boundary": DEFAULT_VARIANCE_BOUNDARY[0],
        }
    se = math.sqrt(variance)
    t = estimate / se
    return {
        "estimate": estimate,
        "se": se,
        "df": n - 1,
        "t": t,
        "p_value": two_sided_t_p(t, n - 1),
        "clusters": n,
        "boundary": "",
    }


def calibrate_utility_effect(own: np.ndarray, target_difference: float) -> float:
    """Solve the utility shift whose expected within-backend difference is the target.

    ``target_difference`` is on the probability scale (the design's declared effect); the
    Gumbel/logit utility coefficient that realises it is smaller because the other fixed
    backends compete for the same choice.  Calibration uses the actual own/other exposure
    matrix, so it is schedule-specific and deterministic.
    """
    def expected_difference(delta: float) -> float:
        utilities = BACKEND_UTILITIES[None, :] + delta * own.astype(float)
        all_utilities = np.concatenate(
            [utilities, np.full((own.shape[0], 1), NO_DELEGATION_UTILITY)], axis=1
        )
        all_utilities = all_utilities - all_utilities.max(axis=1, keepdims=True)
        probabilities = np.exp(all_utilities)
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
        selection = probabilities[:, : len(PEER_SLOTS)]
        diffs = []
        for position in range(len(PEER_SLOTS)):
            is_own = own[:, position]
            if is_own.all() or (~is_own).all():
                continue
            diffs.append(
                float(selection[is_own, position].mean()
                      - selection[~is_own, position].mean())
            )
        return float(np.mean(diffs)) if diffs else 0.0

    if target_difference <= 0:
        return 0.0
    low, high = 0.0, 1.0
    while expected_difference(high) < target_difference and high < 100.0:
        high *= 2.0
    for _ in range(120):
        mid = (low + high) / 2.0
        if expected_difference(mid) < target_difference:
            low = mid
        else:
            high = mid
        if high - low < 1e-10:
            break
    return (low + high) / 2.0


def _expected_label_difference(own: np.ndarray, delta: float) -> float:
    """Deterministic expected within-backend difference at a utility shift."""
    utilities = BACKEND_UTILITIES[None, :] + delta * own.astype(float)
    all_utilities = np.concatenate(
        [utilities, np.full((own.shape[0], 1), NO_DELEGATION_UTILITY)], axis=1
    )
    all_utilities = all_utilities - all_utilities.max(axis=1, keepdims=True)
    probabilities = np.exp(all_utilities)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    selection = probabilities[:, : len(PEER_SLOTS)]
    diffs = []
    for position in range(len(PEER_SLOTS)):
        is_own = own[:, position]
        if is_own.all() or (~is_own).all():
            continue
        diffs.append(
            float(selection[is_own, position].mean()
                  - selection[~is_own, position].mean())
        )
    return float(np.mean(diffs)) if diffs else 0.0


def _label_replicate(
    rng,
    exposures: dict,
    *,
    effect: float,
    icc: float,
    utility_effect: float | None = None,
) -> dict:
    """One simulated label replicate; returns per-model and combined estimates.

    ``effect`` is the declared probability-scale effect; ``utility_effect`` is the
    calibrated logit shift that realises it under the real exposure schedule.
    """
    own = exposures["own"]
    n_rows = own.shape[0]
    n_models = max(1, len(exposures["models"]))
    lineages = exposures["lineages"]
    n_clusters = max(1, len(lineages))
    model_codes = exposures["model_codes"]
    cluster_codes = exposures["lineage_codes"]
    tau = math.sqrt(icc / (1.0 - icc)) * math.pi / math.sqrt(3.0) if icc > 0 else 0.0
    cluster_effect = rng.normal(0.0, tau, n_clusters) if tau else np.zeros(n_clusters)
    shared = cluster_effect[cluster_codes]
    delta = (
        float(utility_effect) if utility_effect is not None
        else calibrate_utility_effect(own, float(effect))
    )
    utilities = (
        shared[:, None]
        + BACKEND_UTILITIES[None, :]
        + delta * own.astype(float)
        + rng.gumbel(size=(n_rows, len(PEER_SLOTS)))
    )
    nodeleg_utility = (
        NO_DELEGATION_UTILITY + 0.35 * shared + rng.gumbel(size=n_rows)
    )
    choices = np.argmax(
        np.concatenate([utilities, nodeleg_utility[:, None]], axis=1), axis=1
    )
    selected = (choices[:, None] == np.arange(len(PEER_SLOTS))[None, :]).astype(float)

    # Accumulate per (model, lineage, slot): selected under own and other exposure.
    group = (cluster_codes[:, None] * n_models + model_codes[:, None])
    slot_index = np.arange(len(PEER_SLOTS))[None, :]
    flat_group = (group * len(PEER_SLOTS) + slot_index).ravel()
    size = n_clusters * n_models * len(PEER_SLOTS)
    y_own = np.zeros(size)
    y_other = np.zeros(size)
    n_own = np.zeros(size)
    n_other = np.zeros(size)
    np.add.at(y_own, flat_group, (selected * own).ravel())
    np.add.at(y_other, flat_group, (selected * (~own)).ravel())
    np.add.at(n_own, flat_group, own.ravel().astype(float))
    np.add.at(n_other, flat_group, (~own).ravel().astype(float))
    y_own = y_own.reshape(n_clusters, n_models, len(PEER_SLOTS))
    y_other = y_other.reshape(n_clusters, n_models, len(PEER_SLOTS))
    n_own = n_own.reshape(n_clusters, n_models, len(PEER_SLOTS))
    n_other = n_other.reshape(n_clusters, n_models, len(PEER_SLOTS))

    with np.errstate(divide="ignore", invalid="ignore"):
        diff = y_own / n_own - y_other / n_other
    valid = (n_own > 0) & (n_other > 0)
    diff = np.where(valid, diff, np.nan)

    per_model = []
    model_estimates = []
    model_variances = []
    model_dfs = []
    for model in range(n_models):
        cells = diff[:, model, :]
        weights = valid[:, model, :]
        # Equal weight over backends within a lineage, then equal weight over lineages.
        with np.errstate(invalid="ignore"):
            per_lineage = np.where(
                weights.any(axis=1),
                np.nansum(np.where(weights, cells, 0.0), axis=1)
                / np.maximum(weights.sum(axis=1), 1),
                np.nan,
            )
        present = ~np.isnan(per_lineage)
        values = per_lineage[present]
        codes = np.flatnonzero(present)
        result = _cluster_estimator(values, codes, n_clusters)
        per_model.append(
            {
                "model_slot": exposures["models"][model],
                "estimate": result["estimate"],
                "se": result["se"],
                "df": result["df"],
                "p_value": result["p_value"],
                "clusters": result["clusters"],
                "boundary": result["boundary"],
            }
        )
        if math.isfinite(result["estimate"]) and math.isfinite(result["se"]):
            model_estimates.append(result["estimate"])
            model_variances.append(result["se"] ** 2)
            model_dfs.append(result["df"])
    if not model_estimates:
        return {"estimate": float("nan"), "se": float("nan"), "p_value": 1.0,
                "per_model": per_model, "df": 0, "boundary": "NO_MODEL_ESTIMATE"}
    estimate = float(np.mean(model_estimates))
    # Fixed equal-model aggregate: independent model estimates, equal weights.
    variance = float(np.mean(np.array(model_variances) / len(model_estimates)))
    se = math.sqrt(variance) if variance > 0 else 0.0
    df = min(model_dfs) if model_dfs else 0
    if se == 0.0:
        # JOB56/Astra finding 7: same rule in the aggregate/per-model path.
        p_value = 1.0
        boundary = DEFAULT_VARIANCE_BOUNDARY[0]
    else:
        p_value = two_sided_t_p(estimate / se, df) if df > 0 else 1.0
        boundary = ""
    return {
        "estimate": estimate,
        "se": se,
        "df": df,
        "p_value": p_value,
        "per_model": per_model,
        "boundary": boundary,
    }


def _detention_replicate(
    rng,
    *,
    n_selected: int,
    a_rate: float,
    effects: dict,
    icc: float,
    n_clusters: int,
    repeats_per_failure: int = 1,
    within_failure_icc: float | None = None,
    failure_clusters=None,
) -> dict:
    """One paired four-arm Detention replicate on the common endpoint.

    ``repeats_per_failure`` is the number of identical retry opportunities per selected
    failed state and arm.  A repeated retry shares the failed state, so it also shares a
    failure-level intercept with the declared ``within_failure_icc`` (default: the same
    ICC).  With the default ``repeats_per_failure=1`` this is the JOB38 paired design and
    the random stream is unchanged.

    ``failure_clusters`` optionally supplies the construction-lineage codes of the
    selected failures (used by the cap study so the same failures are reused across caps).
    """
    n_selected = int(n_selected)
    repeats = max(1, int(repeats_per_failure))
    if n_selected < 2:
        return {
            "n_selected": n_selected,
            "contrasts": {},
            "boundary": "FEWER_THAN_TWO_PAIRED_FAILURES",
        }
    n_clusters = max(n_clusters, 1)
    tau = math.sqrt(icc / (1.0 - icc)) * math.pi / math.sqrt(3.0) if icc > 0 else 0.0
    cluster_effect = rng.normal(0.0, tau, n_clusters)
    # Sample the construction lineage of each selected failure from the frozen population.
    if failure_clusters is None:
        failure_clusters = rng.integers(0, n_clusters, n_selected)
    else:
        failure_clusters = np.asarray(failure_clusters)[:n_selected]
    shared = cluster_effect[failure_clusters]
    failure_effect = np.zeros(n_selected)
    if repeats > 1:
        failure_icc = (
            DEFAULT_WITHIN_FAILURE_ICC
            if within_failure_icc is None else float(within_failure_icc)
        )
        tau_failure = (
            math.sqrt(failure_icc / (1.0 - failure_icc)) * math.pi / math.sqrt(3.0)
            if failure_icc > 0 else 0.0
        )
        if tau_failure > 0:
            failure_effect = rng.normal(0.0, tau_failure, n_selected)
    rates = {
        "A": a_rate,
        "B": max(1e-6, min(1 - 1e-6, a_rate - effects["A_minus_B"])),
        "C": max(1e-6, min(1 - 1e-6, a_rate - effects["A_minus_C"])),
        "D": max(1e-6, min(1 - 1e-6, a_rate - effects["A_minus_D"])),
    }
    outcomes = {}
    for arm, rate in rates.items():
        logit = math.log(rate / (1.0 - rate))
        probability = 1.0 / (1.0 + np.exp(-(logit + shared + failure_effect)))
        if repeats == 1:
            outcomes[arm] = (rng.random(n_selected) < probability).astype(float)
        else:
            draws = rng.random((n_selected, repeats)) < probability[:, None]
            outcomes[arm] = draws.mean(axis=1)
    contrasts = {}
    for arm, name in (("B", "A_minus_B"), ("C", "A_minus_C"), ("D", "A_minus_D")):
        values = outcomes["A"] - outcomes[arm]
        result = _cluster_estimator(values, failure_clusters, n_clusters)
        contrasts[name] = result
    return {
        "n_selected": n_selected,
        "repeats_per_failure": repeats,
        "contrasts": contrasts,
        "boundary": "",
    }


def _quantile(values, q):
    if not len(values):
        return float("nan")
    return float(np.quantile(np.asarray(values, dtype=float), q))


def _rate(rejects) -> float:
    return float(np.mean(np.asarray(rejects, dtype=float))) if len(rejects) else float("nan")


# Failure yield
def simulate_failure_yield(
    *,
    episodes_by_repeats: dict[int, int],
    failure_rates=(0.10, 0.20, 0.30),
    reachable_fraction: float = 0.85,
    icc: float = 0.10,
    n_clusters: int = 47,
    cap: int = 120,
    warning_below: int = 60,
    repeats_per_failure: int = 1,
    sims: int = 200,
    seed: int = 20261114,
) -> dict:
    """P(insufficient eligible failures) under the frozen Detention cap.

    The cap is an upper bound on selected failed states.  If the initial grid yields fewer
    than ``cap`` eligible failures the frozen rule selects every eligible failure and the
    shortfall is reported; nothing is extended after main outcomes.  ``expected_shortfall``
    is the mean of ``max(0, cap - eligible)`` and ``probability_shortfall`` is its
    probability, so a cap that the grid cannot fill is visible rather than assumed away.
    """
    rng = np.random.default_rng(seed)
    tau = math.sqrt(icc / (1.0 - icc)) * math.pi / math.sqrt(3.0) if icc > 0 else 0.0
    rows = []
    for repeats in sorted(episodes_by_repeats):
        episodes = int(episodes_by_repeats[repeats])
        for failure_rate in failure_rates:
            eligible_counts = []
            selected_counts = []
            shortfall_counts = []
            for _ in range(int(sims)):
                cluster_effect = rng.normal(0.0, tau, max(n_clusters, 1))
                cluster_of_episode = rng.integers(0, max(n_clusters, 1), episodes)
                logit = math.log(failure_rate / (1.0 - failure_rate))
                probability = 1.0 / (1.0 + np.exp(-(logit + cluster_effect[cluster_of_episode])))
                failures = int(np.sum(rng.random(episodes) < probability))
                eligible = int(round(failures * reachable_fraction))
                eligible_counts.append(eligible)
                selected_counts.append(min(cap, eligible))
                shortfall_counts.append(max(0, cap - eligible))
            eligible_counts = np.asarray(eligible_counts)
            rows.append({
                "repeats": repeats,
                "episodes": episodes,
                "failure_rate": failure_rate,
                "reachable_fraction": reachable_fraction,
                "icc": icc,
                "expected_failures": float(episodes * failure_rate),
                "expected_eligible": float(eligible_counts.mean()),
                "expected_selected": float(np.mean(selected_counts)),
                "expected_shortfall": float(np.mean(shortfall_counts)),
                "probability_shortfall": float(np.mean(eligible_counts < cap)),
                "probability_eligible_zero": float(np.mean(eligible_counts == 0)),
                "probability_eligible_below_warning": float(
                    np.mean(eligible_counts < warning_below)
                ),
                "probability_eligible_at_least_warning": float(
                    np.mean(eligible_counts >= warning_below)
                ),
                "probability_eligible_at_cap": float(np.mean(eligible_counts >= cap)),
                "median_eligible": int(np.median(eligible_counts)),
                "p10_eligible": int(np.quantile(eligible_counts, 0.10)),
                "cap": cap,
                "repeats_per_failure": int(repeats_per_failure),
                "warning_below": warning_below,
            })
    return {
        "kind": "vera.v13_failure_yield_simulation.v2",
        "sims": sims,
        "seed": seed,
        "cluster_unit": "construction lineage (mechanism/template)",
        "shortage_rule": (
            "select min(cap, eligible) with the seeded stratified rule; when eligible < cap "
            "the study proceeds on every eligible failure and reports the achieved n, the "
            "shortfall, the realized precision and the realized power.  The cap is never "
            "raised and acquisition is never extended after main outcomes."
        ),
        "repeats_per_failure": int(repeats_per_failure),
        "rows": rows,
    }


# Detention cap study (JOB46 owner decision: raise F to adequate power)
def wilson_lower_bound(successes: int, trials: int, z: float = 1.96) -> float:
    """95% Wilson score lower bound for a simulated rejection rate."""
    if trials <= 0:
        return float("nan")
    p = float(successes) / float(trials)
    denom = 1.0 + z * z / trials
    centre = p + z * z / (2.0 * trials)
    margin = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials))
    return max(0.0, (centre - margin) / denom)


def detention_power_curve(
    *,
    caps,
    scenario: dict | None = None,
    iccs=(0.20,),
    n_clusters: int = 28,
    repeats_per_failure: int = 1,
    within_failure_icc: float | None = None,
    sims: int = 2000,
    seed: int = DETENTION_CAP_STUDY_SEED,
    alpha: float = HOLM_ALPHA,
    exposures: dict | None = None,
    label_effect: float = 0.15,
    label_utility_effect: float | None = None,
    reuse_failures_across_caps: bool = True,
) -> dict:
    """Per-test power of the predeclared four-test Holm family as a function of F.

    The Detention sample is ``F`` selected failed states with one paired A/B/C/D retry each
    (``repeats_per_failure`` adds identical retry opportunities and a shared failure-level
    intercept).  The label test is included in the Holm family exactly as in
    ``simulate_primary_family``: its p values are drawn once and reused for every cap, so a
    cap's power is not inflated by dropping the label test from the family.

    The same selected failures are reused across caps when ``reuse_failures_across_caps``
    is true (common random numbers), which makes the F curve monotone in expectation rather
    than a sequence of unrelated draws.
    """
    scenario = dict(scenario or {
        "scenario": "central",
        "A_rate": 0.50,
        "A_minus_B": 0.15,
        "A_minus_C": 0.10,
        "A_minus_D": 0.05,
    })
    caps = tuple(sorted({int(cap) for cap in caps}))
    iccs = tuple(float(icc) for icc in iccs)
    req = int(repeats_per_failure)
    within = (
        DEFAULT_WITHIN_FAILURE_ICC
        if within_failure_icc is None else float(within_failure_icc)
    )

    label_p_values_by_icc: dict[float, list[float]] = {}
    label_utility = None
    if exposures is not None:
        label_utility = (
            float(label_utility_effect)
            if label_utility_effect is not None
            else calibrate_utility_effect(exposures["own"], float(label_effect))
        )
    for icc in iccs:
        if exposures is None:
            label_p_values_by_icc[icc] = [1.0] * int(sims)
            continue
        label_rng = np.random.default_rng(
            np.random.SeedSequence([int(seed), 11, int(round(icc * 1000))])
        )
        label_p_values_by_icc[icc] = [
            _label_replicate(
                label_rng, exposures, effect=float(label_effect), icc=icc,
                utility_effect=label_utility,
            )["p_value"]
            for _ in range(int(sims))
        ]

    max_cap = max(caps) if caps else 0
    shared_failures: dict[float, np.ndarray] = {}
    for icc in iccs:
        rng_failures = np.random.default_rng(
            np.random.SeedSequence([int(seed), 23, int(round(icc * 1000))])
        )
        shared_failures[icc] = rng_failures.integers(0, max(n_clusters, 1), max_cap)

    rows = []
    for icc in iccs:
        label_p_values = label_p_values_by_icc[icc]
        for cap in caps:
            rng = np.random.default_rng(
                np.random.SeedSequence(
                    [int(seed), 37, int(round(icc * 1000)), cap, req, int(round(within * 1000))]
                )
            )
            failure_clusters = (
                shared_failures[icc][:cap] if reuse_failures_across_caps else None
            )
            contrast_p_values = defaultdict(list)
            contrast_results = defaultdict(list)
            boundary = 0
            for index in range(int(sims)):
                replicate = _detention_replicate(
                    rng,
                    n_selected=cap,
                    a_rate=float(scenario["A_rate"]),
                    effects={
                        "A_minus_B": float(scenario["A_minus_B"]),
                        "A_minus_C": float(scenario["A_minus_C"]),
                        "A_minus_D": float(scenario["A_minus_D"]),
                    },
                    icc=icc,
                    n_clusters=n_clusters,
                    repeats_per_failure=req,
                    within_failure_icc=within,
                    failure_clusters=failure_clusters,
                )
                if replicate["boundary"]:
                    boundary += 1
                    for name in ("A_minus_B", "A_minus_C", "A_minus_D"):
                        contrast_p_values[name].append(1.0)
                        contrast_results[name].append({
                            "estimate": float("nan"), "se": float("nan"),
                            "df": 0, "p_value": 1.0,
                        })
                    continue
                for name, result in replicate["contrasts"].items():
                    contrast_p_values[name].append(result["p_value"])
                    contrast_results[name].append(result)
            family_rejections = defaultdict(int)
            any_rejection = 0
            for index in range(int(sims)):
                result = holm(
                    [
                        label_p_values[index],
                        _at(contrast_p_values["A_minus_B"], index),
                        _at(contrast_p_values["A_minus_C"], index),
                        _at(contrast_p_values["A_minus_D"], index),
                    ],
                    alpha=alpha,
                )
                for test_index, test in enumerate(PRIMARY_FAMILY):
                    family_rejections[test] += int(result["reject"][test_index])
                any_rejection += int(result["any_reject"])
            row = {
                "cap": int(cap),
                "icc": icc,
                "scenario": scenario["scenario"],
                "repeats_per_failure": req,
                "within_failure_icc": within,
                "n_selected": int(cap),
                "A_rate": float(scenario["A_rate"]),
                "B_rate": round(float(scenario["A_rate"]) - float(scenario["A_minus_B"]), 6),
                "C_rate": round(float(scenario["A_rate"]) - float(scenario["A_minus_C"]), 6),
                "D_rate": round(float(scenario["A_rate"]) - float(scenario["A_minus_D"]), 6),
                "true_difference_A_minus_B": float(scenario["A_minus_B"]),
                "true_difference_A_minus_C": float(scenario["A_minus_C"]),
                "true_difference_A_minus_D": float(scenario["A_minus_D"]),
                "clusters": int(n_clusters),
                "boundary_replicates": int(boundary),
                "multiplicity": f"Holm two-sided {alpha:.2f} over four aggregate tests",
                "familywise_any_rejection_power": any_rejection / float(sims),
                "per_test_power": {
                    test: family_rejections[test] / float(sims) for test in PRIMARY_FAMILY
                },
                "unadjusted_power": {
                    f"detention_{name}": _rate(
                        [p <= alpha for p in contrast_p_values[name]]
                    )
                    for name in ("A_minus_B", "A_minus_C", "A_minus_D")
                },
                "wilson_lower_95": {
                    test: wilson_lower_bound(family_rejections[test], int(sims))
                    for test in PRIMARY_FAMILY
                },
                "expected_half_width": {
                    f"detention_{name}": _mean([
                        t_critical(item["df"], alpha) * item["se"]
                        for item in contrast_results[name]
                        if item["df"] > 0 and math.isfinite(item["se"])
                    ])
                    for name in ("A_minus_B", "A_minus_C", "A_minus_D")
                },
                "median_p_value": {
                    f"detention_{name}": _quantile(contrast_p_values[name], 0.5)
                    for name in ("A_minus_B", "A_minus_C", "A_minus_D")
                },
            }
            rows.append(row)
    return {
        "kind": "vera.v13_detention_cap_study.v1",
        "sims": int(sims),
        "seed": int(seed),
        "alpha": float(alpha),
        "scenario": scenario,
        "caps": list(caps),
        "iccs": list(iccs),
        "clusters": int(n_clusters),
        "repeats_per_failure": req,
        "within_failure_icc": within,
        "label_family_included": exposures is not None,
        "label_seed": int(seed) if exposures is not None else None,
        "reuse_failures_across_caps": bool(reuse_failures_across_caps),
        "hypothesis_test": (
            "cluster-robust two-sided Student-t on equal-weight lineage-cluster "
            "contributions; Holm step-down over label + A-B + A-C + A-D"
        ),
        "notes": [
            "caps are selected failed states, not retry calls",
            "a repeated retry of the same failed state shares a failure-level intercept "
            "with the declared within-failure ICC",
            "A-D (0.05) is reported, never hidden; it is not a cap-selection criterion",
        ],
        "rows": rows,
    }


def _at(values, index):
    """Index a contrast list (every replicate contributes one entry)."""
    return values[index]


def detention_subgroup_power(
    *,
    n_selected: int,
    scenario: dict | None = None,
    icc: float = 0.20,
    n_clusters: int = 28,
    repeats_per_failure: int = 1,
    within_failure_icc: float | None = None,
    sims: int = 1200,
    seed: int = 20261149,
    alpha: float = HOLM_ALPHA,
    contrasts=("A_minus_B", "A_minus_C"),
) -> dict:
    """Power for a predeclared Detention *subgroup* family at one sample size.

    JOB49's chain-depth moderator asks for the paired A-B and A-C contrasts **within** a
    level, not the confirmatory four-test family.  This function simulates exactly that
    secondary family: the selected contrasts (default A-B and A-C) receive Holm correction
    within the level at ``alpha``.  It is the same paired four-arm generative model and the
    same cluster-robust Student-t estimator as ``detention_power_curve``, so the numbers are
    comparable, but the label test and A-D are deliberately absent.  The output is a design
    study, never a result or a power certificate.
    """
    scenario = dict(scenario or {
        "scenario": "central",
        "A_rate": 0.50,
        "A_minus_B": 0.15,
        "A_minus_C": 0.10,
        "A_minus_D": 0.05,
    })
    names = tuple(str(name) for name in contrasts)
    if not names:
        raise ValueError("at least one contrast is required")
    unknown = [name for name in names if name not in {"A_minus_B", "A_minus_C", "A_minus_D"}]
    if unknown:
        raise ValueError(f"unknown Detention contrast(s): {unknown}")
    req = max(1, int(repeats_per_failure))
    within = (
        DEFAULT_WITHIN_FAILURE_ICC
        if within_failure_icc is None else float(within_failure_icc)
    )
    n_selected = int(n_selected)
    n_clusters = max(1, int(n_clusters))
    if n_selected < 2:
        return {
            "kind": "vera.v13_detention_subgroup_power.v1",
            "n_selected": n_selected,
            "clusters": n_clusters,
            "contrasts": list(names),
            "sims": int(sims),
            "seed": int(seed),
            "alpha": float(alpha),
            "scenario": scenario,
            "repeats_per_failure": req,
            "within_failure_icc": within,
            "boundary_replicates": int(sims),
            "boundary": "FEWER_THAN_TWO_PAIRED_FAILURES",
            "per_test_power": {f"detention_{name}": 0.0 for name in names},
            "unadjusted_power": {f"detention_{name}": 0.0 for name in names},
            "wilson_lower_95": {f"detention_{name}": 0.0 for name in names},
            "expected_half_width": {f"detention_{name}": float("nan") for name in names},
            "median_p_value": {f"detention_{name}": 1.0 for name in names},
            "familywise_any_rejection_power": 0.0,
        }
    rng = np.random.default_rng(
        np.random.SeedSequence(
            [int(seed), 49, n_selected, int(round(icc * 1000)), n_clusters, req,
             int(round(within * 1000))]
        )
    )
    p_values: dict[str, list[float]] = defaultdict(list)
    results: dict[str, list[dict]] = defaultdict(list)
    family_rejections = defaultdict(int)
    any_rejection = 0
    boundary = 0
    for _ in range(int(sims)):
        replicate = _detention_replicate(
            rng,
            n_selected=n_selected,
            a_rate=float(scenario["A_rate"]),
            effects={
                "A_minus_B": float(scenario["A_minus_B"]),
                "A_minus_C": float(scenario["A_minus_C"]),
                "A_minus_D": float(scenario["A_minus_D"]),
            },
            icc=icc,
            n_clusters=n_clusters,
            repeats_per_failure=req,
            within_failure_icc=within,
        )
        if replicate["boundary"]:
            boundary += 1
            for name in names:
                p_values[name].append(1.0)
                results[name].append(
                    {"estimate": float("nan"), "se": float("nan"), "df": 0,
                     "p_value": 1.0}
                )
            continue
        for name in names:
            p_values[name].append(float(replicate["contrasts"][name]["p_value"]))
            results[name].append(replicate["contrasts"][name])
    for index in range(int(sims)):
        adjusted = holm([p_values[name][index] for name in names], alpha=alpha)
        for position, name in enumerate(names):
            family_rejections[name] += int(adjusted["reject"][position])
        any_rejection += int(adjusted["any_reject"])
    return {
        "kind": "vera.v13_detention_subgroup_power.v1",
        "n_selected": n_selected,
        "clusters": n_clusters,
        "contrasts": list(names),
        "sims": int(sims),
        "seed": int(seed),
        "alpha": float(alpha),
        "scenario": scenario,
        "repeats_per_failure": req,
        "within_failure_icc": within,
        "boundary_replicates": int(boundary),
        "boundary": "",
        "multiplicity": (
            f"Holm two-sided {alpha:.2f} over {len(names)} within-level Detention "
            "contrasts"
        ),
        "familywise_any_rejection_power": any_rejection / float(sims),
        "per_test_power": {
            f"detention_{name}": family_rejections[name] / float(sims)
            for name in names
        },
        "unadjusted_power": {
            f"detention_{name}": _rate([p <= alpha for p in p_values[name]])
            for name in names
        },
        "wilson_lower_95": {
            f"detention_{name}": wilson_lower_bound(family_rejections[name], int(sims))
            for name in names
        },
        "expected_half_width": {
            f"detention_{name}": _mean([
                t_critical(item["df"], alpha) * item["se"]
                for item in results[name]
                if item["df"] > 0 and math.isfinite(item["se"])
            ])
            for name in names
        },
        "median_p_value": {
            f"detention_{name}": _quantile(p_values[name], 0.5) for name in names
        },
        "notes": [
            "secondary subgroup family: label test and A-D are not part of these tests",
            "paired within failed state; cluster-robust two-sided Student-t; Holm within "
            "the level",
            "design study only; no model, network, key or holdout access",
        ],
    }


def select_detention_cap(
    study: dict,
    *,
    candidate_caps,
    target_power: float = 0.80,
    icc: float = 0.20,
    scenario: str = "central",
    required_tests=("detention_A_minus_B", "detention_A_minus_C"),
) -> dict:
    """Frozen pre-run rule for the Detention cap.

    Choose the smallest candidate cap whose central-scenario, ICC-0.20 Holm per-test power
    is at least the target for A-B and A-C **and** whose 95% Wilson lower bound is also at
    least the target, so a borderline Monte-Carlo point estimate cannot freeze the cap.
    A-D (0.05) is reported but is not a selection criterion: the declared 0.05 effect is
    below the design's reachable precision and the owner asked for it to be reported
    honestly.  If no candidate qualifies, no cap is chosen and the shortfall is reported.
    """
    candidates = tuple(sorted({int(cap) for cap in candidate_caps}))
    indexed = {
        (int(row["cap"]), float(row["icc"]), str(row["scenario"])): row
        for row in study.get("rows", [])
    }
    decisions = []
    for cap in candidates:
        row = indexed.get((cap, float(icc), str(scenario)))
        if row is None:
            continue
        power = {test: row["per_test_power"].get(test) for test in required_tests}
        lower = {test: row["wilson_lower_95"].get(test) for test in required_tests}
        checks = {
            test: bool(
                power.get(test) is not None and power[test] >= target_power
                and lower.get(test) is not None and lower[test] >= target_power
            )
            for test in required_tests
        }
        decisions.append({
            "cap": cap,
            "per_test_power": power,
            "wilson_lower_95": lower,
            "A_minus_D_power": row["per_test_power"].get("detention_A_minus_D"),
            "meets_required_tests": all(checks.values()),
            "failed_tests": [test for test, ok in checks.items() if not ok],
        })
    qualifying = [row for row in decisions if row["meets_required_tests"]]
    chosen = qualifying[0]["cap"] if qualifying else None
    if qualifying:
        status = "CAP_MEETS_REQUIRED_POWER"
        rationale = (
            f"F={chosen} is the smallest candidate cap whose Holm per-test power for A-B "
            f"and A-C is at least {target_power:.2f} at the central scenario and ICC {icc:.2f} "
            "with a 95% Wilson lower bound also at least the target"
        )
    else:
        status = "NO_CAP_MEETS_REQUIRED_POWER"
        rationale = (
            "no candidate cap meets the A-B and A-C power target at the central scenario; "
            "report the shortfall rather than silently raising the cap after outcomes"
        )
    return {
        "status": status,
        "chosen_cap": chosen,
        "candidate_caps": list(candidates),
        "required_tests": list(required_tests),
        "target_power": float(target_power),
        "icc": float(icc),
        "scenario": str(scenario),
        "selection_rule": (
            "smallest cap in the frozen candidate grid whose central-scenario ICC-0.20 Holm "
            "per-test power for A-B and A-C is at least the target and whose 95% Wilson "
            "lower bound is also at least the target; A-D is reported, not hidden, and is "
            "not a selection criterion because the 0.05 effect is below the reachable "
            "precision; no post-outcome change is permitted"
        ),
        "A_minus_D_declared": (
            "A-D (0.05) is expected to remain underpowered at the supply-feasible cap; its "
            "power is reported as a predeclared limitation, never as a null effect"
        ),
        "decisions": decisions,
        "rationale": rationale,
    }


# Main simulation
def simulate_primary_family(
    *,
    exposures: dict,
    repeats_grid=(1, 2, 3, 4),
    iccs=(0.05, 0.20, 0.50),
    label_effects=(0.0, 0.10, 0.15, 0.20),
    detention_scenarios=None,
    failure_rates=(0.10, 0.20, 0.30),
    episodes_by_repeats: dict[int, int] | None = None,
    failure_yield: dict | None = None,
    sims: int = 200,
    seed: int = 20261114,
    target_power: float = 0.80,
    target_half_width: float = 0.10,
    central_effect: float = 0.15,
    alpha: float = HOLM_ALPHA,
    detention_repeats_per_failure: int = 1,
    within_failure_icc: float | None = None,
) -> dict:
    """Prospective power over the actual schedule, real tests and Holm family."""
    rng = np.random.default_rng(seed)
    detention_scenarios = list(detention_scenarios or [
        {
            "scenario": "central",
            "A_rate": 0.50,
            "A_minus_B": 0.15,
            "A_minus_C": 0.10,
            "A_minus_D": 0.05,
        },
        {
            "scenario": "optimistic",
            "A_rate": 0.55,
            "A_minus_B": 0.20,
            "A_minus_C": 0.12,
            "A_minus_D": 0.08,
        },
        {
            "scenario": "pessimistic",
            "A_rate": 0.45,
            "A_minus_B": 0.10,
            "A_minus_C": 0.06,
            "A_minus_D": 0.03,
        },
    ])
    n_lineages = max(1, len(exposures.get("lineages") or []))
    calibrations = {
        float(effect): calibrate_utility_effect(
            exposures["own"], float(effect)
        ) for effect in label_effects
    }
    label_rows = []
    for repeats in repeats_grid:
        # The assignment rows scale with repeats; scale the exposure tensor by r=1 base.
        scale = max(1, int(repeats))
        for icc in iccs:
            for effect in label_effects:
                estimates, half_widths, p_values, boundaries = [], [], [], []
                per_model_half_widths = defaultdict(list)
                scaled = _scaled_exposures(exposures, scale)
                for _ in range(int(sims)):
                    replicate = _label_replicate(
                        rng, scaled,
                        effect=float(effect), icc=float(icc),
                        utility_effect=calibrations[float(effect)],
                    )
                    estimates.append(replicate["estimate"])
                    p_values.append(replicate["p_value"])
                    boundaries.append(replicate["boundary"])
                    for model_row in replicate.get("per_model", []):
                        df = model_row.get("df") or 0
                        se = model_row.get("se") or 0.0
                        if df > 0 and math.isfinite(se):
                            per_model_half_widths[model_row["model_slot"]].append(
                                t_critical(df, alpha) * se
                            )
                    df = replicate.get("df") or 0
                    se = replicate.get("se") or 0.0
                    half_widths.append(
                        t_critical(df, alpha) * se if df > 0 else float("nan")
                    )
                finite = [value for value in estimates if math.isfinite(value)]
                label_rows.append({
                    "repeats": int(repeats),
                    "icc": float(icc),
                    "label_effect": float(effect),
                    "own_rate": round(0.5 + float(effect) / 2.0, 6),
                    "other_rate": round(0.5 - float(effect) / 2.0, 6),
                    "true_difference": float(effect),
                    "utility_effect": calibrations[float(effect)],
                    "realized_estimand_at_zero_cluster_effect": _expected_label_difference(
                        exposures["own"], calibrations[float(effect)]
                    ),
                    "mean_estimate": _quantile(finite, 0.5) if finite else float("nan"),
                    "realized_mean_estimate": _mean(estimates),
                    "interval_lo": _quantile(finite, 0.025),
                    "interval_hi": _quantile(finite, 0.975),
                    "expected_half_width": _mean(half_widths),
                    "model_specific_half_width_worst": _mean(
                        [
                            max(values) if values else float("nan")
                            for values in per_model_half_widths.values()
                        ]
                    ),
                    "power_at_0_15": _rate([p <= 0.05 for p in p_values]),
                    "power_label_test": _rate([p <= 0.05 for p in p_values]),
                    "median_p_value": _quantile(p_values, 0.5),
                    "boundary_replicates": int(sum(1 for item in boundaries if item)),
                    "clusters": n_lineages,
                    "rows": int(exposures.get("rows") or 0) * scale,
                    "test": "cluster-robust two-sided t on fixed-weight within-backend effect",
                })

    # Detention power uses the selected Detention sample from the failure-yield study.
    # The cap binds for every plausible repeat count; report the cap and the warning floor.
    detention_rows = []
    selected_by_repeats: dict[int, int] = {}
    if failure_yield:
        for row in failure_yield.get("rows", []):
            if abs(row["failure_rate"] - 0.20) < 1e-9:
                selected_by_repeats[int(row["repeats"])] = int(row["expected_selected"])
    for repeats in repeats_grid:
        for icc in iccs:
            for scenario in detention_scenarios:
                n_selected = selected_by_repeats.get(int(repeats), 120)
                contrast_samples = defaultdict(list)
                n_boundary = 0
                for _ in range(int(sims)):
                    replicate = _detention_replicate(
                        rng,
                        n_selected=n_selected,
                        a_rate=float(scenario["A_rate"]),
                        effects={
                            "A_minus_B": float(scenario["A_minus_B"]),
                            "A_minus_C": float(scenario["A_minus_C"]),
                            "A_minus_D": float(scenario["A_minus_D"]),
                        },
                        icc=float(icc),
                        n_clusters=n_lineages,
                        repeats_per_failure=detention_repeats_per_failure,
                        within_failure_icc=within_failure_icc,
                    )
                    if replicate["boundary"]:
                        n_boundary += 1
                        continue
                    for name, result in replicate["contrasts"].items():
                        contrast_samples[name].append(result)
                row = {
                    "repeats": int(repeats),
                    "icc": float(icc),
                    "scenario": scenario["scenario"],
                    "n_selected": int(n_selected),
                    "A_rate": float(scenario["A_rate"]),
                    "B_rate": round(float(scenario["A_rate"]) - float(scenario["A_minus_B"]), 6),
                    "C_rate": round(float(scenario["A_rate"]) - float(scenario["A_minus_C"]), 6),
                    "D_rate": round(float(scenario["A_rate"]) - float(scenario["A_minus_D"]), 6),
                    "true_difference": float(scenario["A_minus_B"]),
                    "clusters": n_lineages,
                    "repeats_per_failure": int(detention_repeats_per_failure),
                    "boundary_replicates": n_boundary,
                }
                for name, label in (
                    ("A_minus_B", "detention_A_minus_B"),
                    ("A_minus_C", "detention_A_minus_C"),
                    ("A_minus_D", "detention_A_minus_D"),
                ):
                    samples = contrast_samples[name]
                    estimates = [item["estimate"] for item in samples]
                    row[f"{label}_power"] = _rate(
                        [item["p_value"] <= 0.05 for item in samples]
                    )
                    row[f"{label}_mean_estimate"] = _mean(estimates)
                    row[f"{label}_interval_lo"] = _quantile(estimates, 0.025)
                    row[f"{label}_interval_hi"] = _quantile(estimates, 0.975)
                    row[f"{label}_expected_half_width"] = _mean([
                        t_critical(item["df"], alpha) * item["se"]
                        for item in samples if item["df"] > 0 and math.isfinite(item["se"])
                    ])
                row["power_at_0_15"] = row.get("detention_A_minus_B_power")
                detention_rows.append(row)

    # Joint four-test family: combine a label replicate and a Detention replicate with the
    # same sims count, then apply Holm.  This is the actual predeclared family.
    family_rows = []
    central_scenarios = [
        scenario for scenario in detention_scenarios if scenario["scenario"] == "central"
    ] or detention_scenarios[:1]
    for repeats in repeats_grid:
        for icc in iccs:
            label_row = next(
                row for row in label_rows
                if row["repeats"] == int(repeats) and row["icc"] == float(icc)
                and abs(row["label_effect"] - central_effect) < 1e-9
            )
            n_selected = selected_by_repeats.get(int(repeats), 120)
            label_ps, b_ps, c_ps, d_ps = [], [], [], []
            family_exposures = _scaled_exposures(exposures, max(1, int(repeats)))
            for _ in range(int(sims)):
                label_rep = _label_replicate(
                    rng, family_exposures, effect=central_effect, icc=float(icc),
                    utility_effect=calibrations[float(central_effect)],
                )
                label_ps.append(label_rep["p_value"])
                detention_rep = _detention_replicate(
                    rng,
                    n_selected=n_selected,
                    a_rate=float(central_scenarios[0]["A_rate"]),
                    effects={
                        "A_minus_B": float(central_scenarios[0]["A_minus_B"]),
                        "A_minus_C": float(central_scenarios[0]["A_minus_C"]),
                        "A_minus_D": float(central_scenarios[0]["A_minus_D"]),
                    },
                    icc=float(icc),
                    n_clusters=n_lineages,
                    repeats_per_failure=detention_repeats_per_failure,
                    within_failure_icc=within_failure_icc,
                )
                contrasts = detention_rep.get("contrasts") or {}
                b_ps.append(contrasts.get("A_minus_B", {}).get("p_value", 1.0))
                c_ps.append(contrasts.get("A_minus_C", {}).get("p_value", 1.0))
                d_ps.append(contrasts.get("A_minus_D", {}).get("p_value", 1.0))
            family_rejections = defaultdict(int)
            any_rejections = 0
            for index in range(int(sims)):
                result = holm(
                    [label_ps[index], b_ps[index], c_ps[index], d_ps[index]], alpha=alpha
                )
                for test_index, test in enumerate(PRIMARY_FAMILY):
                    family_rejections[test] += int(result["reject"][test_index])
                any_rejections += int(result["any_reject"])
            family_rows.append({
                "repeats": int(repeats),
                "icc": float(icc),
                "label_effect": float(central_effect),
                "n_selected_detention": int(n_selected),
                "detention_repeats_per_failure": int(detention_repeats_per_failure),
                "multiplicity": f"Holm two-sided {alpha:.2f} over four aggregate tests",
                "familywise_any_rejection_power": any_rejections / float(sims),
                "per_test_power": {
                    test: family_rejections[test] / float(sims) for test in PRIMARY_FAMILY
                },
                "label_only_half_width": label_row["expected_half_width"],
                "label_model_specific_half_width_worst": label_row[
                    "model_specific_half_width_worst"
                ],
            })

    return {
        "label_rows": label_rows,
        "detention_rows": detention_rows,
        "family_rows": family_rows,
        "primary_family": list(PRIMARY_FAMILY),
        "alpha": alpha,
        "targets": {"power": target_power, "half_width": target_half_width,
                    "central_effect": central_effect},
    }


def _scaled_exposures(exposures: dict, scale: int) -> dict:
    """Repeats add independent rows from the same frozen schedule (r=1 base)."""
    if scale <= 1 or not exposures.get("rows"):
        return exposures
    own = np.vstack([exposures["own"] for _ in range(scale)])
    model_codes = np.concatenate([exposures["model_codes"] for _ in range(scale)])
    lineage_codes = np.concatenate([exposures["lineage_codes"] for _ in range(scale)])
    mechanism_codes = np.concatenate([exposures["mechanism_codes"] for _ in range(scale)])
    return {
        **exposures,
        "rows": int(exposures["rows"]) * scale,
        "own": own,
        "model_codes": model_codes,
        "lineage_codes": lineage_codes,
        "mechanism_codes": mechanism_codes,
    }


def _mean(values) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else float("nan")


__all__ = [
    "DETENTION_CAP_STUDY_SEED",
    "DETENTION_TESTS",
    "DEFAULT_WITHIN_FAILURE_ICC",
    "HOLM_ALPHA",
    "LABEL_TEST",
    "PRIMARY_FAMILY",
    "detention_power_curve",
    "exposure_summary",
    "holm",
    "label_exposures",
    "simulate_failure_yield",
    "select_detention_cap",
    "simulate_primary_family",
    "t_critical",
    "two_sided_t_p",
    "wilson_lower_bound",
]
