"""Aggregated statistics for the dpo-reasoning-transfer project.

Used primarily by Notebook 05 to turn the per-seed per-condition eval JSONs
into publishable numbers and figures. Everything here is CPU-only and
depends only on numpy + scipy + standard library.

Statistical protocol (from RESEARCH_PLAN §4):

- Primary per-seed test: exact McNemar on discordant pairs. α = 0.05.
- Pooled test: sum discordant pairs across the 3 seeds and run McNemar
  on the pooled counts. Also Cochran's Q for the joint hypothesis that
  baseline and all seed conditions have equal marginal accuracy.
- Effect size: paired-difference-in-accuracy with 95% percentile
  bootstrap CI (10,000 resamples).
- Multiple comparisons: two families (A vs baseline, B vs baseline),
  so Bonferroni-adjusted α = 0.025 on the pooled tests.
"""

from __future__ import annotations

import json
from math import comb
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# McNemar exact test
# ---------------------------------------------------------------------------

def mcnemar_pvalue(b: int, c: int) -> float:
    """Two-tailed exact McNemar p-value on discordant pairs.

    b: # problems where reference was correct and treatment was wrong.
    c: # problems where reference was wrong and treatment was correct.
    """
    n = b + c
    if n == 0:
        return 1.0
    x = min(b, c)
    tail = sum(comb(n, k) for k in range(x + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


# ---------------------------------------------------------------------------
# Paired bootstrap CI on accuracy delta
# ---------------------------------------------------------------------------

def paired_bootstrap_delta_ci(
    baseline_correct: list[int],
    treatment_correct: list[int],
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI on (treatment_acc − baseline_acc) in pp.

    Resamples problems (not seeds) with replacement, so the CI captures
    per-problem variance for a single (baseline, treatment) pair.

    Returns (point_delta_pp, ci_low_pp, ci_high_pp).
    """
    if len(baseline_correct) != len(treatment_correct):
        raise ValueError(
            f"length mismatch: {len(baseline_correct)} vs {len(treatment_correct)}"
        )
    n = len(baseline_correct)
    b = np.asarray(baseline_correct, dtype=int)
    t = np.asarray(treatment_correct, dtype=int)
    diff = t - b   # in {-1, 0, +1}
    point_pp = float(diff.mean() * 100)

    rng = np.random.default_rng(seed)
    # Vectorized: draw an (n_bootstrap, n) index matrix in chunks to save memory
    chunk = min(n_bootstrap, 2000)
    boot_deltas = np.empty(n_bootstrap, dtype=float)
    idx_offset = 0
    for start in range(0, n_bootstrap, chunk):
        end = min(start + chunk, n_bootstrap)
        this_chunk = end - start
        idx = rng.integers(0, n, size=(this_chunk, n))
        boot_deltas[start:end] = diff[idx].mean(axis=1)
    low_pp = float(np.quantile(boot_deltas, alpha / 2) * 100)
    high_pp = float(np.quantile(boot_deltas, 1 - alpha / 2) * 100)
    return point_pp, low_pp, high_pp


# ---------------------------------------------------------------------------
# Cochran's Q
# ---------------------------------------------------------------------------

def cochrans_q(matrix: np.ndarray) -> tuple[float, float]:
    """Cochran's Q test for k related binary outcomes.

    Args:
        matrix: shape (n_problems, k_conditions), 0/1 correct.

    Returns:
        (Q, p_value). Under H0 that all k conditions have equal marginal
        accuracy, Q is chi-square distributed with df = k−1.
    """
    from scipy import stats

    m = np.asarray(matrix, dtype=int)
    if m.ndim != 2 or m.shape[1] < 2:
        raise ValueError("matrix must be (n_problems, k>=2)")
    n, k = m.shape
    col_sums = m.sum(axis=0)
    row_sums = m.sum(axis=1)
    numer = (k - 1) * (k * (col_sums ** 2).sum() - (col_sums.sum()) ** 2)
    denom = k * row_sums.sum() - (row_sums ** 2).sum()
    if denom == 0:
        return 0.0, 1.0
    Q = numer / denom
    p = float(1 - stats.chi2.cdf(Q, df=k - 1))
    return float(Q), p


# ---------------------------------------------------------------------------
# Loading eval JSONs
# ---------------------------------------------------------------------------

def load_result(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def load_all_results(
    project_root: str, seeds: tuple[int, ...] = (1, 2, 3)
) -> dict:
    """Load baseline + all requested seeded runs for both conditions.

    Missing files are silently skipped (returned dict just won't contain
    them). Notebook 05 checks the shape and reports what's missing.

    Returns:
        {
            'baseline': dict,
            'A': {seed: dict, ...},
            'B': {seed: dict, ...},
        }
    """
    root = Path(project_root) / "results"
    out: dict = {"A": {}, "B": {}}

    baseline_path = root / "baseline_gsm8k_500.json"
    if baseline_path.exists():
        out["baseline"] = load_result(baseline_path)

    for cond in ("A", "B"):
        for seed in seeds:
            p = root / f"dpo_{cond}_seed{seed}.json"
            if p.exists():
                out[cond][seed] = load_result(p)
    return out


def check_completeness(all_results: dict, expected_seeds=(1, 2, 3)) -> list[str]:
    """Return a list of human-readable warnings about missing artifacts."""
    warnings = []
    if "baseline" not in all_results:
        warnings.append("MISSING: results/baseline_gsm8k_500.json (run Notebook 01)")
    for cond in ("A", "B"):
        present = set(all_results.get(cond, {}))
        missing = set(expected_seeds) - present
        for s in sorted(missing):
            nb = "03" if cond == "A" else "04"
            warnings.append(
                f"MISSING: results/dpo_{cond}_seed{s}.json "
                f"(run Notebook {nb} with SEED={s})"
            )
    return warnings


# ---------------------------------------------------------------------------
# Per-problem alignment
# ---------------------------------------------------------------------------

def per_problem_correctness(results: dict) -> np.ndarray:
    """Extract 0/1 correctness aligned by problem index, shape (n_samples,)."""
    by_ix = {r["index"]: int(r["correct"]) for r in results["results"]}
    n = results["n_samples"]
    return np.array([by_ix.get(i, 0) for i in range(n)], dtype=int)


def per_problem_response_lengths(results: dict) -> np.ndarray:
    by_ix = {r["index"]: len(r["response"]) for r in results["results"]}
    n = results["n_samples"]
    return np.array([by_ix.get(i, 0) for i in range(n)], dtype=int)


# ---------------------------------------------------------------------------
# Full condition-vs-baseline comparison
# ---------------------------------------------------------------------------

def full_comparison(
    baseline: dict,
    condition_dict: dict,
    alpha: float = 0.05,
    bootstrap_seed: int = 0,
) -> dict:
    """Compute all statistics comparing a condition (across seeds) to baseline.

    Args:
        baseline: baseline eval JSON.
        condition_dict: {seed_id: eval JSON, ...} for one condition.

    Returns dict with per-seed and pooled statistics.
    """
    b_correct = per_problem_correctness(baseline)
    n_problems = len(b_correct)

    per_seed = {}
    pool_b_right_d_wrong = 0
    pool_b_wrong_d_right = 0

    matrix_cols = [b_correct]

    for seed_id in sorted(condition_dict.keys()):
        dpo_res = condition_dict[seed_id]
        d_correct = per_problem_correctness(dpo_res)
        if len(d_correct) != n_problems:
            raise ValueError(
                f"seed {seed_id}: n_samples mismatch "
                f"({len(d_correct)} vs baseline {n_problems})"
            )
        matrix_cols.append(d_correct)

        b_right_d_wrong = int(((b_correct == 1) & (d_correct == 0)).sum())
        b_wrong_d_right = int(((b_correct == 0) & (d_correct == 1)).sum())
        both_right = int(((b_correct == 1) & (d_correct == 1)).sum())
        both_wrong = int(((b_correct == 0) & (d_correct == 0)).sum())

        p_mcn = mcnemar_pvalue(b_right_d_wrong, b_wrong_d_right)
        delta_pp, ci_low, ci_high = paired_bootstrap_delta_ci(
            b_correct.tolist(), d_correct.tolist(),
            alpha=alpha, seed=bootstrap_seed + seed_id,
        )

        per_seed[seed_id] = {
            "accuracy": dpo_res["accuracy"],
            "delta_pp": delta_pp,
            "ci_low_pp": ci_low,
            "ci_high_pp": ci_high,
            "both_right": both_right,
            "both_wrong": both_wrong,
            "b_right_d_wrong": b_right_d_wrong,
            "b_wrong_d_right": b_wrong_d_right,
            "mcnemar_p": p_mcn,
        }
        pool_b_right_d_wrong += b_right_d_wrong
        pool_b_wrong_d_right += b_wrong_d_right

    # Pooled tests
    matrix = np.column_stack(matrix_cols)   # (n_problems, 1 + n_seeds)
    Q, p_Q = cochrans_q(matrix)
    pooled_mcn_p = mcnemar_pvalue(pool_b_right_d_wrong, pool_b_wrong_d_right)

    accs = [r["accuracy"] for r in condition_dict.values()]
    deltas = [r["delta_pp"] for r in per_seed.values()]

    return {
        "baseline_accuracy": baseline["accuracy"],
        "n_seeds": len(condition_dict),
        "n_problems": n_problems,
        "per_seed": per_seed,
        "mean_dpo_accuracy": float(np.mean(accs)),
        "sd_dpo_accuracy": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
        "mean_delta_pp": float(np.mean(deltas)),
        "sd_delta_pp": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
        "pooled_b_right_d_wrong": pool_b_right_d_wrong,
        "pooled_b_wrong_d_right": pool_b_wrong_d_right,
        "pooled_mcnemar_p": pooled_mcn_p,
        "cochrans_q": Q,
        "cochrans_q_p": p_Q,
    }


# ---------------------------------------------------------------------------
# Length-shift statistics
# ---------------------------------------------------------------------------

def length_shift_stats(baseline: dict, treatment: dict) -> dict:
    """Paired response-length shift stats between baseline and one treatment run."""
    b_lens = per_problem_response_lengths(baseline)
    t_lens = per_problem_response_lengths(treatment)
    diff = t_lens - b_lens
    return {
        "baseline_mean_len": float(b_lens.mean()),
        "treatment_mean_len": float(t_lens.mean()),
        "mean_delta": float(diff.mean()),
        "median_delta": float(np.median(diff)),
        "sd_delta": float(diff.std(ddof=1)),
        "n_longer": int((diff > 0).sum()),
        "n_shorter": int((diff < 0).sum()),
        "n_same": int((diff == 0).sum()),
    }


# ---------------------------------------------------------------------------
# Hypothesis-check decision (per RESEARCH_PLAN §4 pre-registered rule)
# ---------------------------------------------------------------------------

def decide_h2(comparison_B: dict, alpha_family: float = 0.025) -> dict:
    """Pre-registered decision on H2 (Condition B improves reasoning).

    H2 supported iff:
        - mean(Δ) across seeds ≥ +3 pp, AND
        - pooled McNemar p < α_family (default 0.025 after Bonferroni), AND
        - each per-seed bootstrap 95% CI excludes zero (or, softer, the
          mean-of-per-seed-CIs excludes zero).

    Returns:
        dict with 'supported' (bool), 'reasons' (list of strings).
    """
    reasons = []
    checks = []

    mean_delta = comparison_B["mean_delta_pp"]
    checks.append(mean_delta >= 3.0)
    reasons.append(
        f"mean delta = {mean_delta:+.2f} pp  "
        f"(threshold: ≥ +3.0 pp)  "
        f"{'PASS' if checks[-1] else 'FAIL'}"
    )

    pooled_p = comparison_B["pooled_mcnemar_p"]
    checks.append(pooled_p < alpha_family)
    reasons.append(
        f"pooled McNemar p = {pooled_p:.4f}  "
        f"(threshold: < {alpha_family})  "
        f"{'PASS' if checks[-1] else 'FAIL'}"
    )

    # Per-seed CI-exclusion check: does each seed's CI exclude zero?
    all_excludes = all(
        (s["ci_low_pp"] > 0) or (s["ci_high_pp"] < 0)
        for s in comparison_B["per_seed"].values()
    )
    checks.append(all_excludes)
    reasons.append(
        f"every per-seed 95% bootstrap CI excludes zero  "
        f"{'PASS' if all_excludes else 'FAIL'}"
    )

    return {
        "supported": all(checks),
        "reasons": reasons,
    }
