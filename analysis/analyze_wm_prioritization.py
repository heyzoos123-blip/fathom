# -*- coding: utf-8 -*-
"""
analyze_wm_prioritization.py -- PRE-REGISTERED analyzer for the
in-silico working-memory prioritization replication of Weng et al. 2026.

This analyzer implements the sealed decision rule in
prereg/wm_prioritization_prereg.md. It reads a results JSON produced
by runners/run_wm_prioritization.py, computes S_early per trial,
pairs trials by pair_id, and applies the locked tests.

LOCKED PARAMETERS (do not modify -- pre-registered):
  S_early_window       = (0, 7)
  tau                  = 0.005
  n_permutations       = 10000
  permutation_seed     = 42
  alpha_primary        = 0.05 (one-sided)
  alpha_secondary      = 0.10 (one-sided, H2 delay scaling)
  min_d_primary        = 0.3

Usage:
    python analysis/analyze_wm_prioritization.py \
        truthfulqa_results/wm_prioritization_gemma-2-2b-it.json \
        --outdir truthfulqa_results
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy import stats

# LOCKED by pre-registration
S_EARLY_WINDOW = (0, 7)
TAU = 0.005
N_PERMUTATIONS = 10000
PERM_SEED = 42
ALPHA_PRIMARY = 0.05
ALPHA_SECONDARY = 0.10
MIN_D_PRIMARY = 0.30


def compute_s_early(trajectory, start=S_EARLY_WINDOW[0], end=S_EARLY_WINDOW[1],
                    tau=TAU):
    """Fathom S_early: max(window) / count(window > tau). Zero if no spikes."""
    if not trajectory:
        return 0.0
    window = list(trajectory[start:end])
    if not window:
        return 0.0
    spikes = [c for c in window if c > tau]
    if not spikes:
        return 0.0
    return float(max(window) / len(spikes))


def compute_h_early(entropy_trajectory, start=S_EARLY_WINDOW[0],
                    end=S_EARLY_WINDOW[1]):
    """Mean early-token logit entropy (the negative-control signal)."""
    if not entropy_trajectory:
        return 0.0
    window = list(entropy_trajectory[start:end])
    if not window:
        return 0.0
    return float(np.mean(window))


def paired_permutation_one_sided(diffs, n_perm=N_PERMUTATIONS, seed=PERM_SEED):
    """One-sided paired permutation test.

    H0: mean(diffs) <= 0.
    H1: mean(diffs) > 0.

    Null distribution: randomly flip sign of each paired difference,
    recompute mean. p = fraction of null means >= observed mean.
    """
    diffs = np.asarray(diffs, dtype=float)
    obs_mean = float(diffs.mean())
    rng = np.random.default_rng(seed)
    n = len(diffs)
    null_means = []
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=n)
        null_means.append(float((diffs * signs).mean()))
    null_means = np.asarray(null_means)
    p = float(np.mean(null_means >= obs_mean))
    return {
        "observed_mean_diff": obs_mean,
        "null_mean": float(null_means.mean()),
        "null_p95": float(np.percentile(null_means, 95)),
        "p_one_sided": p,
        "n_perm": n_perm,
    }


def cohens_dz_paired(diffs):
    diffs = np.asarray(diffs, dtype=float)
    if diffs.std(ddof=1) == 0:
        return 0.0
    return float(diffs.mean() / diffs.std(ddof=1))


def analyze(results_path, outdir):
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    model = data.get("model", "unknown")
    n_total = len(data.get("results", []))

    print("=" * 76)
    print("WM PRIORITIZATION ANALYSIS -- PRE-REGISTERED")
    print("=" * 76)
    print(f"Source:   {results_path}")
    print(f"Model:    {model}")
    print(f"N trials: {n_total}")
    print(f"Pre-reg:  prereg/wm_prioritization_prereg.md")
    print(f"Locked:   S_early window [0,7), tau=0.005, n_perm=10000, seed=42")
    print()

    # Index by pair_id, probe_target
    by_pair = {}
    excluded = []
    for r in data["results"]:
        gen = r.get("generation", {})
        if "error" in gen or gen.get("total_tokens", 0) < 8:
            excluded.append(r.get("trial_id"))
            continue
        pid = r["pair_id"]
        probe = r["probe_target"]
        if pid not in by_pair:
            by_pair[pid] = {}
        by_pair[pid][probe] = r

    print(f"Excluded (error or <8 tokens): {len(excluded)}")
    print()

    # Build paired differences
    paired_s_diffs = []    # S_early_A - S_early_B (within-pair)
    paired_h_diffs = []    # H_early_A - H_early_B (negative control)
    paired_by_delay = {"short": [], "long": []}
    paired_h_by_delay = {"short": [], "long": []}
    pair_records = []

    for pid, probes in by_pair.items():
        if "A" not in probes or "B" not in probes:
            continue
        s_a = compute_s_early(probes["A"]["generation"]["c_delta_trajectory"])
        s_b = compute_s_early(probes["B"]["generation"]["c_delta_trajectory"])
        h_a = compute_h_early(probes["A"]["generation"].get("entropy_trajectory", []))
        h_b = compute_h_early(probes["B"]["generation"].get("entropy_trajectory", []))
        delay = probes["A"]["delay_condition"]
        s_diff = s_a - s_b
        h_diff = h_a - h_b
        paired_s_diffs.append(s_diff)
        paired_h_diffs.append(h_diff)
        paired_by_delay[delay].append(s_diff)
        paired_h_by_delay[delay].append(h_diff)
        pair_records.append({
            "pair_id": pid,
            "delay": delay,
            "category": probes["A"]["category"],
            "s_early_A": s_a,
            "s_early_B": s_b,
            "s_diff": s_diff,
            "h_early_A": h_a,
            "h_early_B": h_b,
            "h_diff": h_diff,
        })

    n_pairs = len(paired_s_diffs)
    print(f"Valid paired observations: {n_pairs}")
    print(f"  short delay: {len(paired_by_delay['short'])}")
    print(f"  long delay:  {len(paired_by_delay['long'])}")
    print()

    if n_pairs < 10:
        print("[!] Insufficient paired data (need >= 10). Aborting.")
        return None

    # ----- Primary test (H1) -----
    print("-" * 76)
    print("H1: PRIMARY TEST (S_early_A > S_early_B, paired permutation)")
    print("-" * 76)
    h1_perm = paired_permutation_one_sided(paired_s_diffs)
    h1_dz = cohens_dz_paired(paired_s_diffs)
    print(f"mean S_early_A: {np.mean([p['s_early_A'] for p in pair_records]):.6f}")
    print(f"mean S_early_B: {np.mean([p['s_early_B'] for p in pair_records]):.6f}")
    print(f"mean diff:      {h1_perm['observed_mean_diff']:+.6f}")
    print(f"Cohen's d_z:    {h1_dz:+.3f}")
    print(f"one-sided p:    {h1_perm['p_one_sided']:.4f}")

    h1_pass_p = h1_perm["p_one_sided"] < ALPHA_PRIMARY
    h1_pass_d = h1_dz >= MIN_D_PRIMARY
    print(f"  H1 pass (p < 0.05):      {h1_pass_p}")
    print(f"  H1 pass (d_z >= 0.30):   {h1_pass_d}")
    print()

    # ----- Negative control (H4): entropy on the same pairs -----
    print("-" * 76)
    print("H4: NEGATIVE CONTROL (H_early_A vs H_early_B, paired permutation)")
    print("-" * 76)
    h4_perm = paired_permutation_one_sided(paired_h_diffs)
    h4_dz = cohens_dz_paired(paired_h_diffs)
    print(f"mean H_early_A: {np.mean([p['h_early_A'] for p in pair_records]):.6f}")
    print(f"mean H_early_B: {np.mean([p['h_early_B'] for p in pair_records]):.6f}")
    print(f"mean diff:      {h4_perm['observed_mean_diff']:+.6f}")
    print(f"Cohen's d_z:    {h4_dz:+.3f}")
    print(f"one-sided p:    {h4_perm['p_one_sided']:.4f}")
    h4_pass = abs(h4_dz) < MIN_D_PRIMARY
    print(f"  H4 negative control satisfied (|d_z| < 0.30): {h4_pass}")
    print()

    # ----- Delay scaling (H2) -----
    print("-" * 76)
    print("H2: DELAY SCALING (long-delay diff > short-delay diff)")
    print("-" * 76)
    mean_short = float(np.mean(paired_by_delay["short"])) if paired_by_delay["short"] else 0.0
    mean_long = float(np.mean(paired_by_delay["long"])) if paired_by_delay["long"] else 0.0
    print(f"mean short-delay diff: {mean_short:+.6f}")
    print(f"mean long-delay diff:  {mean_long:+.6f}")
    h2_obs = mean_long - mean_short
    print(f"long - short:          {h2_obs:+.6f}")

    # Unpaired permutation on diffs between delay conditions
    all_diffs = np.concatenate([paired_by_delay["short"],
                                paired_by_delay["long"]])
    n_short = len(paired_by_delay["short"])
    rng = np.random.default_rng(PERM_SEED)
    null_h2 = []
    for _ in range(N_PERMUTATIONS):
        perm = rng.permutation(len(all_diffs))
        s = all_diffs[perm[:n_short]].mean()
        l = all_diffs[perm[n_short:]].mean()
        null_h2.append(l - s)
    null_h2 = np.asarray(null_h2)
    h2_p = float(np.mean(null_h2 >= h2_obs))
    print(f"one-sided p: {h2_p:.4f}")
    h2_pass = h2_p < ALPHA_SECONDARY
    print(f"  H2 pass (p < 0.10): {h2_pass}")
    print()

    # ----- Sealed decision -----
    print("=" * 76)
    print("SEALED DECISION RULE EVALUATION")
    print("=" * 76)
    print(f"H1 (S_early_A > S_early_B, p < 0.05, d_z >= 0.30): "
          f"{h1_pass_p and h1_pass_d}")
    print(f"H4 (entropy negative control, |d_z| < 0.30):       {h4_pass}")
    print(f"H2 (delay scaling, p < 0.10):                      {h2_pass}")
    print()
    h1_supported = h1_pass_p and h1_pass_d and h4_pass
    print(f"H1 SUPPORTED: {h1_supported}")
    print()

    # Save
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    slug = os.path.basename(results_path).replace(".json", "")
    out_path = outdir / f"{slug}_analysis.json"
    out = {
        "source": str(results_path),
        "model": model,
        "pre_reg": "prereg/wm_prioritization_prereg.md",
        "locked_params": {
            "s_early_window": list(S_EARLY_WINDOW),
            "tau": TAU,
            "n_permutations": N_PERMUTATIONS,
            "perm_seed": PERM_SEED,
            "alpha_primary": ALPHA_PRIMARY,
            "alpha_secondary": ALPHA_SECONDARY,
            "min_d_primary": MIN_D_PRIMARY,
        },
        "n_pairs": n_pairs,
        "n_excluded": len(excluded),
        "pair_records": pair_records,
        "H1_primary": {
            "permutation": h1_perm,
            "cohens_dz": h1_dz,
            "pass_p": h1_pass_p,
            "pass_d": h1_pass_d,
        },
        "H4_negative_control": {
            "permutation": h4_perm,
            "cohens_dz": h4_dz,
            "pass": h4_pass,
        },
        "H2_delay_scaling": {
            "mean_short_diff": mean_short,
            "mean_long_diff": mean_long,
            "long_minus_short": h2_obs,
            "one_sided_p": h2_p,
            "pass": h2_pass,
        },
        "H1_SUPPORTED": h1_supported,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"[saved] {out_path}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_file")
    ap.add_argument("--outdir", default="truthfulqa_results")
    args = ap.parse_args()
    analyze(args.results_file, args.outdir)


if __name__ == "__main__":
    main()
