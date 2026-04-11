# -*- coding: utf-8 -*-
"""
analyze_S_spectral_controls.py -- Head-to-head spectral analysis on
C_delta vs. logit entropy vs. logprob vs. top2_margin trajectories.

Purpose: if the spectral hallucination signature we found in C_delta
(correct > hallucinated at period 4-8 tokens) also appears in raw
logit-based trajectories, then the spectral story is "any trajectory
has it" and is not specific to SAE coherence. If the signature is
present in C_delta but absent (or weaker) in entropy / logprob /
top2_margin, then we have evidence that SAE-derived coherence
trajectories carry spectral information that raw uncertainty proxies
do not.

This is the same control structure that underpins the Fathom paper's
head-to-head claim: S_early beats entropy as a POINTWISE statistic
(AUC 0.663 vs 0.607). We are now asking the same question at the
SPECTRAL level.

Requires: S_baselines_n200.json (Gemma-2-2b-IT) which was captured
with the extended CoherenceSteererExt and has all four trajectories
attached to each generation.

Usage:
    python analysis/analyze_S_spectral_controls.py \
        [baselines_file.json] [--window_len L]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from analyze_S_truthfulqa import classify_generation
from analyze_S_spectral import compute_spectrum, welch, permutation_max_d_null


SIGNAL_KEYS = [
    ("c_delta_trajectory",    "C_delta (SAE coherence)"),
    ("entropy_trajectory",    "logit entropy"),
    ("logprob_trajectory",    "negative log-prob"),
    ("top2_margin_trajectory", "top-2 margin"),
]


def analyze_signal(results, signal_key, L, min_tokens):
    """Return per-item spectra for one signal key, split by label."""
    correct_specs = []
    halluc_specs = []
    for r in results:
        n_tokens = r["generation"]["total_tokens"]
        if n_tokens < min_tokens:
            continue
        traj = r["generation"].get(signal_key)
        if traj is None or len(traj) < L:
            continue
        label, _ = classify_generation(
            r["generation"]["text"],
            r["correct_answers"],
            r["incorrect_answers"],
            r.get("best_answer", ""),
        )
        spec = compute_spectrum(traj, L)
        if spec is None:
            continue
        if label == "correct":
            correct_specs.append(spec)
        elif label == "hallucinated":
            halluc_specs.append(spec)
    return correct_specs, halluc_specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "baselines_file",
        nargs="?",
        default="truthfulqa_results/S_baselines_n200.json",
    )
    ap.add_argument("--window_len", type=int, default=40)
    ap.add_argument("--min_tokens", type=int, default=40)
    ap.add_argument("--outdir", default="truthfulqa_results")
    args = ap.parse_args()

    with open(args.baselines_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("=" * 76)
    print("SPECTRAL CONTROLS: C_delta vs raw baseline trajectories")
    print("=" * 76)
    print(f"Source:     {args.baselines_file}")
    print(f"Model:      {data.get('model')}")
    print(f"N items:    {len(data.get('results', []))}")
    print(f"Window len: {args.window_len}")
    print()

    summary = {}
    for key, label in SIGNAL_KEYS:
        print("-" * 76)
        print(f"SIGNAL: {label}  ({key})")
        print("-" * 76)
        c_specs, h_specs = analyze_signal(
            data["results"], key, args.window_len, args.min_tokens
        )
        n_c, n_h = len(c_specs), len(h_specs)
        print(f"Labeled: {n_c} correct, {n_h} hallucinated")
        if n_c < 5 or n_h < 5:
            print("[!] Insufficient data for this signal, skipping")
            print()
            summary[key] = {"skipped": True, "n_c": n_c, "n_h": n_h}
            continue

        # Per-bin comparison
        c_power_mat = np.array([s["fft_power"] for s in c_specs])
        h_power_mat = np.array([s["fft_power"] for s in h_specs])
        n_bins = c_power_mat.shape[1]

        per_bin = []
        for k in range(1, n_bins):
            st = welch(c_power_mat[:, k], h_power_mat[:, k])
            per_bin.append({"bin": k, "period": args.window_len / k, **st})

        # Family-wise permutation
        perm = permutation_max_d_null(c_power_mat, h_power_mat,
                                      n_perm=2000, seed=42)

        # Aggregate stats
        total_c = c_power_mat[:, 1:].sum(axis=1)
        total_h = h_power_mat[:, 1:].sum(axis=1)
        st_total = welch(total_c, total_h)

        # Print key bins
        sig_bins = [b for b in per_bin if b["p"] < 0.05]
        print(f"n significant bins (p<0.05): {len(sig_bins)}")
        if sig_bins:
            print("Significant bins (bin, period, d, AUC, p):")
            for b in sig_bins:
                print(f"  bin {b['bin']:>2}  period {b['period']:>6.2f}  "
                      f"d={b['d']:+.3f}  AUC={b['auc']:.3f}  p={b['p']:.4f}")
        else:
            # top-3 by absolute d
            top = sorted(per_bin, key=lambda x: -abs(x["d"]))[:3]
            print("Top 3 bins by |d| (none reach p<0.05):")
            for b in top:
                print(f"  bin {b['bin']:>2}  period {b['period']:>6.2f}  "
                      f"d={b['d']:+.3f}  AUC={b['auc']:.3f}  p={b['p']:.4f}")

        print(f"Total power:        d={st_total['d']:+.3f}, "
              f"p={st_total['p']:.4f}")
        print(f"Family-wise max|d|: {perm['observed_max_abs_d']:.3f}, "
              f"global p={perm['global_p']:.4f}")
        print()

        summary[key] = {
            "label": label,
            "n_c": n_c,
            "n_h": n_h,
            "per_bin": per_bin,
            "total_power": st_total,
            "family_wise_perm": perm,
            "n_sig_bins": len(sig_bins),
        }

    # --------------------------------------------------------------
    # Head-to-head verdict
    # --------------------------------------------------------------
    print("=" * 76)
    print("HEAD-TO-HEAD VERDICT")
    print("=" * 76)
    print(f"{'signal':<28} {'max|d|':>8} {'global_p':>10} {'sig_bins':>10}")
    for key, _ in SIGNAL_KEYS:
        s = summary.get(key)
        if s is None or s.get("skipped"):
            continue
        print(f"{s['label']:<28} "
              f"{s['family_wise_perm']['observed_max_abs_d']:>8.3f} "
              f"{s['family_wise_perm']['global_p']:>10.4f} "
              f"{s['n_sig_bins']:>10d}")
    print()
    print("Interpretation: if C_delta has significant spectral structure")
    print("AND raw-uncertainty trajectories (entropy, logprob, top2_margin)")
    print("do not, then the spectral hallucination signature is specific to")
    print("SAE-derived coherence trajectories.")

    # Save
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    slug = os.path.basename(args.baselines_file).replace(".json", "")
    out_path = outdir / f"{slug}_spectral_controls.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "source": args.baselines_file,
            "model": data.get("model"),
            "window_len": args.window_len,
            "min_tokens": args.min_tokens,
            "signals": summary,
        }, f, indent=2, default=float)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
