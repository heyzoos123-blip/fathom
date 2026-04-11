# -*- coding: utf-8 -*-
"""
analyze_S_spectral_v22.py -- Spectral analysis under v22 (semantic-relabeled)
labels for the pre-registered sensitivity analysis on the C_delta
hallucination signature.

This is the confirmatory analyzer for the pre-registration sealed in
prereg/spectral_sensitivity_prereg.md (git commit 90a6f7d).

It re-runs the exact same spectral pipeline as analyze_S_spectral.py,
but reads item labels from the *_v22_audit.json file instead of calling
classify_generation on the fly. The audit file is produced by
analysis/semantic_relabel.py with the pre-registered embedding model
(all-MiniLM-L6-v2) and margin threshold (0.15).

LOCKED PARAMETERS (from prereg/spectral_sensitivity_prereg.md):
  L = 40 tokens
  min_tokens = 40
  window = Hann, normalized for power preservation
  permutation n_perm = 2000, seed = 42

Usage:
    python analysis/analyze_S_spectral_v22.py \
        truthfulqa_results/S_baselines_n200.json \
        truthfulqa_results/S_baselines_n200_v22_audit.json \
        --outdir truthfulqa_results
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from analyze_S_spectral import compute_spectrum, welch, permutation_max_d_null, BAND_ORDER, period_to_band


# LOCKED by pre-registration (do not change)
WINDOW_LEN = 40
MIN_TOKENS = 40
N_PERM = 2000
PERM_SEED = 42


def load_v22_labels(audit_path):
    """Load v22 audit file and return a dict: hf_index -> combined label.

    Combined label is one of {correct, hallucinated, other}.
    "correct" = v22_label in {correct, correct_semantic}
    "hallucinated" = v22_label in {hallucinated, hallucinated_semantic}
    "other" = everything else (ambiguous, refusal, too_short, etc.)
    """
    with open(audit_path, "r", encoding="utf-8") as f:
        audit = json.load(f)
    details = audit["details"]
    labels = {}
    for d in details:
        lbl = d.get("v22_label", "unknown")
        if lbl in ("correct", "correct_semantic"):
            combined = "correct"
        elif lbl in ("hallucinated", "hallucinated_semantic"):
            combined = "hallucinated"
        else:
            combined = "other"
        labels[d["hf_index"]] = combined
    return labels


def analyze(results_path, audit_path, outdir):
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    labels_by_idx = load_v22_labels(audit_path)

    model = data.get("model", "unknown")

    print("=" * 76)
    print("SPECTRAL SENSITIVITY (v22 labels) -- PRE-REGISTERED")
    print("=" * 76)
    print(f"Trajectory:  {results_path}")
    print(f"v22 audit:   {audit_path}")
    print(f"Model:       {model}")
    print(f"Window L:    {WINDOW_LEN}  (LOCKED)")
    print(f"Permutation: n={N_PERM}, seed={PERM_SEED}  (LOCKED)")
    print(f"Pre-reg:     prereg/spectral_sensitivity_prereg.md")
    print()

    correct_specs = []
    halluc_specs = []
    unused = {"short": 0, "other": 0, "missing_label": 0}

    for r in data["results"]:
        n_tokens = r["generation"]["total_tokens"]
        if n_tokens < MIN_TOKENS:
            unused["short"] += 1
            continue
        hf_idx = r.get("hf_index")
        combined = labels_by_idx.get(hf_idx)
        if combined is None:
            unused["missing_label"] += 1
            continue
        traj = r["generation"]["c_delta_trajectory"]
        spec = compute_spectrum(traj, WINDOW_LEN)
        if spec is None:
            unused["short"] += 1
            continue
        if combined == "correct":
            correct_specs.append(spec)
        elif combined == "hallucinated":
            halluc_specs.append(spec)
        else:
            unused["other"] += 1

    n_c = len(correct_specs)
    n_h = len(halluc_specs)
    print(f"Labeled: {n_c} correct, {n_h} hallucinated  "
          f"({unused['other']} other, {unused['short']} too short, "
          f"{unused['missing_label']} missing from audit)")
    print()

    if n_c < 5 or n_h < 5:
        print("[!] Insufficient labeled data.")
        return None

    # Per-bin comparison
    print("-" * 76)
    print("PER-BIN POWER COMPARISON (L=40, v22 labels)")
    print("-" * 76)
    c_power_mat = np.array([s["fft_power"] for s in correct_specs])
    h_power_mat = np.array([s["fft_power"] for s in halluc_specs])
    n_bins = c_power_mat.shape[1]

    print(f"{'bin':>4} {'period':>10} {'corr_mean':>12} {'hall_mean':>12} "
          f"{'d':>8} {'AUC':>8} {'p':>10}")
    per_bin_stats = []
    for k in range(1, n_bins):
        st = welch(c_power_mat[:, k], h_power_mat[:, k])
        per_bin_stats.append({"bin": k, "period_tokens": WINDOW_LEN / k, **st})
        marker = " *" if st["p"] < 0.05 else "  "
        print(f"{k:>4} {WINDOW_LEN/k:>10.2f} {st['a_mean']:>12.2e} "
              f"{st['b_mean']:>12.2e} {st['d']:>+8.3f} "
              f"{st['auc']:>8.3f} {st['p']:>10.4f}{marker}")
    print()

    # Family-wise max|d| permutation
    print("-" * 76)
    print("FAMILY-WISE MAX|d| PERMUTATION (LOCKED)")
    print("-" * 76)
    perm = permutation_max_d_null(
        c_power_mat, h_power_mat, n_perm=N_PERM, seed=PERM_SEED,
    )
    print(f"observed max |d|:  {perm['observed_max_abs_d']:.3f}")
    print(f"null mean max |d|: {perm['null_mean_max_d']:.3f}")
    print(f"null 95 percentile:{perm['null_p95_max_d']:.3f}")
    print(f"family-wise global p: {perm['global_p']:.4f}")
    print()

    # Aggregate
    total_c = c_power_mat[:, 1:].sum(axis=1)
    total_h = h_power_mat[:, 1:].sum(axis=1)
    st_total = welch(total_c, total_h)
    print(f"total power: d={st_total['d']:+.3f}, p={st_total['p']:.4f}")

    # Peak bin info
    peak = max(per_bin_stats, key=lambda x: abs(x["d"]))
    print(f"peak bin: {peak['bin']} (period {peak['period_tokens']:.2f} tok), "
          f"d={peak['d']:+.3f}, p={peak['p']:.4f}")
    print()

    # Sealed decision
    fwise_p = perm["global_p"]
    direction_ok = peak["d"] < 0  # correct > hallucinated
    print("=" * 76)
    print("SEALED DECISION RULE EVALUATION")
    print("=" * 76)
    print(f"family-wise p < 0.05:  {fwise_p < 0.05}  (observed {fwise_p:.4f})")
    print(f"peak direction correct > halluc:  {direction_ok}")
    print()

    # Save
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    slug = os.path.basename(results_path).replace(".json", "")
    out_path = outdir / f"{slug}_spectral_v22.json"
    out = {
        "source_trajectory": str(results_path),
        "source_audit":      str(audit_path),
        "model":             model,
        "window_len":        WINDOW_LEN,
        "min_tokens":        MIN_TOKENS,
        "n_perm":             N_PERM,
        "perm_seed":          PERM_SEED,
        "n_correct":          n_c,
        "n_hallucinated":     n_h,
        "n_other":            unused["other"],
        "n_short":            unused["short"],
        "n_missing_label":    unused["missing_label"],
        "per_bin_stats":      per_bin_stats,
        "family_wise_perm":   perm,
        "total_power":        st_total,
        "peak_bin":           peak,
        "correct_mean_spectrum": np.mean(c_power_mat, axis=0).tolist(),
        "halluc_mean_spectrum":  np.mean(h_power_mat, axis=0).tolist(),
        "correct_sem_spectrum": (
            np.std(c_power_mat, axis=0, ddof=1) / np.sqrt(n_c)
        ).tolist(),
        "halluc_sem_spectrum": (
            np.std(h_power_mat, axis=0, ddof=1) / np.sqrt(n_h)
        ).tolist(),
        "sealed_decision": {
            "fwise_p_lt_0p05": bool(fwise_p < 0.05),
            "direction_correct_gt_halluc": bool(direction_ok),
            "observed_fwise_p": float(fwise_p),
        },
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"[saved] {out_path}")

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_file")
    ap.add_argument("audit_file")
    ap.add_argument("--outdir", default="truthfulqa_results")
    args = ap.parse_args()
    analyze(args.results_file, args.audit_file, args.outdir)


if __name__ == "__main__":
    main()
