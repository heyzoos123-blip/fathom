# -*- coding: utf-8 -*-
"""
analyze_S_spectral.py -- Spectral analysis of C_delta trajectories.

The question: do transformer coherence trajectories have oscillatory
structure in the token-frequency domain, and does that structure differ
between correct and hallucinated generations?

Why we ask: Weng, Borst & Akyuerek 2026 (Imaging Neuroscience) showed
that sustained alpha-band EEG power tracks the prioritized item in a
working-memory task. That is a *frequency-domain* localization of
cognitive selection. All prior Fathom measurements of commitment
(S_early = IPR of events, Gini of per-layer coherence, etc.) are
*time-localization* statistics computed on a trajectory's values.
Nobody has ever asked whether the C_delta(t) sequence itself has
characteristic token-frequency peaks -- the analog of brain oscillations
indexed to token position rather than to seconds.

This script runs that analysis on the n=200 TruthfulQA C_delta
trajectory data Fathom already owns on disk. Zero new data collection.

Methodology:
  1. Load a results file with per-item c_delta_trajectory arrays.
  2. Classify each item as correct / hallucinated using the same
     helper the biphasic analyzer uses (classify_generation from
     analyze_S_truthfulqa).
  3. Truncate all trajectories to a common length L (default: 30 tokens).
     Drop items shorter than L.
  4. For each item, compute:
       - Mean-subtracted trajectory (DC-removed).
       - Real FFT magnitude spectrum.
       - Power in each of 5 token-frequency bands:
           gamma:  k in [L/2 ..] (period ~ 2 tokens, token-local)
           beta:   k in [L/3 .. L/2) (period ~ 3 tokens)
           alpha:  k in [L/7 .. L/3) (period ~ 4-7 tokens)  <-- WM prioritization analog
           theta:  k in [L/15 .. L/7) (period ~ 8-15 tokens)
           delta:  k in [1 .. L/15) (period >= 16 tokens, discourse-level)
         Band edges are documented inline.
  5. Compare per-band power between correct and hallucinated groups:
       - Welch t-test, Cohen's d, AUC
       - Also compare total power and spectral centroid
  6. Save raw per-item spectra + group stats as JSON.
  7. Emit a two-panel matplotlib figure:
       (a) mean power spectrum, correct vs hallucinated, with shaded SEM
       (b) per-band AUC bar chart with significance markers

Pre-registration note: this is an EXPLORATORY pilot. Bands are chosen
before seeing the correct/hallucinated comparison (they match the
standard EEG convention reinterpreted for token-frequency). Effect
sizes are reported honestly. A positive pilot motivates a
pre-registered confirmatory run; a null pilot is reported as-is and
informs the paper's spectral section scope.

Usage:
    python analysis/analyze_S_spectral.py [results_file.json]
         [--window_len L] [--min_tokens M] [--outdir DIR]

Defaults:
    results_file: truthfulqa_results/S_truthfulqa_llama_n200.json
    window_len:   30
    min_tokens:   30
    outdir:       truthfulqa_results/
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
from analyze_S_truthfulqa import classify_generation


# -----------------------------------------------------------------
# Band definitions (token-frequency, analogous to EEG bands)
# -----------------------------------------------------------------
# For a trajectory of length L with DC removed, the real FFT produces
# bins at frequencies k/L for k in [0, L//2]. The physical "period" at
# bin k is L/k tokens.
#
# We define bands by token-period so they are architecture-agnostic:
#   gamma: period 2-3 tokens   (token-local commitment)
#   beta:  period 3-4 tokens   (n-gram local syntax)
#   alpha: period 4-7 tokens   (phrase-level attractor prioritization)
#              <-- hypothesized analog of WM prioritization alpha
#   theta: period 8-15 tokens  (clause / working-memory chunking)
#   delta: period >= 16 tokens (discourse / stance)
# -----------------------------------------------------------------
def period_to_band(period):
    if period < 3:       return "gamma"
    if period < 4:       return "beta"
    if period < 8:       return "alpha"
    if period < 16:      return "theta"
    return "delta"


BAND_ORDER = ["delta", "theta", "alpha", "beta", "gamma"]


def compute_spectrum(traj, L, window="hann"):
    """Return (freqs, magnitude, power, per_band_power) for a single trajectory.

    The trajectory is truncated to L, mean-subtracted, optionally
    windowed (Hann by default to reduce spectral leakage), and run
    through a real FFT. Power is |FFT|^2 summed per band. Bin k=0
    (DC) is included in arrays with power 0 for plotting convenience.
    """
    x = np.asarray(traj[:L], dtype=float)
    if len(x) < L:
        return None
    x = x - x.mean()  # DC removal
    if window == "hann":
        w = np.hanning(L)
        # normalize to preserve power
        w = w / np.sqrt(np.mean(w ** 2))
        x = x * w
    elif window != "none":
        raise ValueError(f"unknown window: {window}")
    fft = np.fft.rfft(x)          # length L//2 + 1
    mag = np.abs(fft)             # magnitude
    power = mag ** 2              # power

    # Per-bin token period. Bin 0 is DC; bin k > 0 has period L/k.
    periods = np.array([np.inf] + [L / k for k in range(1, len(fft))])

    # Per-band power (ignore DC).
    band_power = {b: 0.0 for b in BAND_ORDER}
    for k in range(1, len(fft)):
        b = period_to_band(periods[k])
        band_power[b] += float(power[k])

    return {
        "fft_mag": mag.tolist(),
        "fft_power": power.tolist(),
        "periods": periods.tolist(),
        "band_power": band_power,
        "total_power": float(power[1:].sum()),
        "spectral_centroid": float(
            np.sum(np.arange(1, len(power)) * power[1:]) / (power[1:].sum() + 1e-12)
        ),
    }


def welch(a, b):
    """Welch t-test + Cohen's d + AUC. a = group 0, b = group 1."""
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    _, p = stats.ttest_ind(a_arr, b_arr, equal_var=False)
    pooled = np.sqrt((a_arr.var(ddof=1) + b_arr.var(ddof=1)) / 2)
    d = (b_arr.mean() - a_arr.mean()) / pooled if pooled > 0 else 0.0
    auc = roc_auc_score([0] * len(a_arr) + [1] * len(b_arr),
                        list(a_arr) + list(b_arr))
    return {
        "n_a": int(len(a_arr)),
        "n_b": int(len(b_arr)),
        "a_mean": float(a_arr.mean()),
        "b_mean": float(b_arr.mean()),
        "a_std": float(a_arr.std(ddof=1)),
        "b_std": float(b_arr.std(ddof=1)),
        "p": float(p),
        "d": float(d),
        "auc": float(auc),
        "ratio_b_over_a": float(b_arr.mean() / a_arr.mean())
            if a_arr.mean() > 0 else None,
    }


def permutation_max_d_null(correct_vals_by_bin, halluc_vals_by_bin, n_perm=2000, seed=42):
    """Family-wise permutation test on per-bin Cohen's d.

    Under the null (no correct/hallucinated difference) we shuffle
    labels and recompute the max |d| across bins. If the observed
    max |d| is in the extreme tail of that null distribution, the
    overall signal survives multi-comparison correction.

    Returns: {
      observed_max_abs_d: float,
      max_abs_d_null:     list[float],
      global_p:           float,
    }
    """
    rng = np.random.default_rng(seed)
    all_vals = np.concatenate([correct_vals_by_bin, halluc_vals_by_bin], axis=0)
    n_c = correct_vals_by_bin.shape[0]
    n_total = all_vals.shape[0]

    def max_abs_d(labels):
        a = all_vals[~labels]
        b = all_vals[labels]
        mean_a = a.mean(axis=0)
        mean_b = b.mean(axis=0)
        var_a = a.var(axis=0, ddof=1)
        var_b = b.var(axis=0, ddof=1)
        pooled = np.sqrt((var_a + var_b) / 2)
        d = np.where(pooled > 0, (mean_b - mean_a) / pooled, 0.0)
        # Skip DC bin (index 0)
        return float(np.max(np.abs(d[1:])))

    # observed
    obs_labels = np.zeros(n_total, dtype=bool)
    obs_labels[n_c:] = True
    obs_max_d = max_abs_d(obs_labels)

    # null
    null_max_ds = []
    for _ in range(n_perm):
        perm = rng.permutation(n_total)
        null_labels = obs_labels[perm]
        null_max_ds.append(max_abs_d(null_labels))
    null_max_ds = np.asarray(null_max_ds)
    global_p = float(np.mean(null_max_ds >= obs_max_d))

    return {
        "observed_max_abs_d": obs_max_d,
        "n_perm": n_perm,
        "null_mean_max_d": float(null_max_ds.mean()),
        "null_p95_max_d": float(np.percentile(null_max_ds, 95)),
        "global_p": global_p,
    }


def analyze(results_path, window_len, min_tokens, outdir):
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    model = data.get("model", "unknown")
    n_items = len(data.get("results", []))

    print("=" * 76)
    print("SPECTRAL PILOT: C_delta token-frequency analysis")
    print("=" * 76)
    print(f"Source:      {results_path}")
    print(f"Model:       {model}")
    print(f"N items:     {n_items}")
    print(f"Window len:  {window_len} tokens")
    print(f"Min tokens:  {min_tokens}")
    print()

    correct_specs = []
    halluc_specs = []
    unused = {"short": 0, "unclassified": 0}

    for r in data["results"]:
        n_tokens = r["generation"]["total_tokens"]
        if n_tokens < min_tokens:
            unused["short"] += 1
            continue
        traj = r["generation"]["c_delta_trajectory"]
        label, _ = classify_generation(
            r["generation"]["text"],
            r["correct_answers"],
            r["incorrect_answers"],
            r.get("best_answer", ""),
        )
        spec = compute_spectrum(traj, window_len)
        if spec is None:
            unused["short"] += 1
            continue
        if label == "correct":
            correct_specs.append(spec)
        elif label == "hallucinated":
            halluc_specs.append(spec)
        else:
            unused["unclassified"] += 1

    n_c = len(correct_specs)
    n_h = len(halluc_specs)
    print(f"Labeled:     {n_c} correct, {n_h} hallucinated "
          f"({unused['unclassified']} unclassified, "
          f"{unused['short']} too short)")
    print()

    if n_c < 5 or n_h < 5:
        print("[!] Insufficient labeled data (need at least 5+5).")
        return None

    # --------------------------------------------------------------
    # Per-band comparison
    # --------------------------------------------------------------
    print("-" * 76)
    print("PER-BAND POWER COMPARISON (correct vs hallucinated)")
    print("-" * 76)
    print(f"{'band':>7} {'period':>12} {'corr_mean':>12} {'hall_mean':>12} "
          f"{'d':>8} {'AUC':>8} {'p':>10}")

    per_band_stats = {}
    for b in BAND_ORDER:
        c_vals = [s["band_power"][b] for s in correct_specs]
        h_vals = [s["band_power"][b] for s in halluc_specs]
        st = welch(c_vals, h_vals)
        per_band_stats[b] = st
        period_label = {
            "delta": ">=16 tok",
            "theta": "8-15 tok",
            "alpha": "4-7 tok",
            "beta":  "3-4 tok",
            "gamma": "2-3 tok",
        }[b]
        print(f"{b:>7} {period_label:>12} {st['a_mean']:>12.2e} "
              f"{st['b_mean']:>12.2e} {st['d']:>+8.3f} "
              f"{st['auc']:>8.3f} {st['p']:>10.4f}")
    print()

    # --------------------------------------------------------------
    # Total power + spectral centroid
    # --------------------------------------------------------------
    print("-" * 76)
    print("AGGREGATE SPECTRAL STATS")
    print("-" * 76)
    total_c = [s["total_power"] for s in correct_specs]
    total_h = [s["total_power"] for s in halluc_specs]
    st_total = welch(total_c, total_h)
    print(f"total power:       corr={st_total['a_mean']:.2e} "
          f"hall={st_total['b_mean']:.2e} d={st_total['d']:+.3f} "
          f"AUC={st_total['auc']:.3f} p={st_total['p']:.4f}")

    cent_c = [s["spectral_centroid"] for s in correct_specs]
    cent_h = [s["spectral_centroid"] for s in halluc_specs]
    st_cent = welch(cent_c, cent_h)
    print(f"spectral centroid: corr={st_cent['a_mean']:.3f} "
          f"hall={st_cent['b_mean']:.3f} d={st_cent['d']:+.3f} "
          f"AUC={st_cent['auc']:.3f} p={st_cent['p']:.4f}")
    print()

    # --------------------------------------------------------------
    # Per-bin comparison (full spectrum)
    # --------------------------------------------------------------
    print("-" * 76)
    print("PER-BIN POWER COMPARISON (bin k, period L/k tokens)")
    print("-" * 76)
    n_bins = len(correct_specs[0]["fft_power"])
    per_bin_stats = []
    print(f"{'bin':>4} {'period':>10} {'corr_mean':>12} {'hall_mean':>12} "
          f"{'d':>8} {'AUC':>8} {'p':>10}")
    for k in range(1, n_bins):  # skip DC
        c_vals = [s["fft_power"][k] for s in correct_specs]
        h_vals = [s["fft_power"][k] for s in halluc_specs]
        st = welch(c_vals, h_vals)
        per_bin_stats.append({"bin": k, "period_tokens": window_len / k, **st})
        marker = " *" if st["p"] < 0.05 else "  "
        print(f"{k:>4} {window_len/k:>10.2f} {st['a_mean']:>12.2e} "
              f"{st['b_mean']:>12.2e} {st['d']:>+8.3f} "
              f"{st['auc']:>8.3f} {st['p']:>10.4f}{marker}")
    print()

    # --------------------------------------------------------------
    # Family-wise multi-comparison correction (permutation on max |d|)
    # --------------------------------------------------------------
    print("-" * 76)
    print("FAMILY-WISE MAX-|d| PERMUTATION TEST")
    print("-" * 76)
    c_power_mat = np.array([s["fft_power"] for s in correct_specs])
    h_power_mat = np.array([s["fft_power"] for s in halluc_specs])
    perm_result = permutation_max_d_null(c_power_mat, h_power_mat,
                                         n_perm=2000, seed=42)
    print(f"observed max |d|:  {perm_result['observed_max_abs_d']:.3f}")
    print(f"null mean max |d|: {perm_result['null_mean_max_d']:.3f}")
    print(f"null 95%ile:       {perm_result['null_p95_max_d']:.3f}")
    print(f"global p (family-wise): {perm_result['global_p']:.4f}")
    print()

    # --------------------------------------------------------------
    # Save
    # --------------------------------------------------------------
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    slug = os.path.basename(results_path).replace(".json", "")
    out_path = outdir / f"{slug}_spectral.json"

    out = {
        "source": str(results_path),
        "model": model,
        "window_len": window_len,
        "min_tokens": min_tokens,
        "n_correct": n_c,
        "n_hallucinated": n_h,
        "n_unclassified": unused["unclassified"],
        "n_short": unused["short"],
        "bands": {
            "delta": "period >= 16 tokens",
            "theta": "period 8-15 tokens",
            "alpha": "period 4-7 tokens (WM prioritization analog)",
            "beta":  "period 3-4 tokens",
            "gamma": "period 2-3 tokens",
        },
        "per_band_stats": per_band_stats,
        "aggregate": {
            "total_power": st_total,
            "spectral_centroid": st_cent,
        },
        "per_bin_stats": per_bin_stats,
        "family_wise_perm": perm_result,
        "correct_mean_spectrum": np.mean(
            [s["fft_power"] for s in correct_specs], axis=0
        ).tolist(),
        "halluc_mean_spectrum": np.mean(
            [s["fft_power"] for s in halluc_specs], axis=0
        ).tolist(),
        "correct_sem_spectrum": (
            np.std([s["fft_power"] for s in correct_specs], axis=0, ddof=1)
            / np.sqrt(n_c)
        ).tolist(),
        "halluc_sem_spectrum": (
            np.std([s["fft_power"] for s in halluc_specs], axis=0, ddof=1)
            / np.sqrt(n_h)
        ).tolist(),
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=float)

    print(f"[saved] {out_path}")

    # --------------------------------------------------------------
    # Figure
    # --------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

        # (a) mean power spectrum
        ax = axes[0]
        bins = np.arange(n_bins)
        mean_c = np.array(out["correct_mean_spectrum"])
        mean_h = np.array(out["halluc_mean_spectrum"])
        sem_c = np.array(out["correct_sem_spectrum"])
        sem_h = np.array(out["halluc_sem_spectrum"])

        # skip DC bin for plotting
        x = bins[1:]
        periods = window_len / x
        ax.plot(periods, mean_c[1:], label=f"correct (n={n_c})",
                color="#1f77b4", linewidth=2)
        ax.fill_between(periods, mean_c[1:] - sem_c[1:], mean_c[1:] + sem_c[1:],
                        alpha=0.25, color="#1f77b4")
        ax.plot(periods, mean_h[1:], label=f"hallucinated (n={n_h})",
                color="#d62728", linewidth=2)
        ax.fill_between(periods, mean_h[1:] - sem_h[1:], mean_h[1:] + sem_h[1:],
                        alpha=0.25, color="#d62728")
        ax.set_xlabel("token period (L / k)")
        ax.set_ylabel("mean FFT power")
        ax.set_title(f"spectral power -- {os.path.basename(model)}")
        ax.invert_xaxis()
        ax.legend(frameon=False)
        ax.grid(alpha=0.3)

        # (b) per-band AUC bar chart
        ax = axes[1]
        band_labels = BAND_ORDER
        band_aucs = [per_band_stats[b]["auc"] for b in band_labels]
        band_ps = [per_band_stats[b]["p"] for b in band_labels]
        colors = ["#ff7f0e" if p < 0.05 else "#808080" for p in band_ps]
        ax.barh(band_labels, band_aucs, color=colors, alpha=0.85)
        ax.axvline(0.5, color="black", linestyle="--", alpha=0.5, linewidth=1)
        for i, (auc, p) in enumerate(zip(band_aucs, band_ps)):
            marker = "*" if p < 0.05 else ""
            ax.text(auc + 0.005, i, f"AUC={auc:.3f}{marker}", va="center",
                    fontsize=9)
        ax.set_xlim(0.3, 0.85)
        ax.set_xlabel("AUC (correct vs hallucinated)")
        ax.set_title("per-band discriminability")
        ax.grid(axis="x", alpha=0.3)

        fig.suptitle(
            f"C_delta spectral analysis -- {os.path.basename(results_path)}",
            fontsize=11, y=1.02,
        )
        fig.tight_layout()
        fig_path = outdir / f"{slug}_spectral.png"
        fig.savefig(fig_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"[saved] {fig_path}")
    except Exception as e:
        print(f"[warn] figure generation failed: {e}")

    # --------------------------------------------------------------
    # Summary line
    # --------------------------------------------------------------
    print()
    print("=" * 76)
    print("HEADLINE")
    print("=" * 76)
    sig_bands = [b for b, st in per_band_stats.items() if st["p"] < 0.05]
    if sig_bands:
        print(f"Significant bands (p < 0.05): {sig_bands}")
        best = max(per_band_stats.items(), key=lambda x: abs(x[1]["d"]))
        print(f"Largest effect: {best[0]} band, "
              f"d={best[1]['d']:+.3f}, AUC={best[1]['auc']:.3f}, "
              f"p={best[1]['p']:.4f}")
    else:
        print("No band reaches p < 0.05 at this sample size.")
        best = max(per_band_stats.items(), key=lambda x: abs(x[1]["d"]))
        print(f"Largest trend: {best[0]} band, "
              f"d={best[1]['d']:+.3f}, AUC={best[1]['auc']:.3f}, "
              f"p={best[1]['p']:.4f}")

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "results_file",
        nargs="?",
        default="truthfulqa_results/S_truthfulqa_llama_n200.json",
    )
    ap.add_argument("--window_len", type=int, default=30)
    ap.add_argument("--min_tokens", type=int, default=30)
    ap.add_argument("--outdir", default="truthfulqa_results")
    args = ap.parse_args()

    analyze(args.results_file, args.window_len, args.min_tokens, args.outdir)


if __name__ == "__main__":
    main()
