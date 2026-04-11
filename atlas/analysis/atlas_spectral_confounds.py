# -*- coding: utf-8 -*-
"""
atlas_spectral_confounds.py -- Adversarial sanity checks on the
cross-architecture atlas spectral category finding.

The headline: 12/12 open-weight models show reasoning-vs-hallucination
separation at mean |d| > 1.0 on entropy/logprob trajectory spectra.
Before we build an entire product on top of this, we need to rule out
four trivial confounds. If ANY confound explains the effect, we pivot.

CONFOUNDS TESTED:

  C1. Prompt length / token count bias.
      Do category prompts differ in prompt length or generated-token
      count in a way that trivially predicts category?
      Test: category separability using only the scalar log(n_tokens)
      as a feature. If this alone separates categories with d > 1.0,
      the spectral method might be picking up nothing but length.

  C2. DC / mean-level confound.
      Does the separation live entirely in the overall mean (DC bin)
      of entropy / logprob, which would make the spectral claim
      trivial — i.e. "reasoning prompts produce lower-entropy outputs
      on average" would be a one-number summary that requires no FFT.
      Test: rerun the per-bin separability WITHOUT removing DC, and
      also test DC-only. If DC carries all the signal, the spectral
      shape adds nothing and the wrapper's value collapses to "measure
      mean entropy," which is a single scalar competitors already use.

  C3. Surface-statistic confound.
      Does the full trajectory mean + variance alone match the
      per-bin spectral separability? I.e., is FFT adding anything
      over (mean, std) scalars of the trajectory?
      Test: compute (mean, std) of each trajectory, compute Cohen's d
      between categories on each scalar, compare to the spectral
      max|d|. If (mean, std) beats or matches spectral, we ship a
      much simpler product (and a much smaller claim).

  C4. Baseline-entropy confound.
      Does the signal simply reflect that different prompt categories
      induce different overall entropy magnitudes (i.e., reasoning
      prompts have lower baseline entropy than hallucination prompts)?
      Test: z-score each trajectory (subtract mean, divide by std)
      BEFORE computing the spectrum. If the signal vanishes, it was
      magnitude; if it survives, it's shape.

These are not "nice to haves". They are the gatekeepers between
"we discovered something" and "we're building on an artifact."

Usage:
    python atlas/analysis/atlas_spectral_confounds.py \
        --captures-dir atlas/captures \
        --outdir atlas/analysis
"""

import argparse
import json
import os
import sys
from pathlib import Path
from itertools import combinations

import numpy as np
from scipy import stats

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
from analyze_S_spectral import compute_spectrum


WINDOW_LEN = 20
MIN_TOKENS = 20
SIGNAL_KEYS = ["D", "entropy", "logprob", "top2_margin"]
CATEGORIES = [
    "retrieval", "reasoning", "refusal",
    "creative", "adversarial", "hallucination",
]


def load_capture(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def cohen_d(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return 0.0
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return (b.mean() - a.mean()) / pooled if pooled > 0 else 0.0


def extract_trajectory_set(capture, signal):
    """Return dict: category -> list of trajectory arrays (raw, length L).
    Also returns the prompt and response token counts.
    """
    out = {}
    for cat in CATEGORIES:
        cat_list = []
        probes = capture.get("probes", {}).get(cat, {})
        for probe_id, probe in probes.items():
            traj = probe.get("trajectories", {}).get(signal)
            if traj is None or len(traj) < MIN_TOKENS:
                continue
            cat_list.append({
                "id": probe_id,
                "trajectory": np.asarray(traj[:WINDOW_LEN], dtype=float),
                "full_length": len(traj),
                "prompt_len": len(probe.get("text", "")),
                "tokens_generated": len(probe.get("tokens", [])),
            })
        out[cat] = cat_list
    return out


# -----------------------------------------------------------------
# Confound tests -- each returns a dict of pair -> |d| on the
# scalar/alternative feature defined by the confound. We compare
# each to the spectral max|d| to see if the confound explains it.
# -----------------------------------------------------------------

def c1_length_confound(traj_by_cat):
    """Cohen's d between categories using only log(tokens_generated)."""
    out = {}
    cats = list(traj_by_cat.keys())
    for c1, c2 in combinations(cats, 2):
        v1 = np.array([np.log(t["tokens_generated"] + 1)
                       for t in traj_by_cat[c1]])
        v2 = np.array([np.log(t["tokens_generated"] + 1)
                       for t in traj_by_cat[c2]])
        out[(c1, c2)] = abs(cohen_d(v1, v2))
    return out


def c2_dc_only(traj_by_cat):
    """Cohen's d using only the trajectory mean (DC component)."""
    out = {}
    cats = list(traj_by_cat.keys())
    for c1, c2 in combinations(cats, 2):
        v1 = np.array([t["trajectory"].mean() for t in traj_by_cat[c1]])
        v2 = np.array([t["trajectory"].mean() for t in traj_by_cat[c2]])
        out[(c1, c2)] = abs(cohen_d(v1, v2))
    return out


def c3_scalar_surface(traj_by_cat):
    """Cohen's d using only (mean, std) of the raw trajectory.
    We take the max over the two scalars to be fair.
    """
    out = {}
    cats = list(traj_by_cat.keys())
    for c1, c2 in combinations(cats, 2):
        means1 = np.array([t["trajectory"].mean() for t in traj_by_cat[c1]])
        means2 = np.array([t["trajectory"].mean() for t in traj_by_cat[c2]])
        stds1 = np.array([t["trajectory"].std(ddof=1) for t in traj_by_cat[c1]])
        stds2 = np.array([t["trajectory"].std(ddof=1) for t in traj_by_cat[c2]])
        d_mean = abs(cohen_d(means1, means2))
        d_std = abs(cohen_d(stds1, stds2))
        out[(c1, c2)] = max(d_mean, d_std)
    return out


def c4_zscored_spectral(traj_by_cat):
    """Rerun per-bin spectral separability after z-scoring each trajectory.

    If the effect vanishes, it was magnitude/baseline (pointless for
    the product). If it survives, it's genuine trajectory SHAPE.
    """
    out = {}
    cats = list(traj_by_cat.keys())
    n_bins = WINDOW_LEN // 2 + 1

    def zscore_and_fft(traj):
        t = np.asarray(traj, dtype=float)
        if t.std(ddof=1) < 1e-12:
            return None
        z = (t - t.mean()) / t.std(ddof=1)
        spec = compute_spectrum(z.tolist(), WINDOW_LEN, window="hann")
        return None if spec is None else np.array(spec["fft_power"])

    cat_spec_mats = {}
    for c in cats:
        specs = []
        for t in traj_by_cat[c]:
            spec = zscore_and_fft(t["trajectory"])
            if spec is not None:
                specs.append(spec)
        cat_spec_mats[c] = np.array(specs) if specs else None

    for c1, c2 in combinations(cats, 2):
        m1 = cat_spec_mats.get(c1)
        m2 = cat_spec_mats.get(c2)
        if m1 is None or m2 is None or len(m1) < 2 or len(m2) < 2:
            out[(c1, c2)] = 0.0
            continue
        max_d = 0.0
        for k in range(1, n_bins):
            d = abs(cohen_d(m1[:, k], m2[:, k]))
            if d > max_d:
                max_d = d
        out[(c1, c2)] = max_d
    return out


# -----------------------------------------------------------------
# Reference: the original per-bin spectral max|d| (what we want to
# compare each confound against).
# -----------------------------------------------------------------

def reference_spectral(traj_by_cat):
    out = {}
    cats = list(traj_by_cat.keys())
    n_bins = WINDOW_LEN // 2 + 1

    cat_spec_mats = {}
    for c in cats:
        specs = []
        for t in traj_by_cat[c]:
            spec = compute_spectrum(
                t["trajectory"].tolist(), WINDOW_LEN, window="hann"
            )
            if spec is not None:
                specs.append(np.array(spec["fft_power"]))
        cat_spec_mats[c] = np.array(specs) if specs else None

    for c1, c2 in combinations(cats, 2):
        m1 = cat_spec_mats.get(c1)
        m2 = cat_spec_mats.get(c2)
        if m1 is None or m2 is None or len(m1) < 2 or len(m2) < 2:
            out[(c1, c2)] = 0.0
            continue
        max_d = 0.0
        for k in range(1, n_bins):
            d = abs(cohen_d(m1[:, k], m2[:, k]))
            if d > max_d:
                max_d = d
        out[(c1, c2)] = max_d
    return out


# -----------------------------------------------------------------
# Main
# -----------------------------------------------------------------

def summarize_against_reference(label, confound, reference):
    """Return diagnostic stats comparing a confound feature against
    the reference spectral max|d|.

    If the confound's max|d| per pair tracks or exceeds the reference,
    the confound explains (part of) the spectral signal. If it is
    systematically smaller, the spectral signal is real and lives
    beyond the confound.
    """
    ref_vals = []
    conf_vals = []
    per_pair = []
    for pair in reference:
        r = reference[pair]
        c = confound.get(pair, 0.0)
        ref_vals.append(r)
        conf_vals.append(c)
        per_pair.append({
            "pair": f"{pair[0]} vs {pair[1]}",
            "spectral_max_abs_d": r,
            f"{label}_max_abs_d": c,
            "confound_explains_pct": (c / r * 100.0) if r > 0 else 0.0,
        })
    ref_vals = np.array(ref_vals)
    conf_vals = np.array(conf_vals)
    # Correlation between pair-wise confound and pair-wise reference
    if len(ref_vals) > 2 and ref_vals.std() > 0 and conf_vals.std() > 0:
        r_corr = float(np.corrcoef(ref_vals, conf_vals)[0, 1])
    else:
        r_corr = None
    return {
        "confound_label": label,
        "mean_spectral_d": float(ref_vals.mean()),
        f"mean_{label}_d": float(conf_vals.mean()),
        "ratio_confound_to_spectral": (
            float(conf_vals.mean() / ref_vals.mean())
            if ref_vals.mean() > 0 else None
        ),
        "correlation_pairwise_r": r_corr,
        "n_pairs": len(ref_vals),
        "per_pair": per_pair,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures-dir", default="atlas/captures")
    ap.add_argument("--outdir", default="atlas/analysis")
    args = ap.parse_args()

    captures_dir = Path(args.captures_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    capture_paths = sorted([
        p for p in captures_dir.glob("*.json")
        if "gpt-4o-mini" not in p.name.lower()
    ])

    print("=" * 76)
    print("ATLAS SPECTRAL CONFOUND CHECKS")
    print("=" * 76)
    print(f"Captures: {len(capture_paths)} open-weight models")
    print(f"Window L: {WINDOW_LEN}")
    print()

    all_results = {}

    for signal in SIGNAL_KEYS:
        print("-" * 76)
        print(f"SIGNAL: {signal}")
        print("-" * 76)
        per_model = []
        for cap_path in capture_paths:
            capture = load_capture(cap_path)
            model = capture.get("model", cap_path.name)
            traj_by_cat = extract_trajectory_set(capture, signal)
            if not any(len(v) > 0 for v in traj_by_cat.values()):
                print(f"  [skip] {model}: no data")
                continue

            ref = reference_spectral(traj_by_cat)
            c1 = c1_length_confound(traj_by_cat)
            c2 = c2_dc_only(traj_by_cat)
            c3 = c3_scalar_surface(traj_by_cat)
            c4 = c4_zscored_spectral(traj_by_cat)

            per_model.append({
                "model": model,
                "reference": {str(k): v for k, v in ref.items()},
                "C1_length": summarize_against_reference(
                    "C1_length", c1, ref
                ),
                "C2_dc_only": summarize_against_reference(
                    "C2_dc_only", c2, ref
                ),
                "C3_scalar_surface": summarize_against_reference(
                    "C3_scalar_surface", c3, ref
                ),
                "C4_zscored_spectral": summarize_against_reference(
                    "C4_zscored_spectral", c4, ref
                ),
            })
            print(
                f"  {model:<36}  "
                f"ref={per_model[-1]['C1_length']['mean_spectral_d']:.2f}  "
                f"C1_len={per_model[-1]['C1_length']['mean_C1_length_d']:.2f}  "
                f"C2_dc={per_model[-1]['C2_dc_only']['mean_C2_dc_only_d']:.2f}  "
                f"C3_mstd={per_model[-1]['C3_scalar_surface']['mean_C3_scalar_surface_d']:.2f}  "
                f"C4_zFFT={per_model[-1]['C4_zscored_spectral']['mean_C4_zscored_spectral_d']:.2f}"
            )

        # Cross-model summary per signal
        def avg(metric_key, subkey):
            vals = [m[metric_key][subkey] for m in per_model
                    if m[metric_key][subkey] is not None]
            return float(np.mean(vals)) if vals else None

        signal_summary = {
            "signal": signal,
            "n_models": len(per_model),
            "mean_spectral_d_across_models": avg(
                "C1_length", "mean_spectral_d"
            ),
            "C1_length": {
                "mean_confound_d": avg("C1_length", "mean_C1_length_d"),
                "mean_ratio": avg("C1_length", "ratio_confound_to_spectral"),
                "mean_pairwise_r": avg("C1_length", "correlation_pairwise_r"),
            },
            "C2_dc_only": {
                "mean_confound_d": avg("C2_dc_only", "mean_C2_dc_only_d"),
                "mean_ratio": avg("C2_dc_only", "ratio_confound_to_spectral"),
                "mean_pairwise_r": avg("C2_dc_only", "correlation_pairwise_r"),
            },
            "C3_scalar_surface": {
                "mean_confound_d": avg("C3_scalar_surface", "mean_C3_scalar_surface_d"),
                "mean_ratio": avg("C3_scalar_surface", "ratio_confound_to_spectral"),
                "mean_pairwise_r": avg("C3_scalar_surface", "correlation_pairwise_r"),
            },
            "C4_zscored_spectral": {
                "mean_confound_d": avg("C4_zscored_spectral", "mean_C4_zscored_spectral_d"),
                "mean_ratio": avg("C4_zscored_spectral", "ratio_confound_to_spectral"),
                "mean_pairwise_r": avg("C4_zscored_spectral", "correlation_pairwise_r"),
            },
        }
        all_results[signal] = {
            "summary": signal_summary,
            "per_model": per_model,
        }
        print()
        print(f"  CROSS-MODEL SUMMARY for {signal}:")
        print(f"    spectral  mean d = {signal_summary['mean_spectral_d_across_models']:.3f}")
        for c_label in ["C1_length", "C2_dc_only", "C3_scalar_surface", "C4_zscored_spectral"]:
            s = signal_summary[c_label]
            r_pw = s["mean_pairwise_r"]
            r_pw_str = f"{r_pw:+.2f}" if r_pw is not None else "n/a"
            print(
                f"    {c_label:<22} mean d = {s['mean_confound_d']:.3f}  "
                f"ratio={s['mean_ratio']:.2f}  pair_r={r_pw_str}"
            )
        print()

    # -------------------------------------------------------------
    # Verdict
    # -------------------------------------------------------------
    print("=" * 76)
    print("VERDICT")
    print("=" * 76)
    verdicts = {}
    for signal in SIGNAL_KEYS:
        s = all_results[signal]["summary"]
        ref_d = s["mean_spectral_d_across_models"] or 0.0
        verdicts[signal] = {}
        for c_label, desc in [
            ("C1_length",           "token-length alone"),
            ("C2_dc_only",          "trajectory mean alone"),
            ("C3_scalar_surface",   "trajectory (mean, std) alone"),
            ("C4_zscored_spectral", "per-trajectory z-scored spectrum"),
        ]:
            cs = s[c_label]
            conf_d = cs["mean_confound_d"] or 0.0
            # Heuristics:
            # - if confound mean d >= 0.80 * reference, confound LIKELY explains
            # - if confound mean d < 0.50 * reference, spectral has MORE info
            # - C4 is different: if zscored mean d ~= spectral, then shape is real
            if c_label == "C4_zscored_spectral":
                survives = conf_d >= 0.50 * ref_d
                print(
                    f"  [{signal:<12}] C4 z-scored spectrum:  "
                    f"{conf_d:.3f} vs ref {ref_d:.3f}  →  "
                    f"{'SURVIVES (shape is real)' if survives else 'FAILS (was magnitude)'}"
                )
                verdicts[signal][c_label] = {
                    "survives_shape": bool(survives),
                    "ratio": conf_d / ref_d if ref_d > 0 else None,
                }
            else:
                explained = conf_d >= 0.80 * ref_d
                ratio = conf_d / ref_d if ref_d > 0 else None
                print(
                    f"  [{signal:<12}] {c_label:<22} ({desc}):  "
                    f"{conf_d:.3f} vs ref {ref_d:.3f}  →  "
                    f"{'CONFOUND LIKELY EXPLAINS' if explained else 'spectral still larger'}"
                )
                verdicts[signal][c_label] = {
                    "explains": bool(explained),
                    "ratio": ratio,
                }
    print()

    # Save everything
    out_path = outdir / f"atlas_spectral_confounds_L{WINDOW_LEN}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "window_len": WINDOW_LEN,
            "signals": SIGNAL_KEYS,
            "categories": CATEGORIES,
            "per_signal": all_results,
            "verdicts": verdicts,
        }, f, indent=2, default=float)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
