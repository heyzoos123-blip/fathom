# -*- coding: utf-8 -*-
"""
atlas_spectral_categories.py -- Cognitive state fingerprinting via
spectral analysis of atlas trajectory data.

The REFRAME: Fathom is not a hallucination detector. It is a
measurement instrument for transformer cognitive state. Hallucination
is ONE binary split on ONE axis. The atlas v0.3 data contains 12
open-weight models captured on a hand-curated probe set of 90 prompts
across 6 categories (retrieval, reasoning, refusal, creative,
adversarial, hallucination), with per-token D / entropy / logprob /
top2_margin trajectories at 30 tokens each.

This script computes the spectral fingerprint of each
(model, category, signal) combination, then asks:

  Q1  Do categories separate in spectral space within a single model?
      (within-model cognitive state classification)

  Q2  Are category spectral signatures universal across architectures?
      (cross-model cognitive geometry)

  Q3  Which signal (D, entropy, logprob, top2_margin) gives the
      cleanest category separation?
      (head-to-head at the spectral level, not pointwise)

  Q4  For any given category, which token-period carries the most
      category-discriminative power? (band localization)

The result is a cognitive state fingerprint matrix that has never
been computed before. It is bigger than hallucination detection
because:
  - it operates on 6 categories, not a binary
  - the category labels are design facts, not post-hoc classifications
  - it uses architecturally universal signals (D is SAE-free)
  - it reveals a FULL cognitive state space, not a single detector

Usage:
    python atlas/analysis/atlas_spectral_categories.py \
        --captures-dir atlas/captures \
        --outdir atlas/analysis \
        [--window-len 20]
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


# LOCKED parameters for this analysis
# ----------------------------------------
# Atlas captures are 30 tokens per probe (except creative which is
# sometimes shorter). We use L=20 so every probe contributes a
# spectrum of the same length, and no creative probes drop out.
# Bins: k in [1, L/2+1] = 11 frequency bins after DC removal.
WINDOW_LEN = 20
MIN_TOKENS = 20
WINDOW_TYPE = "hann"

# Signals in each probe's trajectories dict
SIGNAL_KEYS = ["D", "entropy", "logprob", "top2_margin"]

CATEGORIES = [
    "retrieval",
    "reasoning",
    "refusal",
    "creative",
    "adversarial",
    "hallucination",
]


def load_capture(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_spectra(capture, signal):
    """Return dict: category -> list of per-probe spectra (FFT power).

    Each spectrum is a length-(L//2 + 1) vector.
    """
    out = {}
    for cat in CATEGORIES:
        cat_specs = []
        probes = capture.get("probes", {}).get(cat, {})
        for probe_id, probe in probes.items():
            traj = probe.get("trajectories", {}).get(signal)
            if traj is None or len(traj) < MIN_TOKENS:
                continue
            spec = compute_spectrum(traj, WINDOW_LEN, window=WINDOW_TYPE)
            if spec is None:
                continue
            cat_specs.append(np.array(spec["fft_power"]))
        out[cat] = cat_specs
    return out


def cohen_d(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) if len(a) > 1 and len(b) > 1 else 0.0
    return (b.mean() - a.mean()) / pooled if pooled > 0 else 0.0


def pairwise_category_separability(cat_specs, n_bins):
    """For each pair of categories, for each bin, compute Cohen's d.
    Return the maximum |d| across bins per pair, and the bin where
    that max was achieved.

    cat_specs: dict category -> list of spectrum arrays
    """
    out = []
    cats = list(cat_specs.keys())
    for c1, c2 in combinations(cats, 2):
        s1 = cat_specs[c1]
        s2 = cat_specs[c2]
        if len(s1) < 3 or len(s2) < 3:
            continue
        mat1 = np.array(s1)
        mat2 = np.array(s2)
        max_d = 0.0
        best_bin = -1
        for k in range(1, n_bins):  # skip DC
            d = cohen_d(mat1[:, k], mat2[:, k])
            if abs(d) > abs(max_d):
                max_d = d
                best_bin = k
        out.append({
            "cat_a": c1,
            "cat_b": c2,
            "max_abs_d": float(abs(max_d)),
            "signed_d": float(max_d),
            "best_bin": best_bin,
            "best_period_tokens": float(WINDOW_LEN / best_bin) if best_bin > 0 else 0.0,
            "n_a": int(len(s1)),
            "n_b": int(len(s2)),
        })
    return out


def silhouette_like_score(cat_specs, n_bins):
    """Compute a per-category separability score by measuring how
    close each probe's spectrum is to its own category centroid vs
    the nearest other-category centroid.

    Returns: dict cat -> average "silhouette" score in [-1, 1].
    Skips DC bin.
    """
    cats = list(cat_specs.keys())
    centroids = {}
    for c in cats:
        if len(cat_specs[c]) < 2:
            centroids[c] = None
            continue
        centroids[c] = np.mean(np.array(cat_specs[c]), axis=0)

    scores_by_cat = {c: [] for c in cats}
    for c in cats:
        own_centroid = centroids[c]
        if own_centroid is None:
            continue
        other_centroids = {oc: v for oc, v in centroids.items()
                           if oc != c and v is not None}
        for spec in cat_specs[c]:
            spec = np.asarray(spec)
            own_dist = np.linalg.norm(spec[1:n_bins] - own_centroid[1:n_bins])
            nearest_other_dist = min(
                np.linalg.norm(spec[1:n_bins] - v[1:n_bins])
                for v in other_centroids.values()
            )
            denom = max(own_dist, nearest_other_dist)
            if denom > 0:
                scores_by_cat[c].append(
                    (nearest_other_dist - own_dist) / denom
                )
    return {c: float(np.mean(v)) if v else None for c, v in scores_by_cat.items()}


def analyze_one_model(capture_path, signal):
    capture = load_capture(capture_path)
    model = capture.get("model", os.path.basename(capture_path))
    cat_specs = extract_spectra(capture, signal)
    n_bins = WINDOW_LEN // 2 + 1

    # Per-category n probes
    n_by_cat = {c: len(s) for c, s in cat_specs.items()}

    # Pairwise separability
    pairwise = pairwise_category_separability(cat_specs, n_bins)

    # Silhouette-like score per category
    silh = silhouette_like_score(cat_specs, n_bins)

    # Mean spectrum per category (for figure generation downstream)
    mean_by_cat = {
        c: (np.mean(np.array(s), axis=0).tolist() if len(s) >= 2 else None)
        for c, s in cat_specs.items()
    }
    sem_by_cat = {
        c: ((np.std(np.array(s), axis=0, ddof=1) / np.sqrt(len(s))).tolist()
            if len(s) >= 2 else None)
        for c, s in cat_specs.items()
    }

    return {
        "model": model,
        "signal": signal,
        "window_len": WINDOW_LEN,
        "min_tokens": MIN_TOKENS,
        "n_bins": n_bins,
        "n_by_category": n_by_cat,
        "pairwise_separability": pairwise,
        "silhouette_by_category": silh,
        "mean_spectrum_by_category": mean_by_cat,
        "sem_spectrum_by_category": sem_by_cat,
    }


def main():
    global WINDOW_LEN, MIN_TOKENS
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures-dir", default="atlas/captures")
    ap.add_argument("--outdir", default="atlas/analysis")
    ap.add_argument("--window-len", type=int, default=WINDOW_LEN)
    args = ap.parse_args()

    WINDOW_LEN = args.window_len
    MIN_TOKENS = args.window_len

    captures_dir = Path(args.captures_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Collect all open-weight captures (skip the GPT-4o-mini API capture
    # since that has only logprob, not the same signal set)
    capture_paths = sorted([
        p for p in captures_dir.glob("*.json")
        if "gpt-4o-mini" not in p.name.lower()
    ])
    print("=" * 76)
    print("ATLAS SPECTRAL CATEGORY FINGERPRINT")
    print("=" * 76)
    print(f"Captures dir: {captures_dir}")
    print(f"Found {len(capture_paths)} open-weight captures:")
    for p in capture_paths:
        print(f"  - {p.name}")
    print(f"Window L: {WINDOW_LEN} (min_tokens={MIN_TOKENS})")
    print(f"Signals:  {SIGNAL_KEYS}")
    print(f"Categories: {CATEGORIES}")
    print()

    all_results = {}
    for signal in SIGNAL_KEYS:
        print("-" * 76)
        print(f"SIGNAL: {signal}")
        print("-" * 76)
        signal_results = []
        for cap_path in capture_paths:
            try:
                result = analyze_one_model(cap_path, signal)
            except Exception as e:
                print(f"  [FAIL] {cap_path.name}: {e}")
                continue
            signal_results.append(result)
            # Compact print
            pairs = result["pairwise_separability"]
            best_pair = max(pairs, key=lambda x: x["max_abs_d"]) if pairs else None
            mean_silh = np.mean([
                v for v in result["silhouette_by_category"].values()
                if v is not None
            ]) if result["silhouette_by_category"] else 0.0
            n_cat = result["n_by_category"]
            if best_pair:
                print(
                    f"  {result['model']:<38} "
                    f"best_pair: {best_pair['cat_a']} vs {best_pair['cat_b']} "
                    f"(|d|={best_pair['max_abs_d']:.2f} @ period "
                    f"{best_pair['best_period_tokens']:.1f}) "
                    f"mean_silh={mean_silh:+.2f}"
                )
        all_results[signal] = signal_results
        print()

    # --------------------------------------------------------------
    # Aggregate across models: which pairwise separations are UNIVERSAL?
    # --------------------------------------------------------------
    print("=" * 76)
    print("CROSS-MODEL AGGREGATE: which category pairs separate universally?")
    print("=" * 76)
    universal_summary = {}
    for signal in SIGNAL_KEYS:
        print(f"\n[{signal}]")
        signal_results = all_results[signal]
        # For each pair (c1, c2), collect max_abs_d across models
        pair_max_ds = {}
        pair_directions = {}
        for result in signal_results:
            for p in result["pairwise_separability"]:
                key = (p["cat_a"], p["cat_b"])
                pair_max_ds.setdefault(key, []).append(p["max_abs_d"])
                pair_directions.setdefault(key, []).append(
                    np.sign(p["signed_d"])
                )
        # Rank pairs by mean max_abs_d
        ranked = sorted(
            pair_max_ds.items(),
            key=lambda kv: -np.mean(kv[1]),
        )
        print(f"{'pair':<35} {'mean_|d|':>10} {'min_|d|':>10} "
              f"{'n_models':>10} {'direction':>12}")
        for key, ds in ranked[:8]:
            direction_consistency = (
                np.mean(pair_directions[key]) if pair_directions[key] else 0.0
            )
            print(f"{key[0]:<15} vs {key[1]:<15} "
                  f"{np.mean(ds):>10.3f} {min(ds):>10.3f} "
                  f"{len(ds):>10d} "
                  f"{direction_consistency:>+12.2f}")
        universal_summary[signal] = {
            str(k): {
                "mean_max_abs_d": float(np.mean(v)),
                "min_max_abs_d": float(min(v)),
                "max_max_abs_d": float(max(v)),
                "n_models": len(v),
                "direction_consistency": float(np.mean(pair_directions[k])),
            }
            for k, v in pair_max_ds.items()
        }

    # --------------------------------------------------------------
    # Head-to-head: which signal gives the cleanest separation overall?
    # --------------------------------------------------------------
    print()
    print("=" * 76)
    print("HEAD-TO-HEAD: best signal for cognitive state fingerprinting")
    print("=" * 76)
    print(f"{'signal':<15} {'mean_silh':>12} {'mean_best_|d|':>16} "
          f"{'n_models':>10}")
    head_to_head = {}
    for signal in SIGNAL_KEYS:
        signal_results = all_results[signal]
        silhs = []
        best_ds = []
        for result in signal_results:
            s = [v for v in result["silhouette_by_category"].values()
                 if v is not None]
            if s:
                silhs.append(np.mean(s))
            pairs = result["pairwise_separability"]
            if pairs:
                best_ds.append(max(p["max_abs_d"] for p in pairs))
        if silhs and best_ds:
            print(f"{signal:<15} {np.mean(silhs):>+12.3f} "
                  f"{np.mean(best_ds):>16.3f} {len(silhs):>10d}")
            head_to_head[signal] = {
                "mean_silhouette": float(np.mean(silhs)),
                "mean_best_abs_d": float(np.mean(best_ds)),
                "n_models": len(silhs),
            }
    print()

    # Save full results
    out_path = outdir / f"atlas_spectral_categories_L{WINDOW_LEN}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "window_len": WINDOW_LEN,
            "min_tokens": MIN_TOKENS,
            "signals": SIGNAL_KEYS,
            "categories": CATEGORIES,
            "per_model_results": all_results,
            "cross_model_pair_summary": universal_summary,
            "head_to_head_signals": head_to_head,
        }, f, indent=2, default=float)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
