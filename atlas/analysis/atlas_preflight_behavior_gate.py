# -*- coding: utf-8 -*-
"""
atlas_preflight_behavior_gate.py -- THE GATE TEST.

Does a confident token-0 pre-flight reading predict the ACTUAL
BEHAVIOR of the model over the full generation, or only the design
label of the prompt?

This is the test that decides whether fathom-agent is a GATING
architecture (can refuse / reroute / abort before generation) or a
REPORTING architecture (emits vitals alongside output). Both are
valuable; the first is dramatically more valuable.

Method:
    Atlas data has, for every (model, category, probe):
      - a TRUE category label   (design fact)
      - a 30-token trajectory   (D / entropy / logprob / top2_margin)
      - the generated TEXT      (what the model actually did)

    For each probe, compute two things:
      1. PRE-FLIGHT PREDICTION: what a token-0 nearest-centroid
         classifier (trained on all other models, leave-one-out)
         predicts the category to be, from token-0 features alone.
      2. FULL-GENERATION BEHAVIOR: what a richer classifier trained
         on the FULL 30-token trajectory feature vector (means,
         stds, early-vs-late contrasts) predicts the category to
         be, using full trajectory information. This is a proxy
         for the model's actual behavior because it is based on
         what the model did, not what the prompt looked like.

    Then ask:
      * AGREEMENT RATE: how often do pre-flight and full-generation
        agree with each other?
      * BEHAVIOR-CONDITIONED AGREEMENT: when pre-flight is CONFIDENT
        (nearest-centroid margin large), does that predict full-
        generation agreement better than random chance?
      * GATE POWER: if we gate only on high-confidence pre-flight
        predictions (top X% margin), what's the accuracy of
        pre-flight vs full-generation behavior on that subset?

    If high-confidence pre-flight predictions agree with full-
    generation behavior well above chance, pre-flight is a real
    behavior predictor and the GATE architecture is viable.

    If pre-flight correlates with the prompt label but NOT with
    full-generation behavior, pre-flight only reads prompt surface
    and we pivot to the REPORTING architecture.

Usage:
    python atlas/analysis/atlas_preflight_behavior_gate.py \
        --captures-dir atlas/captures
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

CATEGORIES = [
    "retrieval", "reasoning", "refusal",
    "creative", "adversarial", "hallucination",
]

SIGNALS = ["D", "entropy", "logprob", "top2_margin"]


# =================================================================
# Feature extractors
# =================================================================

def extract_token0_features(probe):
    """Pre-flight: token-0 values + mean of tokens 0-2."""
    trajs = probe.get("trajectories", {})
    feats = []
    for s in SIGNALS:
        t = trajs.get(s, [])
        feats.append(t[0] if len(t) >= 1 else 0.0)
    for s in SIGNALS:
        t = trajs.get(s, [])
        if len(t) >= 3:
            feats.append(float(np.mean(t[:3])))
        elif t:
            feats.append(float(np.mean(t)))
        else:
            feats.append(0.0)
    return np.array(feats, dtype=float)


def extract_full_trajectory_features(probe):
    """Full generation: rich summary of the full trajectory.

    This is the "behavior" feature set — what the model actually
    DID across its whole generation. Per-signal: mean, std,
    early-mean (tokens 0-9), late-mean (tokens 15+), early-late
    delta, min, max.
    """
    trajs = probe.get("trajectories", {})
    feats = []
    for s in SIGNALS:
        t = np.asarray(trajs.get(s, []), dtype=float)
        if len(t) == 0:
            feats.extend([0.0] * 7)
            continue
        early = t[:10] if len(t) >= 10 else t[:max(1, len(t)//2)]
        late = t[15:] if len(t) >= 16 else t[max(1, len(t)//2):]
        feats.extend([
            float(t.mean()),
            float(t.std(ddof=1)) if len(t) > 1 else 0.0,
            float(early.mean()),
            float(late.mean()) if len(late) > 0 else float(t.mean()),
            float(late.mean() - early.mean()) if len(late) > 0 else 0.0,
            float(t.min()),
            float(t.max()),
        ])
    return np.array(feats, dtype=float)


# =================================================================
# Classifier helpers
# =================================================================

def zscore_fit(X):
    X = np.asarray(X, dtype=float)
    mu = X.mean(axis=0)
    sigma = X.std(axis=0, ddof=1)
    sigma = np.where(sigma > 1e-12, sigma, 1.0)
    return mu, sigma


def zscore_apply(X, mu, sigma):
    return (np.asarray(X, dtype=float) - mu) / sigma


def train_centroids(X_train, y_train):
    """Fit z-score + per-category centroids in z-space.

    Returns (centroids_dict, mu, sigma)."""
    mu, sigma = zscore_fit(X_train)
    Z = zscore_apply(X_train, mu, sigma)
    centroids = {}
    for cat in CATEGORIES:
        idx = [i for i, l in enumerate(y_train) if l == cat]
        if idx:
            centroids[cat] = Z[idx].mean(axis=0)
    return centroids, mu, sigma


def predict_with_margin(centroids, mu, sigma, feat_vec):
    """Return (predicted_category, margin) where margin is the
    z-space distance to the 2nd-nearest centroid minus the z-space
    distance to the nearest. Larger margin = more confident."""
    z = zscore_apply(feat_vec[None, :], mu, sigma)[0]
    dists = []
    for cat in CATEGORIES:
        c = centroids.get(cat)
        if c is None:
            dists.append((cat, float("inf")))
            continue
        d = float(np.linalg.norm(z - c))
        dists.append((cat, d))
    dists.sort(key=lambda x: x[1])
    best_cat, best_d = dists[0]
    second_d = dists[1][1] if len(dists) > 1 else best_d
    margin = second_d - best_d
    return best_cat, margin


# =================================================================
# Data collection
# =================================================================

def load_capture(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_all(captures_dir):
    """Return dict: model_name -> {category -> list of
    (token0_feats, full_feats, probe_id, text)}
    """
    captures_dir = Path(captures_dir)
    out = {}
    paths = sorted([
        p for p in captures_dir.glob("*.json")
        if "gpt-4o-mini" not in p.name.lower()
    ])
    for p in paths:
        capture = load_capture(p)
        model = capture.get("model", p.name)
        by_cat = {}
        for cat in CATEGORIES:
            probes = capture.get("probes", {}).get(cat, {})
            items = []
            for pid, probe in probes.items():
                if len(probe.get("trajectories", {}).get("entropy", [])) < 10:
                    continue
                items.append((
                    extract_token0_features(probe),
                    extract_full_trajectory_features(probe),
                    pid,
                    probe.get("generated", ""),
                ))
            by_cat[cat] = items
        out[model] = by_cat
    return out


# =================================================================
# Main experiment
# =================================================================

def run_leave_one_out(data):
    """For each held-out model, train two classifiers:
      - PRE  on token-0 features from all other models
      - FULL on full-trajectory features from all other models
    Then predict each held-out probe with BOTH classifiers.
    Return per-probe records with (true, pre_pred, pre_margin,
    full_pred, full_margin).
    """
    models = list(data.keys())
    records = []

    for held_out in models:
        # Build train sets from all other models
        X_pre_tr, X_full_tr, y_tr = [], [], []
        for m in models:
            if m == held_out:
                continue
            for cat in CATEGORIES:
                for t0, ft, pid, text in data[m][cat]:
                    X_pre_tr.append(t0)
                    X_full_tr.append(ft)
                    y_tr.append(cat)
        X_pre_tr = np.array(X_pre_tr)
        X_full_tr = np.array(X_full_tr)

        pre_cent, pre_mu, pre_sig = train_centroids(X_pre_tr, y_tr)
        full_cent, full_mu, full_sig = train_centroids(X_full_tr, y_tr)

        for cat in CATEGORIES:
            for t0, ft, pid, text in data[held_out][cat]:
                pre_pred, pre_margin = predict_with_margin(
                    pre_cent, pre_mu, pre_sig, t0
                )
                full_pred, full_margin = predict_with_margin(
                    full_cent, full_mu, full_sig, ft
                )
                records.append({
                    "held_out_model": held_out,
                    "probe_id": pid,
                    "true_category": cat,
                    "pre_pred": pre_pred,
                    "pre_margin": pre_margin,
                    "full_pred": full_pred,
                    "full_margin": full_margin,
                    "pre_correct": pre_pred == cat,
                    "full_correct": full_pred == cat,
                    "pre_full_agree": pre_pred == full_pred,
                })
    return records


def analyze_gate_power(records):
    """Analyze the key question: when pre-flight is confident, does
    it predict full-generation behavior?

    We measure:
      - overall agreement rate between pre-flight and full-gen
      - agreement rate at different pre-flight confidence percentiles
      - pre-flight accuracy vs TRUE label, gated by confidence
      - full-gen accuracy vs TRUE label (the upper bound for what a
        trajectory-based approach can do on this data)
    """
    n = len(records)
    pre_accs = sum(r["pre_correct"] for r in records) / n
    full_accs = sum(r["full_correct"] for r in records) / n
    agree_rate = sum(r["pre_full_agree"] for r in records) / n

    # Bin by pre-flight confidence percentile
    margins = np.array([r["pre_margin"] for r in records])
    percentiles = [25, 50, 75, 90]
    gated_stats = []
    for pct in percentiles:
        thresh = np.percentile(margins, pct)
        subset = [r for r in records if r["pre_margin"] >= thresh]
        if not subset:
            continue
        gated_stats.append({
            "percentile": pct,
            "threshold": float(thresh),
            "n_subset": len(subset),
            "pct_of_total": len(subset) / n,
            "pre_acc_vs_true": sum(r["pre_correct"] for r in subset) / len(subset),
            "full_acc_vs_true": sum(r["full_correct"] for r in subset) / len(subset),
            "agree_rate": sum(r["pre_full_agree"] for r in subset) / len(subset),
        })

    # Per-category breakdown
    per_cat = {}
    for cat in CATEGORIES:
        cat_records = [r for r in records if r["true_category"] == cat]
        if not cat_records:
            continue
        per_cat[cat] = {
            "n": len(cat_records),
            "pre_acc": sum(r["pre_correct"] for r in cat_records) / len(cat_records),
            "full_acc": sum(r["full_correct"] for r in cat_records) / len(cat_records),
            "agree_rate": sum(r["pre_full_agree"] for r in cat_records) / len(cat_records),
        }

    # Conditional: given pre-flight says X with confidence, what does
    # full-gen actually say?
    cond = {}
    for pred_cat in CATEGORIES:
        subset = [r for r in records if r["pre_pred"] == pred_cat]
        if not subset:
            continue
        dist = defaultdict(int)
        for r in subset:
            dist[r["full_pred"]] += 1
        total = len(subset)
        cond[pred_cat] = {
            "n_pre_predictions": total,
            "full_distribution": {k: v / total for k, v in dist.items()},
            "agreement": dist[pred_cat] / total,
        }

    return {
        "n_records": n,
        "pre_accuracy_vs_true": pre_accs,
        "full_accuracy_vs_true": full_accs,
        "pre_full_agreement_rate": agree_rate,
        "gated_by_confidence": gated_stats,
        "per_category": per_cat,
        "conditional_on_pre_prediction": cond,
        "chance": 1.0 / 6,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures-dir", default="atlas/captures")
    ap.add_argument("--outdir", default="atlas/analysis")
    args = ap.parse_args()

    print("=" * 76)
    print("PRE-FLIGHT BEHAVIOR GATE TEST")
    print("=" * 76)
    print()
    print("Question: does a confident token-0 prediction agree with")
    print("what the full-trajectory classifier says the model DID?")
    print()

    data = collect_all(args.captures_dir)
    print(f"Models: {len(data)}")
    print(f"Categories: {CATEGORIES}")
    print()

    records = run_leave_one_out(data)
    print(f"Total leave-one-model-out records: {len(records)}")
    print()

    results = analyze_gate_power(records)

    print("OVERALL METRICS")
    print("-" * 76)
    print(f"  pre-flight accuracy vs TRUE label:    {results['pre_accuracy_vs_true']:.3f}")
    print(f"  full-gen accuracy vs TRUE label:      {results['full_accuracy_vs_true']:.3f}")
    print(f"  pre-flight <-> full-gen agreement:    {results['pre_full_agreement_rate']:.3f}")
    print(f"  chance baseline:                      {results['chance']:.3f}")
    print()

    print("GATED BY PRE-FLIGHT CONFIDENCE PERCENTILE")
    print("-" * 76)
    print(f"{'pct':>4} {'thresh':>9} {'n':>6} {'pct_all':>9} "
          f"{'pre_acc':>10} {'full_acc':>10} {'agree':>10}")
    for g in results["gated_by_confidence"]:
        print(
            f"{g['percentile']:>4d} {g['threshold']:>9.4f} "
            f"{g['n_subset']:>6d} {g['pct_of_total']:>9.2f} "
            f"{g['pre_acc_vs_true']:>10.3f} "
            f"{g['full_acc_vs_true']:>10.3f} "
            f"{g['agree_rate']:>10.3f}"
        )
    print()

    print("PER-CATEGORY")
    print("-" * 76)
    print(f"{'category':<15} {'n':>5} {'pre_acc':>10} {'full_acc':>10} {'agree':>10}")
    for cat in CATEGORIES:
        pc = results["per_category"].get(cat)
        if not pc:
            continue
        print(
            f"{cat:<15} {pc['n']:>5d} "
            f"{pc['pre_acc']:>10.3f} {pc['full_acc']:>10.3f} {pc['agree_rate']:>10.3f}"
        )
    print()

    print("CONDITIONAL: when pre-flight predicts X, what does full-gen say?")
    print("-" * 76)
    for pred_cat in CATEGORIES:
        cond = results["conditional_on_pre_prediction"].get(pred_cat)
        if not cond:
            continue
        print(f"  pre={pred_cat:<15} (n={cond['n_pre_predictions']:>3d})")
        dist = cond["full_distribution"]
        top = sorted(dist.items(), key=lambda x: -x[1])[:3]
        for full_cat, frac in top:
            marker = "  *" if full_cat == pred_cat else "   "
            print(f"     {marker} full={full_cat:<15} {frac:.2f}")
    print()

    # --- Headline verdict ---
    print("=" * 76)
    print("VERDICT")
    print("=" * 76)
    gate_at_90 = [g for g in results["gated_by_confidence"] if g["percentile"] == 90]
    gate_at_75 = [g for g in results["gated_by_confidence"] if g["percentile"] == 75]
    if gate_at_90:
        g = gate_at_90[0]
        print(f"Top 10% confident pre-flight predictions:")
        print(f"  agree with full-gen behavior: {g['agree_rate']:.3f}")
        print(f"  accurate vs true label:       {g['pre_acc_vs_true']:.3f}")
    if gate_at_75:
        g = gate_at_75[0]
        print(f"Top 25% confident pre-flight predictions:")
        print(f"  agree with full-gen behavior: {g['agree_rate']:.3f}")
        print(f"  accurate vs true label:       {g['pre_acc_vs_true']:.3f}")
    print()
    print("INTERPRETATION")
    print("  If high-confidence agreement >> chance (0.167), pre-flight")
    print("  predicts what the model is about to DO, not just what the")
    print("  prompt looks like. The GATE architecture is viable.")
    print()

    # Save
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / "atlas_preflight_behavior_gate.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
