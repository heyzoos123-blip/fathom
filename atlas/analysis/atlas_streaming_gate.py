# -*- coding: utf-8 -*-
"""
atlas_streaming_gate.py -- THE STREAMING GATE TEST.

Question: at what token position does an intra-flight classifier
reach gate-worthy accuracy, and does confidence at that point
become informative enough to drive an abort decision?

This is the experiment that decides the exact token threshold at
which the fathom agent's intra-flight gate fires.

Method:
    For each token position t in [1, 2, 3, 5, 7, 10, 15, 20, 25, 30],
    build a leave-one-model-out nearest-centroid classifier trained
    on features computed from tokens [0, t). The feature vector
    captures mean, std, early/late structure, and cumulative
    coherence-delta statistics of the trajectory SO FAR at each
    position.

    Report for each t:
      - cross-model accuracy vs TRUE category
      - agreement with the full-30-token classifier (the behavior proxy)
      - accuracy on the top-10% most confident predictions
      - per-category accuracy for the three categories fathom
        cares about: reasoning (routing), hallucination (gate),
        adversarial (gate)

    The gate token is the smallest t where:
      - overall accuracy >= 0.45 (2.7x chance), AND
      - top-10% confident accuracy >= 0.60, AND
      - hallucination OR adversarial per-cat accuracy >= 0.50

    If no t meets this bar, the gate must be driven by threshold
    rules on raw vital values rather than by a classifier, and
    we design the runtime accordingly.

Usage:
    python atlas/analysis/atlas_streaming_gate.py \
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

# Token positions to test (each is the RIGHT edge of the window [0, t))
TOKEN_CUTOFFS = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30]


def load_capture(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def features_up_to_token(probe, t_cutoff):
    """Rich feature vector using only tokens [0, t_cutoff)."""
    trajs = probe.get("trajectories", {})
    feats = []
    for s in SIGNALS:
        full = np.asarray(trajs.get(s, []), dtype=float)
        if len(full) == 0:
            feats.extend([0.0] * 6)
            continue
        t = full[:t_cutoff]
        if len(t) == 0:
            feats.extend([0.0] * 6)
            continue
        feats.append(float(t.mean()))
        feats.append(float(t.std(ddof=1)) if len(t) > 1 else 0.0)
        feats.append(float(t.min()))
        feats.append(float(t.max()))
        # early half and late half of the window
        mid = max(1, len(t) // 2)
        feats.append(float(t[:mid].mean()))
        feats.append(float(t[mid:].mean()) if len(t) > mid else float(t.mean()))
    return np.array(feats, dtype=float)


def zscore_fit(X):
    X = np.asarray(X, dtype=float)
    mu = X.mean(axis=0)
    sigma = X.std(axis=0, ddof=1)
    sigma = np.where(sigma > 1e-12, sigma, 1.0)
    return mu, sigma


def zscore_apply(X, mu, sigma):
    return (np.asarray(X, dtype=float) - mu) / sigma


def train_centroids(X_train, y_train):
    mu, sigma = zscore_fit(X_train)
    Z = zscore_apply(X_train, mu, sigma)
    centroids = {}
    for cat in CATEGORIES:
        idx = [i for i, l in enumerate(y_train) if l == cat]
        if idx:
            centroids[cat] = Z[idx].mean(axis=0)
    return centroids, mu, sigma


def predict_with_margin(centroids, mu, sigma, feat_vec):
    z = zscore_apply(feat_vec[None, :], mu, sigma)[0]
    dists = []
    for cat in CATEGORIES:
        c = centroids.get(cat)
        if c is None:
            continue
        dists.append((cat, float(np.linalg.norm(z - c))))
    dists.sort(key=lambda x: x[1])
    best_cat, best_d = dists[0]
    second_d = dists[1][1] if len(dists) > 1 else best_d
    return best_cat, (second_d - best_d)


def collect_all(captures_dir):
    captures_dir = Path(captures_dir)
    out = {}
    paths = sorted([
        p for p in captures_dir.glob("*.json")
        if "gpt-4o-mini" not in p.name.lower()
    ])
    for p in paths:
        cap = load_capture(p)
        model = cap.get("model", p.name)
        by_cat = {}
        for cat in CATEGORIES:
            probes = cap.get("probes", {}).get(cat, {})
            items = []
            for pid, probe in probes.items():
                if len(probe.get("trajectories", {}).get("entropy", [])) < 5:
                    continue
                items.append((probe, pid))
            by_cat[cat] = items
        out[model] = by_cat
    return out


def run_for_cutoff(data, t_cutoff):
    """LOO across models, classify each probe using only tokens [0, t)."""
    models = list(data.keys())
    records = []
    for held_out in models:
        X_tr, y_tr = [], []
        for m in models:
            if m == held_out:
                continue
            for cat in CATEGORIES:
                for probe, _ in data[m][cat]:
                    X_tr.append(features_up_to_token(probe, t_cutoff))
                    y_tr.append(cat)
        X_tr = np.array(X_tr)
        cent, mu, sig = train_centroids(X_tr, y_tr)
        for cat in CATEGORIES:
            for probe, pid in data[held_out][cat]:
                f = features_up_to_token(probe, t_cutoff)
                pred, margin = predict_with_margin(cent, mu, sig, f)
                records.append({
                    "model": held_out,
                    "probe_id": pid,
                    "true": cat,
                    "pred": pred,
                    "margin": margin,
                    "correct": pred == cat,
                })
    return records


def summarize_records(records, t_cutoff):
    n = len(records)
    overall_acc = sum(r["correct"] for r in records) / n

    # Per-category
    per_cat = {}
    for cat in CATEGORIES:
        cat_r = [r for r in records if r["true"] == cat]
        if cat_r:
            per_cat[cat] = sum(r["correct"] for r in cat_r) / len(cat_r)
        else:
            per_cat[cat] = 0.0

    # Top-10% confidence
    margins = np.array([r["margin"] for r in records])
    thresh = np.percentile(margins, 90)
    top10 = [r for r in records if r["margin"] >= thresh]
    top10_acc = (
        sum(r["correct"] for r in top10) / len(top10)
        if top10 else 0.0
    )

    # Top-25% for a softer cut
    thresh25 = np.percentile(margins, 75)
    top25 = [r for r in records if r["margin"] >= thresh25]
    top25_acc = (
        sum(r["correct"] for r in top25) / len(top25)
        if top25 else 0.0
    )

    return {
        "t_cutoff": t_cutoff,
        "n_records": n,
        "overall_accuracy": overall_acc,
        "top10_accuracy": top10_acc,
        "top25_accuracy": top25_acc,
        "per_category_recall": per_cat,
        "chance": 1.0 / 6,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures-dir", default="atlas/captures")
    ap.add_argument("--outdir", default="atlas/analysis")
    args = ap.parse_args()

    print("=" * 78)
    print("ATLAS STREAMING GATE TEST (cross-model LOO at each token cutoff)")
    print("=" * 78)
    print()
    data = collect_all(args.captures_dir)
    print(f"Models: {len(data)}")
    print(f"Categories: {CATEGORIES}")
    print(f"Token cutoffs: {TOKEN_CUTOFFS}")
    print(f"Chance: {1/6:.3f}")
    print()

    all_results = []
    header = (
        f"{'t':>4} {'n':>6} {'overall':>9} {'top25':>9} {'top10':>9}  "
        f"{'retr':>6} {'reas':>6} {'refu':>6} {'crea':>6} {'adve':>6} {'hall':>6}"
    )
    print(header)
    print("-" * len(header))
    for t in TOKEN_CUTOFFS:
        records = run_for_cutoff(data, t)
        summary = summarize_records(records, t)
        all_results.append(summary)
        pc = summary["per_category_recall"]
        line = (
            f"{t:>4} {summary['n_records']:>6} "
            f"{summary['overall_accuracy']:>9.3f} "
            f"{summary['top25_accuracy']:>9.3f} "
            f"{summary['top10_accuracy']:>9.3f}  "
            f"{pc['retrieval']:>6.3f} {pc['reasoning']:>6.3f} "
            f"{pc['refusal']:>6.3f} {pc['creative']:>6.3f} "
            f"{pc['adversarial']:>6.3f} {pc['hallucination']:>6.3f}"
        )
        print(line)
    print()

    # Pick the gate token
    print("=" * 78)
    print("GATE TOKEN SELECTION")
    print("=" * 78)
    gate_candidates = []
    for s in all_results:
        crit1 = s["overall_accuracy"] >= 0.45
        crit2 = s["top10_accuracy"] >= 0.60
        crit3 = (
            s["per_category_recall"]["hallucination"] >= 0.50
            or s["per_category_recall"]["adversarial"] >= 0.50
        )
        crit_all = crit1 and crit2 and crit3
        gate_candidates.append({
            "t": s["t_cutoff"],
            "overall_>=0.45": crit1,
            "top10_>=0.60": crit2,
            "hall_or_adv_>=0.50": crit3,
            "gate_worthy": crit_all,
        })
        marker = " ★" if crit_all else "  "
        print(
            f"  t={s['t_cutoff']:>3}: "
            f"overall={s['overall_accuracy']:.3f} [{'Y' if crit1 else 'N'}]  "
            f"top10={s['top10_accuracy']:.3f} [{'Y' if crit2 else 'N'}]  "
            f"hall={s['per_category_recall']['hallucination']:.3f} "
            f"adv={s['per_category_recall']['adversarial']:.3f} "
            f"[{'Y' if crit3 else 'N'}] {marker}"
        )
    print()
    gate_t = next(
        (g["t"] for g in gate_candidates if g["gate_worthy"]), None
    )
    if gate_t is not None:
        print(f"[GATE TOKEN FOUND] t = {gate_t}")
        print(f"  at t={gate_t}, the intra-flight classifier is")
        print(f"  gate-worthy: can drive abort decisions mid-stream.")
    else:
        print("[NO GATE TOKEN] no cutoff meets the criteria for a")
        print("  classifier-driven abort gate. The runtime must use")
        print("  threshold rules on raw vitals instead.")
    print()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / "atlas_streaming_gate.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "token_cutoffs": TOKEN_CUTOFFS,
            "results": all_results,
            "gate_candidates": gate_candidates,
            "selected_gate_token": gate_t,
        }, f, indent=2, default=float)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
