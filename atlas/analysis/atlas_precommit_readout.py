# -*- coding: utf-8 -*-
"""
atlas_precommit_readout.py -- THE CRITICAL TEST.

Can we read a prompt's cognitive category from cognitive-geometry
features extracted at TOKEN 0 alone?

Token 0 in the atlas captures is the model's commitment given only
the prompt. Nothing downstream has been generated yet. The residual
stream at that step is the prompt's full representation. Every feature
we extract from it (D, entropy, logprob, top2_margin) is a *pre-output*
cognitive state readout.

If token-0 features classify prompt category with above-chance
accuracy on HELD-OUT MODELS, then:

  1. The measurement works before the model commits.
  2. The measurement is architecturally universal (cross-model).
  3. The measurement is training-free (no probes, no fit per model).
  4. The measurement is real-time (1 forward pass, no FFT on trajectories).

This is the cleanest possible test of the pre-output cognitive state
hypothesis on data we already have on disk.

We use LEAVE-ONE-MODEL-OUT CROSS VALIDATION:

  for each held-out model m:
    1. compute token-0 features (D, entropy, logprob, top2_margin,
       plus early-mean, late-mean, std from the summaries if
       available) for all models except m
    2. compute the per-category centroid in feature space
    3. predict each held-out probe's category by nearest centroid
    4. report per-category accuracy + 6-class accuracy

Chance = 1/6 = 16.67%. Any result > ~30% is meaningful.
>50% is strong. >65% is a product.

The classifier uses NO TRAINING, just z-scored feature distances
from category centroids in leave-one-out fashion.

Usage:
    python atlas/analysis/atlas_precommit_readout.py \
        --captures-dir atlas/captures
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from sklearn.metrics import confusion_matrix

CATEGORIES = [
    "retrieval", "reasoning", "refusal",
    "creative", "adversarial", "hallucination",
]

# Features extracted at token 0 for every probe
SIGNALS = ["D", "entropy", "logprob", "top2_margin"]


def load_capture(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_token0_features(probe):
    """Return a flat feature vector at token 0 for one probe.

    We include token-0 values of all signals, plus the first-few-token
    means (up to token 3) to give a little context without losing
    the pre-commitment framing. Token 0 alone is still emphasized
    by being first in the vector.
    """
    trajs = probe.get("trajectories", {})
    feats = []
    # token 0 value per signal
    for s in SIGNALS:
        t = trajs.get(s, [])
        feats.append(t[0] if len(t) >= 1 else 0.0)
    # mean of tokens 0-2 per signal (pre-commitment window)
    for s in SIGNALS:
        t = trajs.get(s, [])
        if len(t) >= 3:
            feats.append(float(np.mean(t[:3])))
        elif t:
            feats.append(float(np.mean(t)))
        else:
            feats.append(0.0)
    return np.array(feats, dtype=float)


def collect_by_model_and_category(captures_dir):
    """Return dict: model_name -> {category -> list of (feature_vec, probe_id)}."""
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
                feats = extract_token0_features(probe)
                items.append((feats, pid))
            by_cat[cat] = items
        out[model] = by_cat
    return out


def zscore_fit(X):
    """Compute per-feature mean + std from training matrix X."""
    X = np.asarray(X, dtype=float)
    mu = X.mean(axis=0)
    sigma = X.std(axis=0, ddof=1)
    sigma = np.where(sigma > 1e-12, sigma, 1.0)
    return mu, sigma


def zscore_apply(X, mu, sigma):
    return (np.asarray(X, dtype=float) - mu) / sigma


def centroids_from_training(train_by_cat):
    """Return a centroid (in z-scored feature space) per category.

    train_by_cat: dict category -> list of feature vectors
    Also returns the z-score mu/sigma fit on the concatenation of all
    training data.
    """
    all_X = []
    labels = []
    for cat in CATEGORIES:
        for v in train_by_cat.get(cat, []):
            all_X.append(v)
            labels.append(cat)
    all_X = np.array(all_X, dtype=float)
    if len(all_X) < 2:
        return None, None, None
    mu, sigma = zscore_fit(all_X)
    Z = zscore_apply(all_X, mu, sigma)
    centroids = {}
    for cat in CATEGORIES:
        idx = [i for i, l in enumerate(labels) if l == cat]
        if idx:
            centroids[cat] = Z[idx].mean(axis=0)
    return centroids, mu, sigma


def classify(centroids, mu, sigma, feat_vec):
    """Return predicted category = nearest centroid (euclidean) in z-space."""
    z = zscore_apply(feat_vec[None, :], mu, sigma)[0]
    best = None
    best_d = float("inf")
    for cat in CATEGORIES:
        c = centroids.get(cat)
        if c is None:
            continue
        d = float(np.linalg.norm(z - c))
        if d < best_d:
            best_d = d
            best = cat
    return best, best_d


def run_loo(data_by_model):
    """Leave-one-model-out nearest-centroid classification.

    Returns: dict with per-model confusion matrices + aggregated metrics.
    """
    models = list(data_by_model.keys())
    results_per_model = {}
    all_true = []
    all_pred = []
    for held_out in models:
        # Build training set from all OTHER models
        train_by_cat = defaultdict(list)
        for m in models:
            if m == held_out:
                continue
            for cat in CATEGORIES:
                for feats, _ in data_by_model[m][cat]:
                    train_by_cat[cat].append(feats)
        centroids, mu, sigma = centroids_from_training(train_by_cat)
        if centroids is None:
            continue
        # Predict held-out probes
        y_true = []
        y_pred = []
        for cat in CATEGORIES:
            for feats, pid in data_by_model[held_out][cat]:
                pred, _ = classify(centroids, mu, sigma, feats)
                y_true.append(cat)
                y_pred.append(pred)
        acc = float(np.mean([t == p for t, p in zip(y_true, y_pred)]))
        cm = confusion_matrix(y_true, y_pred, labels=CATEGORIES)
        # Per-category recall
        per_cat_acc = {}
        for i, cat in enumerate(CATEGORIES):
            row = cm[i]
            denom = row.sum()
            per_cat_acc[cat] = float(row[i] / denom) if denom > 0 else 0.0
        results_per_model[held_out] = {
            "accuracy": acc,
            "per_category_recall": per_cat_acc,
            "confusion": cm.tolist(),
        }
        all_true.extend(y_true)
        all_pred.extend(y_pred)
    overall_acc = float(np.mean([t == p for t, p in zip(all_true, all_pred)]))
    overall_cm = confusion_matrix(all_true, all_pred, labels=CATEGORIES)
    per_cat_overall = {}
    for i, cat in enumerate(CATEGORIES):
        row = overall_cm[i]
        denom = row.sum()
        per_cat_overall[cat] = float(row[i] / denom) if denom > 0 else 0.0
    return {
        "overall_accuracy": overall_acc,
        "overall_per_category_recall": per_cat_overall,
        "overall_confusion": overall_cm.tolist(),
        "per_held_out_model": results_per_model,
        "n_models": len(models),
        "n_classes": len(CATEGORIES),
        "chance": 1.0 / len(CATEGORIES),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures-dir", default="atlas/captures")
    ap.add_argument("--outdir", default="atlas/analysis")
    args = ap.parse_args()

    print("=" * 76)
    print("ATLAS PRE-COMMITMENT READOUT (token-0 features, LOO cross-model)")
    print("=" * 76)
    print()

    data = collect_by_model_and_category(args.captures_dir)
    print(f"Models: {len(data)}")
    print(f"Categories: {CATEGORIES}")
    print(f"Features: token-0 D/entropy/logprob/top2_margin + first-3-token means")
    print(f"Classifier: leave-one-model-out nearest centroid (z-scored, no training)")
    print(f"Chance: {1/6:.3f}")
    print()

    results = run_loo(data)

    print(f"OVERALL CROSS-MODEL ACCURACY: {results['overall_accuracy']:.3f}")
    print(f"  chance: {results['chance']:.3f}")
    print(f"  n_classes: {results['n_classes']}")
    print(f"  lift over chance: {results['overall_accuracy'] - results['chance']:+.3f}")
    print()

    print("Per-category recall (overall):")
    for cat in CATEGORIES:
        r = results["overall_per_category_recall"][cat]
        print(f"  {cat:<15} {r:.3f}")
    print()

    print("Per-held-out-model accuracy:")
    for m, r in results["per_held_out_model"].items():
        print(f"  {m:<38} acc = {r['accuracy']:.3f}")
    print()

    print("Overall confusion matrix (rows=true, cols=pred):")
    cm = np.array(results["overall_confusion"])
    header = "          " + " ".join(f"{c[:8]:>8}" for c in CATEGORIES)
    print(header)
    for i, cat in enumerate(CATEGORIES):
        row = " ".join(f"{v:>8d}" for v in cm[i])
        print(f"  {cat[:8]:<8}" + row)
    print()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / "atlas_precommit_readout.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
