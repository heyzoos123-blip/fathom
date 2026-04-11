# -*- coding: utf-8 -*-
"""
make_spectral_figure.py -- Publication-grade figure for the spectral
hallucination signature finding.

Produces a 3-panel figure:
  (a) Gemma C_delta mean spectrum, correct vs hallucinated, shaded SEM
  (b) Llama  C_delta mean spectrum, correct vs hallucinated, shaded SEM
  (c) head-to-head signal comparison on Gemma: per-signal max|d|
      (C_delta vs entropy vs logprob vs top2_margin)

Uses the existing *_spectral.json and S_baselines_n200_spectral_controls.json
files produced by analyze_S_spectral.py and analyze_S_spectral_controls.py.
"""

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_spectral(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    root = Path(__file__).resolve().parent.parent
    tq = root / "truthfulqa_results"
    outdir = root / "figures"
    outdir.mkdir(parents=True, exist_ok=True)

    # These files are produced by running analyze_S_spectral.py with L=40
    # for each input trajectory file.
    gemma_path = tq / "S_baselines_n200_spectral.json"
    llama_path = tq / "S_truthfulqa_llama_n200_spectral.json"
    controls_path = tq / "S_baselines_n200_spectral_controls.json"

    for p in (gemma_path, llama_path, controls_path):
        if not p.exists():
            print(f"[MISSING] {p}", file=sys.stderr)
            print("Run analyze_S_spectral.py and analyze_S_spectral_controls.py first.",
                  file=sys.stderr)
            sys.exit(1)

    gemma = load_spectral(gemma_path)
    llama = load_spectral(llama_path)
    controls = load_spectral(controls_path)

    # ----- figure -----
    fig = plt.figure(figsize=(13, 4.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.9], wspace=0.35)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    for ax, data, title in [
        (axes[0], gemma, f"gemma-2-2b-it  (n={gemma['n_correct']}+{gemma['n_hallucinated']})"),
        (axes[1], llama, f"llama-3.2-1b-instruct  (n={llama['n_correct']}+{llama['n_hallucinated']})"),
    ]:
        mean_c = np.array(data["correct_mean_spectrum"])
        mean_h = np.array(data["halluc_mean_spectrum"])
        sem_c = np.array(data["correct_sem_spectrum"])
        sem_h = np.array(data["halluc_sem_spectrum"])
        L = data["window_len"]
        n_bins = len(mean_c)
        bins = np.arange(n_bins)

        # skip DC (bin 0)
        x = bins[1:]
        periods = L / x

        ax.plot(periods, mean_c[1:], label="correct",
                color="#1f77b4", linewidth=2)
        ax.fill_between(periods, mean_c[1:] - sem_c[1:], mean_c[1:] + sem_c[1:],
                        alpha=0.25, color="#1f77b4")
        ax.plot(periods, mean_h[1:], label="hallucinated",
                color="#d62728", linewidth=2)
        ax.fill_between(periods, mean_h[1:] - sem_h[1:], mean_h[1:] + sem_h[1:],
                        alpha=0.25, color="#d62728")

        # Mark significant bins with a star
        for b in data["per_bin_stats"]:
            if b["p"] < 0.05:
                p_period = b.get("period_tokens") or (L / b["bin"])
                y = max(mean_c[b["bin"]], mean_h[b["bin"]])
                ax.annotate(
                    "*",
                    xy=(p_period, y),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    fontsize=16,
                    color="black",
                )

        ax.set_xlabel("token period (L / k)")
        ax.set_ylabel("mean FFT power")
        ax.set_title(title, fontsize=11)
        ax.invert_xaxis()
        ax.legend(frameon=False, fontsize=9)
        ax.grid(alpha=0.3)

    # Panel (c): head-to-head comparison on Gemma
    ax = axes[2]
    sig_order = [
        ("c_delta_trajectory",     "C_delta (SAE)"),
        ("entropy_trajectory",     "logit entropy"),
        ("logprob_trajectory",     "neg log-prob"),
        ("top2_margin_trajectory", "top-2 margin"),
    ]
    max_ds = []
    global_ps = []
    n_sigs = []
    labels = []
    for key, label in sig_order:
        s = controls["signals"].get(key)
        if not s or s.get("skipped"):
            continue
        labels.append(label)
        max_ds.append(s["family_wise_perm"]["observed_max_abs_d"])
        global_ps.append(s["family_wise_perm"]["global_p"])
        n_sigs.append(s["n_sig_bins"])

    y_pos = np.arange(len(labels))
    colors = ["#ff7f0e" if p < 0.1 else "#808080" for p in global_ps]
    ax.barh(y_pos, max_ds, color=colors, alpha=0.9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    for i, (d, p, k) in enumerate(zip(max_ds, global_ps, n_sigs)):
        ax.text(d + 0.02, i,
                f"max|d|={d:.2f}, p={p:.3f}, {k} sig bins",
                va="center", fontsize=8)
    ax.set_xlim(0, max(max_ds) * 1.6)
    ax.set_xlabel("family-wise max | Cohen's d |")
    ax.set_title(f"head-to-head  (gemma, L={controls['window_len']})",
                 fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()

    fig.suptitle(
        "C_delta spectral structure discriminates correct vs. hallucinated  "
        "(*  $p<0.05$ per bin)",
        fontsize=12, y=1.03,
    )

    fig.tight_layout()
    png_path = outdir / "spectral_hallucination_signature.png"
    pdf_path = outdir / "spectral_hallucination_signature.pdf"
    fig.savefig(png_path, dpi=160, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {png_path}")
    print(f"[saved] {pdf_path}")


if __name__ == "__main__":
    main()
