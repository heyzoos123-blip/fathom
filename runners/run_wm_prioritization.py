# -*- coding: utf-8 -*-
"""
run_wm_prioritization.py -- Run the Weng-inspired working-memory
prioritization experiment on a transformer, using the Fathom
CoherenceSteererExt pipeline to capture per-token C_delta, entropy,
logprob, and top2_margin trajectories.

This script is the experiment runner for the pre-registration in
prereg/wm_prioritization_prereg.md. It is GPU-heavy (loads a
HuggingFace model + the Fathom SAE transcoder set) and should be
executed on a machine with the standard Fathom pipeline available.

Usage:
    python runners/run_wm_prioritization.py \
        --model google/gemma-2-2b-it \
        --prompts prompts/wm_prioritization_prompts.json \
        --out truthfulqa_results/wm_prioritization_gemma-2-2b-it.json \
        --max-tokens 40

Pre-registered invariants (do not override without a pre-reg amendment):
    max_tokens = 40
    temperature = 0  (greedy)
    the prompts file must be the committed pre-registered battery
    S_early window = tokens [0, 7)
    tau = 0.005

This script does NOT run the analysis. The analyzer is in
analysis/analyze_wm_prioritization.py and reads the JSON this script
writes.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
sys.path.insert(0, str(ROOT / "api"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-2-2b-it",
                    help="HuggingFace model ID. Primary: google/gemma-2-2b-it. "
                         "Replication: meta-llama/Llama-3.2-1B-Instruct.")
    ap.add_argument("--prompts", default="prompts/wm_prioritization_prompts.json",
                    help="Path to the pre-registered prompt battery.")
    ap.add_argument("--out", required=True,
                    help="Where to write the raw per-trial capture JSON.")
    ap.add_argument("--max-tokens", type=int, default=40,
                    help="Max generation length (LOCKED by pre-reg at 40).")
    args = ap.parse_args()

    # Load prompts
    prompts_path = Path(args.prompts)
    with open(prompts_path, "r", encoding="utf-8") as f:
        battery = json.load(f)

    assert battery["n_trials"] == 60, \
        f"Expected 60 trials, found {battery['n_trials']}"
    assert battery["n_short"] == 30 and battery["n_long"] == 30, \
        "Delay balance broken"
    assert battery["probe_target_counts"]["A"] == 30, \
        "Probe A count != 30"
    assert battery["probe_target_counts"]["B"] == 30, \
        "Probe B count != 30"

    print(f"Loaded {battery['n_trials']} trials from {prompts_path}")
    print(f"Model: {args.model}")
    print(f"Max tokens: {args.max_tokens}")
    print()

    # Load CoherenceSteererExt from the api/ package
    try:
        from coherence_steerer_ext import CoherenceSteererExt
    except ImportError as e:
        print(f"[FATAL] cannot import CoherenceSteererExt from api/: {e}")
        print("Check that api/coherence_steerer_ext.py is on sys.path.")
        sys.exit(1)

    print("Initializing CoherenceSteererExt... (this loads the model + SAEs)")
    t0 = time.time()
    steerer = CoherenceSteererExt(model_name=args.model)
    print(f"Model loaded in {time.time()-t0:.1f}s")
    print()

    # Run each trial
    results = []
    for i, trial in enumerate(battery["trials"]):
        t0 = time.time()
        print(f"[{i+1}/{len(battery['trials'])}] "
              f"pair={trial['pair_id']:>2d} "
              f"delay={trial['delay_condition']:<5} "
              f"probe={trial['probe_target']} "
              f"cat={trial['category']:<9}", end=" ", flush=True)

        try:
            gen = steerer.generate_with_entropy(
                trial["prompt"],
                max_tokens=args.max_tokens,
            )
            # Expect gen to be a dataclass-like object with trajectory attrs
            trial_out = dict(trial)
            trial_out["generation"] = {
                "text": getattr(gen, "text", ""),
                "total_tokens": getattr(gen, "total_tokens", 0),
                "c_delta_trajectory": list(getattr(gen, "c_delta_trajectory", [])),
                "entropy_trajectory": list(getattr(gen, "entropy_trajectory", [])),
                "logprob_trajectory": list(getattr(gen, "logprob_trajectory", [])),
                "top2_margin_trajectory": list(getattr(gen, "top2_margin_trajectory", [])),
                "tokens": list(getattr(gen, "tokens", [])),
                "generation_time_s": time.time() - t0,
            }
            results.append(trial_out)
            print(f"ok ({time.time()-t0:.1f}s, "
                  f"{trial_out['generation']['total_tokens']} tok)")
        except Exception as e:
            print(f"FAIL: {e}")
            trial_out = dict(trial)
            trial_out["generation"] = {
                "error": str(e),
                "generation_time_s": time.time() - t0,
            }
            results.append(trial_out)

    # Write output
    out = {
        "experiment": "wm_prioritization_in_silico_replication",
        "prereg":     "prereg/wm_prioritization_prereg.md",
        "model":      args.model,
        "prompts_file": str(prompts_path),
        "max_tokens": args.max_tokens,
        "n_trials":   len(results),
        "results":    results,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # Summary
    ok = sum(1 for r in results if "error" not in r["generation"])
    fail = sum(1 for r in results if "error" in r["generation"])
    print()
    print(f"Wrote {out_path}")
    print(f"  OK:   {ok} / {len(results)}")
    print(f"  FAIL: {fail} / {len(results)}")
    print()
    print("Next step: python analysis/analyze_wm_prioritization.py " + str(out_path))


if __name__ == "__main__":
    main()
