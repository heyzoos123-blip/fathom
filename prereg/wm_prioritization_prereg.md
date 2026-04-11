# Pre-registration: In-silico working-memory prioritization on transformer C_delta trajectories

**Status:** SEALED draft. Will be committed to git **before** any
`run_wm_prioritization.py` invocation that writes real data.
**Authors:** Alexander Rodabaugh
**Project:** Fathom / sae-reasoning-depth
**Date drafted:** 2026-04-11
**Biological parallel (motivation):**
[Weng, Borst & Akyurek (2026)](https://doi.org/10.1162/imag.a.1199),
*Sustained alpha oscillations serve attentional prioritization in
working memory, not maintenance*, Imaging Neuroscience.
**Related findings:**
- `findings/findings_2026_04_11_alpha_bridge.md`
- `findings/findings_2026_04_11_spectral_pilot.md`
- `findings/findings_2026_04_11_spectral_sensitivity_v22.md`
**Related paper:** Fathom v15, `10.5281/zenodo.19504993`
(concept `10.5281/zenodo.19326174`)

---

## Background

Weng et al. (2026) showed in a human two-item working-memory paradigm
that sustained alpha-band EEG power tracks only the **prioritized**
item (the one tested first in the trial) across both short (1 s) and
long (3 s) delays. The **deprioritized** item transitions to an
"activity-quiescent state" that is not indexed by sustained alpha --
the item is still in memory (it can be retrieved if probed) but it
is not actively represented in alpha-band activity. The dissociation
is between **active prioritization** (alpha-indexed) and **passive
storage** (not alpha-indexed).

Fathom's `S_early` commitment intensity is structurally analogous: it
is an IPR-derived localization measure of commitment events on the
early generation window (tokens 0-7), and it is orthogonal to the
scalar depth measure `K` in the same way Weng's sustained alpha is
orthogonal to contralateral delay activity. We have independent
evidence that `S_early` operates on a different dimension than
standard uncertainty (TruthfulQA pointwise head-to-head: `S_early`
AUC 0.663 vs logit entropy AUC 0.607, `r = -0.17` between the two
signals).

This pre-registration tests whether `S_early` can be validated as the
transformer-side analog of sustained alpha prioritization by
replicating the Weng dissociation in-silico: a transformer committing
to the prioritized item should show **elevated `S_early` on
generations triggered by the prioritized probe** and near-baseline
`S_early` on generations triggered by the deprioritized probe, even
when the model can answer both correctly.

## Hypotheses

**H1 (primary): prioritization difference.**
On a controlled 2-item working-memory prompt battery with
prioritized item A (tested with probability 0.8) and deprioritized
item B (tested with probability 0.2):
- `S_early_A` (the `S_early` of the model's response when the probe
  asks about item A) is greater than `S_early_B` (the `S_early` when
  the probe asks about item B) with **Cohen's d >= 0.3 and
  one-sided paired permutation p < 0.05** at n >= 60 matched trial
  pairs.

**H2 (secondary): delay scaling.**
The effect in H1 is larger for long-delay trials (5+ intervening
distractor sentences) than for short-delay trials (1 intervening
distractor sentence), mirroring Weng's finding that sustained alpha
prioritization is more pronounced at the longer delay:
- `(S_early_A - S_early_B)_long > (S_early_A - S_early_B)_short`
  with a one-sided p < 0.10 at n >= 30 trials per delay condition.

**H3 (tertiary): storage dissociation.**
Deprioritized items are still retrievable: `accuracy_B` (fraction of
correct B responses when the trial probes B) is >= 0.50, even though
their `S_early` is near the prioritized-baseline. This is the
in-silico analog of Weng's "activity-quiescent but retrievable"
finding.

**H4 (negative control): entropy does not discriminate.**
Running the same contrast on mean early-token logit entropy
(`H_early_A` vs `H_early_B`) does NOT yield Cohen's d >= 0.3. If
entropy also shows the effect, the `S_early`-specific interpretation
weakens and we scope the claim.

## Pre-locked parameters

### Model
- **Primary:** `google/gemma-2-2b-it` (matches the Fathom paper v14
  baselines run; has full SAE infrastructure; 26 layers).
- **Replication model (for condition 1, if primary passes):**
  `meta-llama/Llama-3.2-1B-Instruct` (matches the other architecture
  in the spectral pilot; has SAE-free probe infrastructure; 16 layers).

### Prompt battery
- **Source:** `prompts/wm_prioritization_prompts.json` (to be
  committed with this pre-registration).
- **Content:** 60 trials, each consisting of:
  1. A two-item encode phase: `"FACT_A: <topic_A> <fact_A>. FACT_B: <topic_B> <fact_B>."`
  2. A prioritization cue: `"You will be tested first on FACT_A."`
  3. A distractor task of variable length (1-5 sentences of a neutral
     arithmetic or paraphrase task).
  4. A single probe: with probability 0.8, `"QUESTION: What is <fact_A>?"`;
     with probability 0.2, `"QUESTION: What is <fact_B>?"`. The probe
     target is selected by a fixed seed (seed=42) at prompt-battery
     construction time so the same (trial, probe_target) pairings
     are used across models.
- **Balancing:** `fact_A` and `fact_B` are drawn from topic-matched
  pairs (same semantic category, same token count within +/- 3
  tokens, same prompt template). Neither fact is in the model's
  training data verbatim (the facts are hand-constructed counterfactual
  statements of the form "X is Y years old" or "W was invented by Z"
  with controlled novelty).
- **Delay manipulation:** 30 trials have a 1-sentence distractor
  (short delay); 30 trials have a 5-sentence distractor (long delay).

### Measurement
- **Pipeline:** `api/coherence_steerer_ext.CoherenceSteererExt` with
  `max_tokens=40`, `temperature=0` (greedy). Captures
  `c_delta_trajectory`, `entropy_trajectory`, `logprob_trajectory`,
  `top2_margin_trajectory` per generation.
- **S_early window:** tokens 0-7 (locked from the existing Fathom
  pipeline).
- **S_early threshold:** `tau = 0.005` (locked from
  `analyze_S_truthfulqa.THR_PRIMARY`).
- **Formula:** `S_early(traj) = max(traj[0:7]) / count(traj[0:7] > tau)`
  (the Fathom definition, verified to be equal to
  `M * IPR(event_locations)` on every trajectory).

### Statistical test
- **Primary (H1):** one-sided paired permutation test on
  `S_early_A - S_early_B` across trial pairs matched by delay
  condition. Test statistic: mean difference. Null: labels shuffled
  within-trial. n_perm = 10000, seed = 42. alpha = 0.05.
- **Effect size:** Cohen's d_z (paired) on the difference.
- **H2 (delay scaling):** one-sided unpaired test on the
  delay-condition difference-of-differences. n_perm = 10000, seed = 42.
  alpha = 0.10 (exploratory secondary).
- **H3 (retrievability):** fraction correct on B-probe trials,
  compared to 0.50 with exact binomial test.
- **H4 (negative control):** same pipeline, but replace `S_early`
  with `H_early = mean(entropy[0:7])`.

### Exclusion criteria
- Trials where the model refuses or produces < 8 tokens on either
  probe target are excluded.
- Trials where the model's response does not mention the target fact
  at all are flagged in the audit but NOT excluded -- they still
  contribute to the `S_early` measurement (this is the key test: is
  `S_early` high even when the output is incorrect?).

## Sealed decision rule

H1 is SUPPORTED iff:
1. `S_early_A > S_early_B` paired permutation p < 0.05 on the primary
   model (Gemma-2-2b-IT).
2. Cohen's `d_z >= 0.3` on the same test.
3. `S_early_A > S_early_B` AND `H_early_A - H_early_B` does NOT have
   Cohen's `|d_z| >= 0.3` (negative control satisfied).

H2 is SUPPORTED iff:
4. `(S_early_A - S_early_B)_long > (S_early_A - S_early_B)_short`
   one-sided p < 0.10.

H3 is SUPPORTED iff:
5. Fraction correct on B-probe trials is >= 0.50 (binomial p > 0.05
   for the test `p_correct = 0.5`).

## What falsifies the hypothesis

- **S_early_A ~= S_early_B:** the transformer does not implement a
  prioritization/storage dissociation in the same way humans do.
  Reported as a publishable null against the biological parallel.
- **S_early_B > S_early_A:** the model is prioritizing the wrong item.
  Investigate as a task-design problem OR a genuine failure of the
  prioritization analog.
- **Entropy shows the same effect:** the distinction between S_early
  and standard uncertainty is weaker in this paradigm than on
  TruthfulQA. Scope the head-to-head claim.
- **Retrievability fails:** the deprioritized item is not actually
  stored. The "activity-quiescent" analog fails. Reframe.

## Deviations allowed

- Minor engineering (file format fixes, GPU memory management) must
  be documented in the results findings doc.
- If the primary run returns noisy trajectories (e.g. most trials
  have `S_early < tau` baseline), we may re-run with the model's
  `temperature = 0.1` instead of 0 to break ties in greedy decoding.
  This deviation must be documented.

## Outputs

- `prompts/wm_prioritization_prompts.json` (hand-curated prompt battery)
- `runners/run_wm_prioritization.py` (the experiment runner, uses
  `coherence_steerer_ext`)
- `truthfulqa_results/wm_prioritization_<model_slug>.json` (raw
  per-trial captures)
- `analysis/analyze_wm_prioritization.py` (the locked analyzer)
- `findings/findings_<date>_wm_prioritization.md` (the results)

## Commitment

This pre-registration is sealed in git before any model invocation
that writes real data. The analyzer, the S_early formula, the
statistical tests, the prompt battery, and the permutation seeds
are all locked. The results will be reported against the sealed
decision rule above without modification, whether they support
H1-H4 or not.

**Seal commit:** (to be filled in after this file is committed)
