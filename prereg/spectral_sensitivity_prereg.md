# Pre-registration: spectral sensitivity analysis under v22 semantic labeling

**Status:** SEALED (committed to git BEFORE running any v22 spectral analysis)
**Authors:** Alexander Rodabaugh
**Project:** Fathom / sae-reasoning-depth
**Date sealed:** 2026-04-11
**Related findings docs:**
- `findings/findings_2026_04_11_spectral_pilot.md` (the exploratory pilot that motivates this test)
- `findings/findings_2026_04_11_alpha_bridge.md` (the biological parallel framing)
**Related published paper:** Fathom v15, `10.5281/zenodo.19504993` (concept `10.5281/zenodo.19326174`)

---

## Background and motivation

On 2026-04-11 we ran an exploratory spectral pilot on the n=200 TruthfulQA
C_delta trajectory data already on disk. Using the v21 text-match
classifier (`classify_generation` from `analyze_S_truthfulqa.py`), we found:

| Model | n (correct / hallucinated) | peak bin period | d | family-wise p |
|---|---|---|---|---|
| Llama-3.2-1B-Instruct | 33/38 | 8.0 tokens | -0.811 | **0.016** |
| Gemma-2-2b-IT | 38/30 | 4.0 tokens | -0.710 | 0.074 |

Llama passed family-wise correction at alpha=0.05. Gemma was borderline.
The pilot was explicitly reported as exploratory.

The Fathom team has a history of running v22 semantic relabeling as a
sensitivity analysis on the n=200 TruthfulQA runs (see
`osf_prereg_semantic_relabel.md`, sealed 2026-04-05, embedding model
`sentence-transformers/all-MiniLM-L6-v2`, margin threshold 0.15). The
v22 protocol catches items the n-gram classifier misses and typically
expands the labeled subset by 30-40%.

This pre-registration locks the spectral sensitivity analysis under v22
labels BEFORE running it. The purpose is to prevent any post-hoc tweaking
of the labeling / window / statistics after seeing the result.

## Primary hypothesis

**H1:** The spectral hallucination signature observed in the v21 pilot
replicates under the v22 semantic labeling on both Gemma-2-2b-IT and
Llama-3.2-1B-Instruct, with "replicates" defined by the sealed decision
rule below.

## Pre-locked parameters

These parameters are fixed and will not be adjusted after seeing results.

### Labeling
- **Source script:** `analysis/semantic_relabel.py` (existing, unmodified,
  sealed by git in commit `f3a7943` or earlier).
- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Margin threshold:** 0.15
- **Label set:** combined correct = `correct` + `correct_semantic`;
  combined hallucinated = `hallucinated` + `hallucinated_semantic`.
- **Join key:** `hf_index` (matches trajectory files to the audit files
  exactly).
- **Trajectory files to relabel:**
  - `truthfulqa_results/S_baselines_n200.json` (Gemma-2-2b-IT)
  - `truthfulqa_results/S_truthfulqa_llama_n200.json` (Llama-3.2-1B-Instruct)

### Spectral analysis
- **Source script:** `analysis/analyze_S_spectral.py` (existing, unmodified
  from the pilot) + a new `analysis/analyze_S_spectral_v22.py` wrapper
  that reads v22 audit files to assign labels instead of calling
  `classify_generation` directly.
- **Window length:** `L = 40` (locked from the pilot sweep).
- **Min tokens per item:** 40.
- **Window function:** Hann, normalized to preserve power.
- **DC removal:** mean-subtract before FFT.
- **Test statistic per bin:** Cohen's d (correct vs hallucinated), Welch t-test.
- **Family-wise correction:** max |d| permutation test, 2000 permutations,
  fixed seed 42.
- **Signal tested:** `c_delta_trajectory` (primary). Head-to-head controls
  on `entropy_trajectory`, `logprob_trajectory`, `top2_margin_trajectory`
  (Gemma only, since Llama file does not have these).

## Sealed decision rule

H1 is SUPPORTED if and only if ALL of the following conditions hold on
the PRIMARY signal (C_delta):

1. **Gemma-2-2b-IT family-wise p (C_delta, L=40) < 0.05** at the v22
   labeled set.
2. **Llama-3.2-1B-Instruct family-wise p (C_delta, L=40) < 0.05** at the
   v22 labeled set (i.e. the Llama pilot result replicates under the
   v22 labeling).
3. **Direction of the peak bin is consistent with the pilot** on both
   models, i.e. correct > hallucinated in power (Cohen's d negative
   because the test statistic is correct-mean minus halluc-mean).

If conditions 1 and 2 pass AND condition 3 passes, we report H1
SUPPORTED and integrate the v22 numbers as the confirmatory row of the
spectral section table in paper v16.

If condition 1 fails (Gemma family-wise p >= 0.05) but conditions 2 and
3 pass, we report H1 PARTIALLY SUPPORTED (Llama replicates, Gemma does
not reach significance even under v22 labeling) and keep the finding
scoped as "pilot-level with one-architecture replication."

If condition 2 fails (Llama family-wise p >= 0.05), we report H1 NOT
SUPPORTED and downgrade the paper language from "replicates on two
architectures" to "pilot finding on one architecture (Llama v21) that
did not replicate under v22 labeling."

If condition 3 fails (direction flip on either model), we report H1
NOT SUPPORTED and investigate whether the v21 finding was a labeling
artefact.

## Deviations allowed

- None for the primary test. The labeling, window length, window function,
  permutation seed, and number of permutations are locked.
- Minor engineering deviations (e.g. fixing a bug in the audit file join,
  handling an unexpected file format) must be documented in the results
  section of the findings doc and not affect the decision.

## Outputs

On completion, the following files will be written (names locked):

- `truthfulqa_results/S_baselines_n200_v22_audit.json` (Gemma v22 labels)
- `truthfulqa_results/S_truthfulqa_llama_n200_v22_audit.json` (Llama v22 labels)
- `truthfulqa_results/S_baselines_n200_spectral_v22.json` (Gemma spectral)
- `truthfulqa_results/S_truthfulqa_llama_n200_spectral_v22.json` (Llama spectral)
- `findings/findings_2026_04_11_spectral_sensitivity_v22.md` (the results)

## Commitment

This pre-registration is sealed in git. The sealing commit will be
referenced in the results findings doc. The analyzer, the semantic
relabeler, the window length, and the permutation seed are all locked.
The results will be reported against the decision rule above without
modification, whether they support H1 or not.

**Seal commit:** (to be filled in after this file is committed, before
any v22 analysis is run)
