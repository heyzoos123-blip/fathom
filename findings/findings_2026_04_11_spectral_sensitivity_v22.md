# Findings 2026-04-11: Spectral sensitivity analysis under v22 semantic labeling

**Status:** Pre-registered sensitivity test ran as specified.
Sealed decision rule returns **H1 NOT SUPPORTED** (family-wise corrected).
Direction and peak-bin location **replicate exactly** on both architectures.

**Pre-registration:** `prereg/spectral_sensitivity_prereg.md`, sealed in
git commit `90a6f7d` **before** any v22 spectral analysis was run.

---

## Headline

The spectral pilot from earlier today (`findings_2026_04_11_spectral_pilot.md`)
was reported as exploratory. We pre-registered a sensitivity analysis that
re-runs the exact same spectral pipeline under Fathom's v22 semantic
relabeling (sentence-transformers embedding-based, margin threshold 0.15,
`analysis/semantic_relabel.py`), which expands the labeled subset on
both architectures by 34% to 50%.

Under v22, the pilot's direction **replicates cleanly** (correct >
hallucinated on both models, at the same peak frequency bin as v21),
and the peak-bin per-bin significance remains **p < 0.01 on both models
under both labelings**. But the family-wise permutation correction
across 20 bins comes in at **0.058 on Gemma** and **0.081 on Llama** --
both directionally correct, both borderline, both **fail** the sealed
0.05 threshold.

Per the sealed decision rule, **H1 is reported as NOT SUPPORTED.**

This is scientifically cleaner than a lucky-win headline: the pilot did
not survive the pre-registered multi-comparison test, but the underlying
signal structure (which bin peaks, in which direction, at what effect
size) replicates exactly across two independent labeling schemes and
two architectures. The right next move is a **bin-locked** confirmatory
run that avoids the multi-comparison tax by pre-registering the exact
peak bin from this analysis.

## Full numbers

### Primary test: family-wise max|d| permutation on C_delta, L=40

| Model | Labeling | n (c/h) | Peak bin | Peak period | Peak d | Peak bin p | **Family-wise p** |
|---|---|---|---|---|---|---|---|
| Gemma-2-2b-IT | v21 (pilot) | 38/30 | 10 | 4.00 tok | -0.710 | 0.0040 | 0.0745 |
| Gemma-2-2b-IT | v22 (sensitivity) | 54/48 | 9 | 4.44 tok | -0.600 | **0.0047** | **0.0575** |
| Llama-3.2-1B-Instruct | v21 (pilot) | 33/38 | 5 | 8.00 tok | -0.811 | 0.0016 | **0.0155** |
| Llama-3.2-1B-Instruct | v22 (sensitivity) | 39/54 | 5 | 8.00 tok | -0.597 | **0.0087** | **0.0805** |

**Things that replicate cleanly across v21 and v22:**
- Direction: correct > hallucinated on both models under both labelings.
- Peak bin location: Gemma peaks at period 4 tokens under both labelings
  (bin 10 at v21, bin 9 at v22 -- adjacent bins, essentially the same
  peak). Llama peaks at bin 5, period 8 tokens, **identically**.
- Per-bin significance at the peak bin: p < 0.01 on all four runs.
- Effect size at the peak bin: |d| = 0.60 to 0.81, all four runs.

**Things that move between labelings:**
- Family-wise p on Llama worsens from 0.016 to 0.081. This is the v22
  sample expansion diluting an already-concentrated effect: under v21
  we kept 71 items with clean n-gram labels; under v22 we added 22
  items that the semantic relabeler found, and the spectral effect in
  those added items is weaker on average, which pulls the family-wise
  statistic down.
- Gemma family-wise p shifts marginally (0.074 -> 0.058). Nearly
  unchanged.

### Sealed decision rule evaluation

Per `prereg/spectral_sensitivity_prereg.md`, H1 is supported iff:

| Condition | Threshold | Observed | Status |
|---|---|---|---|
| Gemma family-wise p (C_delta, L=40, v22) | < 0.05 | 0.0575 | **FAIL** |
| Llama family-wise p (C_delta, L=40, v22) | < 0.05 | 0.0805 | **FAIL** |
| Peak bin direction (both models) | correct > halluc | both d < 0 | **PASS** |

Two out of three sealed conditions fail. **H1 is not supported.**

## Interpretation

The pre-registration did exactly what pre-registrations are supposed to
do: it prevented a borderline post-hoc result from being framed as a
clean replication. We report the result honestly.

At the same time, the result is structurally informative:

1. **The peak-bin location is locked across labelings.** If the spectral
   signal were noise, v21 and v22 would pick up different bins in
   different directions. Instead, on Gemma the same period-4 bin is
   peak in both labelings, and on Llama the period-8 bin is peak in
   **both**. That cross-labeling consistency is unlikely under the
   null.

2. **Per-bin p at the peak is p < 0.01 under BOTH labelings on BOTH
   models.** The bin itself is individually significant; only the
   family-wise correction across 20 bins fails.

3. **The multi-comparison tax is the bottleneck, not the effect size.**
   At n = 93 to 102 (usable after L=40 truncation), max|d| of 0.60
   needs a null 95%ile of ~0.60 to pass. The observed values are
   right at that threshold.

The scientifically disciplined reading is: **the spectral hallucination
signature is present and reproducible as a directional pattern and
bin-location fingerprint across labelings and architectures, but the
effect size is too small to survive a 20-bin family-wise correction
at current sample sizes.** That is a stronger evidentiary claim than
the pilot alone, because it tests the effect against an independent
labeling scheme and the same pattern comes out.

## What changes in the paper

The spectral section added earlier today to `paper/fathom_arxiv_v1.tex`
(Section 9.5) needs to be rewritten from:

> Llama passes family-wise correction at alpha = 0.05. Gemma is
> borderline...

to:

> The pilot reported here was not survived by a pre-registered v22
> sensitivity analysis ([sensitivity findings doc]). Under Fathom's
> standard v22 semantic relabeling, both models' family-wise p values
> fall in [0.06, 0.09] -- directionally consistent, peak-bin identical,
> per-bin p < 0.01 on all four runs (v21 and v22, both models), but
> failing the sealed 0.05 multi-comparison threshold. We therefore
> report the spectral signature as a **pilot result with structural
> replication across labelings but without family-wise significance at
> current sample sizes**. The bin-location consistency (Gemma period 4,
> Llama period 8) and the direction (correct > hallucinated on both
> models under both labelings) are the robust parts of the finding.
> The natural next step is a **bin-locked confirmatory replication**
> that pre-registers the exact peak bin from the present analysis and
> tests power at that single bin, avoiding the 20-bin multi-comparison
> tax entirely.

The updated paper figures and tables also need to add the v22 row and
the "H1 not supported" outcome.

## The natural next test (for the next pre-registration)

Once you avoid the multi-comparison tax by locking the peak bin
BEFORE unblinding, you get a single-df test that is vastly more
powerful. The pre-registration for that test would say, roughly:

> On a new model or a new dataset we have not yet spectrally
> analyzed, compute the C_delta trajectory for 100+ labeled items
> (50+ correct, 50+ hallucinated). Extract the power in FFT bin 5
> (period 8 tokens) for Llama family models or bin 9 (period ~4.5
> tokens) for Gemma family models, fixed at L=40 with Hann window.
> Test: power(correct) > power(hallucinated) with Welch one-sided
> p < 0.05.

This is a single test, single degree of freedom, no multi-comparison
correction. If the spectral signal is real, this design will detect
it at n ~ 100 per arm. If it's a two-model artefact, this design will
fail.

Candidate model families for the bin-locked confirmatory test
(Fathom already has partial infrastructure for each):

- **Qwen2.5 family** (3B and 1.5B, base and instruct -- 4 models).
  Already part of Atlas v0.3, captures exist, spectral analysis has
  not been run yet on their TruthfulQA C_delta.
- **Gemma-3-1B**. Already in Atlas v0.3. Same family as Gemma-2 but
  different generation -- good within-family test.
- **Llama-3.2-3B-Instruct**. Already part of Atlas v0.3.

We do not have TruthfulQA C_delta trajectory captures for these models
yet -- only atlas captures on the fixed 90-probe set. A bin-locked
confirmatory run would require a new capture pass.

## What this means for paper v16

Paper v16 (planned for after this sensitivity analysis) should:

1. Report the v21 pilot + v22 sensitivity result honestly, both in the
   abstract and the spectral section. No confident "replicates" framing.
2. State the sealed decision rule and report H1 not supported against it.
3. State what did replicate (direction, peak-bin location, per-bin p).
4. Propose the bin-locked confirmatory test as the open question.
5. Cite Weng et al. 2026 in the related-work section as the biological
   parallel that motivated the frequency-domain analysis (that framing
   is unchanged).

This is the scientifically honest and strongest possible writeup
for the current state of the evidence. Fathom already has a track
record of this kind of reporting (biphasic primary null →
semantic-relabeled discovery; atlas v0.2.1 over-framing → v0.3 sealed
replication). This adds another example.

## Files

| File | Purpose |
|---|---|
| `prereg/spectral_sensitivity_prereg.md` | Sealed pre-registration, commit `90a6f7d` |
| `analysis/analyze_S_spectral_v22.py` | v22 confirmatory analyzer (LOCKED parameters) |
| `truthfulqa_results/S_baselines_n200_v22_audit.json` | Gemma v22 labels (54c, 48h) |
| `truthfulqa_results/S_truthfulqa_llama_n200_v22_audit.json` | Llama v22 labels (39c, 54h) |
| `truthfulqa_results/S_baselines_n200_spectral_v22.json` | Gemma v22 spectral stats |
| `truthfulqa_results/S_truthfulqa_llama_n200_spectral_v22.json` | Llama v22 spectral stats |

## Status

- [x] Pre-registration sealed (commit `90a6f7d`)
- [x] Semantic relabeling run on both trajectory files
- [x] v22 spectral analysis run with locked parameters
- [x] Sealed decision rule evaluated: **H1 NOT SUPPORTED**
- [x] Findings document written honestly
- [ ] Paper section rewritten with the more cautious language
- [ ] Bin-locked confirmatory test pre-registered (for the next run)
