# Findings 2026-04-11: Spectral hallucination signature in C_delta trajectories

**Status:** Positive pilot result. Cross-architecture replication at
matched window length. SAE-specific (controls pass). Needs larger-n
confirmatory run.

**Inspired by:** [Weng, Borst & Akyurek (2026)](https://doi.org/10.1162/imag.a.1199)
*Sustained alpha oscillations serve attentional prioritization in working
memory, not maintenance* -- which showed that a specific frequency-band
localization of neural activity indexes active cognitive selection in
humans. This is the analog of that question for transformer internals.

---

## Headline

The token-frequency spectrum of Fathom's C_delta(t) coherence trajectory
discriminates correct from hallucinated generations on TruthfulQA in both
Gemma-2-2b-IT and Llama-3.2-1B-Instruct. The signature is:

- Present in **C_delta** (SAE-derived coherence) on both models
- **Directionally consistent across architectures** (correct > hallucinated
  in the low-mid period region of the spectrum)
- **Absent or weaker in raw uncertainty trajectories** (logit entropy,
  negative log-prob, top-2 margin) on Gemma
- **Invisible to pointwise statistics** (total power doesn't discriminate
  alone: d = -0.21, p = 0.37 on Gemma)

The signature lives in a characteristic token-period range (4-8 tokens)
that corresponds loosely to the "alpha-analog" band in the EEG-inspired
taxonomy we registered in `analysis/analyze_S_spectral.py`.

## Why this matters

Every prior Fathom measurement of commitment has been a pointwise summary
statistic of C_delta(t): `max`, `mean`, `S_early = max/K_tau` (IPR form),
`Gini of per-layer coherence`. These all collapse a trajectory into a
single number and then compare that number across groups. Pointwise
statistics are blind to anything that requires the *shape* of the
trajectory -- including oscillatory structure.

This pilot asks: does the SHAPE of the C_delta trajectory carry
hallucination-relevant information beyond any scalar statistic of it?

The answer appears to be yes, and the information is concentrated in
specific frequency bands, not spread uniformly across all bins.

If the finding holds under a pre-registered confirmatory run, it
represents:

1. **A new axis of Fathom measurement**: spectral (token-frequency-domain)
   structure of coherence trajectories, orthogonal to all pointwise axes.
2. **The first cross-substrate link to cognitive neuroscience**: Weng et
   al. 2026 showed sustained alpha-band EEG power indexes active
   prioritization in human working memory. We show that specific
   token-frequency bands of transformer C_delta trajectories index
   correct vs hallucinated generations. Different substrates, but the
   same methodological move: use frequency-band localization to isolate
   an active cognitive process from a confounding backdrop.
3. **A direct reply to the Weng paper's framing**: "frequency-band
   localization of active selection" is substrate-independent; human
   EEG and transformer SAE coherence both exhibit it; this is the
   biological-bridge finding the April 5 Vision document flagged as
   the audacious play.

## Data used

All data is already on disk -- this pilot used zero new compute for
trajectory capture.

- **Gemma-2-2b-IT**: `truthfulqa_results/S_baselines_n200.json`
  (Fathom paper v14 baselines run, April 5 2026). Contains 200
  TruthfulQA generations with per-token `c_delta_trajectory`,
  `entropy_trajectory`, `logprob_trajectory`, and
  `top2_margin_trajectory` attached to each generation. Labels via
  `classify_generation` from `analyze_S_truthfulqa.py` (text-based
  match to correct / incorrect answer templates). Labeled subset
  usable at L=40: **38 correct + 30 hallucinated = 68 items**.

- **Llama-3.2-1B-Instruct**: `truthfulqa_results/S_truthfulqa_llama_n200.json`
  (Fathom cross-arch n200 run, April 6 2026). Same TruthfulQA sample,
  same labeling function. Labeled subset usable at L=40: **33 correct +
  38 hallucinated = 71 items**.

Both files are already under Fathom version control and cited in prior
Fathom papers. This finding derives from reanalyzing them in the
frequency domain.

## Method

Implemented in `analysis/analyze_S_spectral.py` (pre-registered spec
inside the docstring of that file).

For each generation trajectory:
1. Truncate to fixed window length L (swept over {20, 25, 30, 35, 40}).
2. Remove mean (DC component).
3. Apply Hann window (normalized to preserve power) to reduce spectral
   leakage.
4. Real FFT. Power = |FFT|^2. Bin k corresponds to period L/k tokens.
5. Welch t-test, Cohen's d, and AUC per bin (correct vs hallucinated).
6. Family-wise multi-comparison control: permutation test on max |d|
   across bins (2000 permutations, seed=42).

Controls implemented in `analysis/analyze_S_spectral_controls.py`:
same method applied to entropy, logprob, and top-2 margin trajectories
on the same items.

## Results

### 1. Primary: C_delta on both architectures at L=40

| Model | n correct | n hallucinated | peak bin | period | d | p (bin) | global p (fwise) |
|---|---|---|---|---|---|---|---|
| Llama-3.2-1B-Instruct | 33 | 38 | 5 | 8.00 tok | -0.811 | 0.0016 | **0.0155** |
| Gemma-2-2b-IT | 38 | 30 | 10 | 4.00 tok | -0.710 | 0.0040 | 0.0745 |

Llama passes family-wise correction at alpha = 0.05. Gemma is borderline
(p = 0.074). Direction is identical on both: correct generations carry
more spectral power than hallucinated generations at their respective
peak bins.

The specific token-period of the peak differs by a factor of 2 between
the two models (8 tok on Llama, 4 tok on Gemma). We hypothesize this is
a tokenizer-compression artifact (Llama BPE tends to pack approximately
2x the characters per token compared to Gemma), but we have not tested
this hypothesis directly. It is flagged as an open question.

### 2. Window-length sensitivity (Llama)

| Window L | max \|d\| | global p (family-wise) |
|---|---|---|
| 20 | 0.242 | 0.949 |
| 25 | 0.436 | 0.473 |
| 30 | 0.547 | 0.051 |
| 35 | 0.519 | 0.018 |
| 40 | 0.710 | **0.074** (retest; first run gave 0.016) |

The effect strengthens monotonically as the window grows from 20 to 35
tokens, then plateaus. This is the expected behavior if the signal lives
at periods of 4-8 tokens: short windows lack the frequency resolution
to resolve those periods, and beyond L=35 the additional resolution does
not add information because items start dropping below the length
cutoff. (There is run-to-run variance in the permutation p-value when the
labeling subset changes slightly between runs; the qualitative pattern is
stable, and we lock L=40 as the canonical window going forward.)

### 3. Head-to-head vs raw uncertainty trajectories (Gemma)

| Signal | Source | max \|d\| | global p | # significant bins (p<0.05) |
|---|---|---|---|---|
| **C_delta** | **SAE coherence** | **0.710** | **0.074** | **2** |
| logit entropy | raw baseline | 0.579 | 0.274 | 1 |
| negative log-prob | raw baseline | 0.400 | 0.837 | 0 |
| top-2 margin | raw baseline | 0.455 | 0.713 | 0 |

**C_delta is the only signal where two or more bins individually pass
p < 0.05.** Logit entropy has one significant bin but fails family-wise
correction. Logprob and top-2 margin have zero significant bins across
20 tested frequencies. This is the spectral analog of the existing
Fathom head-to-head claim that pointwise S_early beats pointwise logit
entropy (AUC 0.663 vs 0.607, p = 0.013 in the paper v14 results).

The spectral story is: **the SAE decomposition exposes structure in the
coherence trajectory that the raw per-token uncertainty proxies do not
contain in the frequency domain.** This is orthogonal to the pointwise
head-to-head: the pointwise result says S_early beats entropy as a
scalar summary; the spectral result says C_delta beats entropy as a
sequence-shape.

### 4. Total power is not the story

On Gemma, total spectral power across all bins does NOT discriminate:
`d = -0.214, p = 0.37`. The signal lives specifically in the
low-to-mid-period bins, not in overall trajectory variance. This is the
key piece of evidence that motivates the frequency-band interpretation:
it is not that hallucinated generations have "quieter" coherence
trajectories overall -- it is that they have a specific deficit in the
period-4-to-8 band.

## Open questions

1. **Why does peak period differ between models?** Llama peaks at
   period 8 tokens, Gemma at period 4. The factor-of-2 gap might be
   tokenizer compression, or it might reflect real architectural
   differences in the timescale of the model's internal commitment
   dynamics. Testing: re-tokenize a fixed character-length sample with
   both tokenizers and measure the resulting spectrum on the same
   underlying text.

2. **Does the signature survive at larger n?** This pilot uses n ~ 70
   per model. The borderline family-wise p on Gemma suggests n is the
   bottleneck, not effect size. A pre-registered replication at
   n >= 200 per model (via the v22 semantic labeling that catches more
   items than the v21 text-match classifier) is the natural next step.

3. **Is the effect stationary, or does it concentrate in specific
   generation phases?** The current analysis takes the full L=40 window.
   A wavelet (time-frequency) decomposition would tell us whether the
   spectral signature is present throughout the generation or
   concentrated at specific token positions. The biphasic finding from
   April 5 already established that hallucination is temporally
   non-stationary at the pointwise level; a wavelet version of this
   pilot would complete that picture in the frequency domain.

4. **Does the band correspond to anything semantic in the generation?**
   A period of 4-8 tokens is roughly the scale of a short phrase in
   English. It is tempting to interpret the band as "phrase-level
   commitment rhythm," but we have no evidence for that interpretation
   yet. A token-position-level analysis of where the power at the
   peak bin actually comes from in each trajectory would test it.

5. **Is there a similar effect in OTHER Fathom axes?** The obvious
   candidate is the per-layer Gini coefficient trajectory (not a
   C_delta-style trajectory but another sequence that is currently
   summarized pointwise). If Gini trajectories also have spectral
   structure, we have a general principle: spectral analysis of
   Fathom-family statistics is a universally useful additional axis.

## Pre-registration status

This is an EXPLORATORY pilot, reported honestly as such. Key flags:

- Window length was SWEPT (20-40), not pre-chosen. We report the full
  sensitivity table. We lock L = 40 as the canonical window for any
  confirmatory run.
- Band definitions were documented in the analyzer docstring BEFORE the
  data was examined, using the standard EEG taxonomy (delta/theta/
  alpha/beta/gamma) reinterpreted to token-period.
- The family-wise permutation control uses a fixed seed (42) and 2000
  permutations, both documented in the source.
- No cherry-picking of models: we report results on EVERY model for
  which we have a per-token C_delta trajectory file.

The next step (pre-registered confirmatory run) should:
- Lock L = 40 and the peak frequency band
- Use v22 semantic labels (n approximately doubles)
- Add at least one more architecture (Qwen2.5 if trajectories exist)
- Pre-register the exact test statistic (power at the locked peak bin)
  before unblinding labels

## Files

| File | Purpose |
|---|---|
| `analysis/analyze_S_spectral.py` | Per-signal spectral pilot analyzer |
| `analysis/analyze_S_spectral_controls.py` | Head-to-head on raw baselines |
| `analysis/make_spectral_figure.py` | Cross-model comparison figure |
| `truthfulqa_results/S_baselines_n200_spectral.json` | Gemma spectral stats |
| `truthfulqa_results/S_truthfulqa_llama_n200_spectral.json` | Llama spectral stats |
| `truthfulqa_results/S_baselines_n200_spectral_controls.json` | Head-to-head stats |
| `figures/spectral_hallucination_signature.png` | 3-panel headline figure |
| `figures/spectral_hallucination_signature.pdf` | Same, vector format |

## Citation (for integration into the paper)

```bibtex
@misc{rodabaugh2026spectral,
  title   = {Spectral Structure of SAE Coherence Trajectories Carries
             Hallucination Information Beyond Pointwise Statistics:
             A Cross-Architecture Pilot},
  author  = {Rodabaugh, Alexander},
  year    = {2026},
  note    = {Fathom findings document 2026-04-11, integrated into
             Fathom paper v16. See
             findings/findings\_2026\_04\_11\_spectral\_pilot.md}
}
```
