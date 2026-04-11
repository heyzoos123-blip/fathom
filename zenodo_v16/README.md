# Fathom Paper v16 — Release notes

**Status:** Draft v16 bundle, ready for upload review.
**Not yet published to Zenodo.** The Zenodo concept DOI
(`10.5281/zenodo.19326174`) currently resolves to v15
(`10.5281/zenodo.19504993`, published 2026-04-10).

## Delta vs v15

### New scientific content

1. **Spectral pilot of C_delta trajectories** (exploratory).
   Fourier analysis of the C_delta(t) coherence trajectories for
   n=200 TruthfulQA items on Gemma-2-2b-IT and
   Llama-3.2-1B-Instruct. Correct generations carry more spectral
   power than hallucinated ones at a characteristic low-mid-period
   peak bin on both models. Direction and peak-bin location
   replicate exactly under Fathom's v21 text-match labeling.

2. **Pre-registered sensitivity analysis under v22 semantic labeling.**
   Sealed in git commit `90a6f7d` before any v22 spectral analysis
   was run. Sealed decision rule required family-wise p < 0.05 on
   both models and direction correct > hallucinated on both. Result:
   **H1 NOT SUPPORTED.** The direction and peak-bin location
   replicate under v22 (per-bin p < 0.01 on all four runs), but the
   family-wise correction across 20 bins returns p = 0.058 on Gemma
   and p = 0.081 on Llama -- both borderline, both fail the sealed
   0.05 threshold. Reported honestly against the sealed rule.

3. **Biological parallel framing.** The frequency-domain analysis is
   motivated by Weng, Borst & Akyurek (2026, *Imaging Neuroscience*,
   doi:10.1162/imag.a.1199) -- "Sustained alpha oscillations serve
   attentional prioritization in working memory, not maintenance."
   Their sustained alpha dissociation between prioritization (alpha-
   indexed) and passive storage (synaptic, not alpha-indexed) is
   structurally analogous to Fathom's S_early / K dissociation. We
   make no causal-identity claim, but the substrate-independent
   cognitive construct ("localization of active selection") is the
   first explicit bridge from Fathom to the frequency-domain tradition
   in cognitive neuroscience. Citation added to the Related Work
   section of both the long-form arXiv paper and the ICML v2 workshop
   paper.

4. **Pre-registered in-silico WM prioritization experiment.** Sealed
   pre-registration, deterministic prompt battery (60 trials, 30
   fact pairs, balanced short/long delay and probe target), runner
   script, and locked analyzer, ready to run on a GPU workstation.
   The experiment replicates the Weng two-item working-memory paradigm
   in a transformer by testing whether S_early is elevated on probes
   of the prioritized fact vs the deprioritized fact. NOT YET RUN --
   this is shipped as scaffolding with a sealed decision rule for
   when GPU time is available.

5. **The long-form arXiv paper is included for the first time.** v14
   and v15 shipped only the ICML short paper as the PDF artefact in
   the Zenodo record. v16 ships the 36-page long-form `fathom_arxiv_v1`
   as the headline paper (with the new Section 9.5 for the spectral
   finding) alongside the ICML short paper, which is retained for
   workshop-format citation.

### What is unchanged vs v15

- The atlas v0.3 data bundle is byte-identical to v15 and is
  re-uploaded as a sibling file on the v16 record so the unified
  release pattern is preserved.
- The Fathom v14 measurement instruments (K, C, C_delta, S, D) are
  unchanged. No new axis is added in v16.
- The v0.3 atlas RLHF attractor replication results (n=6 families,
  mean LOO cosine +0.769, permutation p=0.0315) are unchanged.

## Files

Both files are attached as sibling files on the v16 record under the
Fathom concept DOI `10.5281/zenodo.19326174`:

| File | Size | Contents |
|---|---|---|
| `fathom_paper_v16.zip` | ~2.5 MB | Long-form paper + short paper + pre-regs + findings + new analysis |
| `fathom_atlas_v0.3.zip` | ~3.4 MB | Unchanged from v15: probe set, 12 captures, pre-reg, audit, verify script |

## Citation

```bibtex
@article{rodabaugh2026fathom,
  title   = {Fathom: Cognitive Measurement Instruments for
             Transformer Internals via SAE Feature Coherence Geometry},
  author  = {Rodabaugh, Alexander},
  year    = {2026},
  note    = {Zenodo v16 (paper v16, unified release: long-form paper +
             atlas v0.3 data bundle). doi:10.5281/zenodo.<TBD>
             (concept DOI: 10.5281/zenodo.19326174)}
}
```

## Supersedes

- Paper v15: `10.5281/zenodo.19504993` (2026-04-10)
- Paper v14: `10.5281/zenodo.19502710` (2026-04-10)
- Paper concept history: v1 .. v15 under concept `10.5281/zenodo.19326174`

## Related

- Weng, Borst & Akyurek (2026). *Sustained alpha oscillations serve
  attentional prioritization in working memory, not maintenance.*
  Imaging Neuroscience. https://doi.org/10.1162/imag.a.1199
- OSF mirror: https://osf.io/wtkzg/
- Fathom site: https://fathom.darkflobi.com
- Code: https://github.com/heyzoos123-blip/fathom

## Patents (unchanged vs v14/v15)

- US Provisional 64/020,489
- US Provisional 64/021,113
- US Provisional 64/026,964

All scientific artefacts (paper, data bundles, analysis code, findings,
pre-registrations) remain under CC-BY-4.0 on data and MIT on code.
