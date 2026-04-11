# Findings 2026-04-11: The Alpha Oscillation Bridge

**Status:** Theoretical finding + experimental proposal (no data yet)
**Trigger:** [Weng, Borst, Akyürek 2026](https://doi.org/10.1162/imag.a.1199)
published in *Imaging Neuroscience* (preprint: bioRxiv 10.1101/2025.10.16.682668)

---

## Summary

Human cognitive neuroscience has independently characterized a measurement
that is structurally analogous to Fathom's `S_early`. The parallel is
strong enough that we now treat "localization of active selection" as a
substrate-independent cognitive construct with two validated operational
definitions:

| Substrate         | Measurement                          | Dissociation                                  |
|-------------------|--------------------------------------|-----------------------------------------------|
| Human EEG         | Sustained alpha-band power on a     | prioritization ⊥ storage (sustained alpha vs. |
|                   | prioritized item across a delay      | contralateral delay activity)                 |
| Transformer SAE   | `S_early` = max(C∆)/K_τ on the       | selection ⊥ depth (`S_early` ⊥ K,             |
|                   | first 7 tokens = IPR of commitment   | r = -0.062 at n=400 TruthfulQA)               |
|                   | events on early window               |                                               |

Both are time-localization measures of a cognitive target that the system
is *actively* committing to right now. Both distinguish the active
commitment from passive storage. Both are operationally grounded in
localization math (EEG: power concentration across a delay window;
transformer: IPR on a discrete event distribution).

This is the biological bridge that the April 5 Vision document flagged as
the "audacious play" for Fathom — and we did not have to go looking for
it. It was already in the literature.

## The Weng et al. finding (2026)

**Paradigm.** Two-item working memory task. Subjects encode two items
with different priorities (item A is tested first with high probability;
item B is tested second or not at all). Short delay = 1 s; long delay =
3 s. EEG throughout.

**Result.** Sustained alpha-band power tracks only the prioritized item
(A) through the delay with reliable decodability. The deprioritized item
(B) does not show sustained alpha and transitions into an
"activity-quiescent state" — it is still in memory (it can be retrieved
if probed), but it is not actively represented in alpha-band activity.

**Interpretation.** Sustained alpha is *not* a general working-memory
maintenance signal. It is a **prioritization / attentional selection**
signal. This is a dissociation within working memory: prioritization
(alpha-supported) vs. storage (synaptic, activity-quiescent).

## The mapping to `S_early`

Each property of sustained alpha has a clean `S_early` analog:

```
  PROPERTY              EEG (Weng 2026)           FATHOM (S_early)
  --------              ------------------        -----------------
  target of measure     which WM item is          which attractor is
                        currently prioritized     currently committed
  time structure        sustained power           sustained IPR-concentrated
                        across delay window       commitment on early window
  math                  spectral power            max(C∆)/K_τ
                        concentration over time   = IPR of event locations
  dissociated from      contralateral delay       K (computational depth)
                        activity (storage)        (r = -0.062, n=400)
  substrate             electrodes × time         tokens × layers
  observable            EEG scalp recording       residual stream via SAE
                                                  (IPR probes cross-arch)
```

The mathematical common structure is **localization of a scalar
trajectory over a short window after the onset of a selection event**.
In both cases the cognitive interpretation is "the thing the system is
hard-committed to right now, distinct from what it merely has stored."

## Why this matters for the Fathom research program

### 1. It validates the reframe from "hallucination detector" to "measurement instrument"

The April 5 vision document (`docs/VISION_2026_04_05.md`) argued that
`S_early` should be positioned as a **measurement primitive of commitment
intensity**, not as a hallucination detector. The hallucination
application is one downstream use; the underlying measurement is more
general. Weng et al. provide independent external validation for this
reframe: the human brain has its own realization of the same construct,
and the neuroscience literature has a 20+ year history studying it under
the name "sustained alpha."

### 2. It opens a clean biological-bridge experiment we can actually run

Unlike the speculative TRIBE / Jean-Rémi King collaboration path, the
Weng paradigm gives us a **ready-made task structure** that has already
been validated on humans. We can run the same task in-silico on a
transformer and test whether `S_early` shows the same prioritization/
storage dissociation that sustained alpha showed in the EEG study.

This is a 2-week experiment with a clear pre-registered prediction. If
the prediction holds, we have empirical evidence that the two
measurements are converging on the same cognitive construct — which is
the strongest form of construct validity a cognitive measurement can
have.

### 3. It turns the S framing from "novel" to "grounded"

A reviewer in 2026 reading "we introduce a new measurement instrument
for transformers called commitment intensity" has to take our word that
the construct exists. A reviewer reading "we port a construct that
neuroscience has studied for 20 years (sustained-alpha-indexed
prioritization) to transformers via an architecturally universal SAE-free
probe" has an entire literature backing the construct before they
even start evaluating our specific measurements.

## Proposed experiment: in-silico WM prioritization

### Goal

Test whether `S_early` on a transformer trajectory shows the same
prioritization/storage dissociation that Weng et al. reported for
sustained alpha in the human EEG.

### Task structure (in-silico replication)

For each trial, construct a prompt of the form:

```
You will memorize two facts and then answer a question.
FACT_A (high priority): <content_A>
FACT_B (low priority): <content_B>
Now complete the intervening task: <distractor task, 1-3 sentences>
Question: <probes FACT_A with probability 0.8, FACT_B with probability 0.2>
Answer:
```

Varies:
- `content_A` / `content_B`: factual statements from a controlled pool
  (e.g. "The capital of X is Y"), balanced so mean length / token count /
  entropy of each is matched across the two items.
- `distractor task`: varies length to create a 1-token (short) vs 5-token
  (long) delay analog of Weng's 1s vs 3s manipulation.
- `probe`: asks for FACT_A 80% of the time. The remaining 20% ask FACT_B
  and are used to confirm the deprioritized item is still in memory
  (i.e. model can still answer when probed).

### Measurements

For each trial, capture the full `(C_∆, S_early, K, logit entropy)`
trajectory across the model's response to the question. Compute:

- `S_early_A`: the `S_early` of the generation when the question probes
  FACT_A (prioritized → expected high `S_early`).
- `S_early_B`: the `S_early` of the generation when the question probes
  FACT_B (deprioritized → expected lower `S_early`).
- `accuracy_A` and `accuracy_B`: binary correctness of the response to
  each item.

### Pre-registered predictions

Primary prediction (H1):
- **`S_early_A > S_early_B`** with Cohen's d ≥ 0.3 and p < 0.05 on a
  paired permutation test at n ≥ 60 matched pairs (30 short-delay + 30
  long-delay). The direction is fixed by the analogy to sustained
  alpha; the effect size threshold is chosen to be comparable to the
  existing n=200 TruthfulQA `S_early` result (d = 0.506).

Secondary prediction (H2, delay interaction):
- The `S_early_A - S_early_B` gap is **larger for long-delay trials than
  for short-delay trials**, mirroring Weng's finding that sustained
  alpha decodability of the prioritized item strengthens selectively in
  the long-delay condition.

Tertiary prediction (H3, storage dissociation):
- `accuracy_B` is ≥ 50% (i.e. the deprioritized item is still stored
  and retrievable) even on trials where `S_early_B` is near the
  quiescent baseline. This is the in-silico analog of Weng's
  "activity-quiescent but still retrievable" finding.

### Control: logit entropy

Compute the same contrasts on mean early-token logit entropy. The
prediction is that entropy does NOT show the prioritization effect,
consistent with the `S_early` vs entropy head-to-head result in the
Fathom ICML paper (AUC 0.663 vs 0.607). If entropy also shows the
effect, the `S_early`-specific interpretation weakens.

### Failure modes

- If `S_early_A ≈ S_early_B`: the transformer does not implement a
  prioritization/storage dissociation in the same way humans do. This
  is still a publishable null — it would tell us the cross-substrate
  parallel is weaker than the related-work paragraph currently claims.
- If `S_early_B > S_early_A`: the model is prioritizing the wrong item.
  This would indicate a task design problem or a genuine failure of
  the prioritization hypothesis; we would investigate which.
- If entropy shows the same effect: the distinction between `S_early`
  and standard uncertainty is weaker in this paradigm than we reported
  on TruthfulQA. This would require a scope restriction in the paper's
  head-to-head claim.

### Estimated cost

~2 weeks:
- Week 1: task design, prompt controls, n=10 pilot on Gemma-2-2B-IT to
  verify the task format elicits different generation trajectories for
  A vs. B probes.
- Week 2: n=60 pre-registered run, analysis, writeup.

## Integration with existing Fathom work

This finding should be integrated into:

1. **Paper v16 of the Fathom preprint** — the related-work paragraph
   added to `paper/fathom_arxiv_v1.tex` on 2026-04-11 cites Weng et al.
   and frames the parallel; if and when the in-silico experiment runs,
   a results section can be added.
2. **ICML workshop submission** — the shorter `Biological Parallel`
   paragraph has been added to `paper/fathom_icml2026_v2.tex`.
3. **Atlas roadmap** — the cross-substrate validation becomes a
   formal entry in `atlas/README.md`'s Status and roadmap.
4. **OSF wtkzg** — research-direction update; the experiment, if run,
   should have its own pre-registration under the same project.
5. **Outreach** — Jean-Rémi King (Meta FAIR TRIBE) should receive a
   pointer to both Weng et al. and our cross-substrate framing as part
   of the collaboration pitch.

## Citation

```bibtex
@article{weng2026sustained,
  title   = {Sustained alpha oscillations serve attentional prioritization
             in working memory, not maintenance},
  author  = {Weng, Yuanyuan and Borst, Jelmer P. and Aky{\"u}rek,
             Elkan G.},
  journal = {Imaging Neuroscience},
  year    = {2026},
  doi     = {10.1162/imag.a.1199},
  note    = {Preprint: bioRxiv 10.1101/2025.10.16.682668}
}
```

## Status

- [x] Paper citation added to `paper/references.bib`
- [x] Related-work paragraph added to `paper/fathom_arxiv_v1.tex`
- [x] Related-work paragraph added to `paper/fathom_icml2026_v2.tex`
- [x] Findings document written (this file)
- [ ] In-silico replication experiment designed in detail
- [ ] OSF pre-registration drafted
- [ ] Experiment run
- [ ] Results integrated into paper v16
- [ ] Outreach to Weng / Borst / Akyürek or TRIBE collaborators
