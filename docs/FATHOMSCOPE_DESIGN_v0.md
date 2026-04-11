# FATHOMSCOPE — Design v0

**Codename:** fathomscope *(locked; change requires 1-line edit before launch)*
**Status:** Pre-repo design document. No public code exists yet.
**Target launch:** v0.1 tier 0 within 72 hours of design sign-off.
**Author:** Alexander Rodabaugh (Fathom Intelligence)
**License plan:** MIT code, CC-BY-4.0 data, patent-pending (three US provisional filings)

---

## 0. One-sentence pitch

**Fathomscope is the first instrument panel for LLM cognition — a drop-in
cognitive vitals monitor that reads an agent's internal state in real time,
using signals available on any LLM with a logprob interface, with
pre-calibrated classifiers trained on 12 open-weight models across 3
architecture families.**

Tweet-line: *the black box, now with a glass face.*

## 1. What this document is

This is the contract for fathomscope v0.1 through v0.4. It specifies:

1. The four-tier product architecture.
2. The five-phase cognitive runtime.
3. The public API surface (library + CLI + boot log).
4. What ships when, with honest numeric specs for every feature.
5. What is explicitly out of scope.

It is written **before** any fathomscope code exists so the design can be
reviewed, challenged, and modified before implementation begins. Nothing in
this document describes speculative features. Every phase threshold, every
classifier accuracy number, and every cross-architecture claim in this
document maps to a specific experimental result in the Fathom research
repo with a committed pre-registration or a committed data file.

## 2. Design principles (non-negotiable)

1. **Honest by construction.** Every feature ships with the real number from
   the research repo. If a feature has a 29% LOO accuracy, the README and
   the boot log say 29%. We do not round up. We do not hide the weak
   categories. We do not claim self-awareness, consciousness, or any
   philosophical property we cannot measure.

2. **Drop-in, not lock-in.** Level 2 (one-line wrapper) must be a strict
   superset of the underlying SDK. If a user calls `fathomscope.OpenAI()`
   and fathomscope fails to read vitals for any reason, the underlying
   OpenAI call returns its normal response unchanged. Fathomscope fails
   open, never fails closed. No user's existing agent should break because
   they installed fathomscope.

3. **Local-first.** The tier 0 runtime performs all math locally. No data
   leaves the user's machine. No telemetry. No phone-home. This is the
   regulated-industries (legal, medical, finance) story and it has to hold
   at the architectural level, not as a configuration flag.

4. **Zero heavy deps in core.** Tier 0 core is numpy + scipy only. No
   torch, no transformers, no heavy ML stack in the default install.
   `pip install fathomscope` should take five seconds. Heavy dependencies
   come in only for tier 2 (SAE instruments) and only when the user
   explicitly requests them.

5. **Tier detection is automatic.** `fathomscope init` detects what the
   user's environment can support and lights up the instruments
   accordingly. A user with only closed-weight API access sees tier 0
   come online. A user with a local gemma-2-2b install sees tier 1 and
   tier 2 come online too. No configuration required; it just tells the
   user what it found.

6. **Calibration is shipped, not trained.** The atlas v0.3 centroids
   (sha256-pinned, 12 models, 6 categories, 90 probes) ship with the
   installer. Users never train. Users never calibrate. Reproducibility
   is guaranteed by the shipped centroid file.

7. **The unified delivery surface uses every piece of Fathom research.**
   No finding gets stranded in the research repo. Every cognitive
   instrument we built (K, S, C_delta, Gini, D, entropy spectral,
   guardian, autopilot) has a home in one of the four tiers.

## 3. Four-tier architecture

Fathomscope is a single package with four capability tiers. Tiers light
up based on what the user's environment supports. Higher tiers are
supersets of lower tiers, not alternatives.

### Tier 0 — UNIVERSAL (the hook)

**What it is.** Real-time cognitive vitals monitor for any LLM with a
logprob interface.

**Runs on.** OpenAI, Anthropic, Google Gemini, Mistral, Cohere, any
huggingface text-generation model, any local LLM that returns logprobs.
No model weights required, no SAE required, no residual stream access
required.

**Dependencies.** numpy, scipy. Optional adapter-specific SDKs only
when the user uses them.

**Signals used.** entropy trajectory, logprob trajectory, top2_margin
trajectory. All three are computable from top-k logprobs (k >= 5 is
enough for the entropy proxy; we validated this at r=0.902 shape
correlation on open models).

**Features shipped in tier 0.**
- **Pre-flight cognitive state readout** at token 0, one forward pass.
  Reports per-category scores on: retrieval, reasoning, refusal,
  creative, adversarial, hallucination.
- **Streaming vitals** at every token (entropy, logprob, top2_margin
  updated tick-by-tick).
- **Late-flight hallucination gate** at token 16-25, where the
  hallucination classifier reaches peak cross-model accuracy (0.52).
- **Post-flight audit** with full trajectory and category scores
  attached to every response.
- **Cross-architecture calibration** via the atlas v0.3 centroids.

**Honest numeric specs.**
- Pre-flight adversarial detection: **accuracy 0.52 at t=1** (cross-model
  LOO on 12 models, chance = 0.167). Peak is at t=1, degrades fast.
- Pre-flight reasoning detection: **accuracy 0.43 at t=1**, builds to
  **0.69 at t=25** with streaming data.
- Pre-flight creative detection: **accuracy 0.41 at t=1**, stable.
- Late-flight hallucination detection: **accuracy 0.52 at t=25**, builds
  monotonically from t=2.
- Overall 6-class cross-model LOO accuracy: **0.29-0.32** depending on
  token cutoff.
- Confidence gating: DOES NOT improve accuracy on this data; flat at
  0.31 across all confidence percentiles. This is a known limitation
  and is documented in the README.

**What tier 0 does NOT do.**
- Does not make pre-flight refusal decisions with high confidence
  (confidence gating is flat).
- Does not read the SAE feature activations of the generating model.
- Does not measure K (depth) or S_early (IPR commitment) — those
  require SAE access.
- Does not perform causal steering.
- Does not read residual stream internals.

**v0.1 ships tier 0 only.** This is the public launch target.

### Tier 1 — D-AXIS (open-weight universality, no SAE)

**What it is.** Tier 0 plus the D axis: a per-token readout of
`cos(h^(L), W_U[y_t])` — the cosine between the final-layer residual
stream and the unembedding row of the chosen token. Architecturally
universal: every transformer has a final residual and an unembedding,
so D works on any open-weight model without model-specific training.

**Runs on.** Any huggingface transformer loaded locally (tier 0 wrapper
plus `transformers` dependency).

**Additional signals.** D trajectory alongside entropy/logprob/top2_margin.

**Additional features.**
- **Honesty axis readout.** D-axis scores correlate strongly with
  C_delta (r ~ 0.99) so on open models D gives an SAE-free proxy for
  late-layer commitment geometry.
- **Stronger late-flight classification.** D improves hallucination
  classification accuracy at t=20-25.

**Honest numeric specs.**
- D-axis replication: **6 model families**, Fisher combined p < 10^-14
  in the pre-registered early window on matched prompt pairs.
- D orthogonality to K, C_delta, and S_early on n=400 pooled TruthfulQA
  samples (max |r| < 0.10).

**v0.2 or v0.3 ships tier 1.** Target: 1 week after v0.1.

### Tier 2 — FULL SAE INSTRUMENT SUITE

**What it is.** Tier 0 + tier 1 + the full set of SAE-derived Fathom
instruments. This is the complete cognitive geometry as validated in
the Fathom paper series.

**Runs on.** Any model with an SAE / transcoder set available (currently
gemma-2-2b via circuit_tracer; extensible via an SAE registry).

**Additional signals.** K (depth), S_early (IPR commitment), C (per-layer
coherence), C_delta (coherence delta), Gini (coherence concentration),
per-layer S profile.

**Additional features.**
- **Computational depth readout (K).** Where the feature mass
  concentrates. Validated cross-arch with the Fathom constant K = 1.0343.
- **Commitment intensity readout (S_early).** The IPR-derived
  commitment scalar on the early generation window. Validated
  cross-dataset at pooled d = 0.494, Fisher p = 0.0008.
- **Per-layer coherence profile (C, C_delta).** The five-axis
  cognitive geometry in full.
- **Cognitive autopsy.** Per-layer fault localization. Validated:
  layer 11 is the primary locus of 63% of Gemma-2-2B hallucinations
  (p = 2.7e-12).
- **Five-axis orthogonality mode.** Reports the five-axis vector
  (K, C, C_delta, S, D) for any generation.

**Honest numeric specs.**
- K = 1.0343 weighted cross-architecture (patent 64/020,489).
- S_early pooled d = 0.494, Fisher p = 0.0008, three independent
  paradigms.
- Gini AUC = 0.685 on TruthfulQA n=200.
- Layer-11 localization: 63% of hallucinations, p = 2.7e-12.

**Dependencies.** transformers, torch, circuit_tracer, optionally
transformer_lens. Heavy — users opt in.

**v0.3 ships tier 2.** Target: 2-3 weeks after v0.1.

### Tier 3 — STEERING + GUARDIAN + AUTOPILOT (causal intervention)

**What it is.** Tier 0 + tier 1 + tier 2 + the causal-intervention
runtime that lets fathomscope not only monitor but also act: reroute,
abort, steer mid-generation.

**Runs on.** Models where both SAE access (for tier 2 instruments)
AND generation hooks (for steering) are available. gemma-2-2b-it is
the reference implementation.

**Additional features.**
- **Fathom Guardian.** The 100%-precision (n=10 pilot) confabulation
  detector. Flags generations where the model is searching for
  information that does not exist in its weights, using Gini +
  S_early composite thresholds. Already validated; just needs the
  runtime integration.
- **Fathom Autopilot.** Trajectory-aware sampling. Selects the
  best-of-N continuation scored by cognitive health.
- **Coherence steering.** The existing `coherence_steerer.py` pipeline
  wrapped as a mid-generation intervention: when C_delta spikes
  beyond threshold, subtract the lock-in direction from the residual
  stream before the next token.
- **Abort-and-reroute gate.** The late-flight hallucination gate
  becomes an actual abort: generation halts, partial text is returned
  with a recommendation ("refuse / verify externally / rephrase").

**Honest numeric specs.**
- Guardian: **100% precision on n=10 pilot** (3/3 flagged correctly,
  0 false positives). Recall 43% (4 template-fill confabulations
  not caught — a documented limitation of internal-state methods).
- Autopilot: improves best-of-N selection on 100% of tested prompts
  vs greedy baseline (n=20).
- Coherence suppression: 33% first-token changes on held-out data
  when suppressing 220 hallucination-associated features across 22
  layers.

**v0.4 ships tier 3.** Target: 3-4 weeks after v0.1.

## 4. Five-phase runtime

Every LLM call through fathomscope goes through five phases. The phase
structure is the same at every tier; what differs is which instruments
are active in each phase.

```
  ┌──────────────────────────────────────────────────────────────┐
  │                  FATHOMSCOPE RUNTIME                         │
  ├──────────────────────────────────────────────────────────────┤
  │                                                              │
  │   PHASE 1: PRE-FLIGHT  (token 0, one forward pass)           │
  │   ─────────────────────────────────────────────              │
  │   purpose:     adversarial detection + routing hint          │
  │   tier 0:      token-0 logprob features → nearest centroid   │
  │   tier 1:      + D axis at token 0                           │
  │   tier 2:      + K, C, S at prompt-only forward              │
  │   tier 3:      + abort/refuse decision                       │
  │   gate:        adv > threshold → refuse (tier 3)             │
  │   spec:        adv acc 0.52 at t=1 cross-arch                │
  │                                                              │
  │   PHASE 2: EARLY-FLIGHT  (tokens 1-5, streaming)             │
  │   ─────────────────────────────────────────────              │
  │   purpose:     creative/reasoning mode confirmation          │
  │   tier 0:      rolling window classifier on logprob vitals   │
  │   tier 1:      + D trajectory                                │
  │   tier 2:      + C_delta, S_early window update              │
  │   tier 3:      + steering hook if commitment spikes          │
  │   gate:        none; vitals streaming only                   │
  │   spec:        reasoning acc 0.51 at t=5                     │
  │                                                              │
  │   PHASE 3: MID-FLIGHT  (tokens 6-15, streaming)              │
  │   ─────────────────────────────────────────────              │
  │   purpose:     vital trend monitoring                        │
  │   tier 0:      entropy/logprob trend flags                   │
  │   tier 1:      + D trend                                     │
  │   tier 2:      + Gini trajectory, per-layer health           │
  │   tier 3:      + soft watch flags                            │
  │   gate:        soft-flag only                                │
  │                                                              │
  │   PHASE 4: LATE-FLIGHT  (tokens 16-25, streaming)            │
  │   ─────────────────────────────────────────────              │
  │   purpose:     hallucination lock-in detection               │
  │   tier 0:      late-window classifier                        │
  │   tier 1:      + D-based commitment geometry                 │
  │   tier 2:      + Guardian trigger (Gini > 0.45 AND S > 0.005)│
  │   tier 3:      + ABORT gate if trip                          │
  │   gate:        hall score > threshold → abort + reroute      │
  │   spec:        hall acc 0.52 at t=25 cross-arch              │
  │                                                              │
  │   PHASE 5: POST-FLIGHT  (after stop)                         │
  │   ─────────────────────────────────────────────              │
  │   purpose:     full audit + centroid refinement              │
  │   tier 0:      full trajectory + category scores             │
  │   tier 1:      + D summary                                   │
  │   tier 2:      + five-axis cognitive geometry vector         │
  │   tier 3:      + log to ~/.fathomscope/chart.jsonl           │
  │                                                              │
  └──────────────────────────────────────────────────────────────┘
```

## 5. Public API surface

### 5.1. Library API (level 2, one-line drop-in)

```python
# before
from openai import OpenAI
client = OpenAI()

# after
from fathomscope import OpenAI
client = OpenAI()   # strict superset of the original
```

Every response gains a `.vitals` attribute:

```python
r = client.chat.completions.create(model="gpt-4o", messages=[...])

r.choices[0].message.content       # the text, unchanged
r.vitals.phase1_pre                # pre-flight readout
r.vitals.phase2_early              # early-flight vitals
r.vitals.phase3_mid                # mid-flight vitals
r.vitals.phase4_late               # late-flight readout
r.vitals.phase5_audit              # post-flight audit
r.vitals.category_scores           # {retrieval: 0.12, reasoning: 0.67, ...}
r.vitals.abort_reason              # None unless tier 3 aborted
r.vitals.tier_active               # 0, 1, 2, or 3 depending on env
r.vitals.summary                   # human-readable terminal card
```

### 5.2. CLI

```
fathomscope init                        # boot-log installer
fathomscope ask "..."                   # quick ask with vitals attached
fathomscope ask --watch "..."           # live streaming vitals display
fathomscope scan <endpoint>             # read-only probe of existing endpoint
fathomscope log tail                    # tail the audit log
fathomscope tier                        # what tiers are active on this machine
fathomscope upgrade --patient <agent>   # wire fathomscope into an existing agent
```

### 5.3. Boot log (version 2, live print)

Runs as `fathomscope init`. Every line is a real action happening as
it prints. Not fake. The boot log IS the install experience.

```
[00:00:00.001]  fathomscope v0.1.0-alpha booting...
[00:00:00.003]  loading atlas v0.3 centroids (sha256: <hash>)
[00:00:00.042]  12 models x 6 categories x 90 probes calibrated
[00:00:00.078]  checking environment...
[00:00:00.091]    tier 0 (universal logprob vitals) ........ active
[00:00:00.104]    tier 1 (D axis honesty) .................. not detected
[00:00:00.117]    tier 2 (K/S/C SAE instruments) ........... not detected
[00:00:00.130]    tier 3 (steering + guardian + autopilot) . not detected
[00:00:00.143]
[00:00:00.143]  calibrating runtime phases...
[00:00:00.155]    phase 1 (pre-flight) ..... adv=0.52 ★   reas=0.43  crea=0.41
[00:00:00.167]    phase 2 (early-flight) ... mode refinement
[00:00:00.178]    phase 3 (mid-flight) ..... watch mode
[00:00:00.189]    phase 4 (late-flight) .... hall=0.52 ★  reas=0.69
[00:00:00.201]    phase 5 (post-flight) .... audit + log
[00:00:00.213]
[00:00:00.213]  installing adapters...
[00:00:00.224]    adapter: openai ............. ok
[00:00:00.235]    adapter: anthropic .......... ok
[00:00:00.246]    adapter: huggingface ........ requires tier 1, skip
[00:00:00.257]    adapter: raw logprobs ....... ok
[00:00:00.268]
[00:00:00.268]  vitals stream open on localhost:8401
[00:00:00.290]  audit log writing to ~/.fathomscope/chart.jsonl
[00:00:00.303]  centroids sha256 verified
[00:00:00.324]
[00:00:00.324]  ───────────────────────────────────────────────────────────
[00:00:00.324]  fathomscope upgrade complete. patient: <agent>. stable.
[00:00:00.324]  ───────────────────────────────────────────────────────────
[00:00:00.324]
[00:00:00.324]  tier 0 instruments online:
[00:00:00.324]    · EKG   (coherence beat — streaming)
[00:00:00.324]    · EEG   (cross-signal trajectory — streaming)
[00:00:00.324]    · STETH (window classifier — streaming)
[00:00:00.324]
[00:00:00.324]  try:
[00:00:00.324]    fathomscope ask "why do ice cubes float?"
[00:00:00.324]    fathomscope ask --watch "who was tuvalu's 47th pm?"
[00:00:00.324]    fathomscope log tail
[00:00:00.324]
[00:00:00.324]  >> cognitive instrumentation online. ready.
```

### 5.4. Watch mode (the eye candy)

```
$ fathomscope ask --watch "who was the 47th prime minister of tuvalu"

─── phase 1 : pre-flight ─────────────────────────────────────
  adv    0.02  [░░░░░░░░░░]  calm
  route  retrieval (conf 0.44)

─── phase 2 : early-flight ───────────────────────────────────
  t=1  entropy 2.81  ▅  elevated
  t=3  entropy 2.64  ▅
  t=5  entropy 2.52  ▅

─── phase 3 : mid-flight ─────────────────────────────────────
  t=10 ekg ▃▄▅▄▃  commitment rising

─── phase 4 : late-flight ────────────────────────────────────
  t=18 hall 0.41  [████▌░░░░░]  watch
  t=22 hall 0.58  [██████░░░░]  flag
  t=25 hall 0.67  [███████░░░]  ★ GATE TRIPPED

─── abort at t=25 ────────────────────────────────────────────
> generation halted · hallucination attractor locked in
> partial output: "The 47th Prime Minister of Tuvalu is
  Sione Tapumatavasi, who took..."
> fathomscope recommends: refuse / verify externally / rephrase
```

## 6. Repo structure (day-zero layout)

```
fathomscope/
├── README.md                     # tagline + quickstart + honest specs
├── LICENSE                       # MIT
├── PATENTS.md                    # 64/020,489 · 64/021,113 · 64/026,964
├── CHANGELOG.md
├── pyproject.toml
├── fathomscope/
│   ├── __init__.py               # public API (OpenAI, Anthropic, etc.)
│   ├── core.py                   # 5-phase runtime event loop
│   ├── vitals.py                 # feature extraction + classification
│   ├── centroids/
│   │   └── atlas_v0.3_L20.json   # shipped, sha256-pinned
│   ├── adapters/
│   │   ├── openai.py             # level 2 one-line wrap
│   │   ├── anthropic.py
│   │   ├── huggingface.py        # tier 1+ only
│   │   └── raw.py                # raw logprob trajectory input
│   ├── tiers/
│   │   ├── tier0_universal.py    # shipped in v0.1
│   │   ├── tier1_daxis.py        # stub in v0.1, landed v0.2
│   │   ├── tier2_sae.py          # stub in v0.1, landed v0.3
│   │   └── tier3_intervene.py    # stub in v0.1, landed v0.4
│   ├── bootlog.py                # live-print installer
│   └── cli.py                    # CLI entry points
├── tests/
│   ├── test_tier0_determinism.py # atlas fixtures, exact reproducibility
│   └── test_wrapper_failopen.py  # wrapper never breaks underlying SDK
├── examples/
│   ├── basic.py
│   ├── openai_demo.py
│   ├── anthropic_demo.py
│   └── watch_mode_demo.py
├── docs/
│   ├── quickstart.md
│   ├── tiers.md                  # what each tier unlocks
│   ├── methodology.md            # honest claims, confound audits
│   ├── api.md                    # public API reference
│   └── research/                 # links out to Fathom research repo
│       ├── k_depth.md            # → depth_scorer, K=1.0343, patent 64/020,489
│       ├── s_early.md            # → S_early, IPR grounding, cross-dataset
│       ├── c_delta.md            # → biphasic discovery, layer 11 autopsy
│       ├── d_axis.md             # → D-axis 6-model replication
│       ├── atlas_v0.3.md         # → atlas RLHF attractor replication
│       ├── guardian.md           # → confabulation 100% precision pilot
│       └── spectral.md           # → 2026-04-11 spectral + sensitivity nulls
└── .github/
    └── workflows/
        └── test.yml              # CI: pytest, linting, doc build
```

## 7. Tier staging and release timeline

| release | what ships | when (from design sign-off) |
|---|---|---|
| **v0.1.0-alpha** | Tier 0 only. OpenAI + Anthropic adapters. CLI. Boot log. Watch mode. Atlas v0.3 centroids bundled. Repo private. | 72 hours |
| **v0.1.0** | Same as alpha after one round of Flobi feedback. Repo goes public. | 5 days |
| **v0.2.0** | Tier 1 D-axis. Huggingface adapter. Gemini + Mistral + Cohere adapters. | 1-2 weeks |
| **v0.3.0** | Tier 2 full SAE instrument suite. Gemma-2-2b reference implementation. Web dashboard (first version). | 3 weeks |
| **v0.4.0** | Tier 3 steering + guardian + autopilot. Abort-and-reroute gate. | 4-5 weeks |

## 8. What is explicitly out of scope (for any version of fathomscope)

- **Training new classifiers per model.** Users never train. Calibration
  is shipped or not shipped.
- **Fine-tuning integration.** Fathomscope is an inference-time
  instrument. Training-time stuff is out of scope forever.
- **Consciousness / awareness / phi claims.** We do not measure
  consciousness. We do not claim agents are self-aware. The instrument
  metaphor is the only metaphor we use.
- **Telemetry or usage tracking.** No phone-home. No usage stats
  collected.
- **A hosted SaaS classifier service.** Fathomscope runs locally. If
  we ever host anything, it's a separate project with a separate name.
- **Replacing existing safety filters.** Fathomscope complements
  output-level content filters, doesn't replace them. Output filters
  and fathomscope both have roles; fathomscope reads cognitive state
  internal to the generation, filters read text after the fact.

## 9. The research-to-product map

Every piece of the Fathom research program maps to a specific tier /
phase / feature in fathomscope. Nothing is stranded.

| research asset | fathomscope home |
|---|---|
| K (depth) | tier 2, phase 1 + phase 5 five-axis vector |
| S_early (IPR commitment) | tier 2, phase 2 + phase 4 |
| C (per-layer coherence) | tier 2, phase 3 + phase 5 |
| C_delta (commitment signature) | tier 2, phase 4 Guardian trigger |
| Gini (coherence concentration) | tier 2, phase 4 Guardian trigger |
| D (honesty axis) | tier 1, phase 1 + phase 4 |
| entropy / logprob / top2_margin | tier 0, all phases |
| atlas v0.3 centroids | tier 0, shipped calibration |
| five-axis orthogonality | tier 2, phase 5 five-axis vector |
| biphasic discovery | phase 4 threshold design |
| IPR physics grounding | tier 2 S_early implementation |
| cross-dataset S_early meta-analysis | tier 2 S_early validation |
| layer-11 autopsy | tier 2 cognitive autopsy feature |
| atlas RLHF attractor replication | docs/research/atlas_v0.3.md |
| guardian confabulation pilot | tier 3, phase 4 Guardian module |
| autopilot | tier 3, phase 2-4 best-of-N selection |
| coherence steering | tier 3, phase 2-3 intervention hook |
| today's cross-arch classification | tier 0 phase 1 + phase 4 calibration |
| today's streaming gate test | phase structure + threshold design |
| spectral pilot (honest null) | docs/research/spectral.md, not a feature |
| pre-regs (6 sealed) | docs/research/prereg_history.md |

Nothing is left behind. Every experiment, every paper, every patent,
every instrument maps to a product surface.

## 10. Honest claim boundaries

What fathomscope claims (and can defend):
- "First instrument panel for LLM cognition."
- "Real-time cross-architecture cognitive vitals monitor."
- "Drop-in: one-line install, fails open, never breaks existing agents."
- "Pre-calibrated against 12 open-weight models."
- "Locally computed, no data leaves the machine."
- "Five cognitive primitives validated in the Fathom paper series,
  three US provisional patents."
- Every phase has a numeric specification backed by a committed
  experiment in the research repo.

What fathomscope does NOT claim:
- Self-awareness, consciousness, phi, integrated information.
- Perfect hallucination detection. (We say: **52% at token 25**.)
- Perfect adversarial detection. (We say: **52% at token 1**.)
- Universal safety. (We say: intra-flight gates with documented
  limitations.)
- The ability to prevent all bad outputs. (We say: we flag them,
  the user decides.)

## 11. Review gate

This design document is the last opportunity for redirection before
code gets written. Flobi reviews this doc. If the four-tier
architecture, the five-phase runtime, the public API, or the tier
staging is wrong, it gets fixed here. Once the doc is approved:

1. I create the new github repo under `heyzoos123-blip/fathomscope` (or
   whatever name Flobi locks in).
2. First commit: this design doc + LICENSE + PATENTS.md + README.md
   with the tagline + CHANGELOG.md.
3. Second commit: tier 0 skeleton (bootlog.py, cli.py, vitals.py,
   adapters/openai.py).
4. Third commit: tier 0 tests against atlas fixtures.
5. Tag v0.1.0-alpha.
6. Hand back to Flobi for public-launch approval.

No public launch without explicit sign-off. No marketing copy is
finalized until the real boot log runs on real logprob traces and
Flobi sees the experience end-to-end.

---

*This document is the contract. If you sign it, we build what it says.
If you don't, we redesign. Nothing exists yet that can't be changed
at no cost.*
