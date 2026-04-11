# STYXX — Design v0

**Name:** styxx *(final; pypi verified 404 on 2026-04-11)*
**Tagline:** nothing crosses unseen.
**Status:** Pre-repo design. Local staging only. No github repo exists yet.
**Target:** v0.1.0-alpha, tier 0 only, in ~3 hours of focused build.
**Author:** Alexander Rodabaugh (Fathom Intelligence)
**License plan:** MIT code, CC-BY-4.0 data, patent pending (US 64/020,489 · 64/021,113 · 64/026,964)
**Long-form architecture reference:** `docs/FATHOMSCOPE_DESIGN_v0.md` — same four-tier architecture, same five-phase runtime, same honest numeric specs, different name. This doc captures only the styxx-specific decisions.

---

## 1. one-sentence pitch

**styxx is a drop-in cognitive vitals monitor for any LLM-powered agent** — it reads the agent's internal state in real time from signals available on any LLM with a logprob interface, pre-calibrated across 12 open-weight models, locally computed, with five-phase gates (pre-flight / early / mid / late / post-flight) at honest numeric thresholds derived from cross-architecture leave-one-out testing.

**nothing crosses unseen.** every call an agent makes through styxx is a crossing. styxx reads every crossing.

## 2. naming anchors

| surface | value |
|---|---|
| pypi | `pip install styxx` |
| import | `from styxx import OpenAI` (and `Anthropic`, `HuggingFace`, `Raw`) |
| CLI | `styxx init`, `styxx ask`, `styxx ask --watch`, `styxx log tail`, `styxx scan`, `styxx tier`, `styxx upgrade --patient <agent>` |
| config dir | `~/.styxx/` |
| audit log | `~/.styxx/chart.jsonl` |
| vitals stream default port | `localhost:8401` |
| github repo (pending sign-off) | `heyzoos123-blip/styxx` |
| tagline | `nothing crosses unseen.` |
| readme h1 | `styxx — nothing crosses unseen.` |
| tweet line | `the black box, now with a glass face. nothing crosses unseen.` |

## 3. architecture summary (four tiers, five phases)

Tiers (cf. FATHOMSCOPE_DESIGN_v0.md section 3 for full specs):

- **Tier 0 — universal logprob vitals.** Runs on any LLM with a logprob interface. v0.1 ships tier 0 only.
- **Tier 1 — D-axis honesty.** Adds architecturally universal residual-stream readout.
- **Tier 2 — full K/S/C SAE instrument suite.** For SAE-available models.
- **Tier 3 — steering + guardian + autopilot.** Causal intervention.

Phases:

- **Phase 1 — pre-flight** (token 0): adversarial detection + routing. Locked: adv accuracy 0.52 @ t=1 cross-model LOO.
- **Phase 2 — early-flight** (tokens 1-5): mode refinement. Reasoning/creative confirmation.
- **Phase 3 — mid-flight** (tokens 6-15): vital trend monitoring. Watch-only, no hard gates.
- **Phase 4 — late-flight** (tokens 16-25): hallucination lock-in detection. Locked: hall accuracy 0.52 @ t=25 cross-model LOO; reasoning accuracy 0.69 @ t=25.
- **Phase 5 — post-flight**: full audit + centroid refinement.

## 4. v0.1 scope

**In:**
- Tier 0 core (universal logprob vitals)
- OpenAI adapter (level 2 one-line wrap)
- Anthropic adapter (level 2 one-line wrap)
- Raw logprob adapter (user supplies trajectories directly)
- Five-phase runtime event loop
- Atlas v0.3 centroids bundled + sha256-pinned
- CLI: `init`, `ask`, `ask --watch`, `log tail`, `tier`, `scan`
- Live-print bootlog (the upgrade card experience)
- Unit tests against atlas fixtures for exact reproducibility
- Fail-open semantics on every call
- README, LICENSE (MIT), PATENTS.md, CHANGELOG.md

**Explicitly out (deferred to v0.2+):**
- Tier 1 D-axis readout
- Tier 2 SAE instruments (K, S_early, C_delta, Gini)
- Tier 3 steering / guardian / autopilot causal intervention
- Huggingface local model adapter
- Gemini / Mistral / Cohere / Groq adapters
- Web dashboard
- Hosted API service

## 5. honest specs (these go in the README, unredacted)

```
  cross-model LOO at token 1  (pre-flight)
    adversarial .......... 0.52   (chance 0.17)
    reasoning ............ 0.43
    creative ............. 0.41
    retrieval ............ 0.11
    refusal .............. 0.16
    hallucination ........ 0.21

  cross-model LOO at token 25 (late-flight)
    reasoning ............ 0.69
    hallucination ........ 0.52
    retrieval ............ 0.16
    refusal .............. 0.15
    creative ............. 0.29
    adversarial .......... 0.10   (adv peaks at t=1, degrades by t=25)

  what styxx is NOT
    - a perfect hallucination blocker
    - a consciousness measurement
    - a pre-flight refusal gate at high confidence
    - an output-level content filter replacement

  what styxx IS
    - real-time cross-architecture cognitive vitals
    - a streaming instrument panel for LLM cognition
    - locally computed, no data leaves the machine
    - the first thing of its kind
```

## 6. non-negotiable design principles

1. **Honest by construction.** Every number in the README and boot log comes from a committed experiment in the research repo. No rounding up.
2. **Drop-in, fail-open.** The OpenAI adapter is a strict superset of openai-python. If styxx fails for any reason, the underlying call returns normally. We never break the user's existing agent.
3. **Local-first.** No telemetry, no phone-home, no hosted classifier. All math runs on the user's machine.
4. **Zero heavy deps in core.** numpy + scipy only. No torch, no transformers, no circuit_tracer in v0.1.
5. **Calibration shipped, not trained.** Atlas v0.3 centroids bundled, sha256-pinned, versioned.
6. **Tier detection automatic.** `styxx init` detects environment capabilities and lights up tiers accordingly. User configures nothing.

## 7. repo structure (to be created at sign-off)

```
  styxx/
  ├── README.md              # nothing crosses unseen. + quickstart + honest specs
  ├── LICENSE                # MIT
  ├── PATENTS.md             # 3 USPTO provisional filings + license posture
  ├── CHANGELOG.md           # v0.1.0-alpha entry
  ├── pyproject.toml         # name = "styxx", python >= 3.9
  ├── styxx/
  │   ├── __init__.py        # exports OpenAI, Anthropic, Raw, __version__
  │   ├── core.py            # 5-phase runtime event loop
  │   ├── vitals.py          # feature extraction + nearest-centroid classifier
  │   ├── phases.py          # phase 1-5 definitions, thresholds, gate logic
  │   ├── centroids/
  │   │   └── atlas_v0.3.json  # bundled, sha256-pinned
  │   ├── adapters/
  │   │   ├── __init__.py
  │   │   ├── openai.py      # from styxx import OpenAI
  │   │   ├── anthropic.py   # from styxx import Anthropic
  │   │   └── raw.py         # from styxx import Raw
  │   ├── bootlog.py         # functional live-print installer
  │   ├── cli.py             # styxx init, ask, watch, log, tier, scan
  │   └── audit.py           # audit log writer
  ├── tests/
  │   ├── fixtures/          # atlas fixtures for determinism tests
  │   ├── test_determinism.py
  │   ├── test_failopen.py   # wrapper never breaks the underlying SDK
  │   └── test_centroids.py  # centroid file sha256 matches expected
  ├── examples/
  │   ├── basic.py
  │   ├── openai_demo.py
  │   └── watch_mode_demo.py
  ├── docs/
  │   ├── quickstart.md
  │   ├── tiers.md
  │   ├── methodology.md     # honest + confound audits from today
  │   └── research/          # pointers back to the Fathom research repo
  └── .github/
      └── workflows/
          └── test.yml       # pytest on every push
```

## 8. review gate

This design doc is the last redirection opportunity before code lands. Sign off in three moves:

1. **Architecture OK?** Four tiers, five phases, v0.1 scope as above.
2. **Build it?** Explicit greenlight for me to write `styxx/core.py`, `bootlog.py`, `adapters/openai.py`, `cli.py`, tests, and examples under `styxx_staging/` in the research worktree. Local only. No github repo yet.
3. **Staging location OK?** I'll put everything in `styxx_staging/` inside this worktree so it's git-tracked and recoverable. When the repo gets created, files move into the new repo root. Nothing is lost either way.

After sign-off, the next ~3 hours of work are:
- `styxx/core.py` (the five-phase runtime)
- `styxx/vitals.py` (feature extraction + classifier)
- `styxx/centroids/atlas_v0.3.json` (generated from our atlas captures)
- `styxx/adapters/openai.py` (the one-line drop-in)
- `styxx/bootlog.py` (the live-print installer that IS the upgrade card)
- `styxx/cli.py` (the commands)
- `tests/test_determinism.py` (reproducibility guarantee)
- `README.md` draft (with the real numbers)

I will stop after the skeleton is functional and show you the running boot log before any github repo is created.
