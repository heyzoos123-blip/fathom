# -*- coding: utf-8 -*-
"""
styxx — nothing crosses unseen.

a fathom lab product.

The first drop-in cognitive vitals monitor for LLM agents. Reads an
agent's internal state in real time using signals available on any
LLM with a logprob interface (entropy, logprob, top-2 margin),
calibrated cross-architecture on the Fathom Cognitive Atlas v0.3.

Quickstart:

    # Before: from openai import OpenAI
    from styxx import OpenAI
    client = OpenAI()
    r = client.chat.completions.create(model="gpt-4o", messages=[...])

    print(r.choices[0].message.content)   # text, unchanged
    print(r.vitals.phase1_pre)            # pre-flight cognitive state
    print(r.vitals.phase4_late)           # late-flight hallucination read
    print(r.vitals.summary)               # human-readable vitals card

Fail-open: if styxx can't read vitals for any reason, the underlying
SDK call returns its normal response unchanged. styxx never breaks
the user's existing agent.

Honest specs at tier 0 (cross-model LOO, chance = 0.167):
  - phase 1 adversarial     acc 0.52  @ t=1
  - phase 1 reasoning       acc 0.43  @ t=1
  - phase 4 hallucination   acc 0.52  @ t=25
  - phase 4 reasoning       acc 0.69  @ t=25

This is an instrument panel, not a fortune teller.

Research: https://github.com/heyzoos123-blip/fathom
Patents:  US Provisional 64/020,489 · 64/021,113 · 64/026,964
License:  MIT (code), CC-BY-4.0 (atlas data)
"""

__version__ = "0.1.0a0"
__author__ = "flobi"
__license__ = "MIT"
__url__ = "https://github.com/heyzoos123-blip/styxx"
__tagline__ = "nothing crosses unseen."


def OpenAI(*args, **kwargs):
    """Drop-in replacement for openai.OpenAI that emits cognitive vitals.

    Every response gains a .vitals attribute alongside the normal
    .choices. Fails open: if the wrapper can't read vitals, the
    underlying openai call returns its normal response.

    Usage:
        from styxx import OpenAI
        client = OpenAI()  # same interface as openai.OpenAI
        r = client.chat.completions.create(...)
        print(r.vitals.summary)
    """
    from .adapters.openai import OpenAIWithVitals
    return OpenAIWithVitals(*args, **kwargs)


def Raw(*args, **kwargs):
    """Adapter for users who already have a logprob trajectory.

    Usage:
        from styxx import Raw
        styxx = Raw()
        vitals = styxx.read(
            entropy=[...],    # per-token entropy trajectory
            logprob=[...],    # per-token chosen-token logprob
            top2_margin=[...],
        )
        print(vitals.summary)
    """
    from .adapters.raw import RawAdapter
    return RawAdapter(*args, **kwargs)


# Public API
from .core import StyxxRuntime
from .vitals import Vitals, CentroidClassifier

__all__ = [
    "StyxxRuntime",
    "Vitals",
    "CentroidClassifier",
    "OpenAI",
    "Raw",
    "__version__",
    "__tagline__",
]
