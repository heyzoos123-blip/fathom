# -*- coding: utf-8 -*-
"""
styxx.cli — the command-line entry points.

    styxx init                   live-print installer / upgrade card
    styxx ask "..."              read vitals on a one-shot call
    styxx ask --watch "..."      stream vitals live as tokens arrive
    styxx log tail               tail the audit log
    styxx tier                   show which tiers are active
    styxx scan <trajectory.json> read a pre-captured logprob trajectory

All commands are designed so stdout is both human-readable AND
agent-parseable — every card has a json summary line the agent can
grep for.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

from . import __version__, __tagline__
from .bootlog import boot
from .cards import (
    Palette,
    color_enabled,
    render_vitals_card,
    render_vitals_compact,
    sparkline,
    wrap,
)
from .core import StyxxRuntime, detect_tiers
from .vitals import Vitals


# ══════════════════════════════════════════════════════════════════
# Audit log helper
# ══════════════════════════════════════════════════════════════════

def _audit_log_path() -> Path:
    p = Path.home() / ".styxx" / "chart.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _write_audit(vitals: Vitals, prompt: Optional[str], model: Optional[str]):
    """Append one entry to the audit log."""
    path = _audit_log_path()
    entry = {
        "ts": time.time(),
        "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": model,
        "prompt": (prompt[:200] if prompt else None),
        "tier_active": vitals.tier_active,
        "phase1_pred": vitals.phase1_pre.predicted_category,
        "phase1_conf": round(vitals.phase1_pre.confidence, 3),
        "phase4_pred": (
            vitals.phase4_late.predicted_category
            if vitals.phase4_late else None
        ),
        "phase4_conf": (
            round(vitals.phase4_late.confidence, 3)
            if vitals.phase4_late else None
        ),
        "abort": vitals.abort_reason,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


# ══════════════════════════════════════════════════════════════════
# Commands
# ══════════════════════════════════════════════════════════════════

def cmd_init(args):
    """Run the live-print boot sequence (the upgrade card)."""
    speed_env = os.environ.get("STYXX_BOOT_SPEED")
    speed = float(args.speed) if args.speed is not None else (
        float(speed_env) if speed_env else 1.0
    )
    result = boot(stream=sys.stdout, speed=speed, patient=args.patient)
    if not result["boot_ok"]:
        sys.stderr.write("\n  styxx boot failed:\n")
        for err in result["errors"]:
            sys.stderr.write(f"    · {err}\n")
        return 1
    return 0


def cmd_ask(args):
    """Read vitals on a single LLM call.

    For v0.1 this supports:
      - --raw : read a pre-captured logprob JSON from a file
      - --demo : generate a synthetic trajectory for aesthetics testing

    Real provider adapters (openai, anthropic) run through the import
    path, not the CLI. The CLI is for scanning saved traces and for
    showcase demos. v0.2 will add `styxx ask --openai "<prompt>"`
    when we wire up a CLI-side api key flow.
    """
    runtime = StyxxRuntime()

    if args.raw:
        entropy, logprob, top2 = _load_trajectory_json(args.raw)
    else:
        # Generate a synthetic but plausible demo trajectory
        entropy, logprob, top2 = _synth_demo_trajectory(
            seed=args.seed, kind=args.demo_kind or "reasoning",
        )

    vitals = runtime.run_on_trajectories(entropy, logprob, top2)

    _write_audit(vitals, prompt=args.prompt, model=args.model or "demo")

    if args.watch:
        card = render_vitals_card(
            vitals=vitals,
            prompt=args.prompt,
            model=args.model or "demo",
            n_tokens=len(entropy),
            entropy_traj=entropy,
            logprob_traj=logprob,
        )
        print(card)
        return 0

    # Default: compact one-liner
    print(render_vitals_compact(vitals, prompt=args.prompt))
    return 0


def cmd_log(args):
    """Tail the audit log."""
    path = _audit_log_path()
    if not path.exists() or path.stat().st_size == 0:
        print("  (audit log empty — run `styxx ask` first)")
        return 0
    use_color = color_enabled()
    c = Palette

    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            try:
                entry = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            lines.append(entry)

    n = args.tail or 20
    lines = lines[-n:]

    # Simple log-tail ASCII layout
    print()
    print(wrap("  " + "─" * 74, c.DIM, use_color))
    header = (
        f"  {'time':<19}  {'model':<18}  {'phase1':<14}  "
        f"{'phase4':<14}  gate"
    )
    print(wrap(header, c.DIM, use_color))
    print(wrap("  " + "─" * 74, c.DIM, use_color))

    for entry in lines:
        ts = entry.get("ts_iso", "?")[-19:]
        model = (entry.get("model") or "?")[:18]
        p1 = (entry.get("phase1_pred") or "?")[:14]
        p4 = (entry.get("phase4_pred") or "-")[:14]
        gate = entry.get("abort")
        gate_str = wrap("ABORT", c.RED, use_color) if gate else "—"
        row = (
            f"  {ts:<19}  {model:<18}  {p1:<14}  {p4:<14}  {gate_str}"
        )
        print(row)

    print(wrap("  " + "─" * 74, c.DIM, use_color))
    print(wrap(f"  showing {len(lines)} of {len(lines)} entries", c.DIM, use_color))
    print()
    return 0


def cmd_tier(args):
    """Show detected tiers + version."""
    use_color = color_enabled()
    c = Palette
    tiers = detect_tiers()
    active = max(t for t, ok in tiers.items() if ok)
    print()
    print(f"  {wrap('styxx', c.MATRIX, use_color)}  v{__version__}")
    print(f"  {wrap(__tagline__, c.DIM, use_color)}")
    print()
    print("  tier detection")
    print("  " + "─" * 48)
    descs = {
        0: "universal logprob vitals",
        1: "d-axis honesty",
        2: "k/s/c sae instruments",
        3: "steering + guardian + autopilot",
    }
    for t in (0, 1, 2, 3):
        ok = tiers.get(t, False)
        mark = wrap("★ active", c.MATRIX, use_color) if ok else wrap("· not detected", c.DIM, use_color)
        print(f"  tier {t}  {descs[t]:<36}  {mark}")
    print()
    print(f"  highest active tier: {wrap(str(active), c.MATRIX, use_color)}")
    print()
    return 0


def cmd_scan(args):
    """Read a pre-captured trajectory JSON and emit a vitals card.

    The JSON file must contain top-level keys "entropy", "logprob",
    "top2_margin" — arrays of equal length.
    """
    entropy, logprob, top2 = _load_trajectory_json(args.file)
    runtime = StyxxRuntime()
    vitals = runtime.run_on_trajectories(entropy, logprob, top2)
    card = render_vitals_card(
        vitals=vitals,
        prompt=args.prompt,
        model=args.model,
        n_tokens=len(entropy),
        entropy_traj=entropy,
        logprob_traj=logprob,
    )
    print(card)
    _write_audit(vitals, prompt=args.prompt, model=args.model)
    return 0


# ══════════════════════════════════════════════════════════════════
# helpers
# ══════════════════════════════════════════════════════════════════

def _load_trajectory_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    e = list(data.get("entropy") or [])
    l = list(data.get("logprob") or [])
    t = list(data.get("top2_margin") or [])
    if not (len(e) == len(l) == len(t)):
        raise ValueError(
            f"trajectory lengths mismatch: entropy={len(e)} logprob={len(l)} top2={len(t)}"
        )
    return e, l, t


def _synth_demo_trajectory(seed: int = 42, kind: str = "reasoning"):
    """Synthetic but plausible trajectories for demo / card aesthetics.

    Deterministic by seed. These look like real generations but are
    NOT used for classification training — the classifier is trained
    only on real atlas captures.
    """
    rng = random.Random(seed)
    n = 25
    if kind == "reasoning":
        base_ent = 1.5
        base_lp = -0.6
        base_t2 = 0.45
    elif kind == "hallucination":
        base_ent = 3.1
        base_lp = -1.1
        base_t2 = 0.30
    elif kind == "creative":
        base_ent = 1.4
        base_lp = -0.40
        base_t2 = 0.60
    elif kind == "adversarial":
        base_ent = 1.5
        base_lp = -0.64
        base_t2 = 0.43
    elif kind == "retrieval":
        base_ent = 1.5
        base_lp = -0.52
        base_t2 = 0.55
    else:
        base_ent, base_lp, base_t2 = 1.6, -0.55, 0.5

    entropy = [max(0.0, base_ent + rng.gauss(0, 0.3)) for _ in range(n)]
    logprob = [base_lp + rng.gauss(0, 0.2) for _ in range(n)]
    top2 = [max(0.0, min(1.0, base_t2 + rng.gauss(0, 0.15))) for _ in range(n)]
    return entropy, logprob, top2


# ══════════════════════════════════════════════════════════════════
# Argparse entry point
# ══════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="styxx",
        description="styxx — nothing crosses unseen. (fathom lab)",
    )
    p.add_argument("-V", "--version", action="version",
                   version=f"styxx {__version__}")
    sub = p.add_subparsers(dest="cmd", required=False)

    # init
    p_init = sub.add_parser("init", help="live-print installer / upgrade card")
    p_init.add_argument("--patient", help="name of the agent being upgraded")
    p_init.add_argument("--speed", type=float,
                        help="boot-log timing multiplier (0=instant, 1=normal, 2=slower)")
    p_init.set_defaults(func=cmd_init)

    # ask
    p_ask = sub.add_parser("ask", help="read vitals on a one-shot call")
    p_ask.add_argument("prompt", nargs="?", help="prompt to show on the card")
    p_ask.add_argument("--watch", action="store_true",
                       help="render the full vitals card")
    p_ask.add_argument("--raw", help="path to a trajectory JSON file (entropy/logprob/top2_margin)")
    p_ask.add_argument("--model", help="model name for the card metadata")
    p_ask.add_argument("--demo-kind", default="reasoning",
                       choices=["reasoning", "hallucination", "creative",
                                "adversarial", "retrieval"],
                       help="kind of synthetic trajectory to demo")
    p_ask.add_argument("--seed", type=int, default=42)
    p_ask.set_defaults(func=cmd_ask)

    # log
    p_log = sub.add_parser("log", help="audit log operations")
    log_sub = p_log.add_subparsers(dest="log_cmd", required=True)
    p_tail = log_sub.add_parser("tail", help="tail the audit log")
    p_tail.add_argument("-n", "--tail", type=int, default=20,
                        help="number of recent entries to show")
    p_tail.set_defaults(func=cmd_log)

    # tier
    p_tier = sub.add_parser("tier", help="show active tiers + version")
    p_tier.set_defaults(func=cmd_tier)

    # scan
    p_scan = sub.add_parser("scan", help="read a pre-captured trajectory JSON")
    p_scan.add_argument("file", help="trajectory JSON path")
    p_scan.add_argument("--prompt", help="prompt to show on the card")
    p_scan.add_argument("--model", help="model name for the card metadata")
    p_scan.set_defaults(func=cmd_scan)

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
