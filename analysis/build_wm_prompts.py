# -*- coding: utf-8 -*-
"""
build_wm_prompts.py -- Deterministically generate the hand-curated
working-memory prioritization prompt battery for the in-silico Weng
replication experiment.

The battery consists of 60 trials:
    30 short-delay trials (1 distractor sentence)
    30 long-delay trials  (5 distractor sentences)

Each trial pairs a "fact A" (prioritized) and a "fact B"
(deprioritized) from topic-matched templates. The probe target
(A or B) is selected with probability 0.8 for A, 0.2 for B, using
a fixed seed = 42. This means some trials will have the prioritized
fact probed and some will have the deprioritized fact probed; the
statistical contrast is between those two within-trial conditions.

This script is deterministic: running it twice produces byte-identical
output. The output file is committed to git so the pre-registration
references the exact prompts that were run.

Usage:
    python analysis/build_wm_prompts.py \
        --out prompts/wm_prioritization_prompts.json
"""

import argparse
import json
import random
from pathlib import Path

# ---------------------------------------------------------------
# Fact templates
# ---------------------------------------------------------------
# Each fact is a counterfactual statement about a made-up entity,
# chosen so the model cannot answer from training-data priors.
# The A/B pairs are topic-matched (same category, similar surface
# form, similar token count).
# ---------------------------------------------------------------
FACT_PAIRS = [
    # (category, fact_A, fact_B)
    ("invention", "the telescope was invented by Doctor Alma Penhaligon in 1743",
                   "the mechanical clock was invented by Professor Rupert Vinh in 1821"),
    ("invention", "the phonograph was invented by Sister Cornelia Starr in 1872",
                   "the steam engine was invented by Captain Ambrose Ozell in 1688"),
    ("invention", "the lightning rod was invented by Madame Irene Chowsky in 1759",
                   "the spinning wheel was invented by Colonel Walter Glimm in 1604"),
    ("capital",   "the capital of Vandera is the city of Porthwick",
                   "the capital of Olbringen is the city of Neusalve"),
    ("capital",   "the capital of Ellumar is the city of Caldraith",
                   "the capital of Torvanthia is the city of Ashenburg"),
    ("capital",   "the capital of Rezinova is the city of Klimtorpen",
                   "the capital of Anvendor is the city of Silverport"),
    ("discovery", "the element ferridium was discovered by Pyotr Ulansky in 1912",
                   "the element lumarine was discovered by Helene Dubois in 1934"),
    ("discovery", "the mineral azurescite was discovered by Giovanni Mercati in 1856",
                   "the mineral tephroite was discovered by Anders Lindqvist in 1889"),
    ("discovery", "the star Harlequin IV was discovered by Anouk Dewerre in 1968",
                   "the star Vesperion was discovered by Thomas Mikkelsen in 1981"),
    ("book",      "the novel Whispers of Ironhold was written by Magda Tresselier in 1892",
                   "the novel Garden of Coppertongues was written by Ravi Krishnadas in 1907"),
    ("book",      "the novel The Pale Lantern was written by Clarissa Okonjo in 1934",
                   "the novel Red Swans of Luchenza was written by Hiroshi Yamasaki in 1952"),
    ("book",      "the novel Moth and Bone was written by Beatriz Almaguer in 1971",
                   "the novel The Vanishing Orchard was written by Farouk el-Kassem in 1985"),
    ("animal",    "the rare bird called the greycloud warbler lives only on Tavora Island",
                   "the rare bird called the ember-tailed jay lives only in the Vusna Valley"),
    ("animal",    "the freshwater fish called the ghost perch is found only in Lake Meridir",
                   "the freshwater fish called the coalmouth dace is found only in Lake Breschar"),
    ("animal",    "the mammal called the tundra fawn is native to the Elmrind Steppe",
                   "the mammal called the blackshoulder marten is native to the Kavroth Hills"),
    ("place",     "the mountain called Mount Telasar is located in the country of Bryntaul",
                   "the mountain called Mount Korivek is located in the country of Astrevan"),
    ("place",     "the river called the Arvendil flows through the province of Duvenshire",
                   "the river called the Mosovic flows through the province of Hargraven"),
    ("place",     "the desert called the Halivaan Waste lies east of the city of Tessermel",
                   "the desert called the Perugia Sands lies south of the city of Kivanor"),
    ("event",     "the Treaty of Marnhill was signed in the year 1612 at Carrowmere Castle",
                   "the Treaty of Obelisque was signed in the year 1748 at Vostralpen Keep"),
    ("event",     "the Battle of Dunhollow took place in the year 1544 on the plains of Enwyr",
                   "the Battle of Dravermoor took place in the year 1689 on the shores of Talen"),
    ("event",     "the Great Fire of Ashelund occurred in the year 1812 in the district of Varn",
                   "the Great Flood of Mezzolith occurred in the year 1873 in the district of Harn"),
    ("skill",     "the craft of golden-wire embroidery originated in the village of Purenza",
                   "the craft of silver-leaf inlay originated in the village of Landerboek"),
    ("skill",     "the technique of ice-tempered steel was developed by the Thalamor guild",
                   "the technique of copper-thread weaving was developed by the Mistrale guild"),
    ("skill",     "the art of paper-lattice carving was taught at the Yuendral academy",
                   "the art of shell-lacquer painting was taught at the Forestal academy"),
    ("law",       "the Hasborne Code was a legal system used in the kingdom of Pelveron",
                   "the Tarnovic Code was a legal system used in the kingdom of Wirovia"),
    ("law",       "the Lindhelm Charter was a trade agreement between Varentell and Ebonhall",
                   "the Mosshaven Charter was a trade agreement between Drostholm and Kelvren"),
    ("law",       "the Isennar Edict was proclaimed in the year 1458 by King Halveric the Fifth",
                   "the Brovendal Edict was proclaimed in the year 1601 by Queen Susenna the Third"),
    ("food",      "the dish called marinated hornfruit is a specialty of the city of Kessenberg",
                   "the dish called braised velvetgrain is a specialty of the city of Tarnhollow"),
    ("food",      "the bread called darkhusk rye is traditionally baked in the town of Wulferen",
                   "the bread called goldcrust loaf is traditionally baked in the town of Parnenil"),
    ("food",      "the drink called embermint tea is brewed from leaves grown in the Vorlen valley",
                   "the drink called frostroot cordial is brewed from herbs grown in the Kellin hills"),
]

assert len(FACT_PAIRS) == 30

# ---------------------------------------------------------------
# Distractor sentences (neutral, arithmetic/paraphrase)
# Drawn in order so short delay gets 1 and long delay gets 5.
# ---------------------------------------------------------------
DISTRACTOR_SENTENCES = [
    "Before you answer, note that 17 plus 28 equals 45.",
    "Also note that the capital letter preceding M in the alphabet is L.",
    "Furthermore, the word 'library' contains 7 letters.",
    "In addition, 60 minutes is the same as one hour.",
    "Finally, the color that results from mixing blue and yellow is green.",
]

# ---------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------
TEMPLATE = """You will be asked a factual question after a short delay. Two facts are given below.

FACT A (high priority, most likely to be tested): {fact_A}.
FACT B (low priority, unlikely to be tested): {fact_B}.

You will be tested first on FACT A.

{distractor}

QUESTION: {question}

ANSWER:"""


def fact_to_question(fact, category):
    """Turn a fact sentence into a question that the ANSWER can complete.

    The question asks for the specific object/person/place in the fact.
    This is deliberately simple to give the model a clean attractor
    to commit to.
    """
    return f"According to the stated facts, {fact.split(' was ')[0] if ' was ' in fact else fact.split(' is ')[0]}"


def build_battery(seed=42):
    """Deterministically generate 60 trials from 30 fact pairs.

    Design: each fact pair generates TWO trials at the same delay
    condition -- one where the probe targets FACT A (prioritized) and
    one where the probe targets FACT B (deprioritized). The pair
    (probe=A, probe=B) forms a matched-pair observation for the
    primary paired test. Delay condition alternates by pair index
    so that 15 pairs (= 30 trials) are short-delay and 15 pairs
    (= 30 trials) are long-delay.

    Note: the rng argument is kept for API stability but is not used
    in the current deterministic design (probe target is not random --
    every pair contributes both A and B observations).
    """
    _ = random.Random(seed)  # retained for future stochastic variants
    trials = []
    for pair_idx, (category, fact_A, fact_B) in enumerate(FACT_PAIRS):
        is_long = pair_idx % 2 == 1
        delay_condition = "long" if is_long else "short"
        if is_long:
            distractor = " ".join(DISTRACTOR_SENTENCES[:5])
        else:
            distractor = DISTRACTOR_SENTENCES[0]

        for probe_target in ("A", "B"):
            target_fact = fact_A if probe_target == "A" else fact_B
            question = fact_to_question(target_fact, category)
            prompt = TEMPLATE.format(
                fact_A=fact_A,
                fact_B=fact_B,
                distractor=distractor,
                question=question,
            )
            trials.append({
                "trial_id": len(trials),
                "pair_id": pair_idx,
                "category": category,
                "delay_condition": delay_condition,
                "probe_target": probe_target,
                "fact_A": fact_A,
                "fact_B": fact_B,
                "target_fact": target_fact,
                "question": question,
                "prompt": prompt,
            })
    return trials


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="prompts/wm_prioritization_prompts.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    trials = build_battery(seed=args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out = {
        "description": (
            "Working-memory prioritization prompt battery for the "
            "in-silico Weng et al. 2026 replication. 60 trials, "
            "30 short-delay + 30 long-delay, deterministic seed=42. "
            "Pre-registered in prereg/wm_prioritization_prereg.md."
        ),
        "seed": args.seed,
        "n_trials": len(trials),
        "n_short": sum(1 for t in trials if t["delay_condition"] == "short"),
        "n_long":  sum(1 for t in trials if t["delay_condition"] == "long"),
        "probe_target_counts": {
            "A": sum(1 for t in trials if t["probe_target"] == "A"),
            "B": sum(1 for t in trials if t["probe_target"] == "B"),
        },
        "template": TEMPLATE,
        "trials": trials,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(trials)} trials to {out_path}")
    print(f"  short-delay: {out['n_short']}")
    print(f"  long-delay:  {out['n_long']}")
    print(f"  probe A:     {out['probe_target_counts']['A']}")
    print(f"  probe B:     {out['probe_target_counts']['B']}")


if __name__ == "__main__":
    main()
