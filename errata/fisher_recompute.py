"""recompute the pooled p printed in README.md from the two per-model p values printed beside it.

python standard library only. run: python -I -B errata/fisher_recompute.py

README.md, as of 2026-11-09 (line numbers with the 7-line correction banner in place;
the number in brackets is the line before the banner was inserted):
  line 80 [73]: Gemma-2-2B-IT          ... p = 0.013
  line 81 [74]: Llama-3.2-1B-Instruct  ... p = 0.005
  line 83 [76]: POOLED                 ... p = 0.00018   (also line 43 [36]: "Fisher p = 0.00018")

three combinations are computed: fisher's method (the label the README prints),
stouffer's method with one-sided inputs, and stouffer's method with two-sided inputs
(z from p/2, pooled p reported two-sided).
"""
import math
from statistics import NormalDist

P_GEMMA = 0.013    # README.md line 80 [73], as printed
P_LLAMA = 0.005    # README.md line 81 [74], as printed
P_PRINTED = 0.00018  # README.md lines 43 and 83 [36, 76], as printed

# fisher's method, k = 2 tests -> chi-square with 4 degrees of freedom.
# survival function of chi2(4) has the closed form exp(-x/2) * (1 + x/2).
x = -2.0 * (math.log(P_GEMMA) + math.log(P_LLAMA))
p_fisher = math.exp(-x / 2.0) * (1.0 + x / 2.0)

N = NormalDist()

# stouffer's method, equal weights, one-sided inputs, one-sided pooled p.
z1 = (N.inv_cdf(1.0 - P_GEMMA) + N.inv_cdf(1.0 - P_LLAMA)) / math.sqrt(2.0)
p_stouffer1 = 1.0 - N.cdf(z1)

# stouffer's method, equal weights, two-sided inputs (z from p/2), two-sided pooled p.
z2 = (N.inv_cdf(1.0 - P_GEMMA / 2.0) + N.inv_cdf(1.0 - P_LLAMA / 2.0)) / math.sqrt(2.0)
p_stouffer2 = 2.0 * (1.0 - N.cdf(z2))


def matches(p):
    """true when p prints as the README's 0.00018 at two significant figures."""
    return f"{p:.2g}" == f"{P_PRINTED:.2g}"


print("inputs (as printed in README.md):")
print(f"  gemma p = {P_GEMMA}  (line 80 [73])")
print(f"  llama p = {P_LLAMA}  (line 81 [74])")
print(f"  pooled p printed = {P_PRINTED}  (lines 43, 83 [36, 76]; labelled 'Fisher p' at line 43)")
print("fisher's method (chi-square, 4 df):")
print(f"  x = {x:.6f}")
print(f"  p = {p_fisher:.6g}  (3 s.f.: {p_fisher:.3g})  matches printed: {matches(p_fisher)}")
print("stouffer's method (equal weights, one-sided inputs, one-sided pooled p):")
print(f"  z = {z1:.6f}")
print(f"  p = {p_stouffer1:.6g}  (3 s.f.: {p_stouffer1:.3g})  matches printed: {matches(p_stouffer1)}")
print("stouffer's method (equal weights, two-sided inputs: z from p/2, two-sided pooled p):")
print(f"  z = {z2:.6f}")
print(f"  p = {p_stouffer2:.6g}  (3 s.f.: {p_stouffer2:.3g})  matches printed: {matches(p_stouffer2)}")
print("printed pooled p reproduces under the 'Fisher' label:", matches(p_fisher))
print("printed pooled p reproduces as two-sided stouffer:", matches(p_stouffer2))
