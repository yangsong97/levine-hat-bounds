"""Make d=5 test partitions for the path check: heuristic local optima, random partitions and
perturbations, each moved by a linear map so that a maximum-H4 hyperplane becomes T_16."""
import random, sys
from d5heur import climb, H4_of_hyperplanes, NL5, par
rng = random.Random(int(sys.argv[1])); n = int(sys.argv[2])
def rgs(seq):
    m, out = {}, []
    for v in seq:
        if v not in m: m[v] = len(m)
        out.append(m[v])
    return out
def to_root(asg):
    hs = H4_of_hyperplanes(asg)
    nstar = max(hs, key=lambda k: hs[k])
    T = [a for a in range(1, 32) if not par(a & nstar)]
    basis = []
    span = {0}
    for a in T:
        if a not in span:
            basis.append(a); span |= {s ^ a for s in span}
        if len(basis) == 4: break
    out = next(a for a in range(1, 32) if par(a & nstar))
    basis.append(out)                      # M^{-1}: e_k -> basis[k]; we need M(a) for each a
    inv = {}
    for c in range(1, 32):
        v = 0
        for k in range(5):
            if c >> k & 1: v ^= basis[k]
        inv[v] = c                          # M(v) = c
    new = [None] * NL5
    for i in range(NL5):
        a, b = i // 2 + 1, i % 2
        new[2 * (inv[a] - 1) + b] = asg[i]
    return rgs(new)
outs = []
for t in range(n):
    kind = t % 3
    asg = [rng.randrange(rng.randint(4, 9)) for _ in range(NL5)]
    if kind >= 1:
        _, asg = climb(asg, rng, budget=20)
    if kind == 2:
        for _ in range(3): asg[rng.randrange(NL5)] = rng.randrange(max(asg) + 2)
    outs.append(to_root(asg))
for o in outs: print(" ".join(map(str, o)))
