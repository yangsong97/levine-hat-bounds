"""More d=5 test partitions for the path check, concentrated near the best known partitions:
small perturbations of the two partitions with H5 = 1401/3968, random completions of their roots,
and images of the best partitions with each maximal hyperplane moved to T_16."""
import random, sys
from d5heur import H4_of_hyperplanes, NL5, par
rng = random.Random(int(sys.argv[1])); n = int(sys.argv[2])
def rgs(seq):
    m, out = {}, []
    for v in seq:
        if v not in m: m[v] = len(m)
        out.append(m[v])
    return out
def to_root(asg, nstar=None):
    hs = H4_of_hyperplanes(asg)
    if nstar is None:
        best = max(hs.values()); nstar = rng.choice([k for k in hs if hs[k] == best])
    T = [a for a in range(1, 32) if not par(a & nstar)]
    rng.shuffle(T)
    basis, span = [], {0}
    for a in T:
        if a not in span:
            basis.append(a); span |= {s ^ a for s in span}
        if len(basis) == 4: break
    out = rng.choice([a for a in range(1, 32) if par(a & nstar)])
    basis.append(out)
    inv = {}
    for c in range(1, 32):
        v = 0
        for k in range(5):
            if c >> k & 1: v ^= basis[k]
        inv[v] = c
    new = [None] * NL5
    sw = rng.randrange(2)
    for i in range(NL5):
        a, b = i // 2 + 1, i % 2
        new[2 * (inv[a] - 1) + (b ^ sw)] = asg[i]
    return rgs(new)
bests = [list(map(int, open('data/best5.txt').read().split()))]
for line in open('data/D5_4766/c0.log'):
    if line.startswith('best leaf H5'): bests.append(list(map(int, line.split('asg:')[1].split())))
outs = []
for t in range(n):
    base = list(rng.choice(bests)); kind = t % 4
    if kind == 0: pass                                    # random symmetric image of a best partition
    elif kind in (1, 2):                                  # perturb 1..6 labels
        for _ in range(rng.randint(1, 6 if kind == 2 else 2)): base[rng.randrange(NL5)] = rng.randrange(max(base) + 2)
    else:                                                 # keep the root hyperplane, randomize some outside labels
        base = to_root(base)
        for i in rng.sample(range(30, 62), rng.randint(2, 12)): base[i] = rng.randrange(max(base) + 2)
    outs.append(to_root(base))
for o in outs: print(" ".join(map(str, o)))
