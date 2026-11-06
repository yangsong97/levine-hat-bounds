"""Heuristic search for d=5 partitions with high H5 (lower bound on the d=5 relaxation).

Labels (a,b), a in 1..31, b in {0,1}, index 2(a-1)+b; label covers x in F2^5 iff a.x ^ b = 1.
"""
import random, sys, time
import numpy as np
from fractions import Fraction

def par(v): return bin(v).count("1") & 1
NL5 = 62
COV5 = np.array([[par((i // 2 + 1) & x) ^ (i % 2) for x in range(32)] for i in range(NL5)], dtype=np.int64)
BITS = {}

def emax_total(asg, cov, npts):
    ids = {}
    for v in asg:
        if v not in ids: ids[v] = len(ids)
    K = len(ids)
    V = np.zeros((K, npts), dtype=np.int64)
    for i, v in enumerate(asg):
        V[ids[v]] += cov[i]
    if K not in BITS:
        BITS[K] = ((np.arange(1 << K)[:, None] >> np.arange(K)[None, :]) & 1).astype(np.int64)
    return int((BITS[K] @ V).max(axis=1).sum()), K

def H5(asg):
    tot, K = emax_total(asg, COV5, 32)
    return Fraction(tot, 62 << K)

def H4_of_hyperplanes(asg):
    """H4 of the restriction to each hyperplane n-perp, n in 1..31."""
    out = {}
    for n in range(1, 32):
        labs = [i for i in range(NL5) if not par((i // 2 + 1) & n)]
        sub = [asg[i] for i in labs]
        tot, K = emax_total(sub, COV5[labs], 32)
        out[n] = Fraction(tot, 30 << K)
    return out

def climb(asg, rng, maxK=12, budget=None):
    cur = H5(asg)
    t0 = time.time()
    while True:
        if budget and time.time() - t0 > budget: break
        moves = [(i, b) for i in range(NL5) for b in range(min(max(asg) + 2, maxK)) if b != asg[i]]
        rng.shuffle(moves)
        better = False
        for i, b in moves:
            old = asg[i]; asg[i] = b
            v = H5(asg)
            if v > cur:
                cur = v; better = True; break
            asg[i] = old
        if not better: break
    return cur, asg

if __name__ == "__main__":
    rng = random.Random(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
    tops = [list(map(int, l.split()[1:])) for l in open("results/d4/L72_high.txt") if float(l.split()[0]) > 0.3562499]
    print("57/160 =", float(Fraction(57, 160)))
    # Lifts of optimal d=4 partitions: a = 16+y (y != 0) copies the block of (y, b ^ s*?) etc.
    best = (Fraction(0), None)
    seen = set()
    for P4 in tops[:40]:
        key = tuple(P4)
        if key in seen: continue
        seen.add(key)
        K4 = max(P4) + 1
        for flip in (0, 1):
            for top16 in ("new2", "new1", "join"):
                asg = [0] * NL5
                for i in range(30): asg[i] = P4[i]
                for y in range(1, 16):
                    for b in (0, 1):
                        asg[2 * (16 + y - 1) + b] = P4[2 * (y - 1) + (b ^ flip)]
                if top16 == "new2": asg[30], asg[31] = K4, K4 + 1
                elif top16 == "new1": asg[30] = asg[31] = K4
                else: asg[30], asg[31] = P4[0], P4[1]
                h = H5(asg)
                if h > best[0]: best = (h, asg[:])
    print("best plain lift:", best[0], float(best[0]))
    hs = H4_of_hyperplanes(best[1])
    print("  hyperplane H4 values of that lift:", sorted(float(v) for v in hs.values())[:6], "...")
    # climb from lifts and from random starts
    results = []
    starts = [("lift", best[1][:])] + [("rand", [rng.randrange(rng.randint(4, 9)) for _ in range(NL5)]) for _ in range(int(sys.argv[2]) if len(sys.argv) > 2 else 6)]
    for name, s in starts:
        h, a = climb(s, rng)
        results.append((h, a))
        print(f"{name}: local max {float(h):.7f} = {h}  blocks={max(a)+1}", flush=True)
    h, a = max(results, key=lambda t: t[0])
    print("BEST", float(h), h, a)
