"""Independent checks of the branch-and-bound output (pure Python + numpy, no shared code).

1. Rescore the highest surviving leaves exactly (Fractions) and compare with the C++ value.
2. Check H4(P) <= (1/15) sum_W F_W(P) exactly on random partitions.
3. Hill-climb directly on H4 from many random starts; report the best value found.
"""
import random, sys
from fractions import Fraction
import numpy as np
from pyref import H4, frame_scores

high_file = sys.argv[1]
target = Fraction(57, 160)

# 1. rescore
rows = [l.split() for l in open(high_file)]
print(f"{len(rows)} leaves with H4 >= 0.355 recorded")
worst = 0
for r in rows[:400]:
    asg = list(map(int, r[1:]))
    h = H4(asg)
    if abs(float(h) - float(r[0])) > 1e-9:
        print("MISMATCH", r[0], h)
        worst += 1
print("rescored", min(400, len(rows)), "leaves, mismatches:", worst)
if rows:
    h0 = H4(list(map(int, rows[0][1:])))
    print("top leaf exact H4 =", h0, "| equals 57/160:", h0 == target)
    fs = frame_scores(list(map(int, rows[0][1:])))
    print("  its frame scores (x/896):", sorted(int(s) if s.denominator == 1 else float(s) for s in fs.values()))
    print("  frame loss R4 =", sum(324 - s for s in fs.values()))

# 2. frame relaxation inequality on random partitions
rng = random.Random(7)
def rand_part(k):
    return [rng.randrange(k) for _ in range(30)]
viol = 0
for t in range(60):
    p = rand_part(rng.randint(2, 9))
    lhs = H4(p)
    rhs = sum(frame_scores(p).values()) / 896 / 15
    if lhs > rhs:
        viol += 1
print("frame-averaging inequality violations on 60 random partitions:", viol)

# 3. hill-climb
COV = np.array([[((bin((i // 2 + 1) & x).count("1") & 1) ^ (i % 2)) for x in range(16)] for i in range(30)], dtype=np.int16)
def h4_fast(asg):
    labels = sorted(set(asg))
    K = len(labels)
    V = np.zeros((K, 16), dtype=np.int32)
    for i, a in enumerate(asg):
        V[labels.index(a)] += COV[i]
    bits = ((np.arange(1 << K)[:, None] >> np.arange(K)[None, :]) & 1).astype(np.int32)
    return (bits @ V).max(axis=1).mean() / 30

best_overall, best_asg = 0, None
for start in range(int(sys.argv[2]) if len(sys.argv) > 2 else 40):
    k = rng.randint(4, 9)
    asg = rand_part(k)
    cur = h4_fast(asg)
    improved = True
    while improved:
        improved = False
        moves = [(i, b) for i in range(30) for b in range(max(asg) + 2) if b != asg[i]]
        rng.shuffle(moves)
        for i, b in moves:
            if b > 11:
                continue
            old = asg[i]
            asg[i] = b
            v = h4_fast(asg)
            if v > cur + 1e-12:
                cur = v
                improved = True
                break
            asg[i] = old
    if cur > best_overall:
        best_overall, best_asg = cur, asg[:]
    print(f"start {start}: local max {cur:.7f}", flush=True)
print("hill-climb best:", best_overall, "exact:", H4(best_asg), "| exceeds 57/160:", H4(best_asg) > target)
