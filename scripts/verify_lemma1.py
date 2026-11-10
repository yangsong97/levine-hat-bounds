"""Exact small-case check of Lemma 1 (referee reduction): V2(n) <= max_P H_d(P).

Written from the definitions only (no code shared with the searches). For small d and n it
computes, in exact rational arithmetic:
  1. the coupling: for every hidden label J=(a,b), the bits A_i = a.R_i xor b over uniform rows R
     are uniform on {0,1}^n (so the original game is unchanged);
  2. V2(n), the true n-hat game value, by brute force over Bob's pure strategies with Alice's
     best response (a common-payoff game, so pure strategies suffice);
  3. the referee game value: rows R public, Bob also sees J, Alice sees R and Z;
     Bob picks g_R : labels -> [n], Alice picks the best row -- brute force over every g_R;
  4. the same value with Alice allowed every x in F2^d instead of only her rows;
  5. max_P H_d(P) by brute force over all partitions of the labels (d <= 2), or the known
     81/224 for d = 3 (Section 3 of the paper);
and checks V2(n) <= referee value <= relaxed value <= max H_d at every step.
"""
import itertools
from fractions import Fraction
import numpy as np

def dot(a, x): return bin(a & x).count("1") & 1
def labels(d): return [(a, b) for a in range(1, 2 ** d) for b in (0, 1)]

def coupling_ok(d, n):
    for (a, b) in labels(d):
        counts = np.zeros(2 ** n, dtype=np.int64)
        for R in itertools.product(range(2 ** d), repeat=n):
            A = sum((dot(a, R[i]) ^ b) << i for i in range(n))
            counts[A] += 1
        if not (counts == counts[0]).all(): return False
    return True

def V2(n):
    best = 0
    for g in itertools.product(range(n), repeat=2 ** n):          # Bob: Alice's string A -> his index
        tot = 0
        for Z in range(2 ** n):                                   # Bob's string, seen by Alice
            win = [A for A in range(2 ** n) if (Z >> g[A]) & 1]
            tot += max(sum((A >> i) & 1 for A in win) for i in range(n))
        best = max(best, tot)
    return Fraction(best, 4 ** n)

def referee_values(d, n):
    """Returns (referee game value, relaxed value with all x allowed)."""
    L = labels(d); m = len(L)
    cov = np.array([[dot(a, x) ^ b for x in range(2 ** d)] for (a, b) in L], dtype=np.int64)  # m x 2^d
    G = np.array(list(itertools.product(range(n), repeat=m)), dtype=np.int64)                 # all g: labels -> [n]
    ref = rel = 0
    for R in itertools.product(range(2 ** d), repeat=n):
        tot_ref = np.zeros(len(G), dtype=np.int64); tot_rel = np.zeros(len(G), dtype=np.int64)
        for Z in range(2 ** n):
            keep = ((Z >> G) & 1)                                   # label J retained iff Z_{g(J)} = 1
            cnt = keep @ cov                                        # (#g) x 2^d : retained labels covering x
            tot_ref += cnt[:, list(R)].max(1)                       # Alice restricted to her rows R_i
            tot_rel += cnt.max(1)                                   # Alice allowed every x
        ref += int(tot_ref.max()); rel += int(tot_rel.max())
    den = (2 ** d) ** n * 2 ** n * m
    return Fraction(ref, den), Fraction(rel, den)

def set_partitions(k):
    def rec(i, rgs, mx):
        if i == k: yield tuple(rgs); return
        for c in range(mx + 2): yield from rec(i + 1, rgs + [c], max(mx, c))
    yield from rec(1, [0], 0)

def H(d, rgs):
    L = labels(d); m = len(L); K = max(rgs) + 1
    cov = np.array([[dot(a, x) ^ b for x in range(2 ** d)] for (a, b) in L], dtype=np.int64)
    B = np.zeros((K, 2 ** d), dtype=np.int64)
    for l, c in enumerate(rgs): B[c] += cov[l]
    S = ((np.arange(2 ** K)[:, None] >> np.arange(K)) & 1)
    return Fraction(int((S @ B).max(1).sum()), 2 ** K * m)

if __name__ == "__main__":
    maxH = {1: max(H(1, p) for p in set_partitions(2)), 2: max(H(2, p) for p in set_partitions(6)), 3: Fraction(81, 224)}
    print("max H_1 =", maxH[1], " max H_2 =", maxH[2], " (brute force);  max H_3 = 81/224 (paper, Section 3)")
    v = {n: V2(n) for n in (1, 2, 3)}
    print("V2(n) by brute force:", {n: f"{x} = {float(x):.6f}" for n, x in v.items()})
    bad = 0
    for d, n in [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2)]:
        c = coupling_ok(d, n); ref, rel = referee_values(d, n)
        ok = c and v[n] <= ref <= rel <= maxH[d]
        bad += not ok
        print(f"d={d} n={n}: coupling uniform={c}  V2={float(v[n]):.6f} <= referee={float(ref):.6f} ({ref}) "
              f"<= all-x={float(rel):.6f} <= maxH_d={float(maxH[d]):.6f}: {'OK' if ok else 'FAIL'}")
    print("Lemma 1 small-case checks:", "ALL PASS" if bad == 0 else f"{bad} FAILURES")
