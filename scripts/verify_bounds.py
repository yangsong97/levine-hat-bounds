"""Independent exact check of the three node bounds of bb5Z (partial-exact cap, coupled bound, direct bound).

Reads the CHK lines printed by `CHKDUMP=n bb5Z ...`: a search node (labels 0..i placed), a set A of
non-root hyperplanes, and the three values bb5Z computed. Recomputes each from the definitions with
exact rationals and compares. Units: a sum of hyperplane values, scaled by 2^25 * 105.
  pe(T)      = [E max_x #(retained placed labels of T covering x) + 3/4 p2(T) + 1/2 p1(T)] / 30
  coupled(A) = [E max_x sum_{T in A + root} #(retained placed labels of T covering x) + sum_{T in A} (3/4 p2 + 1/2 p1)] / 30
  direct     = 31 * [E max_x #(retained placed labels covering x) + 3/4 p2 + 1/2 p1] / 62
where p2 / p1 count the unplaced complementary pairs / unplaced single labels (of T, or of all labels).
"""
import sys
from fractions import Fraction
import numpy as np
U = Fraction(2 ** 25 * 105)
def par(v): return bin(v).count("1") & 1
cov = np.array([[par((l // 2 + 1) & x) ^ (l % 2) for x in range(32)] for l in range(62)], dtype=np.int64)
normals = [n for n in range(1, 32) if n != 16]                      # hyperplane index h -> normal vector
def in_h(l, n): return par((l // 2 + 1) & n) == 0
def emax(asg, weights):
    """E max_x sum over placed labels of weight * [retained and covers x], exact."""
    ids = sorted(set(asg[l] for l in range(len(asg)) if weights[l])); K = len(ids); idx = {c: k for k, c in enumerate(ids)}
    B = np.zeros((max(K, 1), 32), dtype=np.int64)
    for l in range(len(asg)):
        if weights[l]: B[idx[asg[l]]] += weights[l] * cov[l]
    S = ((np.arange(2 ** K)[:, None] >> np.arange(max(K, 1))) & 1)[:, :max(K, 1)]
    return Fraction(int((S @ B).max(1).sum()), 2 ** K)
def residual(i, member):
    """3/4 per unplaced pair + 1/2 per unplaced single, over labels > i satisfying member(label)."""
    r = Fraction(0)
    for a in range(1, 32):
        l0, l1 = 2 * (a - 1), 2 * (a - 1) + 1
        if not member(l0): continue
        un = (l0 > i) + (l1 > i)
        r += Fraction(3, 4) if un == 2 else Fraction(1, 2) if un == 1 else 0
    return r
bad = n = 0
for line in sys.stdin:
    if not line.startswith("CHK"): continue
    head, a_s, grp, direct, pes = [p.strip() for p in line[3:].split("|")]
    i = int(head); asg = list(map(int, a_s.split())); assert len(asg) == i + 1
    A, v = map(int, grp.split()); direct = int(direct); n += 1
    # partial-exact caps of all 30 hyperplanes
    for tok in pes.split():
        nn, val = tok.split(":"); nn = int(nn); val = int(val)
        w = [1 if in_h(l, nn) else 0 for l in range(i + 1)]
        mine = (emax(asg, w) + residual(i, lambda l: in_h(l, nn))) / 30 * U
        if val < 2 ** 55 and mine != val: bad += 1; print("PE MISMATCH", i, nn, mine, val)
    # coupled bound
    hs = [normals[h] for h in range(30) if A >> h & 1]
    w = [(1 if l < 30 else 0) + sum(in_h(l, nn) for nn in hs) for l in range(i + 1)]
    mine = emax(asg, w)
    for nn in hs: mine += residual(i, lambda l: in_h(l, nn))
    mine = mine / 30 * U
    if v >= 0 and mine != v: bad += 1; print("COUPLED MISMATCH", i, A, mine, v)
    # direct bound
    if direct >= 0:
        mine = (emax(asg, [1] * (i + 1)) + residual(i, lambda l: True)) * 31 / 62 * U
        if mine != direct: bad += 1; print("DIRECT MISMATCH", i, mine, direct)
print(f"checked {n} nodes (30 partial-exact caps, 1 coupled bound and 1 direct bound each): {bad} mismatches")
