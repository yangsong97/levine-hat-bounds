"""Exact check of Lemma 2 (averaging) and the incidence facts its proof uses, in d = 3, 4, 5.

Written from the definitions only. For a subspace W of the coefficient space, F_W(P) is
(1/|labels of W|) E max_{x in F2^d} #{retained (a,b) with a in W, a.x xor b = 1}.
Proof being checked, for each retention state U and point x:
    sum_W count_W(U, x) = c * count(U, x)          (c = # subspaces of the family containing a, constant)
so  c * max_x count(U, x) <= sum_W max_x count_W(U, x);  take expectations, and use that the
retained part of W's labels is exactly a union of independently retained blocks of P|_W.
Checks:
  1. incidence: each nonzero a lies in the same number c of hyperplanes / 3-dim frames; there are
     2^d - 1 hyperplanes and N_d frames; 14 N_d = m_d c; each point is covered by m_d / 2 labels;
  2. restriction isomorphism: for each hyperplane T and a basis of T, (a,b) -> (coords of a, b)
     and x -> (basis . x) carry T's labels and points onto the (d-1)-dim problem, preserving covers,
     and x -> (basis . x) is onto F2^(d-1);
  3. block law: E over P's retention of max_x count_T equals m_(d-1) H_(d-1)(P|_T) computed on its own;
  4. the pointwise inequality for EVERY retention state, and the expectation inequality, exactly,
     for hyperplanes (d -> d-1) and frames (d -> 3), on random partitions with 1..12 blocks and on
     the optimal d = 4 partition and the best known d = 5 partition.
"""
import itertools, os, random, sys
from collections import Counter
from fractions import Fraction
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")   # repository root (script lives in scripts/)

def dot(a, x): return bin(a & x).count("1") & 1
def labels(d): return [(a, b) for a in range(1, 2 ** d) for b in (0, 1)]
def cover(d): return np.array([[dot(a, x) ^ b for x in range(2 ** d)] for (a, b) in labels(d)], dtype=np.int64)

def span(vs):
    s = {0}
    for v in vs: s |= {u ^ v for u in s}
    return frozenset(s)

def subspaces(d, k):
    out = set()
    for vs in itertools.combinations(range(1, 2 ** d), k):
        s = span(vs)
        if len(s) == 2 ** k: out.add(s)
    return sorted(out, key=sorted)

def H_direct(d, rgs):
    """H_d of a partition given as a list of block ids per label (any ids)."""
    ids = sorted(set(rgs)); K = len(ids); idx = {c: i for i, c in enumerate(ids)}
    cov = cover(d); B = np.zeros((K, 2 ** d), dtype=np.int64)
    for l, c in enumerate(rgs): B[idx[c]] += cov[l]
    S = ((np.arange(2 ** K)[:, None] >> np.arange(K)) & 1)
    return Fraction(int((S @ B).max(1).sum()), 2 ** K * len(rgs))

def check_incidence(d):
    m = 2 * (2 ** d - 1); cov = cover(d); ok = True
    ok &= bool((cov.sum(0) == m // 2).all())
    for fam, k in (("hyperplanes", d - 1), ("frames", 3)):
        if k < 1 or k > d: continue
        Ws = subspaces(d, k); c = {a: sum(a in W for W in Ws) for a in range(1, 2 ** d)}
        cs = set(c.values()); ok &= len(cs) == 1
        if fam == "hyperplanes": ok &= len(Ws) == 2 ** d - 1 and cs == {2 ** (d - 1) - 1}
        if fam == "frames": ok &= 14 * len(Ws) == m * cs.pop() and len(Ws) == (2**d-1)*(2**(d-1)-1)*(2**(d-2)-1)//21
        print(f"  d={d} {fam}: {len(Ws)} subspaces, each nonzero a in {sorted(set(c.values()))} of them")
    return ok

def iso_ok(d):
    """Each hyperplane T, with a chosen basis, is a copy of the (d-1)-dim problem."""
    ok = True
    for T in subspaces(d, d - 1):
        basis = []
        for v in sorted(T - {0}):
            if v not in span(basis): basis.append(v)
        coords = {}
        for bits in range(1, 2 ** (d - 1)):
            a = 0
            for k in range(d - 1):
                if bits >> k & 1: a ^= basis[k]
            coords[a] = bits
        img = {x: sum(dot(basis[k], x) << k for k in range(d - 1)) for x in range(2 ** d)}
        ok &= set(img.values()) == set(range(2 ** (d - 1)))                      # onto
        ok &= all(dot(a, x) == dot(coords[a], img[x]) for a in T - {0} for x in range(2 ** d))
    return ok

def family_check(d, rgs, Ws, sub_d=None):
    """Pointwise + expectation check of H_d(P) <= (1/#W) sum_W H_sub(P|_W) for the family Ws.
    Returns (pointwise_ok, expectation_ok, blocklaw_ok, lhs, rhs)."""
    L = labels(d); m = len(L); cov = cover(d)
    ids = sorted(set(rgs)); K = len(ids); idx = {c: i for i, c in enumerate(ids)}
    S = ((np.arange(2 ** K)[:, None] >> np.arange(K)) & 1)                  # all retention states
    keep = S[:, [idx[c] for c in rgs]]                                      # state x label
    full = (keep @ cov).max(1)                                              # max_x count(U, x)
    c = sum(L[0][0] in W for W in Ws)
    sumW = np.zeros(2 ** K, dtype=np.int64); rhs = Fraction(0); blocklaw = True
    for W in Ws:
        inW = np.array([a in W for (a, b) in L])
        mW = int(inW.sum())
        cntW = (keep[:, inW] @ cov[inW]).max(1)                             # max_x count_W(U, x), x over F2^d
        sumW += cntW
        # block law: the same expectation, recomputed on the restricted partition by itself
        sub = [rgs[l] for l in range(m) if inW[l]]
        Ksub = len(set(sub)); sidx = {cc: i for i, cc in enumerate(sorted(set(sub)))}
        Ss = ((np.arange(2 ** Ksub)[:, None] >> np.arange(Ksub)) & 1)
        ks = Ss[:, [sidx[cc] for cc in sub]]
        e_sub = Fraction(int((ks @ cov[inW]).max(1).sum()), 2 ** Ksub)
        blocklaw &= e_sub == Fraction(int(cntW.sum()), 2 ** K)
        rhs += e_sub / mW
    pointwise = bool((c * full <= sumW).all())
    lhs = Fraction(int(full.sum()), 2 ** K * m)
    rhs /= len(Ws)
    return pointwise, lhs <= rhs, blocklaw, lhs, rhs

def random_partition(m, K, rng):
    while True:
        r = [rng.randrange(K) for _ in range(m)]
        if len(set(r)) == K: return r

def best_d5():
    """6-block partition with H5 = 1401/3968 that survived the tau = 4766 search."""
    for line in open(os.path.join(ROOT, "results/d5/D5_4766/c0.log")):
        if line.startswith("best leaf H5"): return list(map(int, line.split("asg:")[1].split()))

def anneal_d5():
    """8-block partition with H5 = 1401/3968 found by simulated annealing."""
    return list(map(int, open(os.path.join(ROOT, "results/heuristics/best5.txt")).read().split()))

def hyperplane_profile(rgs):
    """Exact H4 of each of the 31 hyperplane restrictions, in units of 1/13440."""
    L = labels(5); cov = cover(5); out = []
    for T in subspaces(5, 4):
        rows = [l for l, (a, b) in enumerate(L) if a in T]; sub = [rgs[l] for l in rows]
        ids = sorted(set(sub)); idx = {c: i for i, c in enumerate(ids)}; K = len(ids)
        B = np.zeros((K, 32), dtype=np.int64)
        for j, l in enumerate(rows): B[idx[sub[j]]] += cov[l]
        S = ((np.arange(2 ** K)[:, None] >> np.arange(K)) & 1)
        out.append(Fraction(int((S @ B).max(1).sum()), 2 ** K * 30) * 13440)
    return out

def opt_d4():
    # the 5-block optimum of Section 4 (paper), labels written (a as 4-bit vector, b)
    blocks = [["0001,0","0010,0","0100,0","0111,0","1001,0","1010,0","1100,0","1111,0"],
              ["0001,1","0010,1","0101,0","0110,0","1000,1","1001,1","1010,1","1011,0","1101,0","1110,0"],
              ["0011,0","1011,1"], ["0011,1","1000,0"],
              ["0100,1","0101,1","0110,1","0111,1","1100,1","1101,1","1110,1","1111,1"]]
    where = {}
    for bi, bl in enumerate(blocks):
        for s in bl:
            a, b = s.split(","); where[(int(a, 2), int(b))] = bi
    return [where[l] for l in labels(4)]

if __name__ == "__main__":
    rng = random.Random(20261106); bad = 0
    print("1. incidence counts")
    for d in (3, 4, 5): bad += not check_incidence(d)
    print("2. restriction isomorphism:", {d: iso_ok(d) for d in (3, 4, 5)}); bad += not all(iso_ok(d) for d in (3, 4, 5))
    print("3-4. averaging, pointwise over every retention state and in expectation")
    named = {(4, "optimal d=4 partition (H4 = 57/160)"): opt_d4(), (5, "best known d=5 partition, 6 blocks"): best_d5(),
             (5, "best known d=5 partition, 8 blocks"): anneal_d5()}
    for (d, name), rgs in named.items():
        H = H_direct(d, rgs); print(f"  {name}: H_{d} = {H} = {float(H):.6f}")
    tests = 0
    for d in (3, 4, 5):
        fams = [("hyperplanes", subspaces(d, d - 1))] + ([("frames", subspaces(d, 3))] if d > 3 else [])
        parts = [(f"random K={K}", random_partition(2 * (2 ** d - 1), K, rng)) for K in range(1, 13) for _ in range(25 if d < 5 else 8)]
        parts += [(n, r) for (dd, n), r in named.items() if dd == d]
        for fam, Ws in fams:
            fails = 0; worst = None
            for name, rgs in parts:
                pw, ex, bl, lhs, rhs = family_check(d, rgs, Ws); tests += 1
                if not (pw and ex and bl): fails += 1; print("   FAIL", d, fam, name, pw, ex, bl, lhs, rhs)
                if name.startswith(("optimal", "best")):
                    print(f"  d={d} {fam:11s} {name}: H = {float(lhs):.6f} <= average {float(rhs):.6f} "
                          f"(= {float(rhs*13440):.2f}/13440)")
            print(f"  d={d} {fam:11s}: {len(parts)} partitions, {fails} failures (pointwise, expectation, block law)")
            bad += fails
    print("Lemma 2 checks:", "ALL PASS" if bad == 0 else f"{bad} FAILURES", f"({tests} partition-family tests)")
    print("5. hyperplane profiles of the two partitions with H5 = 1401/3968 (the averaging ceiling, paper Section 7)")
    for name, rgs in (("8 blocks (annealing)", anneal_d5()), ("6 blocks (tau = 4766 survivor)", best_d5())):
        prof = hyperplane_profile(rgs); avg = sum(prof) / 31
        print(f"  {name}: H5 = {H_direct(5, rgs)}; hyperplane H4 x 13440 = "
              + ", ".join(f"{n} x {float(v):g}" for v, n in sorted(Counter(prof).items(), reverse=True))
              + f"; average {float(avg):.2f}")
