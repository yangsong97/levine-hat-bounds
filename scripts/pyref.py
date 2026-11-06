"""Plain-Python reference scorer (direct subset summation, exact fractions).

Labels (a,b): index i <-> a = i//2 + 1, b = i % 2.  Label covers point x iff parity(a & x) ^ b == 1.
"""
from fractions import Fraction
import sys


def par(v):
    return bin(v).count("1") & 1


def emax(blocks, points, cover):
    """E max_x #{retained labels covering x}, blocks retained independently w.p. 1/2."""
    K = len(blocks)
    vecs = [[sum(cover(l, x) for l in B) for x in points] for B in blocks]
    tot = 0
    for S in range(1 << K):
        cnt = [0] * len(points)
        for b in range(K):
            if S >> b & 1:
                v = vecs[b]
                for t in range(len(points)):
                    cnt[t] += v[t]
        tot += max(cnt)
    return Fraction(tot, 1 << K)


def blocks_of(rgs, labels=None):
    if labels is None:
        labels = range(len(rgs))
    d = {}
    for i in labels:
        d.setdefault(rgs[i], []).append(i)
    return list(d.values())


def cov(l, x):
    a, b = l // 2 + 1, l % 2
    return par(a & x) ^ b


def local_N(rgs14):
    """2^14 * E max for a partition of the 14 labels of AG(3,2)."""
    e = emax(blocks_of(rgs14), range(8), cov)
    return e * (1 << 14)


def frame_scores(rgs30):
    """For each direction n in 1..15, the frame score F_W * 896 (W = n-perp)."""
    out = {}
    for n in range(1, 16):
        labs = [i for i in range(30) if par((i // 2 + 1) & n) == 0]
        e = emax(blocks_of(rgs30, labs), range(16), cov)
        out[n] = e / 14 * 896
    return out


def H4(rgs30):
    return emax(blocks_of(rgs30), range(16), cov) / 30


if __name__ == "__main__":
    w = [0,1,2,0,2,1,2,3,2,4,2,5,2,5,3,6,4,6,5,6,5,6,2,6,2,6,1,6,0,6]
    fs = frame_scores(w)
    for n, s in fs.items():
        print(f"normal {n:2d}: score {s}/896" if s.denominator == 1 else f"normal {n:2d}: score {float(s):.4f}/896")
    tot = sum(fs.values())
    print("sum of scores:", tot, " average F =", tot / 896 / 15)
    print("floor-loss R4 =", sum(int(324 - s) if (324 - s).denominator == 1 else int((324 - s) // 1) for s in fs.values()))
    h = H4(w)
    print("H4(witness) =", h, "=", float(h))
    print("241/672 =", float(Fraction(241, 672)), " witness below it:", h < Fraction(241, 672))
