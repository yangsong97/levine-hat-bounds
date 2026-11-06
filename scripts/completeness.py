"""Completeness test for the branch-and-bound.

Find partitions independently (hill-climbing on the frame-score sum, from random starts),
keep those whose exact frame loss is < L, map each by an element of AGL(4,2) x swap so that
a maximum-score frame becomes direction 8 and its restriction to labels 0..13 is one of the
root representatives, and check the image is among the dumped leaves.
"""
import random, sys
from fractions import Fraction
from itertools import product
from pyref import frame_scores

L = int(sys.argv[1]); leaves_file = sys.argv[2]; roots_file = sys.argv[3]; ntests = int(sys.argv[4])
leaves = set(l.strip() for l in open(leaves_file))
roots = set()
for line in open(roots_file):
    t = list(map(int, line.split()))
    roots.add(tuple(t[2:]))
print(f"{len(leaves)} leaves, {len(roots)} root reps")

def par(v): return bin(v).count("1") & 1
def rgs(seq):
    m, out = {}, []
    for v in seq:
        if v not in m: m[v] = len(m)
        out.append(m[v])
    return out

# GL(4,2) as images of basis vectors
mats = []
for u1, u2, u3, u4 in product(range(1, 16), repeat=4):
    span = {0}
    ok = True
    for u in (u1, u2, u3, u4):
        if u in span: ok = False; break
        span |= {s ^ u for s in span}
    if ok: mats.append((u1, u2, u3, u4))
assert len(mats) == 20160
def apply(M, a):
    r = 0
    for k in range(4):
        if a >> k & 1: r ^= M[k]
    return r
# frame n maps to frame n' where {Ma : a.n=0} = {a' : a'.n'=0}
def image_dir(M, n):
    img = {apply(M, a) for a in range(1, 16) if not par(a & n)}
    for n2 in range(1, 16):
        if all(not par(a & n2) for a in img): return n2

def canonical_image(asg, nmax):
    Ms = [M for M in mats if image_dir(M, nmax) == 8]
    for M in Ms:
        for c in range(16):
            for s in (0, 1):
                new = [None] * 30
                for i in range(30):
                    a, b = i // 2 + 1, i % 2
                    a2, b2 = apply(M, a), b ^ par(a & c) ^ s
                    new[2 * (a2 - 1) + b2] = asg[i]
                r = rgs(new)
                if tuple(r[:14]) in roots:
                    return r
    return None

rng = random.Random(2026)
import numpy as np
COV = np.array([[((bin((i // 2 + 1) & x).count("1") & 1) ^ (i % 2)) for x in range(16)] for i in range(30)], dtype=np.int64)
FR = [[i for i in range(30) if not par((i // 2 + 1) & n)] for n in range(16)]
def frame_scores_np(asg):
    out = {}
    for n in range(1, 16):
        labs = FR[n]
        ids = sorted(set(asg[i] for i in labs)); K = len(ids)
        V = np.zeros((K, 16), dtype=np.int64)
        for i in labs: V[ids.index(asg[i])] += COV[i]
        bits = ((np.arange(1 << K)[:, None] >> np.arange(K)[None, :]) & 1)
        tot = int((bits @ V).max(axis=1).sum())          # exact integer
        out[n] = Fraction(tot * 896, 14 * (1 << K))       # score in units of 1/896
    return out
def loss(asg):
    fs = frame_scores_np(asg)
    return 15 * 324 - sum(fs.values()), fs

found = tested = 0
attempts = 0
while tested < ntests and attempts < 400:
    attempts += 1
    asg = [rng.randrange(rng.randint(4, 8)) for _ in range(30)]
    cur, _ = loss(asg)
    improved = True
    while improved and cur >= L:
        improved = False
        moves = [(i, b) for i in range(30) for b in range(max(asg) + 2) if b != asg[i]]
        rng.shuffle(moves)
        for i, b in moves[:120]:
            old = asg[i]; asg[i] = b
            v, _ = loss(asg)
            if v < cur: cur = v; improved = True; break
            asg[i] = old
    if cur >= L:
        continue
    lo, fs = loss(asg)
    nmax = max(fs, key=lambda n: fs[n])
    img = canonical_image(asg, nmax)
    tested += 1
    key = "".join(chr(97 + v) for v in img) if img else None
    ok = key in leaves
    found += ok
    print(f"test {tested}: frame loss {float(lo):.2f}, max frame {nmax} at {float(fs[nmax]):.2f}/896 -> {'FOUND' if ok else 'MISSING'}", flush=True)
print(f"{found}/{tested} independently found partitions with frame loss < {L} are leaves of the search")
