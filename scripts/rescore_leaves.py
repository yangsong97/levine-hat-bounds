"""Rescore the best surviving d=5 partition of each chunk with numpy (exact integers), independent of bb5R."""
import glob, re, sys
from fractions import Fraction
import numpy as np
D = sys.argv[1]; TAU = Fraction(int(sys.argv[2]), 13440)
par = lambda v: bin(v).count("1") & 1
cov = np.array([[par((l // 2 + 1) & x) ^ (l % 2) for x in range(32)] for l in range(62)], dtype=np.int64)
def H5(asg):
    blocks = {}
    for l, c in enumerate(asg): blocks.setdefault(c, []).append(l)
    V = np.array([cov[B].sum(0) for B in blocks.values()], dtype=np.int64)  # K x 32
    K = len(V); tot = 0; step = 1 << min(K, 16)
    lowbits = ((np.arange(step)[:, None] >> np.arange(min(K, 16))) & 1).astype(np.int64)
    low = lowbits @ V[:min(K, 16)]                                         # step x 32
    for hi in range(1 << max(K - 16, 0)):
        add = sum(V[16 + b] for b in range(K - 16) if hi >> b & 1) if K > 16 else 0
        tot += int((low + add).max(1).sum())
    return Fraction(tot, (1 << K) * 62), K
n = worst = 0; best = Fraction(0)
for p in sorted(glob.glob(f"{D}/c*.log")):
    m = re.search(r"^best leaf H5 = ([\d.]+) asg: (.*)$", open(p).read(), re.M)
    if not m: continue
    asg = list(map(int, m.group(2).split())); assert len(asg) == 62
    h, K = H5(asg); n += 1
    if abs(float(h) - float(m.group(1))) > 1e-9: print("MISMATCH", p, float(h), m.group(1)); worst += 1
    if h > TAU: print("ABOVE TAU", p, h); worst += 1
    best = max(best, h)
print(f"rescored {n} chunk-best survivors; mismatches/above-tau: {worst}; max = {best} = {float(best):.9f}; tau = {float(TAU):.9f}")
