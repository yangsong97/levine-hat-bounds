"""Spot-check the C++ local table against direct Python subset summation."""
import random, struct, sys
from functools import lru_cache
from pyref import local_N

DATA = sys.argv[1] if len(sys.argv) > 1 else "data"
NL = 14


@lru_cache(None)
def D(rem, k):
    if rem == 0:
        return 1
    return k * D(rem - 1, k) + D(rem - 1, k + 1)


def rank(rgs):
    r, k = 0, 0
    for p, v in enumerate(rgs):
        for c in range(v):
            r += D(NL - p - 1, k + (1 if c == k else 0))
        if v == k:
            k += 1
    return r


def rand_rgs(rng):
    out, k = [], 0
    for _ in range(NL):
        v = rng.randint(0, k)
        out.append(v)
        if v == k:
            k += 1
    return out


def ntab(f, idx):
    f.seek(4 * idx)
    return struct.unpack("<I", f.read(4))[0]


rng = random.Random(12345)
bad = 0
with open(f"{DATA}/Ntab.bin", "rb") as f:
    # random partitions (biased toward few blocks by the generator, so also try "many-block" ones)
    samples = [rand_rgs(rng) for _ in range(150)]
    # top records
    top = open(f"{DATA}/top.bin", "rb").read()
    recs = [top[i:i + 18] for i in range(0, len(top), 18)]
    for rec in rng.sample(recs, 150):
        idx = struct.unpack("<I", rec[:4])[0]
        rgs = list(rec[4:])
        assert rank(rgs) == idx, "rank mismatch"
        samples.append(rgs)
    for rgs in samples:
        n_c = ntab(f, rank(rgs))
        n_p = local_N(rgs)
        if n_p != n_c:
            bad += 1
            print("MISMATCH", rgs, n_c, n_p)
print(f"checked {len(samples)} partitions, mismatches: {bad}")
print("D(14,0) =", D(14, 0))
