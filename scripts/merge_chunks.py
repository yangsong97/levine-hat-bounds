"""Merge per-chunk outputs of a chunked bb4s run: envelope (max over chunks per frame-loss
value), high leaves (concatenated), and a coverage check that every chunk finished."""
import glob, re, sys
d = sys.argv[1]
total = None
logs = sorted(glob.glob(f"{d}/c*.log"), key=lambda p: int(re.search(r"c(\d+)\.log", p).group(1)))
covered = []
for p in logs:
    s = open(p).read()
    m = re.search(r"^tasks: (\d+)", s, re.M); total = int(m.group(1))
    c = re.search(r"CHUNK tasks \[(\d+),(\d+)\)", s)
    if not re.search(r"^done", s, re.M):
        print("UNFINISHED", p); continue
    covered.append((int(c.group(1)), int(c.group(2))) if c else (0, total))
covered.sort()
pos = 0
for lo, hi in covered:
    if lo != pos: print("GAP before", lo); break
    pos = hi
print(f"total tasks {total}, contiguous coverage [0,{pos}) -> {'COMPLETE' if pos == total else 'INCOMPLETE'}")
env = {}
for p in glob.glob(f"{d}/c*_env.txt"):
    for line in open(p):
        a, b = map(int, line.split()); env[a] = max(env.get(a, -1), b)
with open(f"{d}/merged_env.txt", "w") as f:
    for a in sorted(env): f.write(f"{a} {env[a]}\n")
n = 0
with open(f"{d}/merged_high.txt", "w") as f:
    for p in glob.glob(f"{d}/c*_high.txt"):
        for line in open(p): f.write(line); n += 1
print(f"envelope points {len(env)}, high leaves {n}")
