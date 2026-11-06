"""Check a chunked d=5 proof directory (scripts/run_d5.sh output).

Requires: every chunk finished; all chunks record the same tau, root count, lex-leader depth
and task total; the chunks cover [0, total) contiguously; and no chunk reports a violation
(a surviving complete partition with exact H5 > tau) or a consistency failure (exact H5 above
its own capped hyperplane average). Surviving partitions with exact H5 <= tau are allowed:
bb5R scores every one exactly. scripts/rescore_leaves.py rescores each chunk's best survivor
independently.
Usage: python3 scripts/check_d5.py data/D5_<TAU>
"""
import glob, re, sys

d = sys.argv[1]
cov, nodes, secs, bad, leaves, best = [], 0, 0.0, 0, 0, 0.0
settings = set()
for p in glob.glob(f"{d}/c*.log"):
    s = open(p).read()
    m = re.search(r"^total tasks generated: (\d+)", s, re.M)
    if not m: print("STARTING", p); bad += 1; continue
    t = re.search(r"^tau = (\S+) = [\d.]+; roots = (\d+)", s, re.M)
    lx = re.search(r"^lex-leader depth (\d+)", s, re.M)
    settings.add((t and t.groups(), lx and lx.group(1), int(m.group(1))))
    c = re.search(r"CHUNK tasks \[(\d+),(\d+)\)", s)
    dn = re.search(r"^done in ([\d.]+)s: nodes=(\d+) .*leaves=(\d+) violations\(H5>tau\)=(\d+) consistencyFail=(\d+)", s, re.M)
    if not dn or not c: print("UNFINISHED", p); bad += 1; continue
    secs += float(dn.group(1)); nodes += int(dn.group(2))
    if int(dn.group(4)) or int(dn.group(5)): print("VIOLATIONS in", p, dn.group(0)); bad += 1
    leaves += int(dn.group(3))
    bl = re.search(r"^best leaf H5 = ([\d.]+)", s, re.M)
    if bl: best = max(best, float(bl.group(1)))
    cov.append((int(c.group(1)), int(c.group(2))))
if len(settings) != 1: print("SETTINGS DIFFER between chunks:", sorted(settings, key=str)); bad += 1
(tau_roots, lexdepth, total), = list(settings)[:1] or [((None, None), None, None)]
cov.sort(); pos = 0
for lo, hi in cov:
    if lo != pos: print("GAP or OVERLAP at", pos, "next chunk starts at", lo); bad += 1; break
    pos = hi
ok = bad == 0 and total is not None and pos == total
tau = tau_roots[0] if tau_roots else "?"
roots = tau_roots[1] if tau_roots else "?"
verdict = ('PROVED (no surviving partition)' if leaves == 0 else
           f'PROVED ({leaves} surviving partitions, all scored exactly; best H5 = {best:.9f} <= tau)') if ok else 'NOT COMPLETE'
print(f"{d}: tau {tau}, {roots} roots, lex-leader depth {lexdepth}, {len(cov)} chunks, tasks {total}, "
      f"covered [0,{pos}), nodes {nodes:.3g}, chunk time {secs/60:.1f} min -> {verdict}")
