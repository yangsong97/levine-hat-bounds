# Upper bounds for Levine's two-player hat problem: V₂ ≤ 17/48

An independent, computer-assisted search over the common-row referee relaxations H_d, giving V₂ ≤ 4760/13440 = 17/48 ≈ 0.354167.

Two players each wear an infinite stack of independent fair black or white hats. Each sees only the other's stack and points to one hat in their own; they win if both chosen hats are black. Let V₂ be the best possible winning probability. The best known strategies reach 7/20 = 0.35, and V₂ = 7/20 is conjectured.

This repository proves upper bounds with exact, computer-assisted searches over the common-row referee relaxations H_d (V₂ ≤ max H_d for every d):

| Result | Bound | Decimal |
|---|---|---|
| Freiling et al. (published) | 81/224 | 0.361607 |
| Author's earlier write-up ([`paper/write-up-2026-11-02.md`](paper/write-up-2026-11-02.md)) | 3795217/10582656 | 0.358626 |
| **Theorem 1**: max H₄ = 57/160 exactly | V₂ ≤ 57/160 | 0.356250 |
| Theorem 2 of the first version: max H₅ ≤ 4766/13440 | V₂ ≤ 2383/6720 | 0.354613 |
| **Theorem 2**: max H₅ ≤ 4760/13440 | V₂ ≤ 17/48 | **0.354167** |
| Conjectured value | 7/20 | 0.350000 |

Theorem 2 closes 64% of the gap between 81/224 and 7/20. Heuristic search finds max H₅ ≈ 1401/3968 ≈ 0.353075 and max H₆ ≥ 45/128 ≈ 0.351563, so the relaxations appear to keep falling toward 7/20.

The write-up is in [`paper/paper.md`](paper/paper.md) ([PDF](paper/paper.pdf)). [`COMPUTATION.md`](COMPUTATION.md) records every run, check and intermediate result.

> **Status.** Computer-assisted and not peer reviewed. The referee reduction and the averaging inequalities (paper, Lemmas 1–2) come from the earlier write-up and were checked by hand; `scripts/verify_lemma1.py` and `scripts/verify_lemma2.py` also test them exactly in code on small cases and sampled partitions (paper, Section 6). The write-up's own code package is not included here; everything in `src/` was written independently from its text.

## Layout

| Path | Contents |
|---|---|
| `src/` | C++20 programs (SSE4.1; built with `make` into `bin/`) |
| `scripts/` | Python 3 checks and chunked-run drivers (Python needs only `numpy`) |
| `paper/` | The paper (Markdown, PDF, figure) |
| `results/` | Logs and small result files from every run used in the paper |
| `data/` | Working tables, created by the commands below (ignored by Git, about 2.5 GB) |

Main programs:

| Program | Role |
|---|---|
| `local3b` | Scores all Bell(14) = 190,899,322 frame partitions exactly; writes the full table and prefix maxima |
| `rootclass` | Orbits of top frame patterns under AGL(3,2) × swap |
| `bb4`, `bb4y` | d = 4 branch-and-bound (`bb4y` adds kernel and lex-leader symmetry breaking and streams leaves to disk) |
| `orbit4`, `expand4` | Orbit representatives under AGL(4,2) × swap; all labelled images, sorted |
| `bb5V` | d = 5 hierarchical search with frame, envelope and level caps, plus coupled and direct bounds (Theorem 2); `bb5Z` is the same search without the pair-table speedups |
| `bb5R` | Earlier d = 5 search with hyperplane caps only (the 4766 run) |
| `anneal`, `anneal6` | Simulated annealing for max H_d (d ≤ 6) |

The other files in `src/` are earlier versions kept for the cross-checks described in `COMPUTATION.md`.

## Reproducing the results

Run from the repository root. Times are on 2 cores with 8 GB of memory; every program uses all available cores.

```bash
make
mkdir -p data

# Frame table and root patterns (about 1 min, then about 30 min in total)
bin/local3b data 81152
bin/rootclass data 81664 > data/roots_319plus.txt
bin/rootclass data 81152 > data/roots_317plus.txt
python3 scripts/spotcheck_local.py data

# Theorem 1: max H4 = 57/160 (about 3 min)
bin/bb4 data 72 data/roots_319plus.txt data/L72 4
python3 scripts/verify.py data/L72_high.txt 12

# d = 4 lists above 4760/13440 (about 2 h; resumable)
PROG=bb4y DSYM=8 HIGHF=4760 scripts/run_chunks.sh 104 2000
python3 scripts/merge_chunks.py data/S104
bin/orbit4  < data/S104/merged_high.txt > data/orb_S104.txt
bin/expand4 data/lists_F4760.bin < data/orb_S104.txt
# all 291 orbits in orb_S104.txt have H4 > 4760/13440 (the smallest is 4763.5)

# Theorem 2: max H5 <= 4760/13440 = 17/48 (about 9 h of computing with bb5V; resumable, rerun after a stop)
scripts/run_d5v.sh 4760 data/orb_S104.txt 800 data/S104/merged_env.txt 104 data/lists_F4760.bin 4760
python3 scripts/check_d5.py data/Z5_4760
python3 scripts/rescore_leaves.py data/Z5_4760 4760

# soundness checks of the d = 5 bounds: path replay and exact-rational recomputation
for f in results/d5/paths.txt results/d5/paths_lift.txt results/d5/paths3.txt results/d5/paths2.txt; do
  bin/bb5V data path $f data/pv 3 0 data/S104/merged_env.txt 104 data/lists_F4760.bin 4760 | grep 'path check'; done
CHKDUMP=40 CHKEVERY=5000 TLO=20000 THI=20002 bin/bb5V data 4760 data/orb_S104.txt data/chk 3 0 \
  data/S104/merged_env.txt 104 data/lists_F4760.bin 4760 | grep '^CHK' | python3 scripts/verify_bounds.py

# the earlier bound 2383/6720 (bb5R, about 10 h)
awk '$1>0.3546131' data/orb_S104.txt > data/roots_gt4766.txt   # the 224 orbits with H4 > 4766/13440
ADAPT=1 scripts/run_d5.sh 4766 data/roots_gt4766.txt 50 data/S104/merged_env.txt 104 data/lists_F4760.bin 4760
python3 scripts/check_d5.py data/D5_4766

# Lemmas 1-2 (referee reduction, averaging): exact small-case and sampled checks (about 15 s)
python3 scripts/verify_lemma1.py
python3 scripts/verify_lemma2.py
```

`check_d5.py` reports `PROVED` when every chunk finished with the same settings, the chunks cover every task exactly once, and no complete partition that survived the bounds has exact H₅ above τ. At τ = 4760 one partition survives (17 blocks, task range of `c6350.log`) and is scored exactly: H₅ = 44747/126976 ≈ 0.352405. `rescore_leaves.py` rescores it in numpy, independently of the search code. The first 20 chunks (tasks 0–12,749) ran on `bb5Z` and the other 37 on `bb5V`; the two engines explore the identical search tree and give identical path-replay checksums (see `COMPUTATION.md`). At τ = 4766 with `bb5R`, 2,237 partitions survive the hyperplane caps alone and are scored exactly; the largest is 1401/3968, the best value known. Logs of the earlier targets 4769, 4767 and 4766 are in `results/d5/`.

## Open threads

- **Below 17/48**: the d = 4 partitions with H₄ ≤ 4760 are not enumerated (there are none in (4760, 4763.5)). A target under 4760 needs the d = 4 lists at a lower floor (`bb4y` at a larger L with `HIGHF` below 4760) for both the roots and the level caps, then the same `bb5V` search.
- **Ceiling of the hyperplane average**: two partitions reach the best known H₅ = 1401/3968. The 8-block one found by annealing has hyperplane average 4764.74/13440; the 6-block one that survived the 4766 search has 4766.32/13440. The hyperplane caps alone therefore cannot go below 4766.32; Theorem 2 gets to 4760 because the coupled bound lets groups of hyperplanes share a row (paper, Sections 5 and 7). Triples sharing a row tighten the 8-block partition to 4749.5, so the coupled bound has room left.
