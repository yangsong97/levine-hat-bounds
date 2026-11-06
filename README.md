# Upper bounds for Levine's two-player hat problem: V₂ ≤ 2383/6720

An independent, computer-assisted search over the common-row referee relaxations H_d, giving V₂ ≤ 4766/13440 = 2383/6720 ≈ 0.354613.

Two players each wear an infinite stack of independent fair black or white hats. Each sees only the other's stack and points to one hat in their own; they win if both chosen hats are black. Let V₂ be the best possible winning probability. The best known strategies reach 7/20 = 0.35, and V₂ = 7/20 is conjectured.

This repository proves upper bounds with exact, computer-assisted searches over the common-row referee relaxations H_d (V₂ ≤ max H_d for every d):

| Result | Bound | Decimal |
|---|---|---|
| Freiling et al. (published) | 81/224 | 0.361607 |
| Author's earlier write-up ([`paper/write-up-2026-11-02.md`](paper/write-up-2026-11-02.md)) | 3795217/10582656 | 0.358626 |
| **Theorem 1**: max H₄ = 57/160 exactly | V₂ ≤ 57/160 | 0.356250 |
| **Theorem 2**: max H₅ ≤ 4766/13440 | V₂ ≤ 2383/6720 | **0.354613** |
| Conjectured value | 7/20 | 0.350000 |

Theorem 2 closes 60% of the gap between 81/224 and 7/20. Heuristic search finds max H₅ ≈ 1401/3968 ≈ 0.353075 and max H₆ ≥ 45/128 ≈ 0.351563, so the relaxations appear to keep falling toward 7/20.

The write-up is in [`paper/paper.md`](paper/paper.md) ([PDF](paper/paper.pdf)). [`COMPUTATION.md`](COMPUTATION.md) records every run, check and intermediate result.

> **Status.** Computer-assisted and not peer reviewed. The referee reduction and the averaging inequalities (paper, Lemmas 1–2) come from the earlier write-up and were checked by hand, not in code. The write-up's own code package is not included here; everything in `src/` was written independently from its text.

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
| `bb5R` | d = 5 hierarchical search with frame, envelope and level caps |
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
awk '$1>0.3546131' data/orb_S104.txt > data/roots_gt4766.txt   # the 224 orbits with H4 > 4766/13440

# Theorem 2: max H5 <= 4766/13440 = 2383/6720 (about 10 h; resumable, rerun the same command after a stop)
ADAPT=1 scripts/run_d5.sh 4766 data/roots_gt4766.txt 50 data/S104/merged_env.txt 104 data/lists_F4760.bin 4760
python3 scripts/check_d5.py data/D5_4766
python3 scripts/rescore_leaves.py data/D5_4766 4766
```

`check_d5.py` reports `PROVED` when every chunk finished with the same settings, the chunks cover every task exactly once, and no complete partition that survived the caps has exact H₅ above τ. At τ = 4766, 2,237 partitions survive (all in tasks 0–749) and are scored exactly; the largest is 1401/3968, the best value known. `rescore_leaves.py` rescores the best survivor of each chunk in numpy, independently of `bb5R`. The same commands with 4767 (`awk '$1>0.3547'`, 155 roots) or 4769 reproduce the earlier bounds 227/640 and 4769/13440; their logs are in `results/d5/`.

## Open threads

- **Target 4765/13440 ≈ 0.354539**: the last integer target above the averaging ceiling below. It adds the 34 orbits at 4765.25 as roots and would need exact scoring of many more survivors; not attempted.
- **Ceiling of this method**: the hyperplane average already gives 4764.74/13440 ≈ 0.354520 on the best known d = 5 partition, so this search cannot go much lower. Pairs or triples of hyperplanes forced to share a row tighten that to 4756.8 and 4749.5 (paper, Section 7); building a search on triples is the natural next step.
