# Computation notes

Detailed record of every run behind the results in [README.md](README.md) and [the paper](paper/paper.md). File names refer to `src/` (C++), `scripts/` (Python and shell) and `results/` (logs).

Written from the 2 November 2026 write-up only; no code from the original package was used.

## Setup (same objective as the write-up)

Labels (a,b), a ∈ F₂⁴∖{0}, b ∈ F₂, indexed i = 2(a−1)+b (a read as an integer 1..15).
Label i covers point x ∈ F₂⁴ iff a·x ⊕ b = 1. For a partition P of the 30 labels, retain each
block independently with probability 1/2 and set

H₄(P) = (1/30) · E max_x #{retained labels covering x}.

V₂ ≤ max_P H₄(P) by the write-up's referee argument. A frame is the set of 14 labels with
a·n = 0 for a direction n; F_W is the same objective restricted to a frame, and
H₄(P) ≤ (1/15) Σ_W F_W(P|_W).

## Programs

| File | What it does |
|---|---|
| `local3.cpp` | Scores all Bell(14) = 190,899,322 partitions of one frame exactly (Gray code over block subsets, integer arithmetic). Writes the full table and, for every prefix of the restricted-growth string, the maximum over all completions. |
| `rootclass.cpp` | Orbits of the top local patterns under AGL(3,2) × complement swap (order 2688); checks the score is constant on every orbit. |
| `bb4.cpp` | Branch-and-bound over d = 4 partitions for a loss threshold L (see below). |
| `bb4v.cpp` | Same search with an extra check at every leaf (incremental frame index = direct recomputation) and an optional dump of all leaves. |
| `pyref.py` | Plain-Python reference scorer with exact fractions (direct subset summation). |
| `spotcheck_local.py` | Rescoring of 300 local partitions against the C++ table. |
| `verify.py` | Exact rescoring of the best leaves; frame-averaging inequality on random partitions; independent hill-climb on H₄. |
| `completeness.py` | Finds partitions with small frame loss independently, maps each to the search's canonical form with an element of AGL(4,2) × swap, and checks it was visited as a leaf. |

## Branch-and-bound (`bb4.cpp`)

Fix L and T_L = (4860 − L)/13440 = 81/224 − L/(15·896). Work in units N = 2¹⁴·E max (so a frame scoring 324/896 has N = 82944).

1. Every partition has a frame of maximum score. AGL(4,2) × swap is transitive on frames and induces the full order-2688 group on each frame, so we may assume that frame is direction 8 (labels 0..13) and its restriction is an orbit representative R.
2. Every frame scores at most N_R, so if 15(82944 − N_R) ≥ 256 L the frame bound already gives H₄ ≤ T_L. Only representatives with smaller loss are searched; the program checks against the full histogram that every such level is present.
3. The 16 remaining labels are assigned in a fixed order as a restricted-growth string. For each of the other 14 frames a linear isomorphism to the canonical frame sends its 6 shared labels to positions 0..5 and its 8 new labels to positions 6..13 in assignment order, so its partial partition is always a prefix and the prefix-maximum table bounds its final score. Each bound is clamped at N_R.
4. Prune when the sum of the 15 frame bounds ≤ 256(4860 − L); this implies H₄ ≤ T_L for every completion.
5. At each surviving complete partition, H₄ is computed exactly.

Conclusion for each run: max_P H₄(P) ≤ max(T_L, best exact leaf).

## Results

All runs: 2 cores, x86-64.

**Local table (`local3`).** All 190,899,322 frame partitions scored in 42 s. Levels above 318/896 match the write-up exactly: 3,920 / 16,128 / 1,344 / 159,712 / 75,264 (total 256,368); the next level is 318/896 (545,104 partitions); the maximum is 324/896 = 81/224. 300 partitions rescored in plain Python: 0 mismatches.

**Symmetry (`rootclass`).** All 2,688 elements of AGL(3,2) × swap preserve every one of the 256,368 top scores. Orbits: 5 at 324, 8 at 322, 1 at 321, 79 at 320, 31 at 319.

**Witness.** The published loss-28 witness has 10 frames at 324/896, one at 320, four at 318; average 151/420; loss 28. Its exact H₄ is 45/128 ≈ 0.35156.

**Branch-and-bound (`bb4`).**

| L | Roots | Nodes | Surviving leaves | Best exact leaf | Bound on max H₄ | Time |
|---:|---:|---:|---:|---|---|---:|
| 40 | 13 | 18.0 M | 6,728 | 57/160 | 241/672 ≈ 0.358631 (reproduces the write-up) | 0.4 s |
| 48 | 14 | 79.5 M | 72,792 | 57/160 | 401/1120 ≈ 0.358036 | 1.9 s |
| 56 | 14 | 343.5 M | 530,680 | 57/160 | 1201/3360 ≈ 0.357440 | 7.5 s |
| 64 | 14 | 1.47 B | 3,190,392 | 57/160 | 1199/3360 ≈ 0.356845 | 34 s |
| 72 | 93 | 6.79 B | 16,671,664 | 57/160 (832 leaves) | **57/160 = 0.35625, attained** | 156 s |
| 80 | 124 | 33.0 B | 89,969,576 | 57/160 | all partitions above 239/672 listed | 12 min |
| 88 | 124 | 27.6 B (with kernel symmetry breaking) | 93,598,369 | 57/160 | all partitions above 4772/13440 listed | 21 min |
| 96 | 352 | (`bb4x`, 52 chunks, see `results/d4/S96/`) | | 57/160 | all partitions above 4764/13440 listed | ≈ 45 min |
| 104 | 352 | (`bb4y`, 85 chunks, see `results/d4/S104/`) | | 57/160 | all partitions above 4760/13440 listed | ≈ 3.5 h |

`bb4x.cpp` adds a lex-leader test (depth ≤ 8) under all lifts of the root pattern's stabilizer; at L = 72 it halves the nodes again with an identical envelope (per exact frame-loss value) and the same 11 optimal orbits. At L = 96 the values above 4764/13440 are 4788, 4781, 4774, 4770.5, 4768.75, 4767.875, 4767, 4765.25 (258 orbits, 20,420,960 labelled partitions), and no partition with frame loss ≥ 52 exceeds 4760.

`bb4s.cpp` adds the 16-element kernel symmetry (it acts regularly on labels 14..29; label 14 must have minimal type) and records the envelope; at L = 72 it is 5.5× faster with an identical envelope and the same 11 optimal orbits. Above 4772/13440 there are exactly three values: 4788 (11 orbits), 4781 (48) and 4774 (57).

Since T₇₂ = 57/160 and no leaf exceeds it, **max_P H₄(P) = 57/160 exactly, so V₂ ≤ 57/160 = 0.35625.** Every optimal partition has 12 frames at 324/896, frame loss 36 or 40, and 5 or 6 blocks.

**Checks.**
- Exact Python rescoring of the 400 best leaves and of all 832 optimal leaves: no mismatches.
- H₄ ≤ average frame score on 60 random partitions: no violations.
- Incremental frame bookkeeping equals direct recomputation at all 530,680 leaves of the L = 56 run.
- Completeness: partitions found by an independent Python hill-climb on frame loss, mapped into canonical form by AGL(4,2) × swap, were all leaves: 15/15 at L = 56 (root level 324) and 8/8 at L = 72 (root level 322, using a climb that forbids 324 frames).
- No partition without a 324 frame has frame loss below 56.
- An independent hill-climb on H₄ from 12 random starts peaked at 341/960 and never exceeded 57/160.

**What this does not cover.** The referee reduction (V₂ ≤ max H_d) and the averaging inequalities are taken from the write-up. I checked both proofs by hand, and `scripts/verify_lemma1.py` and `scripts/verify_lemma2.py` test them exactly on small cases and sampled partitions (see *Checks of Lemmas 1 and 2* below). Those tests support the proofs; they do not replace them. This is not peer reviewed.

**Limits of the method.** 57/160 is the exact value of the d = 4 relaxation, so no d = 4 argument can go lower. The write-up's frame-loss recursion to d ≥ 5 cannot help either: the optimum is attained at frame loss 36–40, below the loss 72 at which the frame bound reaches 57/160, so any concave majorant is flat on [40, 72] (equivalently, the best penalty constant is c₄ = 1, which keeps L_d/N_d = 72/15 at every d). Going lower needs a genuine d = 5 computation, e.g. a compatibility search over the 31 hyperplanes of F₂⁵ using the near-optimal d = 4 partitions.

## d = 5

**Objective.** 62 labels (a,b), a ∈ F₂⁵∖{0}, points x ∈ F₂⁵, H₅ as before with 1/62. V₂ ≤ max H₅ ≤ max H₄, and H₅(P) ≤ (1/31) Σ_T H₄(P|_T) over the 31 hyperplanes T of the a-space.

**Search (`bb5.cpp` → `bb5e.cpp` → `bb5L.cpp` → `bb5S.cpp`).** For a target τ = m/13440:
1. Every P has a hyperplane with maximal H₄(P|_T). If that maximum is ≤ τ we are done. Otherwise AGL(5,2) × swap (transitive on hyperplanes, inducing all of AGL(4,2) × swap on each) moves it to T₁₆ (labels 0..29) with restriction equal to an orbit representative. The roots are therefore all d = 4 orbits with H₄ > τ, taken from a complete d = 4 enumeration at a loss L with T_L ≤ τ.
2. The 32 outside labels are assigned in a fixed order. Each of the 30 other hyperplanes is bounded by the minimum of:
   - the root value h₀;
   - its frame bound, using the prefix tables for its 14 non-root frames (140 non-root frames in total);
   - the **envelope** g*(r), the largest exact H₄ among d = 4 partitions with frame loss ≥ r. This comes from the d = 4 run, with 4860 − L as the floor for loss ≥ L;
   - the **level cap**: the d = 4 partitions above the floor fall into a few value levels (4788, 4781, 4774 for L = 88). All their labelled versions (4.43 M for L = 88) are stored, and a hyperplane drops to the next level as soon as its partial assignment, mapped to d = 4 labels, extends none of the partitions at the current level;
   - its exact H₄ once all its labels are placed.
3. Prune when h₀ + Σ bounds ≤ 31τ, or when a completed hyperplane beats h₀ (that partition is covered by another root).
4. Symmetry: the 32-element kernel acts regularly on the outside labels (label 30 must have minimal type), plus a lex-leader test at depth ≤ 12 under all lifts of the root's stabilizer (512–18,432 elements per root). Each lifted element was checked on random completions to preserve H₅ and the root partition.
5. Long runs are split into task chunks (`run_chunks.sh` for d = 4, `run_d5.sh` for d = 5); `merge_chunks.py` and `check_d5.py` verify that the chunks cover every task exactly once with identical settings and that no chunk reports a failed bound. A complete partition that survives the caps (a leaf) is scored exactly; it is a violation only if its exact H₅ exceeds τ.

**Checks.**
- Exact H₅ of a heuristic partition matches Python (1401/3968).
- At every surviving leaf, H₅ ≤ (h₀ + Σ bounds)/31; this never failed.
- Path replay: for 70 full partitions (random, hill-climbed, and lifts of optimal d = 4 partitions with up to 16 hyperplanes ≥ 4781), every hyperplane bound at every depth is at least that hyperplane's true final H₄, and the total bound is at least 31·H₅. 2,310 node checks, 0 violations.
- The d = 4 map of every hyperplane preserves H₄ on random partitions.
- 1,000 random checks that the lifted root symmetries preserve H₅ and the root partition.
- d = 4 lists: orbit sizes from `orbit4` match the number of distinct images from `expand4` for every orbit.

**Results (0 surviving leaves except at 4766, where all 2,237 are scored exactly and none exceeds τ).**

| Target m (τ = m/13440) | τ | Roots | d = 4 data used | Nodes | Time | Log |
|---:|---|---:|---|---:|---:|---|
| 4787 | 0.356176 | 11 | frame bounds only | 1.91 B | 208 s | `results/d5/P4787.log` (`bb5`) |
| 4784 | 0.355952 | 59 | L = 88 lists + envelope | 29 k | 0.1 s | `results/d5/S4784.log` |
| 4780 | 0.355655 | 59 | L = 88 | 3.7 M | 11 s | `results/d5/S4780.log` |
| 4777 | 0.355432 | 59 | L = 88 | 110 M | 294 s (3 jobs on 2 cores) | `results/d5/S4777.log` |
| 4774 | 0.355208 = 341/960 | 59 | L = 96 lists (RMQ levels) + envelope | 209 M | 289 s | `results/d5/R4774.log` (`bb5R`, DSYM=12) |
| 4771 | 0.354985 | 131 | L = 96 | 3.46 B | 54 min | `results/d5/R4771.log` |
| 4769 | 0.354836 | 131 | L = 104 lists (floor 4760) + envelope | 7.94 B | 114 min (13 chunks) | `results/d5/D5_4769/` (`run_d5.sh`, checked by `check_d5.py`) |
| 4767 | 0.354688 = 227/640 | 155 | L = 104 lists (floor 4760) + envelope | 17.8 B | 5.2 h (40 chunks: 6 × 1000 tasks, then 34 × 250) | `results/d5/D5_4767/` |
| **4766** | **0.354613 = 2383/6720** | 224 | L = 104 lists (floor 4760) + envelope | 31.2 B | 9.9 h (133 chunks: 27 × 250, 98 × 50, then 8 adaptive) | `results/d5/D5_4766/` |

Target 4767 ran in 40 chunks across several 2-hour background jobs and one machine restart (a killed chunk is simply rerun from its start). All 40 logs record τ = 4767, 155 roots, the same 23,848,160-partition level lists, 14,341 tasks and lex-leader depth 12, and each reports 0 surviving leaves, 0 violations and 0 consistency failures.

Target 4766 uses 224 roots: the orbits with H₄ ≥ 4767, since no d = 4 value lies strictly between 4765.25 and 4767 (`results/orbits/roots104_gt4766.txt`, equal to `awk '$1>0.3546131'` on `orb_S104.txt`). It ran in 133 chunks over about 13 hours of wall-clock time, with 2-hour background jobs and machine restarts; a stopped chunk is deleted and rerun. Chunk times varied from seconds to 79 minutes, so after task 6,750 the chunks were cut to 50 tasks and after task 11,650 sized adaptively (`ADAPT=1`). All 133 logs record τ = 4766, 224 roots, 19,821 tasks and lex-leader depth 12, with 0 violations and 0 consistency failures (`check_d5.py`). 2,237 complete partitions survived the caps, all in tasks 0–749 (chunks c0, c250, c500: 482, 769 and 986), whose roots lie nearest the best known partition. Each was scored exactly; the largest is 1401/3968 ≈ 0.353075, and each also satisfied H₅ ≤ (capped hyperplane sum)/31, a free check of the caps on 2,237 real partitions. `rescore_leaves.py` rescores the best survivor of each of those chunks in numpy with exact integers and agrees.

**Current result: max_P H₅(P) ≤ 4766/13440 = 2383/6720, so V₂ ≤ 2383/6720 ≈ 0.354613** (below 57/160 = 0.35625, the exact d = 4 value).

The roots for τ ≥ 4774 are the 59 d = 4 orbits at 4788 (11) and 4781 (48); the L = 88 enumeration shows no other d = 4 value above 4772 besides 4774 (57 orbits). A heuristic search finds d = 5 partitions only up to 1401/3968 ≈ 0.35307, so the true d = 5 value is probably well below the proved bound.

## Reproduce

See [README.md](README.md#reproducing-the-results) for build and run commands.

## Checks of Lemmas 1 and 2

Both scripts are written from the definitions alone and share no code with the searches; every quantity is an exact rational.

`verify_lemma1.py` (referee reduction, V₂ ≤ max H_d):
- **Coupling.** For every label (a, b) with a ≠ 0, the bits a·Rᵢ ⊕ b over uniform rows R are uniform on {0,1}ⁿ (d ≤ 3, n ≤ 3).
- **True game.** V₂(n) by brute force over Bob's pure strategies with Alice's best response: V₂(1) = 1/4, V₂(2) = 5/16, V₂(3) = 11/32.
- **Referee game.** Rows public, Bob also sees J, brute force over every map g_R from labels to positions, with Alice restricted to her rows and then allowed every x.
- **Chain.** V₂(n) ≤ referee value ≤ all-x value ≤ max H_d in all 8 cases (d = 1, 2 with n ≤ 3; d = 3 with n ≤ 2). Here max H₁ and max H₂ are found by brute force and max H₃ = 81/224.

`verify_lemma2.py` (averaging):
- **Incidence.** Each nonzero a lies in 2^(d−1) − 1 of the 2^d − 1 hyperplanes and in 7 (d = 4) or 35 (d = 5) of the N_d = 15 or 155 frames; 14 N_d = m_d c; every point is covered by m_d/2 labels.
- **Copies.** Each hyperplane, with a basis, is an exact copy of the (d − 1)-dimensional problem (covers preserved, point map onto).
- **Block law.** The expected maximum over a hyperplane's labels under P's retention equals the same quantity computed on P|_T alone.
- **Inequality.** c · max_x count ≤ Σ_W max_x count_W holds for every retention state, and the expectation inequality holds. Tested for hyperplanes (d = 3, 4, 5) and frames (d = 4, 5) on 1,098 partition–family tests: random partitions with 1–12 blocks, the optimal d = 4 partition, and both best d = 5 partitions. 0 failures.
- **Negative controls.** Five deliberately false variants are each detected: allowing a = 0, overweighting by c + 1, bounding by the minimum hyperplane, corrupting one cover entry, and claiming the referee value is below V₂(n).

## How far can this go? (heuristic ceilings)

`anneal.cpp` / `anneal6.cpp`: simulated annealing on H_d (exact scoring). It finds the d = 4 optimum 57/160 on its first restart.

| d | max H_d | status |
|---:|---|---|
| 1, 2 | 3/8 = 0.375 | exact (brute force) |
| 3 | 81/224 ≈ 0.361607 | exact (local table) |
| 4 | 57/160 = 0.35625 | exact (this work) |
| 5 | ≥ 1401/3968 ≈ 0.353075 | best of 11,360 annealing restarts (20 min); proved ≤ 0.354613 |
| 6 | ≥ 45/128 ≈ 0.351563 | best of 6,493 restarts (25 min), seeded half from d = 5 lifts |

Two partitions reach H₅ = 1401/3968. The 8-block one found by annealing (`results/heuristics/best5.txt`) has hyperplane values 8 × 4788, 20 × 4781, 1 × 4606, 2 × 4588.5 (×1/13440), average 4764.74. The 6-block one that survived the τ = 4766 search (best leaf of `results/d5/D5_4766/c0.log`) has 16 × 4788, 8 × 4781, 4 × 4774, 1 × 4606, 2 × 4599, average 4766.32 (both printed by `verify_lemma2.py`). So the maximum of the hyperplane-average relaxation used by `bb5*` lies between 4766.32 and 4767 (the 4767 proof had no survivors). The 4766 proof is below that ceiling: it holds only because the 2,237 partitions the caps cannot exclude were scored exactly, and every lower target adds roots and survivors. The successive drops of max H_d (0.00536, 0.00318, ≥ 0.00151) are consistent with the relaxations converging to about 0.350, i.e. to the conjectured value 7/20, but that is extrapolation, not proof.
