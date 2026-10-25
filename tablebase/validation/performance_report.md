# Tablebase Generator Performance Experiments

## Decision

No performance experiment produced a sufficiently consistent gain to justify a
production change. The experimental packed Stage 2 path and benchmark-only CLI
were removed. Production worker-count and chunk-size settings were left
unchanged.

The active Stage 2 implementation is again the existing per-state parent merge.
The generator retains its pre-experiment 10,000-candidate Stage 2 chunk size and
its existing worker-count rule (`max(1, cpu_count - TB_THREADS_SUBTRACT)`, with
`TB_THREADS_SUBTRACT = 4`).

## Measurements

### Packed result serialization vs. existing merge

Both paths produced matching output fingerprints and the same state/edge
counts, but timings did not establish a repeatable improvement:

| Sample | Existing path median | Packed path median | Result |
|---|---:|---:|---|
| 13,530 candidates, 8 chunks, 3 rounds | 0.38 s | 0.39 s | Packed 4% slower |
| 53,128 candidates, 32 chunks, 3 rounds | 1.72 s | 1.72 s | Effectively tied |
| 77,128 candidates, 32 chunks, 5 rounds + warm-up | 4.60 s | 4.78 s | Packed 4% slower |

A smaller sample with 2 workers appeared to favor packed serialization, but its
short timings were not corroborated by the larger comparison. The packed path
was therefore not retained.

### Worker count

The worker sweep processed the same 49,376 candidates at each setting, with
matching fingerprints:

| Workers | Median Stage 2 sample time |
|---:|---:|
| 1 | 9.06 s |
| 2 | 6.09 s |
| 4 | 4.70 s |
| 8 | 3.75 s |

A follow-up sweep found:

| Workers | Median Stage 2 sample time |
|---:|---:|
| 8 | 3.72 s |
| 12 | 2.91 s |
| 16 | 3.01 s |

Twelve workers were fastest in these samples. This agrees with the existing
worker-count rule on the tested machine, so no worker setting was changed.
These samples were short and should not be treated as a universal optimum.

### Candidate chunk size

The controlled comparison processed the same 120,000 candidate positions for
each chunk size, with 12 workers. Outputs matched:

| Candidates per chunk | 3-round median | 5-round median |
|---:|---:|---:|
| 1,000 | 3.19 s | 7.62 s |
| 2,500 | 3.16 s | 7.75 s |
| 5,000 | 3.30 s | 7.82 s |
| 10,000 | 4.35 s | 8.24 s |

The first run was noisy; the longer run still showed changing throughput during
the test, including a substantial improvement across all chunk sizes in later
rounds. The apparent difference is not reliable enough to justify changing the
production 10,000-candidate chunk size.

### Transition correctness and profiling

Added `tablebase/bench_transitions.py` to sample deterministic states, verify
transition outputs against finished `_tb8.bin` values, save/compare normalized
golden outputs, and profile the reference Python transition function.

The reference run checked 1,500 sampled states per available default table.
All five passed with zero Bellman mismatches:

| Table | Kind | States/s | Estimated Stage 2 at 12 workers* |
|---|---|---:|---:|
| `K_Rook_Rook_K` | 4-man same-side | 10,686 | 18.4 s |
| `K_Pawn_Rook_K` | 4-man same-side | 13,529 | 1.2 min |
| `K_Rook_vs_Rook_K` | 4-man cross | 10,120 | 39.2 s |
| `K_Knight_Queen_Rook_K` | 5-man same-side | 11,674 | 34.0 min |
| `K_Bishop_Queen_Rook_K` | 5-man same-side | 4,375 | 90.8 min |

The golden baseline was saved as
`tablebase/validation/transition_golden.json`. A profiled rerun of
`K_Knight_Queen_Rook_K` checked the same 1,500 states against both the table and
golden baseline, with zero mismatches.

The profile's top internal-time functions were `is_square_attacked`
(GameLogic.py), `_build_transition_worker_5`, `get_valid_moves`,
`make_move_track`, `unmake_move`, and `is_in_check`. This supports targeting
the object-based move-generation/rules path for any future compiled transition
experiment. The profiler materially reduced the observed rate to 5,974
states/s, so its timing/projection is diagnostic only and is not used as a
performance baseline.

\*These are rough projections from random sampled states and ideal division by
the configured worker count. They are not full-table timings and varied
substantially between runs.

## Context and limitations

- The chunk and worker tests sampled Stage 2 transition generation and parent
  accumulation; they did not generate a complete 5-man table.
- The transition benchmark checks sampled Bellman consistency against table
  values, not complete table correctness or a full generation run.
- CPU model, thermal/power state, background load, and disk conditions were not
  recorded. The measured throughput varied substantially between rounds.
- The user-reported 4-man `K_Rook_Rook_K` run took 33.4 s for Stage 2, 5.6 s
  for Stage 2b, and 42.6 s total. This confirms Stage 2 was the largest stage
  in that run, but it is not a direct measurement of 5-man performance.
- No measured change demonstrated a stable end-to-end 5-man generation
  improvement.

## Follow-up guidance

If performance work resumes, rerun controlled samples on an idle machine and
record CPU model, worker count, power/thermal state, and per-round times. Only
adopt a change after it shows a repeatable gain across multiple longer runs and
preserves identical Stage 2 outputs.
