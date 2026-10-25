#!/usr/bin/env python
"""Benchmark and correctness-check Stage 2 tablebase transitions."""

import argparse
import cProfile
import importlib
import io
import json
import os
import pstats
import signal
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import TablebaseGenerator as TG

DEFAULT_TABLES = [
    "K_Rook_Rook_K",
    "K_Pawn_Rook_K",
    "K_Rook_vs_Rook_K",
    "K_Knight_Queen_Rook_K",
    "K_Bishop_Queen_Rook_K",
]


def fmt_t(seconds):
    return f"{seconds / 60:.1f}m" if seconds >= 60 else f"{seconds:.1f}s"


def make_generator(name):
    classes = TG.PIECE_CLASS_BY_NAME
    if not name.startswith("K_") or not name.endswith("_K"):
        raise ValueError(f"unsupported table name: {name}")
    middle = name[2:-2]

    if "_vs_" in middle:
        white, black = middle.split("_vs_")
        white_pieces = white.split("_")
        if len(white_pieces) == 1:
            kind, generator = "4vs", TG.Generator4Vs(classes[white_pieces[0]], classes[black])
        elif len(white_pieces) == 2:
            kind, generator = "5vs", TG.Generator5Vs(
                classes[white_pieces[0]], classes[white_pieces[1]], classes[black]
            )
        else:
            raise ValueError(f"unsupported table name: {name}")
    else:
        pieces = middle.split("_")
        if len(pieces) == 2:
            kind, generator = "4", TG.Generator4(classes[pieces[0]], classes[pieces[1]])
        elif len(pieces) == 3:
            kind, generator = "5", TG.Generator5(
                classes[pieces[0]], classes[pieces[1]], classes[pieces[2]]
            )
        else:
            raise ValueError(f"unsupported table name: {name}")

    if os.path.basename(generator.filename) != name + TG.TB_SUFFIX:
        raise ValueError(f"{name}: table name is not in canonical piece order")
    return kind, generator


def layout(kind, generator):
    if kind == "4":
        return [generator.p1_name, generator.p2_name, "K"], [(0, 1)]
    if kind == "4vs":
        return [generator.w_name, "K", generator.b_name], []
    if kind == "5":
        return [
            generator.p1_name, generator.p2_name, generator.p3_name, "K"
        ], [(0, 1), (1, 2)]
    return [
        generator.w1_name, generator.w2_name, "K", generator.b_name
    ], [(0, 1)]


def keep_signals(function, *args):
    saved = {sig: signal.getsignal(sig) for sig in TG._CONSOLE_INTERRUPT_SIGNALS}
    try:
        function(*args)
    finally:
        for sig, handler in saved.items():
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass


def default_fn(kind, generator):
    if kind == "4":
        keep_signals(
            TG._init_transition_worker_4,
            generator.p1_name,
            generator.p2_name,
            generator.promo_tb_file,
        )
        return TG._build_transition_worker_4
    if kind == "4vs":
        keep_signals(TG._init_transition_worker_4vs, generator.w_name, generator.b_name)
        return TG._build_transition_worker_4vs
    if kind == "5":
        keep_signals(
            TG._init_transition_worker_5,
            generator.p1_name,
            generator.p2_name,
            generator.p3_name,
        )
        return TG._build_transition_worker_5
    keep_signals(
        TG._init_transition_worker_5vs,
        generator.w1_name,
        generator.w2_name,
        generator.b_name,
    )
    return TG._build_transition_worker_5vs


def valid_mask(flats, names, pairs, has_pawn):
    flats = np.asarray(flats, dtype=np.int64)
    rest = flats // 2
    squares = []
    for _ in names:
        squares.append(rest % 64)
        rest //= 64
    squares.reverse()
    wk_indices = rest
    wk_list = np.asarray(
        TG.PAWN_WK_SQUARES if has_pawn else TG.NON_PAWN_WK_SQUARES
    )
    valid = wk_indices < len(wk_list)
    all_squares = [wk_list[np.minimum(wk_indices, len(wk_list) - 1)]] + squares

    for i in range(len(all_squares)):
        for j in range(i + 1, len(all_squares)):
            valid &= all_squares[i] != all_squares[j]
    for square, piece_name in zip(squares, names):
        if piece_name == "Pawn":
            valid &= (square >= 8) & (square < 56)
    for i, j in pairs:
        if names[i] == names[j]:
            valid &= squares[i] <= squares[j]
        if names[i] == names[j] == "Bishop":
            colour_i = (squares[i] // 8 + squares[i] % 8) % 2
            colour_j = (squares[j] // 8 + squares[j] % 8) % 2
            valid &= colour_i != colour_j
    return valid


def ref_valid(kind, generator, window, chunk_size=1_000_000):
    if kind == "4":
        iterator = TG._gen_valid_4man_indices_numpy(
            window,
            generator.p1_name,
            generator.p2_name,
            generator.same_piece,
            generator.has_pawn,
            chunk_size,
        )
    elif kind == "4vs":
        iterator = TG._gen_valid_4man_vs_indices_numpy(
            window, generator.w_name, generator.b_name, generator.has_pawn, chunk_size
        )
    elif kind == "5":
        iterator = TG._gen_valid_5same_indices_numpy(
            window,
            generator.p1_name,
            generator.p2_name,
            generator.p3_name,
            generator.has_pawn,
            chunk_size,
        )
    else:
        iterator = TG._gen_valid_5vs_indices_numpy(
            window,
            generator.w1_name,
            generator.w2_name,
            generator.b_name,
            generator.has_pawn,
            generator.same_wp,
            chunk_size,
        )
    return iterator


def sample_states(total, names, pairs, has_pawn, count, seed):
    rng = np.random.default_rng(seed)
    samples = []
    drawn = accepted = 0
    while len(samples) < count and drawn < count * 400:
        flats = rng.integers(
            0, total, size=max(4 * count, 20_000), dtype=np.int64
        )
        mask = valid_mask(flats, names, pairs, has_pawn)
        drawn += len(flats)
        accepted += int(mask.sum())
        samples.extend(flats[mask].tolist())

    unique = list(dict.fromkeys(samples))
    return unique[:count], accepted / max(1, drawn)


def normalize(result):
    if result is None:
        return None
    _, transition = result
    tag = transition[0]
    if tag == "w":
        if transition[1]:
            return ["w", 1]
        return [
            "w",
            0,
            sorted(int(value) for value in transition[2]),
            sorted(int(value) for value in transition[3]),
        ]
    if tag == "b":
        if transition[2] or int(transition[1]) == 0:
            return ["b", int(transition[1]), bool(transition[2])]
        return [
            "b",
            int(transition[1]),
            False,
            sorted(int(value) for value in transition[3]),
            sorted(int(value) for value in transition[4]),
        ]
    return [
        "g",
        int(transition[1]),
        sorted(int(value) for value in transition[2]),
        sorted(int(value) for value in transition[3]),
    ]


def expected_value(normalized, table):
    if normalized is None:
        return 0
    tag = normalized[0]
    if tag == "w":
        if normalized[1]:
            return 1
        candidates = list(normalized[2])
        for child in normalized[3]:
            value = int(table[child])
            if value < 0:
                candidates.append(1 - value)
        return min(candidates) if candidates else 0
    if tag == "b":
        if normalized[1] == 0:
            return -1
        if normalized[2]:
            return 0
        candidates = list(normalized[3])
        for child in normalized[4]:
            value = int(table[child])
            if value <= 0:
                return 0
            candidates.append(value)
        return -(max(candidates) + 1) if candidates else None

    terminal = normalized[1]
    if terminal:
        return terminal
    winning_dtm = None
    longest_loss = 0
    has_draw = False
    for value in list(normalized[2]) + [int(table[child]) for child in normalized[3]]:
        if value < 0:
            dtm = 1 - value
            winning_dtm = dtm if winning_dtm is None else min(winning_dtm, dtm)
        elif value > 0:
            longest_loss = max(longest_loss, value)
        else:
            has_draw = True
    if winning_dtm is not None:
        return winning_dtm
    if has_draw:
        return 0
    return -(longest_loss + 1)


def run_case(name, args, budget_seconds, golden_reference, implementation, golden_output):
    kind, generator = make_generator(name)
    if not os.path.exists(generator.filename):
        print(f"[SKIP] {name}: missing {os.path.basename(generator.filename)}")
        return None
    if kind in ("4vs", "5vs") and not os.path.exists(
        generator.filename + TG.TB_WDL_MARKER_SUFFIX
    ):
        print(f"[SKIP] {name}: missing .wdl marker")
        return None

    names, pairs = layout(kind, generator)
    window = min(generator.total_positions, 3_000_000)
    generator_valid = np.zeros(window, dtype=bool)
    for chunk in ref_valid(kind, generator, window):
        generator_valid[np.asarray(chunk, dtype=np.int64)] = True
    benchmark_valid = valid_mask(
        np.arange(window, dtype=np.int64), names, pairs, generator.has_pawn
    )
    if not np.array_equal(generator_valid, benchmark_valid):
        print(f"[ERROR] {name}: benchmark validity mask disagrees with generator")
        return False

    table = np.memmap(generator.filename, dtype=TG.TB_DISK_DTYPE, mode="r")
    if table.size != generator.total_positions:
        print(
            f"[ERROR] {name}: table size {table.size} != "
            f"expected {generator.total_positions}"
        )
        return False

    samples, sample_rate = sample_states(
        generator.total_positions, names, pairs, generator.has_pawn, args.n, args.seed
    )
    if not samples:
        print(f"[ERROR] {name}: no valid sample states found")
        return False
    transition_fn = (
        implementation.get_transition_fn(kind, generator)
        if implementation
        else default_fn(kind, generator)
    )

    profiler = cProfile.Profile() if args.profile else None
    results = []
    started = time.perf_counter()
    deadline = started + budget_seconds
    if profiler:
        profiler.enable()
    for flat in samples:
        if time.perf_counter() >= deadline:
            break
        results.append((flat, transition_fn(flat)))
    if profiler:
        profiler.disable()
    elapsed = time.perf_counter() - started

    mismatches = []
    golden_mismatches = 0
    golden_missing = 0
    rejected = 0
    normalized_results = {}
    for flat, transition in results:
        normalized = normalize(transition)
        rejected += normalized is None
        normalized_results[str(flat)] = normalized
        recomputed = expected_value(normalized, table)
        stored = int(table[flat])
        if recomputed != stored and len(mismatches) < 5:
            mismatches.append((flat, stored, recomputed, normalized))
        elif recomputed != stored:
            mismatches.append(None)
        if golden_reference is not None:
            expected_golden = golden_reference.get(name, {}).get(str(flat))
            if expected_golden is None and str(flat) not in golden_reference.get(name, {}):
                golden_missing += 1
            elif expected_golden != normalized:
                golden_mismatches += 1

    candidate_count = sample_rate * generator.total_positions
    seconds_per_state = elapsed / max(1, len(results))
    projected_single_worker = candidate_count * seconds_per_state
    projected_parallel = projected_single_worker / generator.transition_workers
    mismatch_count = sum(item is not None for item in mismatches)
    passed = (
        mismatch_count == 0
        and golden_mismatches == 0
        and golden_missing == 0
        and bool(results)
    )
    print(
        f"[{'PASS' if passed else 'FAIL'}] {name} ({kind}) "
        f"states {len(results)}/{len(samples)} "
        f"{len(results) / max(elapsed, 1e-9):,.0f} st/s "
        f"rejected-as-in-check {100 * rejected / max(1, len(results)):.0f}% "
        f"bellman-mismatch {mismatch_count} golden-mismatch {golden_mismatches} "
        f"golden-missing {golden_missing}"
    )
    print(
        f"        est candidates {candidate_count / 1e6:,.1f}M -> "
        f"1 core {fmt_t(projected_single_worker)}, "
        f"{generator.transition_workers} workers ~{fmt_t(projected_parallel)} "
        "(Stage 2, ideal scaling)"
    )
    for mismatch in [item for item in mismatches if item][:5]:
        print(
            f"        MISMATCH flat={mismatch[0]} table={mismatch[1]} "
            f"recomputed={mismatch[2]} trans={mismatch[3]}"
        )
    if profiler:
        output = io.StringIO()
        pstats.Stats(profiler, stream=output).sort_stats("tottime").print_stats(18)
        print(output.getvalue())

    golden_output[name] = normalized_results
    del table
    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", action="append", help="table name; may be repeated")
    parser.add_argument("--budget", type=float, default=30.0, help="total timed seconds")
    parser.add_argument("-n", type=int, default=1500, help="maximum samples per table")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--impl", help="module implementing get_transition_fn(kind, generator)")
    parser.add_argument("--golden-save")
    parser.add_argument("--golden-check")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if args.budget <= 0 or args.n <= 0:
        parser.error("--budget and -n must be positive")

    tables = args.table or DEFAULT_TABLES
    implementation = importlib.import_module(args.impl) if args.impl else None
    if args.golden_check:
        with open(args.golden_check, encoding="utf-8") as stream:
            golden_reference = json.load(stream)
    else:
        golden_reference = None

    present = [
        table for table in tables
        if os.path.exists(os.path.join(TG.TB_DIR, table + TG.TB_SUFFIX))
    ]
    if not present:
        parser.error("none of the requested tablebase files are available")

    seconds_per_case = args.budget / len(present)
    print(
        f"impl={args.impl or 'TablebaseGenerator (reference Python)'} | "
        f"{len(present)} available tables | {seconds_per_case:.1f}s timed per table | "
        f"n<={args.n} seed={args.seed}"
    )
    golden_output = {}
    failures = []
    skipped = []
    for name in tables:
        result = run_case(
            name, args, seconds_per_case, golden_reference, implementation, golden_output
        )
        if result is False:
            failures.append(name)
        elif result is None:
            skipped.append(name)

    if args.golden_save:
        with open(args.golden_save, "w", encoding="utf-8") as stream:
            json.dump(golden_output, stream)
        print(f"golden saved -> {args.golden_save}")

    if failures:
        print(f"FAILURES PRESENT: {', '.join(failures)}")
    else:
        suffix = f"; skipped unavailable: {', '.join(skipped)}" if skipped else ""
        print(f"ALL AVAILABLE CASES PASS{suffix}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
