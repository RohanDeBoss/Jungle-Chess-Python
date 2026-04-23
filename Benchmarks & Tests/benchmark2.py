"""Benchmark harness for Jungle Chess search performance.

This benchmark is designed to answer:
1. How fast is the engine on a representative suite?
2. How does it behave across opening, middlegame, and endgame positions?
3. What happened at each iterative-deepening layer?

Defaults favor fair engine-to-engine comparison:
- opening book disabled
- tablebases disabled
- fresh bot/search state per position
- curated multi-phase suite
- JSON + TSV output for later comparison
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

os.chdir(PARENT_DIR)

from AI import ChessBot, board_hash  # noqa: E402
from GameLogic import (  # noqa: E402
    Bishop,
    Board,
    King,
    Knight,
    Pawn,
    Queen,
    Rook,
    get_all_legal_moves,
    is_in_check,
)


LOG_FILE = os.path.join(SCRIPT_DIR, "benchmarks.tsv")
JSON_DIR = os.path.join(SCRIPT_DIR, "benchmark_reports")
BENCHMARK_SCRIPT_VERSION = "2.0"

PIECE_FROM_FEN = {
    "p": Pawn,
    "n": Knight,
    "b": Bishop,
    "r": Rook,
    "q": Queen,
    "k": King,
}

STAGE_ORDER = ("opening", "middlegame", "endgame")


@dataclass(frozen=True)
class BenchmarkPosition:
    key: str
    stage: str
    name: str
    fen: str
    note: str = ""


class DummyQueue:
    def put(self, msg):
        return None

    def empty(self):
        return True

    def get_nowait(self):
        return None


class DummyEvent:
    def is_set(self):
        return False


SUITE = [
    BenchmarkPosition(
        "OP01",
        "opening",
        "Start Position",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w",
        "Baseline full-material position.",
    ),
    BenchmarkPosition(
        "OP02",
        "opening",
        "Open Development",
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w",
        "Light development with open central tension.",
    ),
    BenchmarkPosition(
        "OP03",
        "opening",
        "Bishop Pressure",
        "rnbqk2r/pppp1ppp/5n2/4p3/1bB1P3/5N2/PPPP1PPP/RNBQK2R w",
        "Fast piece development and immediate tactical contact.",
    ),
    BenchmarkPosition(
        "OP04",
        "opening",
        "Asymmetric Setup",
        "r1bqkbnr/ppp2ppp/2np4/4p3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R b",
        "Uneven development with both kings still central.",
    ),
    BenchmarkPosition(
        "MG01",
        "middlegame",
        "Queenside Tension",
        "1rbqk2r/pp1p1ppp/4p3/8/1P1P4/4B3/4PPPP/R3KBNR w",
        "Imbalanced structure with several forcing rook and queen ideas.",
    ),
    BenchmarkPosition(
        "MG02",
        "middlegame",
        "Central Imbalance",
        "r2qk3/pppb4/2npp3/8/6p1/2B1P3/PPPQ3P/R3KB2 b",
        "Compact middlegame with tactical queen and bishop motifs.",
    ),
    BenchmarkPosition(
        "MG03",
        "middlegame",
        "Complex Rook Fight",
        "2b3kr/7p/pq1p4/1r2p1p1/3PP2P/2B5/4BPP1/R3K2R w",
        "Original complex benchmark position with multiple tactical layers.",
    ),
    BenchmarkPosition(
        "MG04",
        "middlegame",
        "Loose Kings",
        "2b3kr/7p/p2p4/3rp2p/8/8/4BPP1/R3K2R w",
        "Reduced material but still rich in tactical king pressure.",
    ),
    BenchmarkPosition(
        "MG05",
        "middlegame",
        "Knight and Pawn Friction",
        "r1bqk2r/ppp3pp/n3pp2/3p4/8/1P2KNP1/P1PP3P/RNBQ4 w",
        "Sharp minor-piece and pawn interaction.",
    ),
    BenchmarkPosition(
        "MG06",
        "middlegame",
        "Open Center Race",
        "r2qk2r/1p2bpp1/2p1bn1p/p7/3P4/2NQ1NP1/PPP2P1P/R1K4R w",
        "Open files and diagonals with both sides ready to break through.",
    ),
    BenchmarkPosition(
        "MG07",
        "middlegame",
        "Rook Lift Defense",
        "3rk2r/b4pp1/5n1p/pp6/8/5NP1/PPP1KP1P/R3R3 b",
        "Defensive coordination and rook activity benchmark.",
    ),
    BenchmarkPosition(
        "MG08",
        "middlegame",
        "Heavy Piece Chaos",
        "3r1n2/2r2k1p/R7/2p1b1p1/1p3p2/2P5/2K1QBP1/1R6 w",
        "High-volatility Jungle Chess tactical position.",
    ),
    BenchmarkPosition(
        "MG09",
        "middlegame",
        "Double Rook Pressure",
        "2r2rk1/pp3ppp/2n1bn2/2qp4/3P4/2NQPN2/PP3PPP/2RR2K1 w",
        "Many candidate moves with a large branching factor.",
    ),
    BenchmarkPosition(
        "MG10",
        "middlegame",
        "Compressed Tactical Net",
        "4r1k1/1p3ppp/p1n1b3/3pP3/3P1P2/2P1B2P/PP4P1/2KR3R b",
        "Reduced board density but still tactically forcing.",
    ),
    BenchmarkPosition(
        "EG01",
        "endgame",
        "Rook and Pawn vs Bishop and Pawn",
        "2b5/8/p2p2k1/2K1p3/8/8/5PPr/R7 w",
        "Original technical endgame benchmark.",
    ),
    BenchmarkPosition(
        "EG02",
        "endgame",
        "Rook vs Pawns",
        "8/8/p7/3ppk2/8/R3K2P/5P2/8 b",
        "Low-material race with practical king activity decisions.",
    ),
    BenchmarkPosition(
        "EG03",
        "endgame",
        "Promotion Race",
        "8/8/8/3p4/8/2k5/8/1K6 w",
        "Tiny-tree race benchmark from the tablebase notes.",
    ),
    BenchmarkPosition(
        "EG04",
        "endgame",
        "Three Queens Micro-Board",
        "7k/8/8/8/Q7/8/4Q3/K2Q4 w",
        "Extreme reduced-material queen mobility stress test.",
    ),
    BenchmarkPosition(
        "EG05",
        "endgame",
        "Rook Beats Bishop?",
        "8/4P3/8/8/8/3k4/4b3/K1R5 w",
        "Reduced-material tactical endgame from local notes.",
    ),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Jungle Chess engine search speed.")
    parser.add_argument("--max-depth", type=int, default=6, help="Iterative deepening target depth.")
    parser.add_argument("--runs", type=int, default=1, help="Full suite passes to execute.")
    parser.add_argument(
        "--warmup-depth",
        type=int,
        default=3,
        help="Optional warmup search depth. Use 0 to disable warmup.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Version tag for reports. Defaults to the version parsed from AI.py.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only benchmark the first N validated positions.",
    )
    parser.add_argument(
        "--stages",
        type=str,
        default="all",
        help="Comma-separated stage filter: opening,middlegame,endgame or all.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Run benchmark without writing JSON/TSV reports.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List validated suite positions and exit.",
    )
    return parser.parse_args()


def load_fen(fen):
    parts = fen.strip().split()
    if len(parts) < 2:
        raise ValueError(f"Invalid FEN-like string: {fen!r}")

    board = Board(setup=False)
    row = 0
    col = 0
    for char in parts[0]:
        if char == "/":
            row += 1
            col = 0
            continue
        if char.isdigit():
            col += int(char)
            continue
        piece_cls = PIECE_FROM_FEN.get(char.lower())
        if piece_cls is None:
            raise ValueError(f"Unknown FEN piece {char!r} in {fen!r}")
        color = "white" if char.isupper() else "black"
        board.add_piece(piece_cls(color), row, col)
        col += 1

    turn = "white" if parts[1].lower() == "w" else "black"
    return board, turn


def detect_engine_version():
    ai_path = os.path.join(PARENT_DIR, "AI.py")
    try:
        with open(ai_path, "r", encoding="utf-8") as handle:
            first_line = handle.readline().strip()
    except OSError:
        return "unknown"

    match = re.search(r"\((v[^)]+)\)", first_line)
    if match:
        return match.group(1)
    return "unknown"


def counts_by_color(board):
    result = {}
    for color in ("white", "black"):
        counts = board.piece_counts[color]
        result[color] = {
            "king": int(counts[King]),
            "queen": int(counts[Queen]),
            "rook": int(counts[Rook]),
            "bishop": int(counts[Bishop]),
            "knight": int(counts[Knight]),
            "pawn": int(counts[Pawn]),
        }
    return result


def total_piece_count(board):
    return len(board.white_pieces) + len(board.black_pieces)


def total_non_king_pieces(board):
    return total_piece_count(board) - int(board.white_king_pos is not None) - int(board.black_king_pos is not None)


def validate_position(position):
    board, turn = load_fen(position.fen)
    if board.white_king_pos is None or board.black_king_pos is None:
        raise ValueError("Both kings must be present.")

    legal_moves = get_all_legal_moves(board, turn)
    if not legal_moves:
        raise ValueError(f"No legal moves for side to move ({turn}).")

    return {
        "root_hash": int(board_hash(board, turn)),
        "side_to_move": turn,
        "legal_moves": len(legal_moves),
        "in_check": bool(is_in_check(board, turn)),
        "total_pieces": total_piece_count(board),
        "non_king_pieces": total_non_king_pieces(board),
        "piece_counts": counts_by_color(board),
    }


def summarize(values):
    values = list(values)
    if not values:
        return {
            "count": 0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "stdev": 0.0,
        }

    stdev = statistics.pstdev(values) if len(values) > 1 else 0.0
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "stdev": stdev,
    }


def fmt_nodes(value):
    return f"{int(round(value)):,}"


def fmt_secs(value):
    return f"{value:.3f}s"


def fmt_knps(nodes, seconds):
    if seconds <= 0:
        return 0.0
    return nodes / seconds / 1000.0


def fmt_cp(score_cp):
    return f"{score_cp / 100:+.2f}"


def select_stages(raw_value):
    raw_value = raw_value.strip().lower()
    if raw_value == "all":
        return set(STAGE_ORDER)
    stages = {item.strip() for item in raw_value.split(",") if item.strip()}
    unknown = stages.difference(STAGE_ORDER)
    if unknown:
        raise ValueError(f"Unknown stages: {', '.join(sorted(unknown))}")
    return stages


def build_bot(board, turn, root_hash):
    return ChessBot(
        board=board,
        color=turn,
        position_counts={root_hash: 1},
        comm_queue=DummyQueue(),
        cancellation_event=DummyEvent(),
        bot_name="Benchmark",
        ply_count=0,
        game_mode="benchmark",
        use_opening_book=False,
        use_tablebase=False,
    )


def run_single_search(position, max_depth):
    board, turn = load_fen(position.fen)
    root_hash = int(board_hash(board, turn))
    root_moves = get_all_legal_moves(board, turn)
    if not root_moves:
        raise ValueError(f"Position {position.key} has no legal moves.")

    bot = build_bot(board, turn, root_hash)
    best_move = root_moves[0]
    prev_score = None
    total_nodes = 0
    depth_results = []

    suite_start = time.perf_counter()
    for depth in range(1, max_depth + 1):
        depth_start = time.perf_counter()
        score_cp, best_move = bot._run_depth_iteration(
            depth,
            root_moves,
            root_hash,
            best_move,
            prev_iter_score=prev_score,
        )
        elapsed = time.perf_counter() - depth_start
        prev_score = score_cp
        depth_nodes = int(bot.nodes_searched)
        total_nodes += depth_nodes

        depth_results.append(
            {
                "depth": depth,
                "elapsed_sec": elapsed,
                "nodes": depth_nodes,
                "knps": fmt_knps(depth_nodes, elapsed),
                "score_cp": int(score_cp),
                "best_move": bot._format_move(board, best_move) if best_move else None,
                "tb_hits": int(bot.tb_hits),
                "used_heuristic_eval": bool(bot.used_heuristic_eval),
            }
        )

    total_elapsed = time.perf_counter() - suite_start
    final_depth = depth_results[-1]
    return {
        "iterative_nodes": total_nodes,
        "iterative_time_sec": total_elapsed,
        "iterative_knps": fmt_knps(total_nodes, total_elapsed),
        "final_depth_nodes": final_depth["nodes"],
        "final_depth_time_sec": final_depth["elapsed_sec"],
        "final_depth_knps": final_depth["knps"],
        "final_score_cp": final_depth["score_cp"],
        "best_move": final_depth["best_move"],
        "depths": depth_results,
    }


def aggregate_position_runs(position_record):
    runs = position_record["runs"]
    best_moves = Counter(run["best_move"] for run in runs if run["best_move"])

    aggregate = {
        "iterative_nodes": summarize(run["iterative_nodes"] for run in runs),
        "iterative_time_sec": summarize(run["iterative_time_sec"] for run in runs),
        "iterative_knps": summarize(run["iterative_knps"] for run in runs),
        "final_depth_nodes": summarize(run["final_depth_nodes"] for run in runs),
        "final_depth_time_sec": summarize(run["final_depth_time_sec"] for run in runs),
        "final_depth_knps": summarize(run["final_depth_knps"] for run in runs),
        "final_score_cp": summarize(run["final_score_cp"] for run in runs),
        "best_moves": [{"move": move, "count": count} for move, count in best_moves.most_common()],
    }

    per_depth = defaultdict(lambda: {"nodes": [], "time": [], "knps": [], "score_cp": []})
    for run in runs:
        for depth_row in run["depths"]:
            slot = per_depth[depth_row["depth"]]
            slot["nodes"].append(depth_row["nodes"])
            slot["time"].append(depth_row["elapsed_sec"])
            slot["knps"].append(depth_row["knps"])
            slot["score_cp"].append(depth_row["score_cp"])

    aggregate["per_depth"] = [
        {
            "depth": depth,
            "nodes": summarize(values["nodes"]),
            "time_sec": summarize(values["time"]),
            "knps": summarize(values["knps"]),
            "score_cp": summarize(values["score_cp"]),
        }
        for depth, values in sorted(per_depth.items())
    ]
    return aggregate


def aggregate_stage(position_records, stage_name):
    stage_positions = [row for row in position_records if row["stage"] == stage_name]
    run_rows = [run for row in stage_positions for run in row["runs"]]

    return {
        "stage": stage_name,
        "positions": len(stage_positions),
        "iterative_nodes": summarize(run["iterative_nodes"] for run in run_rows),
        "iterative_time_sec": summarize(run["iterative_time_sec"] for run in run_rows),
        "iterative_knps": summarize(run["iterative_knps"] for run in run_rows),
        "final_depth_nodes": summarize(run["final_depth_nodes"] for run in run_rows),
        "final_depth_time_sec": summarize(run["final_depth_time_sec"] for run in run_rows),
        "final_depth_knps": summarize(run["final_depth_knps"] for run in run_rows),
    }


def write_tsv_row(log_row):
    header = (
        "Timestamp\tTag\tEngineVersion\tBenchmarkVersion\tDepth\tRuns\tPositions\t"
        "IterNodesMean\tIterTimeMean\tIterkNPSMean\tFinalNodesMean\tFinalTimeMean\tFinalkNPSMean\n"
    )
    if not os.path.isfile(LOG_FILE):
        with open(LOG_FILE, "w", encoding="utf-8") as handle:
            handle.write(header)

    with open(LOG_FILE, "a", encoding="utf-8") as handle:
        handle.write(
            "{timestamp}\t{tag}\t{engine_version}\t{benchmark_version}\t{depth}\t{runs}\t{positions}\t"
            "{iter_nodes:.0f}\t{iter_time:.6f}\t{iter_knps:.3f}\t{final_nodes:.0f}\t"
            "{final_time:.6f}\t{final_knps:.3f}\n".format(**log_row)
        )


def warmup(depth):
    if depth <= 0:
        return None

    warmup_position = SUITE[0]
    run_single_search(warmup_position, depth)
    return warmup_position.key


def print_suite_overview(validated_positions, skipped_positions):
    print("=" * 88)
    print(" JUNGLE CHESS SEARCH BENCHMARK")
    print("=" * 88)
    stage_counts = Counter(row["stage"] for row in validated_positions)
    stage_summary = ", ".join(f"{stage}:{stage_counts[stage]}" for stage in STAGE_ORDER if stage_counts[stage])
    print(f"Validated positions: {len(validated_positions)} ({stage_summary})")
    if skipped_positions:
        print(f"Skipped invalid positions: {len(skipped_positions)}")
        for item in skipped_positions:
            print(f"  - {item['key']} {item['name']}: {item['reason']}")
    print("-" * 88)


def print_position_result(stage, key, name, run_index, run_data):
    print(
        f"[{stage:<10}] {key} {name:<24} run={run_index:<2d} "
        f"iter={fmt_nodes(run_data['iterative_nodes']):>12} nodes "
        f"{fmt_secs(run_data['iterative_time_sec']):>8} "
        f"{run_data['iterative_knps']:>8.1f} kN/s | "
        f"final={fmt_nodes(run_data['final_depth_nodes']):>10} "
        f"{fmt_secs(run_data['final_depth_time_sec']):>8} "
        f"{run_data['final_depth_knps']:>8.1f} kN/s | "
        f"best={run_data['best_move'] or 'None'} eval={fmt_cp(run_data['final_score_cp'])}"
    )


def main():
    args = parse_args()
    stages = select_stages(args.stages)
    engine_version = detect_engine_version()
    tag = args.tag or engine_version
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    validated_positions = []
    skipped_positions = []
    for position in SUITE:
        if position.stage not in stages:
            continue
        try:
            metadata = validate_position(position)
        except Exception as exc:
            skipped_positions.append(
                {"key": position.key, "name": position.name, "stage": position.stage, "reason": str(exc)}
            )
            continue

        validated_positions.append(
            {
                "key": position.key,
                "stage": position.stage,
                "name": position.name,
                "fen": position.fen,
                "note": position.note,
                "metadata": metadata,
                "runs": [],
            }
        )

    validated_positions.sort(key=lambda row: (STAGE_ORDER.index(row["stage"]), row["key"]))
    if args.limit is not None:
        validated_positions = validated_positions[: args.limit]

    print_suite_overview(validated_positions, skipped_positions)

    if args.list:
        for row in validated_positions:
            meta = row["metadata"]
            print(
                f"{row['key']} | {row['stage']:<10} | moves={meta['legal_moves']:<3d} "
                f"pieces={meta['total_pieces']:<2d} | {row['name']}"
            )
        return

    if not validated_positions:
        raise SystemExit("No valid positions remain after filtering.")

    warmed = warmup(args.warmup_depth)
    if warmed:
        print(f"Warmup completed on {warmed} at depth {args.warmup_depth}.")
        print("-" * 88)

    pass_summaries = []
    for run_index in range(1, args.runs + 1):
        print(f"Pass {run_index}/{args.runs}")
        pass_iter_nodes = 0
        pass_iter_time = 0.0
        pass_final_nodes = 0
        pass_final_time = 0.0

        for row in validated_positions:
            run_data = run_single_search(
                BenchmarkPosition(row["key"], row["stage"], row["name"], row["fen"], row["note"]),
                args.max_depth,
            )
            run_data["run_index"] = run_index
            row["runs"].append(run_data)

            pass_iter_nodes += run_data["iterative_nodes"]
            pass_iter_time += run_data["iterative_time_sec"]
            pass_final_nodes += run_data["final_depth_nodes"]
            pass_final_time += run_data["final_depth_time_sec"]

            print_position_result(row["stage"], row["key"], row["name"], run_index, run_data)

        pass_summary = {
            "run_index": run_index,
            "iterative_nodes": pass_iter_nodes,
            "iterative_time_sec": pass_iter_time,
            "iterative_knps": fmt_knps(pass_iter_nodes, pass_iter_time),
            "final_depth_nodes": pass_final_nodes,
            "final_depth_time_sec": pass_final_time,
            "final_depth_knps": fmt_knps(pass_final_nodes, pass_final_time),
        }
        pass_summaries.append(pass_summary)

        print("-" * 88)
        print(
            f"Pass {run_index} summary: "
            f"iter {fmt_nodes(pass_iter_nodes)} nodes in {fmt_secs(pass_iter_time)} "
            f"({pass_summary['iterative_knps']:.1f} kN/s) | "
            f"final {fmt_nodes(pass_final_nodes)} nodes in {fmt_secs(pass_final_time)} "
            f"({pass_summary['final_depth_knps']:.1f} kN/s)"
        )
        print("-" * 88)

    for row in validated_positions:
        row["aggregate"] = aggregate_position_runs(row)

    stage_summary = [aggregate_stage(validated_positions, stage) for stage in STAGE_ORDER if any(row["stage"] == stage for row in validated_positions)]
    overall = {
        "passes": len(pass_summaries),
        "positions": len(validated_positions),
        "iterative_nodes": summarize(row["iterative_nodes"] for row in pass_summaries),
        "iterative_time_sec": summarize(row["iterative_time_sec"] for row in pass_summaries),
        "iterative_knps": summarize(row["iterative_knps"] for row in pass_summaries),
        "final_depth_nodes": summarize(row["final_depth_nodes"] for row in pass_summaries),
        "final_depth_time_sec": summarize(row["final_depth_time_sec"] for row in pass_summaries),
        "final_depth_knps": summarize(row["final_depth_knps"] for row in pass_summaries),
    }

    print("Stage summary")
    for stage_row in stage_summary:
        print(
            f"  {stage_row['stage']:<10} | "
            f"iter {stage_row['iterative_knps']['mean']:.1f} kN/s over {stage_row['positions']} positions | "
            f"final {stage_row['final_depth_knps']['mean']:.1f} kN/s"
        )

    print("=" * 88)
    print("OVERALL")
    print("=" * 88)
    print(
        f"Iterative suite mean: {fmt_nodes(overall['iterative_nodes']['mean'])} nodes | "
        f"{overall['iterative_time_sec']['mean']:.3f}s | "
        f"{overall['iterative_knps']['mean']:.1f} kN/s"
    )
    print(
        f"Final-depth mean:     {fmt_nodes(overall['final_depth_nodes']['mean'])} nodes | "
        f"{overall['final_depth_time_sec']['mean']:.3f}s | "
        f"{overall['final_depth_knps']['mean']:.1f} kN/s"
    )
    print("=" * 88)

    if args.no_save:
        return

    os.makedirs(JSON_DIR, exist_ok=True)
    safe_tag = re.sub(r"[^A-Za-z0-9._-]+", "_", tag).strip("_") or "untagged"
    filename_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(JSON_DIR, f"bench_{safe_tag}_{filename_stamp}.json")

    report = {
        "timestamp": timestamp,
        "tag": tag,
        "engine_version": engine_version,
        "benchmark_script_version": BENCHMARK_SCRIPT_VERSION,
        "config": {
            "max_depth": args.max_depth,
            "runs": args.runs,
            "warmup_depth": args.warmup_depth,
            "stages": sorted(stages),
            "positions_requested": len(SUITE),
            "positions_validated": len(validated_positions),
            "opening_book": False,
            "tablebase": False,
            "fresh_bot_per_position": True,
        },
        "skipped_positions": skipped_positions,
        "pass_summaries": pass_summaries,
        "stage_summary": stage_summary,
        "overall": overall,
        "positions": validated_positions,
    }

    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    write_tsv_row(
        {
            "timestamp": timestamp,
            "tag": tag,
            "engine_version": engine_version,
            "benchmark_version": BENCHMARK_SCRIPT_VERSION,
            "depth": args.max_depth,
            "runs": args.runs,
            "positions": len(validated_positions),
            "iter_nodes": overall["iterative_nodes"]["mean"],
            "iter_time": overall["iterative_time_sec"]["mean"],
            "iter_knps": overall["iterative_knps"]["mean"],
            "final_nodes": overall["final_depth_nodes"]["mean"],
            "final_time": overall["final_depth_time_sec"]["mean"],
            "final_knps": overall["final_depth_knps"]["mean"],
        }
    )

    print(f"Saved JSON report to {report_path}")
    print(f"Updated TSV log at {LOG_FILE}")


if __name__ == "__main__":
    main()
