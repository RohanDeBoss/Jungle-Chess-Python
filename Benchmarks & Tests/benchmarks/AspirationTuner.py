# AspirationTuner.py (v1.3 - AI-series suite support)
"""GUI benchmark for tuning ChessBot.ASP_WINDOW_INIT.

Run this file directly. It compares candidate aspiration windows over a fixed
position suite, five trials by default, and records depths 4, 5, and 6 from
each iterative search. Per candidate/trial/depth, the fastest and slowest
position timings are excluded from timing and trimmed diagnostic summaries.

The tuner changes the aspiration value only in memory and restores it when the
run finishes. Results are checkpointed to a timestamped JSON file beside this
script; AI.py is never edited.
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import queue
import random
import statistics
import sys
import time
import traceback
import tkinter as tk
from datetime import datetime
from tkinter import messagebox, ttk


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
os.chdir(PROJECT_DIR)

import AI  # noqa: E402
import benchmark2  # noqa: E402
from EngineRuntime import SearchCancelledException  # noqa: E402
from GameLogic import get_all_legal_moves, is_in_check  # noqa: E402


DEPTHS_REPORTED = (4, 5, 6)
MAX_DEPTH = max(DEPTHS_REPORTED)
DEFAULT_CANDIDATES = "200, 300, 400, 450, 500, 700"
DEFAULT_BASELINE_WINDOW = 450
DEFAULT_TRIALS = 5
CHECKPOINT_EVERY_POSITIONS = 5
_LAST_RECOVERY_CHECKPOINT: str | None = None


# Built-in fallback positions are reachable examples. A 50-position suite
# harvested from AI-vs-OP series games takes precedence when available.
EXTRA_POSITIONS = [
    benchmark2.BenchmarkPosition(
        "OP05", "opening", "Early Asymmetric Development",
        "rnbqk1nr/pp1pppbp/8/2P5/8/4PNP1/PP1P1P1P/RNBQKB1R w",
        "Reachable 10-ply opening position.",
    ),
    benchmark2.BenchmarkPosition(
        "MG11", "middlegame", "Developed Tactical Contact",
        "r2qk1nr/pp2pp1p/6pb/n1pp1bB1/6P1/2P2P2/PP2PK1P/RN3BNR w",
        "Reachable 18-ply position.",
    ),
    benchmark2.BenchmarkPosition(
        "MG12", "middlegame", "Three-Move Check Evasion",
        "r5nr/p2p1ppp/4pk2/2n5/5P2/6K1/2P3PP/RN1Q1BNR w",
        "Reachable position with the side to move in check.",
    ),
    benchmark2.BenchmarkPosition(
        "MG13", "middlegame", "Open Files and Minor Pieces",
        "4r1n1/n1pk4/p3pp2/8/P4PbP/R2PP1K1/NPPQ4/2B2B1R w",
        "Reachable 34-ply position.",
    ),
    benchmark2.BenchmarkPosition(
        "MG14", "middlegame", "Rook and Bishop Pressure",
        "6nr/rp1k1p1b/p7/3p4/P3BPPp/1PP4P/3PP3/R2K3R w",
        "Reachable 42-ply position.",
    ),
    benchmark2.BenchmarkPosition(
        "MG15", "middlegame", "Mixed Piece Coordination",
        "r2k1bn1/3np1p1/7r/pp6/2p1Pp2/PP5B/R2P1P2/1N2K3 w",
        "Reachable 50-ply position.",
    ),
    benchmark2.BenchmarkPosition(
        "MG16", "middlegame", "Compressed King Attack",
        "r5nr/pp4p1/5n2/1k2bp1p/2p5/1P6/4P1KP/6NR w",
        "Reachable 58-ply position.",
    ),
    benchmark2.BenchmarkPosition(
        "EG06", "endgame", "Rook and Passed Pawns",
        "8/3kbP2/1rpp4/B3P1r1/3P1K2/P7/8/RN6 b",
        "Reachable 14-piece endgame.",
    ),
    benchmark2.BenchmarkPosition(
        "EG07", "endgame", "Queen and Rook versus Rook",
        "4q3/1b1p2K1/r1k5/1p1N4/p3P3/P7/8/R7 b",
        "Reachable reduced-material position; side to move is checked.",
    ),
    benchmark2.BenchmarkPosition(
        "EG08", "endgame", "Rook against Minor Pieces",
        "R1k4r/6bp/5p2/4p3/8/6pP/3P4/1KB3NR b",
        "Reachable reduced-material position; side to move is checked.",
    ),
    benchmark2.BenchmarkPosition(
        "EG09", "endgame", "Rook and Knight versus Bishop",
        "rn4k1/p7/4pp2/2pp2P1/8/p3K2N/3P4/1b6 b",
        "Reachable 14-piece endgame.",
    ),
]

BASE_SUITE = [
    benchmark2.BenchmarkPosition(
        position.key, position.stage, position.name,
        # The stock MG09 setup checks the side that just moved, so replace it
        # here with a reachable double-rook position of similar complexity.
        "r2q1bnr/1bp1p1p1/p6p/n2p1k1P/1p4P1/2P1P3/P7/BN2K1NR w"
        if position.key == "MG09" else position.fen,
        position.note,
    )
    for position in benchmark2.SUITE
]
FALLBACK_SUITE = BASE_SUITE + EXTRA_POSITIONS
SERIES_SUITE_PATH = os.path.join(SCRIPT_DIR, "aspiration_series_positions.json")


def _load_benchmark_suite():
    if not os.path.isfile(SERIES_SUITE_PATH):
        return FALLBACK_SUITE, "built-in fallback suite", {}
    try:
        with open(SERIES_SUITE_PATH, encoding="utf-8") as handle:
            payload = json.load(handle)
        raw_positions = payload.get("positions", [])
        if len(raw_positions) != 50:
            raise ValueError(f"expected 50 harvested positions, found {len(raw_positions)}")
        positions = [
            benchmark2.BenchmarkPosition(
                str(row["key"]),
                str(row["stage"]),
                str(row["name"]),
                str(row["fen"]),
                str(row.get("note", "Sampled from AI-vs-OP series games.")),
            )
            for row in raw_positions
        ]
        return positions, "AI-vs-OP series harvest", payload.get("metadata", {})
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        return FALLBACK_SUITE, f"built-in fallback suite (harvest file invalid: {exc})", {}


SUITE, SUITE_SOURCE, HARVEST_METADATA = _load_benchmark_suite()
SUITE_SIZE = len(SUITE)


def _parse_candidates(value: str) -> list[int]:
    try:
        candidates = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError("Enter aspiration windows as comma-separated whole numbers.") from exc
    if not candidates:
        raise ValueError("Enter at least one aspiration window.")
    if any(candidate <= 0 or candidate >= AI.ChessBot.MATE_SCORE for candidate in candidates):
        raise ValueError("Aspiration windows must be positive and smaller than MATE_SCORE.")
    if len(set(candidates)) != len(candidates):
        raise ValueError("Each aspiration window must be unique.")
    return candidates


def _validate_suite() -> list[dict]:
    if len(SUITE) != SUITE_SIZE:
        raise ValueError(f"The tuner suite size changed unexpectedly: {len(SUITE)}.")
    keys = [position.key for position in SUITE]
    if len(set(keys)) != len(keys):
        raise ValueError("The tuner suite contains duplicate position keys.")

    validated = []
    for position in SUITE:
        board, turn = benchmark2.load_fen(position.fen)
        if board.white_king_pos is None or board.black_king_pos is None:
            raise ValueError(f"{position.key}: both kings must be present.")
        if not get_all_legal_moves(board, turn):
            raise ValueError(f"{position.key}: side to move has no legal moves.")
        passive = "black" if turn == "white" else "white"
        if is_in_check(board, passive):
            raise ValueError(f"{position.key}: side not to move is already in check.")
        validated.append({
            "key": position.key,
            "stage": position.stage,
            "name": position.name,
            "pieces": len(board.white_pieces) + len(board.black_pieces),
            "side_to_move": turn,
        })
    return validated


def _trimmed_depth_summary(position_rows: list[dict], depth: int) -> dict:
    ordered = sorted(
        position_rows,
        key=lambda row: (row["depths"][str(depth)]["elapsed_sec"], row["key"]),
    )
    retained = ordered[1:-1]
    dropped = [ordered[0]["key"], ordered[-1]["key"]]
    elapsed_values = [row["depths"][str(depth)]["elapsed_sec"] for row in retained]
    return {
        # Arithmetic mean estimates total work over this fixed suite. The
        # geometric mean gives every position equal influence on relative speed.
        "trimmed_mean_sec": statistics.fmean(elapsed_values),
        "trimmed_geomean_sec": math.exp(statistics.fmean(
            math.log(max(value, 1e-6)) for value in elapsed_values
        )),
        "trimmed_mean_nodes": statistics.fmean(
            row["depths"][str(depth)]["nodes"] for row in retained
        ),
        "mean_nodes_all_positions": statistics.fmean(
            row["depths"][str(depth)]["nodes"] for row in position_rows
        ),
        "trimmed_mean_search_calls": statistics.fmean(
            row["depths"][str(depth)]["search_calls"] for row in retained
        ),
        "trimmed_mean_researches": statistics.fmean(
            row["depths"][str(depth)]["search_calls"] - 1 for row in retained
        ),
        "mean_researches_all_positions": statistics.fmean(
            row["depths"][str(depth)]["search_calls"] - 1 for row in position_rows
        ),
        "retained_positions": len(retained),
        "dropped_fastest_and_slowest": dropped,
    }


def _atomic_save(path: str, data: dict) -> str:
    """Replace the checkpoint, tolerating transient OneDrive file locks.

    If the destination remains locked, leave the fully written temporary JSON
    in place and return its path so the search can continue without losing the
    latest checkpoint.
    """
    global _LAST_RECOVERY_CHECKPOINT

    temp_path = f"{path}.{os.getpid()}.{time.time_ns()}.tmp"
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())

    for attempt in range(6):
        try:
            os.replace(temp_path, path)
            if _LAST_RECOVERY_CHECKPOINT:
                try:
                    os.remove(_LAST_RECOVERY_CHECKPOINT)
                except OSError:
                    pass
            _LAST_RECOVERY_CHECKPOINT = None
            return path
        except PermissionError:
            if attempt < 5:
                time.sleep(0.1 * (attempt + 1))

    previous_recovery = _LAST_RECOVERY_CHECKPOINT
    _LAST_RECOVERY_CHECKPOINT = temp_path
    if previous_recovery and previous_recovery != temp_path:
        try:
            os.remove(previous_recovery)
        except OSError:
            pass
    return temp_path


def _compare_to_baseline(candidate_trials: list[dict], baseline_trials: list[dict]) -> dict:
    baseline_by_trial = {trial["trial"]: trial for trial in baseline_trials}
    by_depth = {
        str(depth): {
            "comparisons": 0,
            "score_changed": 0,
            "absolute_score_delta_cp": [],
            "best_move_changed": 0,
        }
        for depth in DEPTHS_REPORTED
    }

    for candidate_trial in candidate_trials:
        baseline_trial = baseline_by_trial.get(candidate_trial["trial"])
        if baseline_trial is None:
            continue
        baseline_positions = {row["key"]: row for row in baseline_trial["positions"]}
        for candidate_position in candidate_trial["positions"]:
            baseline_position = baseline_positions.get(candidate_position["key"])
            if baseline_position is None:
                continue
            for depth in DEPTHS_REPORTED:
                key = str(depth)
                candidate_depth = candidate_position["depths"].get(key)
                baseline_depth = baseline_position["depths"].get(key)
                if candidate_depth is None or baseline_depth is None:
                    continue
                stats = by_depth[key]
                score_delta = abs(candidate_depth["score_cp"] - baseline_depth["score_cp"])
                stats["comparisons"] += 1
                stats["score_changed"] += score_delta != 0
                stats["absolute_score_delta_cp"].append(score_delta)
                stats["best_move_changed"] += (
                    candidate_depth["best_move"] != baseline_depth["best_move"]
                )

    for stats in by_depth.values():
        deltas = stats.pop("absolute_score_delta_cp")
        count = stats["comparisons"]
        stats["mean_absolute_score_delta_cp"] = statistics.fmean(deltas) if deltas else None
        stats["max_absolute_score_delta_cp"] = max(deltas) if deltas else None
        stats["score_change_rate"] = stats["score_changed"] / count if count else None
        stats["best_move_change_rate"] = stats["best_move_changed"] / count if count else None
    return {"by_depth": by_depth}


def _node_count_audit(results: dict) -> dict:
    mismatch_count = 0
    repeated_count = 0
    examples = []
    for candidate, candidate_data in results.items():
        values_by_position_depth = {}
        for trial in candidate_data["trials"]:
            for position in trial["positions"]:
                for depth in DEPTHS_REPORTED:
                    key = (position["key"], str(depth))
                    values_by_position_depth.setdefault(key, []).append(
                        position["depths"][str(depth)]["nodes"]
                    )
        for (position_key, depth), values in values_by_position_depth.items():
            if len(values) > 1:
                repeated_count += 1
            if len(set(values)) > 1:
                mismatch_count += 1
                if len(examples) < 20:
                    examples.append({
                        "window": int(candidate),
                        "position": position_key,
                        "depth": int(depth),
                        "node_counts": sorted(set(values)),
                    })
    return {
        "determinism_verified": repeated_count > 0,
        "deterministic_across_completed_trials": repeated_count > 0 and mismatch_count == 0,
        "repeated_position_depth_groups": repeated_count,
        "position_depth_mismatch_count": mismatch_count,
        "examples_first_20": examples,
    }


def _summarize_results(results: dict, baseline_window: int) -> list[dict]:
    summary = []
    baseline_trials = results.get(str(baseline_window), {}).get("trials", [])
    for candidate, candidate_data in results.items():
        trials = candidate_data["trials"]
        if not trials:
            continue
        row = {
            "window": int(candidate),
            "completed_trials": len(trials),
            "median_combined_trimmed_mean_sec": statistics.median(
                trial["combined_trimmed_mean_sec"] for trial in trials
            ),
            "median_combined_trimmed_geomean_sec": statistics.median(
                trial["combined_trimmed_geomean_sec"] for trial in trials
            ),
            "median_combined_mean_nodes_all_positions": statistics.median(
                trial["combined_mean_nodes_all_positions"] for trial in trials
            ),
            "baseline_comparison": _compare_to_baseline(trials, baseline_trials),
            "depths": {},
        }
        for depth in DEPTHS_REPORTED:
            row["depths"][str(depth)] = {
                "median_trimmed_mean_sec": statistics.median(
                    trial["depth_summaries"][str(depth)]["trimmed_mean_sec"]
                    for trial in trials
                ),
                "median_trimmed_geomean_sec": statistics.median(
                    trial["depth_summaries"][str(depth)]["trimmed_geomean_sec"]
                    for trial in trials
                ),
                "median_trimmed_mean_nodes": statistics.median(
                    trial["depth_summaries"][str(depth)]["trimmed_mean_nodes"]
                    for trial in trials
                ),
                "median_mean_nodes_all_positions": statistics.median(
                    trial["depth_summaries"][str(depth)]["mean_nodes_all_positions"]
                    for trial in trials
                ),
                "median_trimmed_mean_researches": statistics.median(
                    trial["depth_summaries"][str(depth)]["trimmed_mean_researches"]
                    for trial in trials
                ),
                "median_mean_researches_all_positions": statistics.median(
                    trial["depth_summaries"][str(depth)]["mean_researches_all_positions"]
                    for trial in trials
                ),
            }
        summary.append(row)
    summary.sort(key=lambda row: (
        row["median_combined_mean_nodes_all_positions"],
        row["median_combined_trimmed_geomean_sec"],
        row["median_combined_trimmed_mean_sec"],
    ))
    return summary


def _run_one_position(position, cancel_event, on_depth) -> dict:
    board, turn = benchmark2.load_fen(position.fen)
    root_hash = int(AI.board_hash(board, turn))
    root_moves = get_all_legal_moves(board, turn)
    if not root_moves:
        raise ValueError(f"{position.key}: no legal root moves.")

    bot = benchmark2.build_bot(board, turn, root_hash)
    bot.cancellation_event = cancel_event
    search_calls = [0]
    original_search = bot._search_at_depth

    def counted_search(*args, **kwargs):
        search_calls[0] += 1
        return original_search(*args, **kwargs)

    bot._search_at_depth = counted_search
    best_move = root_moves[0]
    prev_score = None
    depth_rows = {}
    for depth in range(1, MAX_DEPTH + 1):
        if cancel_event.is_set():
            raise SearchCancelledException()
        search_calls[0] = 0
        start = time.perf_counter()
        score, best_move = bot._run_depth_iteration(
            depth, root_moves, root_hash, best_move, prev_iter_score=prev_score
        )
        elapsed = time.perf_counter() - start
        prev_score = score
        if depth in DEPTHS_REPORTED:
            row = {
                "elapsed_sec": elapsed,
                "nodes": int(bot.nodes_searched),
                "search_calls": search_calls[0],
                "score_cp": int(score),
                "best_move": bot._format_move(board, best_move) if best_move else None,
            }
            depth_rows[str(depth)] = row
        on_depth(depth, position)

    return {"key": position.key, "depths": depth_rows}


def _run_tuning(candidates, trials, cancel_event, emit, result_path):
    original_window = AI.ChessBot.ASP_WINDOW_INIT
    run_started = datetime.now().astimezone().isoformat(timespec="seconds")
    try:
        suite_validation = _validate_suite()
    except Exception:
        error = traceback.format_exc()
        saved_path = _atomic_save(result_path, {
            "started": run_started,
            "candidate_windows": candidates,
            "trials_requested": trials,
            "complete": False,
            "error": error,
        })
        emit(("error", error, saved_path))
        return
    total_steps = len(candidates) * trials * len(SUITE) * MAX_DEPTH + MAX_DEPTH
    done_steps = 0
    results = {str(candidate): {"trials": []} for candidate in candidates}
    baseline_window = (
        DEFAULT_BASELINE_WINDOW if DEFAULT_BASELINE_WINDOW in candidates else candidates[0]
    )
    metadata = {
        "started": run_started,
        "ai_header": "",
        "starting_asp_window": original_window,
        "candidate_windows": candidates,
        "baseline_window_for_score_and_move_comparisons": baseline_window,
        "trials_requested": trials,
        "positions": suite_validation,
        "depths_reported": list(DEPTHS_REPORTED),
        "search_max_depth": MAX_DEPTH,
        "suite_source": SUITE_SOURCE,
        "suite_size": SUITE_SIZE,
        "harvest_metadata": HARVEST_METADATA,
        "outlier_policy": (
            f"For every candidate, trial, and reported depth, sort all {SUITE_SIZE} position "
            "times and exclude the fastest and slowest positions. Calculate both "
            "arithmetic and geometric mean time plus time-trimmed nodes/re-searches "
            f"over the remaining {SUITE_SIZE - 2}. Also report full-suite node/re-search means; "
            "full-suite node means are used for ranking so timing noise cannot "
            "change which positions are excluded from the node metric."
        ),
        "ranking_policy": (
            "Rank first by median summed full-suite arithmetic node count at depths "
            "4, 5, and 6; then use median summed trimmed geometric mean seconds, "
            "then arithmetic mean seconds. Node-count repeatability is audited "
            "across trials because node counts are expected to be deterministic here."
        ),
        "comparison_policy": (
            "For every reported depth, compare each candidate's score_cp and "
            "best_move with the same position and trial searched using the "
            "baseline window."
        ),
        "benchmark_setup": (
            "benchmark2.build_bot disables opening book and tablebase use. "
            "Bot construction is outside the timed depth-iteration region."
        ),
    }
    payload = {
        "metadata": metadata,
        "results": results,
        "completed_candidate_trials": 0,
        "total_candidate_trials": len(candidates) * trials,
        "complete": False,
        "cancelled": False,
    }
    with open(os.path.join(PROJECT_DIR, "AI.py"), encoding="utf-8") as handle:
        metadata["ai_header"] = handle.readline().strip()

    last_checkpoint_recovery = None

    def checkpoint():
        nonlocal last_checkpoint_recovery
        payload["results"] = results
        payload["summary"] = _summarize_results(results, baseline_window)
        payload["node_count_audit"] = _node_count_audit(results)
        payload["completed_candidate_trials"] = sum(
            len(value["trials"]) for value in results.values()
        )
        saved_path = _atomic_save(result_path, payload)
        if saved_path != result_path:
            emit(("checkpoint_fallback", saved_path))
            last_checkpoint_recovery = saved_path
        elif last_checkpoint_recovery is not None:
            emit(("checkpoint_restored", result_path))
            last_checkpoint_recovery = None
        return saved_path

    def report_progress(message):
        emit(("progress", done_steps, total_steps, message))

    try:
        # A short discarded search warms interpreter paths before timing.
        AI.ChessBot.ASP_WINDOW_INIT = original_window
        warm_position = SUITE[0]
        warm_board, warm_turn = benchmark2.load_fen(warm_position.fen)
        warm_hash = int(AI.board_hash(warm_board, warm_turn))
        warm_moves = get_all_legal_moves(warm_board, warm_turn)
        warm_bot = benchmark2.build_bot(warm_board, warm_turn, warm_hash)
        warm_bot.cancellation_event = cancel_event
        warm_best = warm_moves[0]
        warm_score = None
        for warm_depth in range(1, MAX_DEPTH + 1):
            if cancel_event.is_set():
                raise SearchCancelledException()
            warm_score, warm_best = warm_bot._run_depth_iteration(
                warm_depth, warm_moves, warm_hash, warm_best, prev_iter_score=warm_score
            )
            done_steps += 1
            report_progress(f"Warm-up search · depth {warm_depth}")

        for trial_index in range(trials):
            if cancel_event.is_set():
                raise SearchCancelledException()
            candidate_order = candidates[trial_index % len(candidates):] + candidates[:trial_index % len(candidates)]
            position_order = list(SUITE)
            random.Random(20261027 + trial_index).shuffle(position_order)

            for candidate in candidate_order:
                if cancel_event.is_set():
                    raise SearchCancelledException()
                AI.ChessBot.ASP_WINDOW_INIT = candidate
                position_rows = []
                for position_index, position in enumerate(position_order, 1):
                    if cancel_event.is_set():
                        raise SearchCancelledException()

                    def on_depth(depth, current_position):
                        nonlocal done_steps
                        done_steps += 1
                        report_progress(
                            f"Window {candidate} · trial {trial_index + 1}/{trials} · "
                            f"{current_position.key} · depth {depth} · "
                            f"position {position_index}/{len(position_order)}"
                        )

                    position_rows.append(_run_one_position(position, cancel_event, on_depth))

                    if position_index % CHECKPOINT_EVERY_POSITIONS == 0:
                        payload["partial_candidate_trial"] = {
                            "window": candidate,
                            "trial": trial_index + 1,
                            "positions_completed": position_index,
                            "position_results": position_rows[:],
                        }
                        checkpoint()

                depth_summaries = {
                    str(depth): _trimmed_depth_summary(position_rows, depth)
                    for depth in DEPTHS_REPORTED
                }
                combined_sec = sum(
                    depth_summaries[str(depth)]["trimmed_mean_sec"]
                    for depth in DEPTHS_REPORTED
                )
                combined_geomean_sec = sum(
                    depth_summaries[str(depth)]["trimmed_geomean_sec"]
                    for depth in DEPTHS_REPORTED
                )
                combined_nodes = sum(
                    depth_summaries[str(depth)]["mean_nodes_all_positions"]
                    for depth in DEPTHS_REPORTED
                )
                trial_result = {
                    "trial": trial_index + 1,
                    "depth_summaries": depth_summaries,
                    "combined_trimmed_mean_sec": combined_sec,
                    "combined_trimmed_geomean_sec": combined_geomean_sec,
                    "combined_mean_nodes_all_positions": combined_nodes,
                    "positions": position_rows,
                }
                results[str(candidate)]["trials"].append(trial_result)
                payload.pop("partial_candidate_trial", None)
                saved_path = checkpoint()
                emit(("trial_complete", candidate, trial_index + 1, trials, depth_summaries,
                      combined_geomean_sec, combined_sec, combined_nodes, saved_path))

        payload["complete"] = True
        payload["finished"] = datetime.now().astimezone().isoformat(timespec="seconds")
        saved_path = checkpoint()
        emit(("finished", payload, saved_path))
    except SearchCancelledException:
        payload["cancelled"] = True
        payload["finished"] = datetime.now().astimezone().isoformat(timespec="seconds")
        saved_path = checkpoint()
        emit(("cancelled", payload, saved_path))
    except Exception:
        payload["error"] = traceback.format_exc()
        payload["finished"] = datetime.now().astimezone().isoformat(timespec="seconds")
        saved_path = checkpoint()
        emit(("error", payload["error"], saved_path))
    finally:
        AI.ChessBot.ASP_WINDOW_INIT = original_window


def _run_tuning_process(candidates, trials, cancel_event, event_queue, result_path):
    """Process entry point, kept module-level for Windows spawn pickling."""
    _run_tuning(candidates, trials, cancel_event, event_queue.put, result_path)


class AspirationTunerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Jungle Chess · Aspiration Window Tuner")
        self.root.geometry("1100x700")
        self.root.minsize(900, 560)
        self.mp_context = mp.get_context("spawn")
        self.events = self.mp_context.Queue()
        self.cancel_event = None
        self.worker = None
        self.running = False
        self.closing = False
        self._terminal_event_received = False
        self._worker_exit_observed_at = None
        self._checkpoint_fallback_active = False
        self._trial_records = {}

        self.candidates_var = tk.StringVar(value=DEFAULT_CANDIDATES)
        self.trials_var = tk.IntVar(value=DEFAULT_TRIALS)
        self.status_var = tk.StringVar(
            value=f"Ready · {SUITE_SIZE} positions · {SUITE_SOURCE} · depths 4, 5, 6"
        )
        self.progress_var = tk.DoubleVar(value=0)

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(100, self._poll_events)

    def _build_ui(self):
        outer = ttk.Frame(self.root, padding=14)
        outer.pack(fill=tk.BOTH, expand=True)

        ttk.Label(outer, text="Aspiration Window Tuner", font=("Segoe UI", 18, "bold")).pack(anchor=tk.W)
        ttk.Label(
            outer,
            text=(f"Compares 200 / 300 / 400 / 450 / 500 / 700 by default across {SUITE_SIZE} "
                  f"{SUITE_SOURCE}. Each position is searched iteratively through depth 6; "
                  "depths 4, 5, and 6 are recorded. "
                  f"The default run is 30 suite passes ({30 * SUITE_SIZE} position searches), "
                  "so it can take a while."),
            wraplength=1040,
        ).pack(anchor=tk.W, pady=(3, 12))

        controls = ttk.Frame(outer)
        controls.pack(fill=tk.X)
        ttk.Label(controls, text="Candidate windows:").grid(row=0, column=0, sticky=tk.W, padx=(0, 6))
        self.candidates_entry = ttk.Entry(controls, textvariable=self.candidates_var, width=28)
        self.candidates_entry.grid(row=0, column=1, sticky=tk.W, padx=(0, 18))
        ttk.Label(controls, text="Trials per window:").grid(row=0, column=2, sticky=tk.W, padx=(0, 6))
        self.trials_spinbox = ttk.Spinbox(controls, from_=1, to=100, textvariable=self.trials_var, width=7)
        self.trials_spinbox.grid(row=0, column=3, sticky=tk.W, padx=(0, 18))
        self.start_button = ttk.Button(controls, text="Start tuning", command=self._start)
        self.start_button.grid(row=0, column=4, padx=(0, 6))
        self.stop_button = ttk.Button(controls, text="Stop search", command=self._stop, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=5)
        controls.columnconfigure(6, weight=1)

        ttk.Label(
            outer,
            text=("For each trial and depth, the fastest and slowest positions are discarded. The tuner "
                  "reports geometric and arithmetic time means, average nodes, and aspiration "
                  "re-searches. It ranks by nodes first and time second, checks node repeatability, "
                  "and compares scores/moves against 450. Candidate and position order rotate "
                  "between trials; search runs in a separate process."),
            wraplength=1040,
        ).pack(anchor=tk.W, pady=(10, 8))

        self.progress = ttk.Progressbar(outer, variable=self.progress_var, maximum=1, mode="determinate")
        self.progress.pack(fill=tk.X)
        ttk.Label(outer, textvariable=self.status_var).pack(anchor=tk.W, pady=(5, 10))

        columns = ("window", "trials", "d4", "d5", "d6", "combined", "nodes", "retries")
        self.table = ttk.Treeview(outer, columns=columns, show="headings", height=9)
        headings = {
            "window": ("Window", 100),
            "trials": ("Trials", 80),
            "d4": ("D4 geo s / pos", 125),
            "d5": ("D5 geo s / pos", 125),
            "d6": ("D6 geo s / pos", 125),
            "combined": ("D4+D5+D6 geo s", 145),
            "nodes": ("Combined nodes", 145),
            "retries": ("D6 retries / pos", 130),
        }
        for key, (title, width) in headings.items():
            self.table.heading(key, text=title)
            self.table.column(key, width=width, anchor=tk.CENTER, stretch=True)
        self.table.pack(fill=tk.BOTH, expand=True)

        self.log = tk.Text(outer, height=7, wrap=tk.WORD, state=tk.DISABLED)
        self.log.pack(fill=tk.BOTH, expand=False, pady=(10, 0))
        self._log(
            "Results are checkpointed beside this tuner. Transient OneDrive locks are retried; "
            "a recovery JSON is kept if the main file remains locked."
        )
        if SUITE_SOURCE == "built-in fallback suite":
            self._log(
                "Run an AI-vs-OP series to create aspiration_series_positions.json; "
                "the tuner will automatically prefer that 50-position suite next time."
            )
        else:
            self._log(f"Loaded {SUITE_SIZE} positions from the AI-vs-OP series harvest.")
        self._log("Baseline comparisons use 450 when present. The tuner never edits AI.py.")

    def _log(self, message):
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, message + "\n")
        self.log.see(tk.END)
        self.log.configure(state=tk.DISABLED)

    def _start(self):
        if self.running:
            return
        try:
            candidates = _parse_candidates(self.candidates_var.get())
            trials = int(self.trials_var.get())
            if trials < 1:
                raise ValueError("Trials per window must be at least 1.")
        except (ValueError, tk.TclError) as exc:
            messagebox.showerror("Invalid tuning settings", str(exc), parent=self.root)
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_path = os.path.join(SCRIPT_DIR, f"aspiration_tuner_results_{timestamp}.json")
        self._trial_records = {}
        for item in self.table.get_children():
            self.table.delete(item)
        self.running = True
        self._terminal_event_received = False
        self._worker_exit_observed_at = None
        self._checkpoint_fallback_active = False
        self.cancel_event = self.mp_context.Event()
        self.start_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL)
        self.candidates_entry.configure(state=tk.DISABLED)
        self.trials_spinbox.configure(state=tk.DISABLED)
        self.progress.configure(maximum=len(candidates) * trials * len(SUITE) * MAX_DEPTH + MAX_DEPTH)
        self.progress_var.set(0)
        self.status_var.set(f"Validating the {SUITE_SIZE}-position suite…")
        self._log(f"Started: windows={candidates}, trials/window={trials}; output={result_path}")

        self.worker = self.mp_context.Process(
            target=_run_tuning_process,
            args=(candidates, trials, self.cancel_event, self.events, result_path),
            daemon=True,
        )
        try:
            self.worker.start()
        except Exception as exc:
            failed_worker = self.worker
            self.worker = None
            failed_worker.close()
            self._finish(f"Could not start benchmark process: {exc}")
            self._log(traceback.format_exc())
            messagebox.showerror("Could not start tuner", str(exc), parent=self.root)

    def _stop(self):
        if self.cancel_event is not None:
            self.cancel_event.set()
            self.stop_button.configure(state=tk.DISABLED)
            self.status_var.set("Stopping after the current search node…")

    def _poll_events(self):
        try:
            while True:
                event = self.events.get_nowait()
                kind = event[0]
                if kind == "progress":
                    _, done, total, message = event
                    self.progress.configure(maximum=total)
                    self.progress_var.set(done)
                    self.status_var.set(f"{done:,}/{total:,} search depths · {message}")
                elif kind == "checkpoint_fallback":
                    _, recovery_path = event
                    if not self._checkpoint_fallback_active:
                        self._log(
                            "The results JSON is temporarily locked. The tuner is keeping "
                            "a complete recovery checkpoint beside it."
                        )
                    self._checkpoint_fallback_active = True
                    self.status_var.set(
                        f"Results file locked · recovery checkpoint: {recovery_path}"
                    )
                elif kind == "checkpoint_restored":
                    _, saved_path = event
                    if self._checkpoint_fallback_active:
                        self._log(f"Checkpointing resumed to the results file: {saved_path}")
                    self._checkpoint_fallback_active = False
                elif kind == "trial_complete":
                    (_, candidate, trial, total_trials, summaries, combined_geomean_sec,
                     combined_mean_sec, combined_nodes, path) = event
                    self._trial_records.setdefault(candidate, []).append({
                        "trial": trial,
                        "summaries": summaries,
                        "combined_geomean_sec": combined_geomean_sec,
                        "combined_mean_sec": combined_mean_sec,
                        "combined_nodes": combined_nodes,
                    })
                    self._refresh_table()
                    self._log(
                        f"Window {candidate}, trial {trial}/{total_trials}: "
                        f"full-suite D4+D5+D6 = {combined_nodes:,.0f} nodes, "
                        f"{combined_geomean_sec:.3f}s geometric / {combined_mean_sec:.3f}s arithmetic. "
                        f"Checkpoint: {os.path.basename(path)}"
                    )
                elif kind == "finished":
                    _, payload, path = event
                    self._terminal_event_received = True
                    self._finish(f"Complete · saved {path}")
                    self._log_node_audit(payload)
                    self._log_baseline_comparisons(payload)
                    winner = self._winner()
                    if winner is not None:
                        self._log(
                            f"Top candidate by median node count, then timing: {winner}. "
                            "Check the logged score and move changes before choosing it."
                        )
                        baseline = payload.get("metadata", {}).get(
                            "baseline_window_for_score_and_move_comparisons"
                        )
                        baseline_rows = self._trial_records.get(baseline, [])
                        winner_rows = self._trial_records.get(winner, [])
                        if baseline_rows and winner != baseline and winner_rows:
                            base_nodes = statistics.median(
                                row["combined_nodes"] for row in baseline_rows
                            )
                            winner_nodes = statistics.median(
                                row["combined_nodes"] for row in winner_rows
                            )
                            relative_change = abs(winner_nodes - base_nodes) / max(base_nodes, 1)
                            if relative_change < 0.03:
                                self._log(
                                    f"The node difference from {baseline} is under 3% "
                                    "and may not justify changing the default; confirm with an AI-vs-OP series."
                                )
                    self._reap_worker()
                elif kind == "cancelled":
                    _, payload, path = event
                    self._terminal_event_received = True
                    self._finish(f"Stopped · completed trials saved to {path}")
                    self._log(f"Completed candidate trials saved: {payload['completed_candidate_trials']}")
                    self._log_node_audit(payload)
                    self._reap_worker()
                elif kind == "error":
                    _, error, path = event
                    self._terminal_event_received = True
                    self._finish(f"Run failed · progress saved to {path}")
                    self._log(error)
                    messagebox.showerror("Tuner error", error, parent=self.root)
                    self._reap_worker()
        except queue.Empty:
            pass

        if self.worker is not None and not self.worker.is_alive():
            if self._terminal_event_received:
                self._reap_worker()
            elif self.running:
                now = time.monotonic()
                if self._worker_exit_observed_at is None:
                    self._worker_exit_observed_at = now
                elif now - self._worker_exit_observed_at > 1.0:
                    self._finish(
                        f"Benchmark process exited without a result event "
                        f"(exit code {self.worker.exitcode})."
                    )
                    self._log("The result file may contain the last completed checkpoint.")
                    self._reap_worker()

        if self.closing and not self.running:
            self._close_event_queue()
            self.root.destroy()
            return
        self.root.after(100, self._poll_events)

    def _refresh_table(self):
        for item in self.table.get_children():
            self.table.delete(item)
        rows = []
        for candidate, trials in self._trial_records.items():
            by_depth = {}
            for depth in DEPTHS_REPORTED:
                by_depth[depth] = statistics.median(
                    row["summaries"][str(depth)]["trimmed_geomean_sec"] for row in trials
                )
            combined = statistics.median(row["combined_geomean_sec"] for row in trials)
            nodes = statistics.median(row["combined_nodes"] for row in trials)
            retries = statistics.median(
                row["summaries"]["6"]["mean_researches_all_positions"] for row in trials
            )
            rows.append((nodes, combined, candidate, len(trials), by_depth, retries))
        rows.sort(key=lambda row: (row[0], row[1]))
        for rank, (nodes, combined, candidate, count, by_depth, retries) in enumerate(rows):
            self.table.insert("", tk.END, values=(
                f"{candidate}" + ("  ★" if rank == 0 and count else ""),
                f"{count}",
                f"{by_depth[4]:.3f}",
                f"{by_depth[5]:.3f}",
                f"{by_depth[6]:.3f}",
                f"{combined:.3f}",
                f"{nodes:,.0f}",
                f"{retries:.2f}",
            ))

    def _winner(self):
        if not self._trial_records:
            return None
        return min(
            self._trial_records,
            key=lambda candidate: (
                statistics.median(row["combined_nodes"] for row in self._trial_records[candidate]),
                statistics.median(row["combined_geomean_sec"] for row in self._trial_records[candidate]),
                statistics.median(row["combined_mean_sec"] for row in self._trial_records[candidate]),
            ),
        )

    def _log_node_audit(self, payload):
        audit = payload.get("node_count_audit", {})
        if not audit.get("determinism_verified"):
            self._log("Node audit: not enough repeated trial data to confirm determinism.")
        elif audit.get("deterministic_across_completed_trials"):
            self._log("Node audit: counts matched for every repeated candidate/position/depth.")
        else:
            self._log(
                "Node audit: counts varied for "
                f"{audit.get('position_depth_mismatch_count', '?')} candidate/position/depth groups. "
                "Node ranking is less conclusive; inspect JSON examples."
            )

    def _log_baseline_comparisons(self, payload):
        baseline = payload.get("metadata", {}).get(
            "baseline_window_for_score_and_move_comparisons", DEFAULT_BASELINE_WINDOW
        )
        for row in payload.get("summary", []):
            if row["window"] == baseline:
                continue
            depth_comparisons = row.get("baseline_comparison", {}).get("by_depth", {})
            details = []
            for depth in DEPTHS_REPORTED:
                stats = depth_comparisons.get(str(depth), {})
                mean_delta = stats.get("mean_absolute_score_delta_cp")
                move_rate = stats.get("best_move_change_rate")
                if mean_delta is None:
                    details.append(f"D{depth}: no paired results")
                else:
                    details.append(
                        f"D{depth}: |score Δ| {mean_delta:.1f} cp, "
                        f"move changed {move_rate:.1%}"
                    )
            self._log(f"Window {row['window']} vs {baseline}: " + "; ".join(details) + ".")

    def _reap_worker(self):
        if self.worker is None:
            return
        self.worker.join(timeout=0.5)
        if not self.worker.is_alive():
            self.worker.close()
            self.worker = None

    def _close_event_queue(self):
        if self.events is not None:
            self.events.close()
            self.events.join_thread()
            self.events = None

    def _finish(self, status):
        self.running = False
        self.start_button.configure(state=tk.NORMAL)
        self.stop_button.configure(state=tk.DISABLED)
        self.candidates_entry.configure(state=tk.NORMAL)
        self.trials_spinbox.configure(state=tk.NORMAL)
        self.status_var.set(status)

    def _on_close(self):
        if self.running:
            self.closing = True
            self._stop()
            self.status_var.set("Stopping safely before closing…")
        else:
            self._reap_worker()
            self._close_event_queue()
            self.root.destroy()


def main():
    mp.freeze_support()
    root = tk.Tk()
    AspirationTunerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
