# PieceTuner.py (v1.5 - Configurable generation depth + fixed step-size skipping)

import math
import json
import os
import random
import time
import multiprocessing as mp

from GameLogic import (
    Board, Pawn, Knight, Bishop, Rook, Queen, King,
    get_all_legal_moves, is_insufficient_material, ROWS, COLS
)
import AI
from AI import board_hash

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

GAMES_TO_GENERATE    = 1000  # Number of self-play games to generate
GENERATION_DEPTH     = 2     # Search depth per move: 1 = fast, 2 = slower but better quality
MAX_GAME_PLIES       = 200   # End game after this many half-moves (longer = more EG data)
SAMPLE_EVERY_N_PLIES = 2     # Record 1 position per N plies (2 = good balance of volume vs correlation)
DATA_FILE            = "tuner_positions.json"

COORD_ROUNDS         = 20    # Max rounds per step size — early-exit handles convergence
K_CALIBRATION_STEPS  = 40    # Golden-section steps to find optimal sigmoid K
STEP_SIZES           = [50, 30, 20, 10, 5]  # Centipawn step sizes, coarse → fine

NUM_WORKERS = max(1, (mp.cpu_count() or 2) - 1)

# ═══════════════════════════════════════════════════════════════════════════════
# PIECE VALUE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

PIECE_CLASSES = [Pawn, Knight, Bishop, Rook, Queen]
PARAM_NAMES   = ['MG Knight', 'MG Bishop', 'MG Rook', 'MG Queen',
                 'EG Pawn', 'EG Knight', 'EG Bishop', 'EG Rook', 'EG Queen']
PARAM_BOUNDS  = [
    (200, 1800), (200, 1800), (200, 1800), (300, 2000),   # MG
    ( 50,  350), (200, 1800), (200, 1800), (200, 1800), (300, 2000),  # EG
]

def patch_ai_values(mg_vals: dict, eg_vals: dict) -> None:
    """Apply piece value dicts to the live AI module and recompute phase constant."""
    for cls in PIECE_CLASSES:
        AI.MG_PIECE_VALUES[cls] = mg_vals[cls]
        AI.EG_PIECE_VALUES[cls] = eg_vals[cls]
    AI.INITIAL_PHASE_MATERIAL = (
        AI.MG_PIECE_VALUES[Rook]   * 4 +
        AI.MG_PIECE_VALUES[Knight] * 4 +
        AI.MG_PIECE_VALUES[Bishop] * 4 +
        AI.MG_PIECE_VALUES[Queen]  * 2
    )

def vals_to_vec(mg: dict, eg: dict) -> list:
    return [
        mg[Knight], mg[Bishop], mg[Rook], mg[Queen],
        eg[Pawn], eg[Knight], eg[Bishop], eg[Rook], eg[Queen],
    ]

def vec_to_vals(vec: list) -> tuple:
    mg = {Pawn: 100, Knight: vec[0], Bishop: vec[1], Rook: vec[2], Queen: vec[3], King: 20000}
    eg = {Pawn: vec[4], Knight: vec[5], Bishop: vec[6], Rook: vec[7], Queen: vec[8], King: 20000}
    return mg, eg

# ═══════════════════════════════════════════════════════════════════════════════
# BOARD SERIALISATION
# ═══════════════════════════════════════════════════════════════════════════════

_CLS_TO_CHAR = {Pawn: 'P', Knight: 'N', Bishop: 'B', Rook: 'R', Queen: 'Q', King: 'K'}
_CHAR_TO_CLS = {v: k for k, v in _CLS_TO_CHAR.items()}

def board_to_fen(board: Board, turn: str) -> str:
    fen = ''
    for r in range(ROWS):
        empty = 0
        for c in range(COLS):
            piece = board.grid[r][c]
            if piece is None:
                empty += 1
            else:
                if empty:
                    fen += str(empty)
                    empty = 0
                ch = _CLS_TO_CHAR[type(piece)]
                fen += ch if piece.color == 'white' else ch.lower()
        if empty:
            fen += str(empty)
        if r < ROWS - 1:
            fen += '/'
    return fen + (' w' if turn == 'white' else ' b')

def fen_to_board(fen_str: str) -> tuple:
    parts      = fen_str.split(' ')
    board_part = parts[0]
    turn       = 'white' if parts[1] == 'w' else 'black'
    board      = Board(setup=False)
    r = c = 0
    for ch in board_part:
        if ch == '/':
            r += 1; c = 0
        elif ch.isdigit():
            c += int(ch)
        else:
            color = 'white' if ch.isupper() else 'black'
            board.add_piece(_CHAR_TO_CLS[ch.upper()](color), r, c)
            c += 1
    return board, turn

# ═══════════════════════════════════════════════════════════════════════════════
# SELF-PLAY GAME GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

# Defined at top level so multiprocessing can pickle it
class MockTB:
    def probe(self, *args, **kwargs): return None

_bot_singleton = None

def _get_bot() -> AI.ChessBot:
    global _bot_singleton
    if _bot_singleton is None:
        dummy_q = mp.Queue()
        dummy_e = mp.Event()
        _bot_singleton = AI.ChessBot(Board(), 'white', {}, dummy_q, dummy_e, 'Tuner')
        _bot_singleton.tb_manager = MockTB()
    return _bot_singleton

def _play_one_game(args: tuple) -> list:
    """
    Play one self-play game at GENERATION_DEPTH + native Q-Search.
    Depth 1 = fast, Depth 2 = stronger positional play, both avoid horizon blunders
    thanks to Q-Search at the leaves.
    """
    game_idx, max_plies, sample_every, gen_depth = args
    random.seed(game_idx * 31337 + int(time.time() * 1000) % 999_983)

    board          = Board()
    turn           = 'white'
    position_count = {}
    plies          = 0
    sampled_fens   = []
    outcome        = 0.5

    bot = _get_bot()
    bot._initialize_search_state()  # Clear stale TT from prior games in this worker

    while plies < max_plies:
        legal = get_all_legal_moves(board, turn)
        if not legal:
            outcome = 1.0 if turn == 'black' else 0.0
            break
        if is_insufficient_material(board):
            break
        h = board_hash(board, turn)
        position_count[h] = position_count.get(h, 0) + 1
        if position_count[h] >= 3:
            break

        if plies % sample_every == 0:
            sampled_fens.append(board_to_fen(board, turn))

        if plies < 2 or len(legal) == 1:
            # Force random opening moves for variety; also handles forced moves
            move = random.choice(legal)
        else:
            random.shuffle(legal)
            opp        = 'black' if turn == 'white' else 'white'
            root_hash  = board_hash(board, turn)
            best_score = -float('inf')
            best_move  = legal[0]

            for m in legal:
                child = board.clone()
                child.make_move(m[0], m[1])
                score = -bot.negamax(
                    child, gen_depth, -float('inf'), float('inf'),
                    opp, 1, {root_hash},
                    current_hash=board_hash(child, opp)
                )
                score += random.uniform(0, 10)  # Small noise to avoid deterministic repetition
                if score > best_score:
                    best_score, best_move = score, m
            move = best_move

        board.make_move(move[0], move[1])
        turn = 'black' if turn == 'white' else 'white'
        plies += 1

    return [(fen, outcome) for fen in sampled_fens]

def generate_training_data(n_games: int, max_plies: int, sample_every: int,
                           gen_depth: int, n_workers: int) -> list:
    print(f"\nGenerating {n_games} self-play games at Depth {gen_depth} + Q-Search ({n_workers} workers)...")
    args = [(i, max_plies, sample_every, gen_depth) for i in range(n_games)]
    all_positions = []
    start_time = time.time()

    with mp.Pool(n_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(_play_one_game, args, chunksize=2), 1):
            all_positions.extend(result)
            elapsed = time.time() - start_time
            speed   = i / elapsed
            eta     = (n_games - i) / speed if speed > 0 else 0
            print(f"  {i}/{n_games} games  |  {len(all_positions)} positions  |  ETA: {eta/60:.1f}m", flush=True)

    random.shuffle(all_positions)
    print(f"\nTotal training positions: {len(all_positions)}")
    return all_positions

# ═══════════════════════════════════════════════════════════════════════════════
# TEXEL LOSS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_mse(positions_data: list, K: float, mg_vals: dict, eg_vals: dict) -> float:
    patch_ai_values(mg_vals, eg_vals)
    bot   = _get_bot()
    total = 0.0
    count = 0
    for fen_str, outcome in positions_data:
        try:
            board, _ = fen_to_board(fen_str)
            raw  = bot.evaluate_board(board, 'white')
            prob = 1.0 / (1.0 + math.exp(-K * raw / 400.0))
            total += (prob - outcome) ** 2
            count += 1
        except Exception:
            continue
    return total / max(1, count)

def calibrate_K(positions_data: list, mg: dict, eg: dict, steps: int = 40) -> float:
    print("\nCalibrating Texel K constant...")
    phi    = (1 + 5 ** 0.5) / 2
    lo, hi = 0.3, 4.0
    for _ in range(steps):
        k1 = hi - (hi - lo) / phi
        k2 = lo + (hi - lo) / phi
        if compute_mse(positions_data, k1, mg, eg) < compute_mse(positions_data, k2, mg, eg):
            hi = k2
        else:
            lo = k1
        if (hi - lo) < 0.002:
            break
    best_K = (lo + hi) / 2
    print(f"  Optimal K = {best_K:.4f}", flush=True)
    return best_K

def coordinate_descent(positions_data: list, K: float, mg_start: dict, eg_start: dict,
                       n_rounds: int, step_sizes: list) -> tuple:
    vec      = vals_to_vec(mg_start, eg_start)
    best_mse = compute_mse(positions_data, K, *vec_to_vals(vec))
    print(f"\nStarting coordinate descent — initial MSE = {best_mse:.6f}", flush=True)

    for step in step_sizes:
        print(f"\n── Step size: {step} cp ──────────────────────────────────", flush=True)
        for round_num in range(1, n_rounds + 1):
            improved_any = False

            for i, name in enumerate(PARAM_NAMES):
                lo_bound, hi_bound = PARAM_BOUNDS[i]

                v_up  = vec.copy(); v_up[i] += step
                v_dn  = vec.copy(); v_dn[i] -= step
                mse_up = compute_mse(positions_data, K, *vec_to_vals(v_up)) if v_up[i] <= hi_bound else float('inf')
                mse_dn = compute_mse(positions_data, K, *vec_to_vals(v_dn)) if v_dn[i] >= lo_bound else float('inf')

                if mse_up < best_mse and mse_up <= mse_dn:
                    old = vec[i]; vec = v_up; best_mse = mse_up; improved_any = True
                    print(f"  {name:12s}: {old:.0f} → {vec[i]:.0f}  (MSE={best_mse:.6f})", flush=True)
                elif mse_dn < best_mse:
                    old = vec[i]; vec = v_dn; best_mse = mse_dn; improved_any = True
                    print(f"  {name:12s}: {old:.0f} → {vec[i]:.0f}  (MSE={best_mse:.6f})", flush=True)

            print(f"  [round {round_num}] MSE = {best_mse:.6f}" + ("" if improved_any else "  (converged)"), flush=True)
            if not improved_any:
                break  # Move to next step size; does NOT skip remaining step sizes

    return vec_to_vals(vec), best_mse

# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT FORMATTING
# ═══════════════════════════════════════════════════════════════════════════════

def print_ai_dicts(mg: dict, eg: dict, header: str = "") -> None:
    if header:
        print(f"\n{'─' * 50}\n{header}\n{'─' * 50}", flush=True)

    def _fmt(d: dict) -> str:
        lines = [f"    {cls.__name__}: {int(round(d[cls]))}," for cls in PIECE_CLASSES]
        lines.append("    King: 20000")
        return '\n'.join(lines)

    print(f"MG_PIECE_VALUES = {{\n{_fmt(mg)}\n}}", flush=True)
    print(f"\nEG_PIECE_VALUES = {{\n{_fmt(eg)}\n}}", flush=True)

    print("\n  Delta vs original AI.py values:")
    print(f"  {'Piece':<10} {'MG orig':>8} {'MG new':>8} {'MG Δ':>7}  {'EG orig':>8} {'EG new':>8} {'EG Δ':>7}")
    print("  " + "─" * 64)
    orig_mg = {Pawn: 100, Knight: 900, Bishop: 650, Rook: 550, Queen: 850}
    orig_eg = {Pawn: 130, Knight: 800, Bishop: 550, Rook: 600, Queen: 850}
    for cls in PIECE_CLASSES:
        dm = int(round(mg[cls])) - orig_mg[cls]
        de = int(round(eg[cls])) - orig_eg[cls]
        print(f"  {cls.__name__:<10} {orig_mg[cls]:>8} {int(round(mg[cls])):>8} {dm:>+7}  "
              f"{orig_eg[cls]:>8} {int(round(eg[cls])):>8} {de:>+7}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("═" * 60, flush=True)
    print(f"  Jungle Chess Piece Value Auto-Tuner  (v1.5)", flush=True)
    print(f"  Depth {GENERATION_DEPTH} + Q-Search  |  {GAMES_TO_GENERATE} games  |  {NUM_WORKERS} workers", flush=True)
    print("═" * 60, flush=True)

    positions_data = None
    if os.path.exists(DATA_FILE) and os.path.getsize(DATA_FILE) > 0:
        try:
            with open(DATA_FILE) as f:
                raw = json.load(f)
            positions_data = [(d['fen'], d['outcome']) for d in raw]
            print(f"\nFound existing training data: {len(positions_data)} positions.")
            print("  Regenerate from scratch? (y/N): ", end='', flush=True)
            if input().strip().lower() == 'y':
                positions_data = None
        except Exception:
            positions_data = None

    if positions_data is None:
        positions_data = generate_training_data(
            GAMES_TO_GENERATE, MAX_GAME_PLIES, SAMPLE_EVERY_N_PLIES,
            GENERATION_DEPTH, NUM_WORKERS
        )
        print(f"\nSaving to {DATA_FILE}...", flush=True)
        with open(DATA_FILE, 'w') as f:
            json.dump([{'fen': fen, 'outcome': out} for fen, out in positions_data], f)

    if not positions_data:
        print("ERROR: No training positions generated. Exiting.")
        return

    outcomes = [o for _, o in positions_data]
    w = sum(o == 1.0 for o in outcomes) / len(outcomes)
    d = sum(o == 0.5 for o in outcomes) / len(outcomes)
    b = sum(o == 0.0 for o in outcomes) / len(outcomes)
    print(f"\nOutcome distribution:  White {w:.1%}  |  Draw {d:.1%}  |  Black {b:.1%}", flush=True)

    mg_start = {cls: AI.MG_PIECE_VALUES[cls] for cls in PIECE_CLASSES}
    eg_start = {cls: AI.EG_PIECE_VALUES[cls] for cls in PIECE_CLASSES}
    print_ai_dicts(mg_start, eg_start, "Starting values (from AI.py):")

    K = calibrate_K(positions_data, mg_start, eg_start, K_CALIBRATION_STEPS)

    print(f"\nRunning coordinate descent ({COORD_ROUNDS} rounds max × {len(STEP_SIZES)} step sizes)...", flush=True)
    (mg_best, eg_best), final_mse = coordinate_descent(
        positions_data, K, mg_start, eg_start, COORD_ROUNDS, STEP_SIZES
    )

    print(f"\n{'═' * 60}", flush=True)
    print(f"  TUNING COMPLETE  —  final MSE = {final_mse:.6f}  (K = {K:.4f})", flush=True)
    print(f"{'═' * 60}", flush=True)
    print_ai_dicts(mg_best, eg_best, "Optimised values — paste into AI.py:")

    result_file = "tuner_results.json"
    with open(result_file, 'w') as f:
        json.dump({
            'K':                K,
            'final_mse':        final_mse,
            'generation_depth': GENERATION_DEPTH,
            'mg': {cls.__name__: int(round(mg_best[cls])) for cls in PIECE_CLASSES},
            'eg': {cls.__name__: int(round(eg_best[cls])) for cls in PIECE_CLASSES},
            'games_used':       GAMES_TO_GENERATE,
            'positions_used':   len(positions_data),
            'timestamp':        time.strftime('%Y-%m-%d %H:%M:%S'),
        }, f, indent=2)
    print(f"\nResults saved to {result_file}", flush=True)

if __name__ == '__main__':
    mp.freeze_support()
    main()