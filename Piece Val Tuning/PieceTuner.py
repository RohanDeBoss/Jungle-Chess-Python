# PieceTuner.py (v3.3 - Compatible with AI v114.3 and GameLogic v59.1)

import sys
import os
import math
import json
import random
import time
import multiprocessing as mp
from collections import defaultdict

# ═══════════════════════════════════════════════════════════════════════════════
# PATH INJECTION (Allows running from subdirectory without moving main files)
# ═══════════════════════════════════════════════════════════════════════════════
TUNER_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(TUNER_DIR, '..'))

if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

# Change working directory so AI.py finds opening_book.json and tablebases/
os.chdir(PARENT_DIR)

# Now we can safely import from the parent directory
from GameLogic import (
    Board, Pawn, Knight, Bishop, Rook, Queen, King,
    get_all_legal_moves, is_insufficient_material, ROWS, COLS
)
import AI
from AI import board_hash

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

GAMES_TO_GENERATE       = 1000     # Games to generate (or APPEND) per run
GENERATION_DEPTH        = 7      # Search depth per move
RANDOM_OPENING_PLIES    = 4      # Random moves before engine kicks in (diversity)
MAX_GAME_PLIES          = 200    # Hard ply limit
SAMPLE_EVERY_N_PLIES    = 2      # Record 1 position per N plies

# Data files save directly into the 'Piece Val Tuning' folder
DATA_FILE               = os.path.join(TUNER_DIR, "tuner_positions.json")
RESULT_FILE             = os.path.join(TUNER_DIR, "tuner_results.json")

CHECKPOINT_EVERY        = 50     # Flush partial results every N games (crash safety)

OPENING_EVAL_THRESHOLD  = 400    # cp — discard game if post-opening eval exceeds this
OPENING_SCREEN_DEPTH    = 8      # Depth for the opening screen check
MAX_REGENERATION_TRIES  = 15     # Max attempts to find a balanced opening per game slot
TB_ADJUDICATION_PIECES  = 5      # Probe TB and adjudicate if total pieces <= this

COORD_ROUNDS            = 15     # Max rounds per step size
K_CALIBRATION_STEPS     = 40     # Golden-section steps to find optimal sigmoid K
STEP_SIZES              = [50, 30, 20, 10, 5]

NUM_WORKERS = max(1, (mp.cpu_count() or 2) - 1)

# Terminal condition labels
TERM_CHECKMATE    = 'checkmate'
TERM_INSUFFICIENT = 'insufficient_material'
TERM_THREEFOLD    = 'threefold_repetition'
TERM_PLY_LIMIT    = 'ply_limit'
TERM_TB           = 'tb_adjudication'

# ═══════════════════════════════════════════════════════════════════════════════
# PIECE VALUE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

PIECE_CLASSES = [Pawn, Knight, Bishop, Rook, Queen]
PARAM_NAMES   = ['MG Knight', 'MG Bishop', 'MG Rook', 'MG Queen',
                 'EG Pawn', 'EG Knight', 'EG Bishop', 'EG Rook', 'EG Queen']
PARAM_BOUNDS  = [
    (200, 1800), (200, 1800), (200, 1800), (300, 2000),
    ( 50,  350), (200, 1800), (200, 1800), (200, 1800), (300, 2000),
]

def _rebuild_flat_psts() -> None:
    PST = AI.PIECE_SQUARE_TABLES
    for pt in [Pawn, Knight, Bishop, Rook, Queen, King]:
        mg_val = AI.MG_PIECE_VALUES[pt]
        eg_val = AI.EG_PIECE_VALUES[pt]
        if pt == Pawn:
            mg_table = PST[Pawn];          eg_table = PST['pawn_endgame']
        elif pt == King:
            mg_table = PST['king_midgame']; eg_table = PST['king_endgame']
        else:
            mg_table = PST[pt];            eg_table = PST[pt]
        for r in range(8):
            for c in range(8):
                sq_w = r * 8 + c
                sq_b = (7 - r) * 8 + c
                z = pt.z_idx
                AI.FLAT_PST_MG_WHITE[z][sq_w] = mg_val + mg_table[r][c]
                AI.FLAT_PST_MG_BLACK[z][sq_b] = mg_val + mg_table[r][c]
                AI.FLAT_PST_EG_WHITE[z][sq_w] = eg_val + eg_table[r][c]
                AI.FLAT_PST_EG_BLACK[z][sq_b] = eg_val + eg_table[r][c]

def patch_ai_values(mg_vals: dict, eg_vals: dict) -> None:
    for cls in PIECE_CLASSES:
        AI.MG_PIECE_VALUES[cls] = mg_vals[cls]
        AI.EG_PIECE_VALUES[cls] = eg_vals[cls]
    AI.INITIAL_PHASE_MATERIAL = (
        AI.MG_PIECE_VALUES[Rook]   * 4 +
        AI.MG_PIECE_VALUES[Knight] * 4 +
        AI.MG_PIECE_VALUES[Bishop] * 4 +
        AI.MG_PIECE_VALUES[Queen]  * 2
    )
    _rebuild_flat_psts()

def vals_to_vec(mg: dict, eg: dict) -> list:
    return [mg[Knight], mg[Bishop], mg[Rook], mg[Queen],
            eg[Pawn], eg[Knight], eg[Bishop], eg[Rook], eg[Queen]]

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

# Mock classes to prevent "Daemonic Process" MP crashes on Windows
class DummyQueue:
    def put(self, item): pass

class DummyEvent:
    def is_set(self): return False
    def clear(self): pass

_bot_singleton = None

def _get_bot() -> AI.ChessBot:
    global _bot_singleton
    if _bot_singleton is None:
        # Use Dummy objects instead of mp.Queue() inside workers
        _bot_singleton = AI.ChessBot(Board(), 'white', {}, DummyQueue(), DummyEvent(), bot_name='Tuner', ply_count=0, use_opening_book=False, use_tablebase=True)
    return _bot_singleton

def _play_one_game(args: tuple) -> tuple:
    game_idx, max_plies, sample_every, gen_depth, rand_plies = args
    bot = _get_bot()
    
    bot._report_log = lambda msg: None
    bot._report_eval = lambda s, d: None
    bot._report_move = lambda m: None

    def _search_with_id(target_depth, legal_moves, root_h):
        b_move = legal_moves[0]
        p_score = None
        for d in range(1, target_depth + 1):
            p_score, b_move = bot._run_depth_iteration(
                d, legal_moves, root_h, b_move, prev_iter_score=p_score
            )
            if p_score is not None and abs(p_score) > bot.MATE_SCORE - 2000:
                break
        return p_score, b_move

    for attempt in range(MAX_REGENERATION_TRIES):
        seed = game_idx * 31337 + attempt * 7919 + int(time.time() * 1000) % 999_983
        random.seed(seed)

        board          = Board()
        turn           = 'white'
        position_count = {}   
        plies          = 0
        sampled_fens   = []
        outcome        = 0.5
        terminal       = TERM_PLY_LIMIT

        bot._initialize_search_state()
        bot._age_history_table()

        # ── Random opening ───────────────────────────────────────────────────
        while plies < rand_plies:
            legal = get_all_legal_moves(board, turn)
            if not legal:
                break
            chosen = random.choice(legal)
            board.make_move(chosen[0], chosen[1])
            turn  = 'black' if turn == 'white' else 'white'
            plies += 1

        # ── Opening screen ───────────────────────────────────────────────────
        legal = get_all_legal_moves(board, turn)
        if not legal:
            continue

        bot.board           = board
        bot.color           = turn
        bot.opponent_color  = 'black' if turn == 'white' else 'white'
        bot.position_counts = dict(position_count)
        bot.ply_count       = plies
        root_hash = board_hash(board, turn)

        screen_score, _ = _search_with_id(OPENING_SCREEN_DEPTH, legal, root_hash)
        if abs(screen_score) > OPENING_EVAL_THRESHOLD:
            continue  

        # ── Main game loop ────────────────────────────────────────────────────
        while plies < max_plies:
            legal = get_all_legal_moves(board, turn)
            if not legal:
                outcome  = 1.0 if turn == 'black' else 0.0
                terminal = TERM_CHECKMATE
                break

            if is_insufficient_material(board):
                outcome  = 0.5
                terminal = TERM_INSUFFICIENT
                break

            h = board_hash(board, turn)
            position_count[h] = position_count.get(h, 0) + 1
            if position_count[h] >= 3:
                outcome  = 0.5
                terminal = TERM_THREEFOLD
                break

            total_pieces = len(board.white_pieces) + len(board.black_pieces)
            if total_pieces <= TB_ADJUDICATION_PIECES:
                tb_score_abs = bot.tb_manager.probe(board, turn)
                if tb_score_abs is not None:
                    if tb_score_abs > bot.MATE_SCORE - 1000:
                        outcome = 1.0  # White Wins
                    elif tb_score_abs < -(bot.MATE_SCORE - 1000):
                        outcome = 0.0  # Black Wins
                    else:
                        outcome = 0.5  # Draw
                    terminal = TERM_TB
                    break

            if plies % sample_every == 0:
                sampled_fens.append(board_to_fen(board, turn))

            bot.board           = board
            bot.color           = turn
            bot.opponent_color  = 'black' if turn == 'white' else 'white'
            bot.position_counts = dict(position_count)
            bot.ply_count       = plies
            root_hash = board_hash(board, turn)

            _, best_move = _search_with_id(gen_depth, legal, root_hash)
            move = best_move if best_move else random.choice(legal)

            board.make_move(move[0], move[1])
            turn  = 'black' if turn == 'white' else 'white'
            plies += 1

        stats = {'game_length': plies, 'terminal': terminal}
        return [(fen, outcome) for fen in sampled_fens], stats

    stats = {'game_length': 0, 'terminal': 'discarded'}
    return [], stats

def _run_generation(n_games: int, game_id_offset: int, max_plies: int,
                    sample_every: int, gen_depth: int, rand_plies: int,
                    n_workers: int, checkpoint_every: int,
                    partial_file: str) -> tuple:
    print(f"\nGenerating {n_games} self-play games — "
          f"Full AI Depth {gen_depth} ({n_workers} workers)")
    print(f"  Random opening : {rand_plies} plies  |  "
          f"Sample every   : {sample_every} plies  |  "
          f"TB adjudication: ≤{TB_ADJUDICATION_PIECES} pieces\n")

    args = [(game_id_offset + i, max_plies, sample_every, gen_depth, rand_plies)
            for i in range(n_games)]

    all_positions   = []
    terminal_counts = defaultdict(int)
    game_lengths    = []
    start_time      = time.time()

    with mp.Pool(n_workers) as pool:
        for i, (positions, stats) in enumerate(
                pool.imap_unordered(_play_one_game, args, chunksize=1), 1):

            all_positions.extend(positions)
            terminal_counts[stats['terminal']] += 1
            if stats['game_length'] > 0:
                game_lengths.append(stats['game_length'])

            elapsed = time.time() - start_time
            speed   = i / elapsed
            eta     = (n_games - i) / speed if speed > 0 else 0
            print(f"  {i}/{n_games} games  |  "
                  f"{len(all_positions)} positions  |  "
                  f"ETA: {eta/60:.1f}m", flush=True)

            if i % checkpoint_every == 0:
                with open(partial_file, 'w') as f:
                    json.dump([{'fen': fen, 'outcome': out}
                               for fen, out in all_positions], f)

    if os.path.exists(partial_file):
        os.remove(partial_file)

    return all_positions, terminal_counts, game_lengths


def _print_game_stats(terminal_counts: dict, game_lengths: list) -> None:
    total_games = sum(terminal_counts.values())
    if not game_lengths:
        print("  No games recorded.")
        return
    avg_length = sum(game_lengths) / len(game_lengths)
    print(f"\n{'─' * 55}")
    print(f"  Game Statistics  ({total_games} games)")
    print(f"{'─' * 55}")
    print(f"  Average game length : {avg_length:.1f} plies  ({avg_length/2:.1f} moves)")
    print(f"  Min / Max           : {min(game_lengths)} / {max(game_lengths)} plies")
    print()
    order  = [TERM_CHECKMATE, TERM_TB, TERM_THREEFOLD,
              TERM_INSUFFICIENT, TERM_PLY_LIMIT, 'discarded']
    labels = {
        TERM_CHECKMATE:    'Checkmate',
        TERM_TB:           'TB adjudication',
        TERM_THREEFOLD:    'Threefold repetition',
        TERM_INSUFFICIENT: 'Insufficient material',
        TERM_PLY_LIMIT:    'Ply limit (200)',
        'discarded':       'Discarded (lopsided)',
    }
    for key in order:
        count = terminal_counts.get(key, 0)
        if count == 0:
            continue
        pct = count / total_games * 100
        bar = '█' * int(pct / 2)
        print(f"  {labels[key]:<25} {count:>4}  ({pct:5.1f}%)  {bar}")
    print(f"{'─' * 55}")

# ═══════════════════════════════════════════════════════════════════════════════
# POSITION UNIQUENESS DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════════

def report_uniqueness(positions_data: list) -> None:
    fens_only = [fen for fen, _ in positions_data]
    unique    = len(set(fens_only))
    total     = len(fens_only)
    pct       = unique / total if total > 0 else 0
    print(f"  Position uniqueness : {unique}/{total} ({pct:.1%})", flush=True)
    if pct < 0.80:
        print(f"  ⚠  Low uniqueness — consider increasing "
              f"RANDOM_OPENING_PLIES (currently {RANDOM_OPENING_PLIES})", flush=True)
    else:
        print(f"  ✓  Diversity looks healthy", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TEXEL LOSS (OPTIMIZED)
# ═══════════════════════════════════════════════════════════════════════════════

_last_patched_vec = [None]

def compute_mse(parsed_data: list, K: float, mg_vals: dict, eg_vals: dict) -> float:
    vec = vals_to_vec(mg_vals, eg_vals)
    if vec != _last_patched_vec[0]:
        patch_ai_values(mg_vals, eg_vals)
        _last_patched_vec[0] = vec[:]
    bot = _get_bot()
    eval_fn = bot.evaluate_board
    
    total = 0.0
    count = len(parsed_data)
    if count == 0: return 0.0
    
    factor = -K / 400.0
    
    for board, outcome in parsed_data:
        raw = eval_fn(board, 'white')
        x = raw * factor
        if x > 30.0:
            prob = 0.0
        elif x < -30.0:
            prob = 1.0
        else:
            prob = 1.0 / (1.0 + math.exp(x))
            
        diff = prob - outcome
        total += diff * diff
        
    return total / count

def calibrate_K(parsed_data: list, mg: dict, eg: dict, steps: int = 40) -> float:
    print("\nCalibrating Texel K constant...")
    phi    = (1 + 5 ** 0.5) / 2
    lo, hi = 0.3, 4.0
    for _ in range(steps):
        k1 = hi - (hi - lo) / phi
        k2 = lo + (hi - lo) / phi
        if compute_mse(parsed_data, k1, mg, eg) < compute_mse(parsed_data, k2, mg, eg):
            hi = k2
        else:
            lo = k1
        if (hi - lo) < 0.002:
            break
    best_K = (lo + hi) / 2
    print(f"  Optimal K = {best_K:.4f}", flush=True)
    return best_K

def coordinate_descent(parsed_data: list, K: float, mg_start: dict, eg_start: dict,
                       n_rounds: int, step_sizes: list) -> tuple:
    vec      = vals_to_vec(mg_start, eg_start)
    best_mse = compute_mse(parsed_data, K, *vec_to_vals(vec))
    print(f"\nStarting coordinate descent — initial MSE = {best_mse:.6f}", flush=True)

    for step in step_sizes:
        print(f"\n── Step size: {step} cp ──────────────────────────────────", flush=True)
        for round_num in range(1, n_rounds + 1):
            improved_any = False

            for i, name in enumerate(PARAM_NAMES):
                lo_bound, hi_bound = PARAM_BOUNDS[i]

                v_up   = vec.copy(); v_up[i] += step
                v_dn   = vec.copy(); v_dn[i] -= step
                mse_up = compute_mse(parsed_data, K, *vec_to_vals(v_up)) \
                         if v_up[i] <= hi_bound else float('inf')
                mse_dn = compute_mse(parsed_data, K, *vec_to_vals(v_dn)) \
                         if v_dn[i] >= lo_bound else float('inf')

                if mse_up < best_mse and mse_up <= mse_dn:
                    old = vec[i]; vec = v_up; best_mse = mse_up; improved_any = True
                    print(f"  {name:12s}: {old:.0f} → {vec[i]:.0f}"
                          f"  (MSE={best_mse:.6f})", flush=True)
                elif mse_dn < best_mse:
                    old = vec[i]; vec = v_dn; best_mse = mse_dn; improved_any = True
                    print(f"  {name:12s}: {old:.0f} → {vec[i]:.0f}"
                          f"  (MSE={best_mse:.6f})", flush=True)

            print(f"  [round {round_num}] MSE = {best_mse:.6f}"
                  + ("" if improved_any else "  (converged)"), flush=True)
            if not improved_any:
                break

    return vec_to_vals(vec), best_mse

# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT FORMATTING
# ═══════════════════════════════════════════════════════════════════════════════

def print_ai_dicts(mg: dict, eg: dict, header: str = "", orig_mg: dict = None, orig_eg: dict = None) -> None:
    if header:
        print(f"\n{'─' * 50}\n{header}\n{'─' * 50}", flush=True)

    def _fmt(d: dict) -> str:
        lines = [f"    {cls.__name__}: {int(round(d[cls]))}," for cls in PIECE_CLASSES]
        lines.append("    King: 20000")
        return '\n'.join(lines)

    print(f"MG_PIECE_VALUES = {{\n{_fmt(mg)}\n}}", flush=True)
    print(f"\nEG_PIECE_VALUES = {{\n{_fmt(eg)}\n}}", flush=True)

    print("\n  Delta vs baseline AI values:")
    print(f"  {'Piece':<10} {'MG orig':>8} {'MG new':>8} {'MG Δ':>7}"
          f"  {'EG orig':>8} {'EG new':>8} {'EG Δ':>7}")
    print("  " + "─" * 64)
    if orig_mg is None:
        orig_mg = {cls: AI.MG_PIECE_VALUES[cls] for cls in PIECE_CLASSES}
    if orig_eg is None:
        orig_eg = {cls: AI.EG_PIECE_VALUES[cls] for cls in PIECE_CLASSES}
    for cls in PIECE_CLASSES:
        dm = int(round(mg[cls])) - orig_mg[cls]
        de = int(round(eg[cls])) - orig_eg[cls]
        print(f"  {cls.__name__:<10} {orig_mg[cls]:>8} {int(round(mg[cls])):>8} {dm:>+7}"
              f"  {orig_eg[cls]:>8} {int(round(eg[cls])):>8} {de:>+7}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("═" * 60, flush=True)
    print(f"  Jungle Chess Piece Value Auto-Tuner  (v3.1)", flush=True)
    print(f"  Full AI Depth {GENERATION_DEPTH}  |  "
          f"{GAMES_TO_GENERATE} games  |  {NUM_WORKERS} workers", flush=True)
    print("═" * 60, flush=True)

    positions_data  = None
    existing_count  = 0
    game_id_offset  = 0
    partial_file    = DATA_FILE + '.partial'

    if os.path.exists(partial_file) and os.path.getsize(partial_file) > 0:
        print(f"\n⚠  Found partial checkpoint: {partial_file}")
        print("  Resume from checkpoint? (Y/n): ", end='', flush=True)
        if input().strip().lower() != 'n':
            try:
                with open(partial_file) as f:
                    raw = json.load(f)
                positions_data = [(d['fen'], d['outcome']) for d in raw]
                print(f"  Resumed {len(positions_data)} positions from checkpoint.")
            except Exception:
                positions_data = None

    if positions_data is None and os.path.exists(DATA_FILE) \
            and os.path.getsize(DATA_FILE) > 0:
        try:
            with open(DATA_FILE) as f:
                raw = json.load(f)
            existing_positions = [(d['fen'], d['outcome']) for d in raw]
            existing_count     = len(existing_positions)
            print(f"\nFound existing training data: {existing_count} positions.")
            print("  Options:")
            print("    [r] Regenerate from scratch")
            print("    [a] Append new games to existing data")
            print("    [s] Skip generation — tune on existing data only")
            print("  Choice (r/a/S): ", end='', flush=True)
            choice = input().strip().lower()

            if choice == 'r':
                positions_data = None
                game_id_offset = 0
            elif choice == 'a':
                positions_data = existing_positions
                game_id_offset = existing_count
                print(f"  Will append {GAMES_TO_GENERATE} new games to "
                      f"{existing_count} existing positions.")
            else:
                positions_data = existing_positions
                print("  Skipping generation — tuning on existing data.")
        except Exception:
            positions_data = None

    need_generation = (positions_data is None) or \
                      (existing_count > 0 and game_id_offset > 0 and
                       len(positions_data) == existing_count)

    if need_generation:
        new_positions, term_counts, game_lengths = _run_generation(
            GAMES_TO_GENERATE, game_id_offset,
            MAX_GAME_PLIES, SAMPLE_EVERY_N_PLIES,
            GENERATION_DEPTH, RANDOM_OPENING_PLIES,
            NUM_WORKERS, CHECKPOINT_EVERY, partial_file
        )
        _print_game_stats(term_counts, game_lengths)

        if positions_data is None:
            positions_data = new_positions
        else:
            positions_data = positions_data + new_positions

        random.shuffle(positions_data)
        print(f"\nTotal training positions: {len(positions_data)}")
        print(f"Saving to {DATA_FILE}...", flush=True)
        with open(DATA_FILE, 'w') as f:
            json.dump([{'fen': fen, 'outcome': out}
                       for fen, out in positions_data], f)

    if not positions_data:
        print("ERROR: No training positions available. Exiting.")
        return

    outcomes = [o for _, o in positions_data]
    w = sum(o == 1.0 for o in outcomes) / len(outcomes)
    d = sum(o == 0.5 for o in outcomes) / len(outcomes)
    b = sum(o == 0.0 for o in outcomes) / len(outcomes)
    print(f"\nDataset: {len(positions_data)} positions")
    print(f"Outcome distribution:  White {w:.1%}  |  "
          f"Draw {d:.1%}  |  Black {b:.1%}", flush=True)
    report_uniqueness(positions_data)

    print(f"\nPre-parsing {len(positions_data)} board states into RAM to accelerate tuning...", flush=True)
    parsed_data = []
    for fen_str, outcome in positions_data:
        try:
            board, _ = fen_to_board(fen_str)
            parsed_data.append((board, outcome))
        except Exception:
            continue
    print(f"Successfully loaded {len(parsed_data)} positions.", flush=True)

    mg_start = {cls: AI.MG_PIECE_VALUES[cls] for cls in PIECE_CLASSES}
    eg_start = {cls: AI.EG_PIECE_VALUES[cls] for cls in PIECE_CLASSES}
    orig_mg = {cls: AI.MG_PIECE_VALUES[cls] for cls in PIECE_CLASSES}
    orig_eg = {cls: AI.EG_PIECE_VALUES[cls] for cls in PIECE_CLASSES}
    print_ai_dicts(mg_start, eg_start, "Starting values (from AI.py):", orig_mg, orig_eg)

    K = calibrate_K(parsed_data, mg_start, eg_start, K_CALIBRATION_STEPS)

    print(f"\nRunning coordinate descent "
          f"({COORD_ROUNDS} rounds max × {len(STEP_SIZES)} step sizes)...", flush=True)
    (mg_best, eg_best), final_mse = coordinate_descent(
        parsed_data, K, mg_start, eg_start, COORD_ROUNDS, STEP_SIZES
    )

    print(f"\n{'═' * 60}", flush=True)
    print(f"  TUNING COMPLETE  —  final MSE = {final_mse:.6f}  (K = {K:.4f})", flush=True)
    print(f"{'═' * 60}", flush=True)
    print_ai_dicts(mg_best, eg_best, "Optimised values — paste into AI.py:", orig_mg, orig_eg)

    with open(RESULT_FILE, 'w') as f:
        json.dump({
            'K':                      K,
            'final_mse':              final_mse,
            'generation_depth':       GENERATION_DEPTH,
            'random_opening_plies':   RANDOM_OPENING_PLIES,
            'tb_adjudication_pieces': TB_ADJUDICATION_PIECES,
            'mg': {cls.__name__: int(round(mg_best[cls])) for cls in PIECE_CLASSES},
            'eg': {cls.__name__: int(round(eg_best[cls])) for cls in PIECE_CLASSES},
            'total_positions':        len(positions_data),
            'timestamp':              time.strftime('%Y-%m-%d %H:%M:%S'),
        }, f, indent=2)
    print(f"\nResults saved to {RESULT_FILE}", flush=True)

if __name__ == '__main__':
    mp.freeze_support()
    main()