# TablebaseGenerator.py (v9.0 - Absolute Perfection)
# Fixes over v8.2:
#   FIX-5: Enforced "No Dead Kings" rule inline. Exploding a king is an illegal move, 
#          not an immediate win. This perfectly synchronizes with the AI engine rules.

import os
import time
import __main__
import signal
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations_with_replacement
import numpy as np
from GameLogic import *
from array import array
from collections import defaultdict

# --- CONFIGURATION ---
TB_DIR = "tablebases"
os.makedirs(TB_DIR, exist_ok=True)

LONGEST_MATES_NOTE_FILE = os.path.join(TB_DIR, "new_longest_mates.tsv")
LONGEST_MATE_KEY_PREFIX = "regen_"

TB_THREADS_SUBTRACT = 2
EXPECTED_TABLE_ENTRIES = 64 * 64 * 64 * 2
PIECE_CLASS_BY_NAME = {
    "Queen": Queen, "Rook": Rook, "Knight": Knight, "Bishop": Bishop, "Pawn": Pawn,
}

_CONSOLE_INTERRUPT_SIGNALS = [signal.SIGINT]
if hasattr(signal, "SIGBREAK"):
    _CONSOLE_INTERRUPT_SIGNALS.append(signal.SIGBREAK)

# ==============================================================================
# SHARED HELPERS
# ==============================================================================

def _main_interrupt_handler(signum, frame):
    print("\n[Interrupt ignored] Tablebase generation is protected from Ctrl+C. "
          "Close the terminal or kill the process explicitly if you really want to stop it.", flush=True)

def _install_main_interrupt_ignores():
    for sig in _CONSOLE_INTERRUPT_SIGNALS:
        try:
            signal.signal(sig, _main_interrupt_handler)
        except Exception:
            pass

def _install_worker_interrupt_ignores():
    for sig in _CONSOLE_INTERRUPT_SIGNALS:
        try:
            signal.signal(sig, signal.SIG_IGN)
        except Exception:
            pass

def _load_3man_table_file(filename):
    data16 = np.fromfile(filename, dtype=np.int16)
    if data16.size == EXPECTED_TABLE_ENTRIES:
        return data16.reshape((64, 64, 64, 2))
    raise ValueError(f"Invalid 3-man tablebase file (expected int16, size {EXPECTED_TABLE_ENTRIES}): {filename}")

def _flat_idx_raw(i0, i1, i2, i3):
    return (((i0 * 64 + i1) * 64 + i2) * 2 + i3)

def _flat_idx_raw_4(i0, i1, i2, i3, i4):
    return (((((i0 * 64 + i1) * 64 + i2) * 64 + i3) * 2) + i4)

def _flip(pos):
    return (7 - pos[0], pos[1])

def _safe_record_longest_mate(table_key, max_dtm, decisive, remaining, elapsed_seconds):
    try:
        records = {}
        if os.path.exists(LONGEST_MATES_NOTE_FILE):
            with open(LONGEST_MATES_NOTE_FILE, "r", encoding="utf-8") as f:
                for raw in f:
                    line = raw.strip()
                    if not line or line.startswith("#"): continue
                    parts = line.split("\t")
                    if len(parts) >= 6:
                        records[parts[0]] = parts[1:6]
        records[f"{LONGEST_MATE_KEY_PREFIX}{table_key}"] = [
            str(int(max_dtm)), str(int(decisive)), str(int(remaining)),
            f"{elapsed_seconds / 60.0:.1f}", time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        ]
        with open(LONGEST_MATES_NOTE_FILE, "w", encoding="utf-8") as f:
            f.write("# table_key\tmax_dtm\tdecisive\tremaining\telapsed_min\tupdated_utc\n")
            for key in sorted(records.keys()):
                r = records[key]
                f.write(f"{key}\t{r[0]}\t{r[1]}\t{r[2]}\t{r[3]}\t{r[4]}\n")
    except Exception as e:
        print(f"[LongestMate] Warning: failed to update note file ({e})")


# ==============================================================================
# 3-MAN GENERATOR
# ==============================================================================

_W_PIECE_NAME = None
_W_PIECE_CLASS = None
_W_QUEEN_TABLE = None
_W3_BOARD = None
_W3_WK_OBJ = None
_W3_WP_OBJ = None
_W3_BK_OBJ = None

def _init_transition_worker(piece_name, queen_tb_file):
    _install_worker_interrupt_ignores()
    global _W_PIECE_NAME, _W_PIECE_CLASS, _W_QUEEN_TABLE
    global _W3_BOARD, _W3_WK_OBJ, _W3_WP_OBJ, _W3_BK_OBJ
    _W_PIECE_NAME = piece_name
    _W_PIECE_CLASS = PIECE_CLASS_BY_NAME[piece_name]
    _W_QUEEN_TABLE = None
    if piece_name == "Pawn" and queen_tb_file:
        _W_QUEEN_TABLE = _load_3man_table_file(queen_tb_file)
    _W3_BOARD = Board(setup=False)
    _W3_WK_OBJ = King('white')
    _W3_WP_OBJ = _W_PIECE_CLASS('white')
    _W3_BK_OBJ = King('black')
    pc = _W3_BOARD.piece_counts
    pc['white'][King] = 1
    pc['black'][King] = 1
    pc['white'][type(_W3_WP_OBJ)] += 1

def _build_transition_worker(idx):
    i0, i1, i2, i3 = idx
    wk, wp, bk = (i0 // 8, i0 % 8), (i1 // 8, i1 % 8), (i2 // 8, i2 % 8)
    turn_is_white = (i3 == 0)
    opp_turn_idx = 1 - i3

    board = _W3_BOARD
    g = board.grid
    for p in (_W3_WK_OBJ, _W3_WP_OBJ, _W3_BK_OBJ):
        if p.pos:
            g[p.pos[0]][p.pos[1]] = None
    _W3_WK_OBJ.pos, _W3_WP_OBJ.pos, _W3_BK_OBJ.pos = wk, wp, bk
    _W3_WK_OBJ._list_pos, _W3_WP_OBJ._list_pos, _W3_BK_OBJ._list_pos = 0, 1, 0
    g[wk[0]][wk[1]], g[wp[0]][wp[1]], g[bk[0]][bk[1]] = _W3_WK_OBJ, _W3_WP_OBJ, _W3_BK_OBJ
    board.white_king_pos, board.black_king_pos = wk, bk
    board.white_pieces[:] = [_W3_WK_OBJ, _W3_WP_OBJ]
    board.black_pieces[:] = [_W3_BK_OBJ]

    passive_color = 'black' if turn_is_white else 'white'
    if is_in_check(board, passive_color):
        return None

    moves = get_all_pseudo_legal_moves(board, 'white' if turn_is_white else 'black')

    if turn_is_white:
        immediate_win = False
        promo_win_vals = []
        child_flats = []

        for m in moves:
            record = board.make_move_track(m[0], m[1])
            
            # --- FIX: Strict Legality Guard (No Kings can die) ---
            if not board.white_king_pos or not board.black_king_pos or is_in_check(board, 'white'):
                board.unmake_move(record)
                continue

            # Catch True Checkmate
            if not has_legal_moves(board, 'black'):
                immediate_win = True
                board.unmake_move(record)
                break

            if _W_PIECE_NAME == "Pawn":
                wkp, bkp = board.white_king_pos, board.black_king_pos
                nk = next((x for x in board.white_pieces if not isinstance(x, King)), None)
                if nk is not None and isinstance(nk, Queen):
                    q_idx = (wkp[0] * 8 + wkp[1], nk.pos[0] * 8 + nk.pos[1], bkp[0] * 8 + bkp[1], 1)
                    q_val = int(_W_QUEEN_TABLE[q_idx])
                    if q_val < 0:
                        promo_win_vals.append(abs(q_val) + 1)
                    board.unmake_move(record)
                    continue

            p = next((x for x in board.white_pieces if not isinstance(x, King)), None)
            if p is not None:
                c0 = board.white_king_pos[0] * 8 + board.white_king_pos[1]
                c1 = p.pos[0] * 8 + p.pos[1]
                c2 = board.black_king_pos[0] * 8 + board.black_king_pos[1]
                child_flats.append(_flat_idx_raw(c0, c1, c2, opp_turn_idx))
            board.unmake_move(record)

        return (_flat_idx_raw(*idx), ('w', immediate_win, array('H', promo_win_vals), array('I', child_flats)))

    # --- Black to move ---
    legal_moves_count = 0
    has_non_losing_escape = False
    child_flats = []

    for m in moves:
        record = board.make_move_track(m[0], m[1])
        
        # --- FIX: Strict Legality Guard (No Kings can die) ---
        if not board.black_king_pos or not board.white_king_pos or is_in_check(board, 'black'):
            board.unmake_move(record)
            continue

        legal_moves_count += 1

        if len(board.white_pieces) < 2 or not has_legal_moves(board, 'white'):
            has_non_losing_escape = True
            board.unmake_move(record)
            continue

        if _W_PIECE_NAME == "Pawn":
            nk = next((x for x in board.white_pieces if not isinstance(x, King)), None)
            if nk is None or isinstance(nk, Queen):
                has_non_losing_escape = True
                board.unmake_move(record)
                continue

        p = next((x for x in board.white_pieces if not isinstance(x, King)), None)
        if p is None:
            has_non_losing_escape = True
            board.unmake_move(record)
            continue

        c0 = board.white_king_pos[0] * 8 + board.white_king_pos[1]
        c1 = p.pos[0] * 8 + p.pos[1]
        c2 = board.black_king_pos[0] * 8 + board.black_king_pos[1]
        child_flats.append(_flat_idx_raw(c0, c1, c2, opp_turn_idx))
        board.unmake_move(record)

    return (_flat_idx_raw(*idx), ('b', legal_moves_count, has_non_losing_escape, array('I', child_flats)))


class Generator:
    def __init__(self, piece_class):
        self.piece_class = piece_class
        self.piece_name = piece_class.__name__
        self.filename = os.path.join(TB_DIR, f"K_{self.piece_name}_K.bin")
        self.queen_tb_file = os.path.join(TB_DIR, "K_Queen_K.bin")
        self.total_positions = 64 * 64 * 64 * 2
        self.table = np.zeros((64, 64, 64, 2), dtype=np.int16)

        if self.piece_name == "Pawn" and not os.path.exists(self.queen_tb_file):
            print("CRITICAL ERROR: Queen Tablebase not found! Generate Queen first.")
            exit()

        cpu_default = max(1, (os.cpu_count() or 2) - TB_THREADS_SUBTRACT)
        self.transition_workers = min(8, cpu_default)

    def generate(self):
        if os.path.exists(self.filename):
            print(f"[SKIP] {self.filename} already exists.")
            return

        start_time = time.time()
        print(f"\n{'='*60}\n GENERATING: King + {self.piece_name} vs King\n{'='*60}")

        print("[Stage 1] Enumerating candidate positions...")
        raw_candidates = []
        for idx in np.ndindex(self.table.shape):
            wk = (idx[0] // 8, idx[0] % 8)
            wp = (idx[1] // 8, idx[1] % 8)
            bk = (idx[2] // 8, idx[2] % 8)
            if wk == wp or wk == bk or wp == bk:
                continue
            if self.piece_name == "Pawn" and (wp[0] == 0 or wp[0] == 7):
                continue
            raw_candidates.append(idx)
        print(f"[Stage 1] Found {len(raw_candidates):,} structural candidates.")

        print("[Stage 2 & 3] Merged legality filter + transition cache (parallel)...")
        stage23_start = time.time()
        unsolved_flats = array('I')
        transitions = []

        can_parallel = bool(getattr(__main__, "__file__", ""))
        if can_parallel:
            try:
                with ProcessPoolExecutor(
                    max_workers=self.transition_workers,
                    initializer=_init_transition_worker,
                    initargs=(self.piece_name, self.queen_tb_file)
                ) as ex:
                    for done, result in enumerate(
                        ex.map(_build_transition_worker, raw_candidates, chunksize=1024), 1
                    ):
                        if result is not None:
                            unsolved_flats.append(result[0])
                            transitions.append(result[1])
                        if done % 50000 == 0:
                            print(f"  > {done / len(raw_candidates) * 100:.1f}%", end='\r')
                print()
            except Exception as e:
                print(f"\n[{self.piece_name}] Parallel build failed ({e}). Falling back to single-process.")
                unsolved_flats = array('I')
                transitions = []
                can_parallel = False

        if not can_parallel:
            _init_transition_worker(self.piece_name, self.queen_tb_file)
            for done, idx in enumerate(raw_candidates, 1):
                result = _build_transition_worker(idx)
                if result is not None:
                    unsolved_flats.append(result[0])
                    transitions.append(result[1])
                if done % 50000 == 0:
                    print(f"  > {done / len(raw_candidates) * 100:.1f}%", end='\r')
            print()

        candidate_states = len(unsolved_flats)
        print(f"[Stage 2 & 3] Valid states: {candidate_states:,} | Time: {time.time()-stage23_start:.1f}s")

        print("[Stage 4] Pre-solving & building retrograde parent maps...")
        table_flat = self.table.reshape(-1)
        btm_to_wtm = defaultdict(list)
        wtm_to_btm = defaultdict(list)
        pre_solved_set = set()
        work_set = set()
        pre_solved = 0

        for flat, trans in zip(unsolved_flats, transitions):
            t_idx = flat & 1
            if t_idx == 1 and trans[1] == 0:
                table_flat[flat] = -1
                pre_solved += 1
                pre_solved_set.add(flat)
            else:
                if t_idx == 0:
                    for cflat in trans[3]:
                        btm_to_wtm[cflat].append(flat)
                    if trans[1] or len(trans[2]) > 0:
                        work_set.add(flat)
                else:
                    if not trans[2]:
                        for cflat in trans[3]:
                            wtm_to_btm[cflat].append(flat)

        for flat in pre_solved_set:
            for parent in btm_to_wtm.get(flat, []):
                if table_flat[parent] == 0:
                    work_set.add(parent)

        trans_lookup = dict(zip(unsolved_flats, transitions))
        print(f"[Stage 4] Pre-solved: {pre_solved:,} | Initial work_set: {len(work_set):,}")

        print("[Stage 5] O(N) Dirty-Set Retrograde Analysis...")
        iteration = 1
        max_dtm_solved = 0
        solved_count = pre_solved

        while work_set:
            iter_start = time.time()
            changed = 0
            next_work_set = set()
            snapshot = table_flat.copy()

            for flat in work_set:
                if table_flat[flat] != 0:
                    continue
                trans = trans_lookup[flat]

                if (flat & 1) == 0:  # White to move
                    _, imm_win, promo_vals, child_vals = trans
                    best_win = 1 if imm_win else 0
                    for val in promo_vals:
                        if best_win == 0 or val < best_win:
                            best_win = val
                    for cflat in child_vals:
                        res = int(snapshot[cflat])
                        if res < 0:
                            val = abs(res) + 1
                            if best_win == 0 or val < best_win:
                                best_win = val
                    if best_win > 0:
                        table_flat[flat] = best_win
                        changed += 1
                        for parent in wtm_to_btm.get(flat, []):
                            if table_flat[parent] == 0:
                                next_work_set.add(parent)

                else:  # Black to move
                    _, moves_count, escape, child_vals = trans
                    if moves_count == 0:
                        table_flat[flat] = -1
                        changed += 1
                        for parent in btm_to_wtm.get(flat, []):
                            if table_flat[parent] == 0:
                                next_work_set.add(parent)
                        continue
                    if escape:
                        continue
                    max_win = 0
                    all_res = True
                    for cflat in child_vals:
                        res = int(snapshot[cflat])
                        if res <= 0:
                            all_res = False
                            break
                        if res > max_win:
                            max_win = res
                    if all_res:
                        table_flat[flat] = -(max_win + 1)
                        changed += 1
                        for parent in btm_to_wtm.get(flat, []):
                            if table_flat[parent] == 0:
                                next_work_set.add(parent)

            work_set = next_work_set
            solved_count += changed
            if changed > 0:
                max_dtm_solved = iteration
            print(f"[Iteration {iteration}] {changed:,} new | Total: {solved_count:,} | "
                  f"Next: {len(work_set):,} | Time: {time.time()-iter_start:.1f}s")
            iteration += 1

        with open(self.filename, 'wb') as f:
            self.table.tofile(f)

        wins = losses = unresolved = 0
        for cflat in unsolved_flats:
            val = int(table_flat[cflat])
            if val > 0:    wins += 1
            elif val < 0:  losses += 1
            else:           unresolved += 1
        solved = wins + losses
        solve_rate = (solved / candidate_states * 100.0) if candidate_states else 0.0
        elapsed = time.time() - start_time
        print(f"[{self.piece_name}] Candidates={candidate_states:,} | "
              f"Decisive={solved:,} ({solve_rate:.1f}%) | "
              f"Wins={wins:,} | Losses={losses:,} | Draws={unresolved:,}")
        print(f"[{self.piece_name}] Longest decisive DTM: {max_dtm_solved}")
        _safe_record_longest_mate(f"K_{self.piece_name}_K", max_dtm_solved, solved, unresolved, elapsed)
        print(f"\nSUCCESS: Generated in {elapsed / 60:.1f} minutes.")


# ==============================================================================
# 4-MAN GENERATOR  (K + 2 White Pieces vs K)
# ==============================================================================

_PIECE_CANONICAL_ORDER = {Bishop: 0, Knight: 1, Pawn: 2, Queen: 3, Rook: 4}

_W4_P1_CLASS = None
_W4_P2_CLASS = None
_W4_P1_NAME = None
_W4_P2_NAME = None
_W4_3MAN_TABLES = {}
_W4_PROMO_TABLE = None
_W4_SAME_PIECE = False
_W4_PROMO_SAME_PIECE = False
_W4_BOARD = None
_W4_WK_OBJ = None
_W4_P1_OBJ = None
_W4_P2_OBJ = None
_W4_BK_OBJ = None

def _init_transition_worker_4(p1_name, p2_name, promo_tb_file):
    _install_worker_interrupt_ignores()
    global _W4_P1_CLASS, _W4_P2_CLASS, _W4_P1_NAME, _W4_P2_NAME
    global _W4_3MAN_TABLES, _W4_PROMO_TABLE, _W4_SAME_PIECE, _W4_PROMO_SAME_PIECE
    global _W4_BOARD, _W4_WK_OBJ, _W4_P1_OBJ, _W4_P2_OBJ, _W4_BK_OBJ

    _W4_P1_NAME, _W4_P2_NAME = p1_name, p2_name
    _W4_P1_CLASS, _W4_P2_CLASS = PIECE_CLASS_BY_NAME[p1_name], PIECE_CLASS_BY_NAME[p2_name]
    _W4_SAME_PIECE = (p1_name == p2_name)
    _W4_3MAN_TABLES, _W4_PROMO_TABLE = {}, None

    for name in PIECE_CLASS_BY_NAME:
        path = os.path.join(TB_DIR, f"K_{name}_K.bin")
        if os.path.exists(path):
            _W4_3MAN_TABLES[name] = _load_3man_table_file(path)

    if promo_tb_file and os.path.exists(promo_tb_file):
        data16 = np.fromfile(promo_tb_file, dtype=np.int16)
        _W4_PROMO_TABLE = data16.reshape((64, 64, 64, 64, 2))
        parts = os.path.basename(promo_tb_file)[2:-6].split('_')
        _W4_PROMO_SAME_PIECE = (len(parts) == 2 and parts[0] == parts[1])

    _W4_BOARD = Board(setup=False)
    _W4_WK_OBJ = King('white')
    _W4_P1_OBJ = _W4_P1_CLASS('white')
    _W4_P2_OBJ = _W4_P2_CLASS('white')
    _W4_BK_OBJ = King('black')
    pc = _W4_BOARD.piece_counts
    pc['white'][King] = 1
    pc['black'][King] = 1
    pc['white'][type(_W4_P1_OBJ)] += 1
    pc['white'][type(_W4_P2_OBJ)] += 1

def _build_transition_worker_4(flat):
    i4, rest = flat % 2, flat // 2
    i3, rest = rest % 64, rest // 64
    i2, rest = rest % 64, rest // 64
    i1, i0   = rest % 64, rest // 64
    wk, p1, p2, bk = (i0//8, i0%8), (i1//8, i1%8), (i2//8, i2%8), (i3//8, i3%8)
    turn_is_white, opp_turn_idx = (i4 == 0), 1 - i4

    board = _W4_BOARD
    g = board.grid
    for p in (_W4_WK_OBJ, _W4_P1_OBJ, _W4_P2_OBJ, _W4_BK_OBJ):
        if p.pos:
            g[p.pos[0]][p.pos[1]] = None
    _W4_WK_OBJ.pos, _W4_P1_OBJ.pos, _W4_P2_OBJ.pos, _W4_BK_OBJ.pos = wk, p1, p2, bk
    _W4_WK_OBJ._list_pos, _W4_P1_OBJ._list_pos, _W4_P2_OBJ._list_pos, _W4_BK_OBJ._list_pos = 0, 1, 2, 0
    g[wk[0]][wk[1]], g[p1[0]][p1[1]], g[p2[0]][p2[1]], g[bk[0]][bk[1]] = _W4_WK_OBJ, _W4_P1_OBJ, _W4_P2_OBJ, _W4_BK_OBJ
    board.white_king_pos, board.black_king_pos = wk, bk
    board.white_pieces[:] = [_W4_WK_OBJ, _W4_P1_OBJ, _W4_P2_OBJ]
    board.black_pieces[:] = [_W4_BK_OBJ]

    passive_color = 'black' if turn_is_white else 'white'
    if is_in_check(board, passive_color):
        return None

    moves = get_all_pseudo_legal_moves(board, 'white' if turn_is_white else 'black')

    if turn_is_white:
        immediate_win = False
        known_win_vals = []
        child_flats = []

        for start, end in moves:
            record = board.make_move_track(start, end)
            
            # --- FIX: Strict Legality Guard ---
            if not board.white_king_pos or not board.black_king_pos or is_in_check(board, 'white'):
                board.unmake_move(record)
                continue

            wkp, bkp = board.white_king_pos, board.black_king_pos

            # True Checkmate (No legal moves for Black)
            if not has_legal_moves(board, 'black'):
                immediate_win = True
                board.unmake_move(record)
                break

            w_pieces = [p for p in board.white_pieces if not isinstance(p, King)]

            if len(w_pieces) < 2:
                if len(w_pieces) == 1:
                    rem_name = type(w_pieces[0]).__name__
                    if rem_name in _W4_3MAN_TABLES:
                        q_idx = (wkp[0]*8+wkp[1], w_pieces[0].pos[0]*8+w_pieces[0].pos[1],
                                 bkp[0]*8+bkp[1], 1)
                        val = int(_W4_3MAN_TABLES[rem_name][q_idx])
                        if val < 0:
                            known_win_vals.append(abs(val) + 1)
                board.unmake_move(record)
                continue

            p0_ord = _PIECE_CANONICAL_ORDER.get(type(w_pieces[0]), 99)
            p1_ord = _PIECE_CANONICAL_ORDER.get(type(w_pieces[1]), 99)
            if p0_ord > p1_ord:
                w_pieces[0], w_pieces[1] = w_pieces[1], w_pieces[0]

            curr_n1, curr_n2 = type(w_pieces[0]).__name__, type(w_pieces[1]).__name__

            if curr_n1 != _W4_P1_NAME or curr_n2 != _W4_P2_NAME:
                if _W4_PROMO_TABLE is not None:
                    idx1 = w_pieces[0].pos[0]*8 + w_pieces[0].pos[1]
                    idx2 = w_pieces[1].pos[0]*8 + w_pieces[1].pos[1]
                    if _W4_PROMO_SAME_PIECE and idx1 > idx2:
                        idx1, idx2 = idx2, idx1
                    q_idx = (wkp[0]*8+wkp[1], idx1, idx2, bkp[0]*8+bkp[1], 1)
                    val = int(_W4_PROMO_TABLE.flat[_flat_idx_raw_4(*q_idx)])
                    if val < 0:
                        known_win_vals.append(abs(val) + 1)
                board.unmake_move(record)
                continue

            c0 = wkp[0]*8 + wkp[1]
            c1 = w_pieces[0].pos[0]*8 + w_pieces[0].pos[1]
            c2 = w_pieces[1].pos[0]*8 + w_pieces[1].pos[1]
            c3 = bkp[0]*8 + bkp[1]
            if _W4_SAME_PIECE and c1 > c2:
                c1, c2 = c2, c1
            child_flats.append(_flat_idx_raw_4(c0, c1, c2, c3, opp_turn_idx))
            board.unmake_move(record)

        return (flat, ('w', immediate_win, array('H', known_win_vals), array('I', child_flats)))

    # --- Black to move ---
    legal_moves_count = 0
    has_non_losing_escape = False
    known_3man_wins = [] 
    child_flats = []

    for start, end in moves:
        record = board.make_move_track(start, end)
        
        # --- FIX: Strict Legality Guard ---
        if not board.black_king_pos or not board.white_king_pos or is_in_check(board, 'black'):
            board.unmake_move(record)
            continue

        legal_moves_count += 1

        if len(board.white_pieces) < 2 or not has_legal_moves(board, 'white'):
            has_non_losing_escape = True
            board.unmake_move(record)
            continue

        w_pieces = [p for p in board.white_pieces if not isinstance(p, King)]

        if len(w_pieces) < 2:
            if len(w_pieces) == 1:
                rem_name = type(w_pieces[0]).__name__
                if rem_name in _W4_3MAN_TABLES:
                    wkp = board.white_king_pos
                    bkp = board.black_king_pos
                    q_idx = (wkp[0]*8+wkp[1], w_pieces[0].pos[0]*8+w_pieces[0].pos[1],
                             bkp[0]*8+bkp[1], 0)
                    val = int(_W4_3MAN_TABLES[rem_name][q_idx])
                    if val <= 0:
                        has_non_losing_escape = True
                    else:
                        known_3man_wins.append(val)
                else:
                    has_non_losing_escape = True
            else:
                has_non_losing_escape = True
            board.unmake_move(record)
            continue

        p0_ord = _PIECE_CANONICAL_ORDER.get(type(w_pieces[0]), 99)
        p1_ord = _PIECE_CANONICAL_ORDER.get(type(w_pieces[1]), 99)
        if p0_ord > p1_ord:
            w_pieces[0], w_pieces[1] = w_pieces[1], w_pieces[0]

        wkp = board.white_king_pos
        bkp = board.black_king_pos
        c0 = wkp[0]*8 + wkp[1]
        c1 = w_pieces[0].pos[0]*8 + w_pieces[0].pos[1]
        c2 = w_pieces[1].pos[0]*8 + w_pieces[1].pos[1]
        c3 = bkp[0]*8 + bkp[1]
        if _W4_SAME_PIECE and c1 > c2:
            c1, c2 = c2, c1
        child_flats.append(_flat_idx_raw_4(c0, c1, c2, c3, opp_turn_idx))
        board.unmake_move(record)

    return (flat, ('b', legal_moves_count, has_non_losing_escape,
                   array('H', known_3man_wins), array('I', child_flats)))


def _gen_valid_4man_indices_numpy(total_positions, p1_name, p2_name, same_piece, chunk_size=8_000_000):
    for start in range(0, total_positions, chunk_size):
        end_c = min(start + chunk_size, total_positions)
        flat = np.arange(start, end_c, dtype=np.int64)
        t_arr  = flat % 2;      rest = flat // 2
        bk_arr = rest % 64;     rest //= 64
        p2_arr = rest % 64;     rest //= 64
        p1_arr = rest % 64
        wk_arr = rest // 64

        mask = ((wk_arr != p1_arr) & (wk_arr != p2_arr) & (wk_arr != bk_arr) &
                (p1_arr != p2_arr) & (p1_arr != bk_arr) & (p2_arr != bk_arr))

        if p1_name == "Pawn": mask &= (p1_arr >= 8) & (p1_arr < 56)
        if p2_name == "Pawn": mask &= (p2_arr >= 8) & (p2_arr < 56)
        if same_piece:        mask &= (p1_arr <= p2_arr)

        if p1_name == "Bishop" and p2_name == "Bishop":
            p1_r, p1_c = p1_arr // 8, p1_arr % 8
            p2_r, p2_c = p2_arr // 8, p2_arr % 8
            mask &= ((p1_r + p1_c) % 2) != ((p2_r + p2_c) % 2)

        if not mask.any():
            continue
        yield flat[mask].tolist()


class Generator4:
    def __init__(self, p1_class, p2_class):
        names = sorted([p1_class.__name__, p2_class.__name__])
        self.p1_name, self.p2_name = names[0], names[1]
        self.p1_class = PIECE_CLASS_BY_NAME[self.p1_name]
        self.p2_class = PIECE_CLASS_BY_NAME[self.p2_name]
        self.same_piece = (self.p1_name == self.p2_name)

        self.filename = os.path.join(TB_DIR, f"K_{self.p1_name}_{self.p2_name}_K.bin")
        self.total_positions = 64 * 64 * 64 * 64 * 2
        self.table = np.zeros((64, 64, 64, 64, 2), dtype=np.int16)

        self.promo_tb_file = None
        if self.p1_name == "Pawn" or self.p2_name == "Pawn":
            other = self.p2_name if self.p1_name == "Pawn" else self.p1_name
            if self.p1_name == "Pawn" and self.p2_name == "Pawn":
                p_names = sorted(["Pawn", "Queen"])
            else:
                p_names = sorted(["Queen", other])
            self.promo_tb_file = os.path.join(TB_DIR, f"K_{p_names[0]}_{p_names[1]}_K.bin")

        self.transition_workers = max(1, (os.cpu_count() or 2) - TB_THREADS_SUBTRACT)

    def generate(self):
        if os.path.exists(self.filename):
            print(f"[SKIP] {self.filename} already exists.")
            return

        start_time = time.time()
        print(f"\n{'='*60}\n GENERATING: King + {self.p1_name} + {self.p2_name} vs King")
        if self.same_piece:
            print(f" [Symmetry] Same-piece table — using canonical half (p1 <= p2).")
        print(f"{'='*60}")

        print("[Stage 1] Enumerating structural candidates...")
        stage1_start = time.time()
        all_flats = array('I')
        for chunk in _gen_valid_4man_indices_numpy(
                self.total_positions, self.p1_name, self.p2_name, self.same_piece):
            all_flats.extend(chunk)
        print(f"[Stage 1] Found {len(all_flats):,} candidates in {time.time()-stage1_start:.1f}s.")

        print(f"[Stage 2 & 3] Merged legality filter + transition cache ({len(all_flats):,} states)...")
        stage23_start = time.time()
        unsolved_flats = array('I')
        transitions = []
        with ProcessPoolExecutor(
            max_workers=self.transition_workers,
            initializer=_init_transition_worker_4,
            initargs=(self.p1_name, self.p2_name, self.promo_tb_file)
        ) as ex:
            for done, result in enumerate(
                ex.map(_build_transition_worker_4, all_flats, chunksize=4096), 1
            ):
                if result is not None:
                    unsolved_flats.append(result[0])
                    transitions.append(result[1])
                if done % 100_000 == 0:
                    elapsed = max(0.001, time.time() - stage23_start)
                    speed = done / elapsed
                    eta = (len(all_flats) - done) / speed
                    print(f"  > {done / len(all_flats) * 100:.1f}% | "
                          f"{speed:,.0f} st/s | ETA: {eta/60:.1f}m", end='\r', flush=True)
        candidate_states = len(unsolved_flats)
        print(f"\n[Stage 2 & 3] Valid states: {candidate_states:,} | "
              f"Time: {(time.time()-stage23_start)/60:.1f}m")

        print("[Stage 4] Pre-solving & building retrograde parent maps...")
        table_flat = self.table.reshape(-1)
        btm_to_wtm = defaultdict(list)
        wtm_to_btm = defaultdict(list)
        pre_solved_set = set()
        work_set = set()
        pre_solved = 0

        for flat, trans in zip(unsolved_flats, transitions):
            t_idx = flat & 1
            if t_idx == 1 and trans[1] == 0:
                table_flat[flat] = -1
                pre_solved += 1
                pre_solved_set.add(flat)
            else:
                if t_idx == 0:
                    for cflat in trans[3]:
                        btm_to_wtm[cflat].append(flat)
                    if trans[1] or len(trans[2]) > 0:
                        work_set.add(flat)
                else:
                    if not trans[2]:
                        for cflat in trans[4]:
                            wtm_to_btm[cflat].append(flat)

        for flat in pre_solved_set:
            for parent in btm_to_wtm.get(flat, []):
                if table_flat[parent] == 0:
                    work_set.add(parent)

        trans_lookup = dict(zip(unsolved_flats, transitions))
        print(f"[Stage 4] Pre-solved: {pre_solved:,} | Initial work_set: {len(work_set):,}")

        print("[Stage 5] O(N) Dirty-Set Retrograde Analysis...")
        iteration = 1
        max_dtm_solved = 0
        solved_count = pre_solved

        while work_set:
            iter_start = time.time()
            changed = 0
            next_work_set = set()
            snapshot = table_flat.copy()

            for flat in work_set:
                if table_flat[flat] != 0:
                    continue
                trans = trans_lookup[flat]

                if (flat & 1) == 0:  # White to move
                    _, imm_win, known_wins, child_vals = trans
                    best_win = 1 if imm_win else 0
                    for val in known_wins:
                        if best_win == 0 or val < best_win:
                            best_win = val
                    for cflat in child_vals:
                        res = int(snapshot[cflat])
                        if res < 0:
                            val = abs(res) + 1
                            if best_win == 0 or val < best_win:
                                best_win = val
                    if best_win > 0:
                        table_flat[flat] = best_win
                        changed += 1
                        for parent in wtm_to_btm.get(flat, []):
                            if table_flat[parent] == 0:
                                next_work_set.add(parent)

                else:  # Black to move
                    _, moves_count, escape, known_3man, child_vals = trans
                    if moves_count == 0:
                        table_flat[flat] = -1
                        changed += 1
                        for parent in btm_to_wtm.get(flat, []):
                            if table_flat[parent] == 0:
                                next_work_set.add(parent)
                        continue
                    if escape:
                        continue
                    max_win = 0
                    for val in known_3man:
                        if val > max_win:
                            max_win = val
                    all_res = True
                    for cflat in child_vals:
                        res = int(snapshot[cflat])
                        if res <= 0:
                            all_res = False
                            break
                        if res > max_win:
                            max_win = res
                    if all_res:
                        table_flat[flat] = -(max_win + 1)
                        changed += 1
                        for parent in btm_to_wtm.get(flat, []):
                            if table_flat[parent] == 0:
                                next_work_set.add(parent)

            work_set = next_work_set
            solved_count += changed
            if changed > 0:
                max_dtm_solved = iteration
            print(f"[Iteration {iteration}] {changed:,} new | Total: {solved_count:,} | "
                  f"Next: {len(work_set):,} | Time: {time.time()-iter_start:.1f}s")
            iteration += 1

        with open(self.filename, 'wb') as f:
            self.table.tofile(f)
        elapsed = time.time() - start_time
        remaining = candidate_states - solved_count
        _safe_record_longest_mate(
            f"K_{self.p1_name}_{self.p2_name}_K", max_dtm_solved, solved_count, remaining, elapsed)
        print(f"[LongestMate] K_{self.p1_name}_{self.p2_name}_K: "
              f"max_dtm={max_dtm_solved}, decisive={solved_count:,}, remaining={remaining:,}")
        print(f"\nSUCCESS: Generated in {elapsed / 60:.1f} minutes.")


# ==============================================================================
# 4-MAN CROSS GENERATOR  (K + WPiece vs K + BPiece)
# ==============================================================================

_WV_W_NAME = None
_WV_B_NAME = None
_WV_W_CLASS = None
_WV_B_CLASS = None
_WV_3MAN_TABLES = {}
_WV_PROMO_TABLES = {}
_IN_TABLE_SENTINEL = "IN_TABLE"
_W4V_BOARD = None
_W4V_WK_OBJ = None
_W4V_WP_OBJ = None
_W4V_BK_OBJ = None
_W4V_BP_OBJ = None

def _tb_file_4man_vs(w_name, b_name):
    names = sorted([w_name, b_name])
    return os.path.join(TB_DIR, f"K_{names[0]}_vs_{names[1]}_K.bin")

def _white_win_dtm_from_raw_tb_value(val, turn_idx):
    if val == 0:
        return None
    if (turn_idx == 0 and val > 0) or (turn_idx == 1 and val < 0):
        return abs(val)
    return None

def _init_transition_worker_4vs(w_name, b_name):
    _install_worker_interrupt_ignores()
    global _WV_W_NAME, _WV_B_NAME, _WV_W_CLASS, _WV_B_CLASS
    global _WV_3MAN_TABLES, _WV_PROMO_TABLES
    global _W4V_BOARD, _W4V_WK_OBJ, _W4V_WP_OBJ, _W4V_BK_OBJ, _W4V_BP_OBJ

    names = sorted([w_name, b_name])
    _WV_W_NAME, _WV_B_NAME = names[0], names[1]
    _WV_W_CLASS = PIECE_CLASS_BY_NAME[_WV_W_NAME]
    _WV_B_CLASS = PIECE_CLASS_BY_NAME[_WV_B_NAME]

    _WV_3MAN_TABLES = {}
    names_needed = {w_name, b_name}
    if w_name == "Pawn" or b_name == "Pawn":
        names_needed.add("Queen")
    for name in names_needed:
        path = os.path.join(TB_DIR, f"K_{name}_K.bin")
        if os.path.exists(path):
            _WV_3MAN_TABLES[name] = _load_3man_table_file(path)

    _WV_PROMO_TABLES = {}
    promo_targets = set()
    if w_name == "Pawn":
        promo_targets.add(("Queen", b_name))
    if b_name == "Pawn":
        promo_targets.add((w_name, "Queen"))
    for key in promo_targets:
        sorted_key = tuple(sorted(key))
        path = _tb_file_4man_vs(key[0], key[1])
        if os.path.exists(path):
            data16 = np.fromfile(path, dtype=np.int16)
            _WV_PROMO_TABLES[sorted_key] = data16.reshape((64, 64, 64, 64, 2))

    _W4V_BOARD = Board(setup=False)
    _W4V_WK_OBJ = King('white')
    _W4V_WP_OBJ = _WV_W_CLASS('white')
    _W4V_BK_OBJ = King('black')
    _W4V_BP_OBJ = _WV_B_CLASS('black')
    pc = _W4V_BOARD.piece_counts
    pc['white'][King] = 1
    pc['black'][King] = 1
    pc['white'][type(_W4V_WP_OBJ)] += 1
    pc['black'][type(_W4V_BP_OBJ)] += 1

def _white_win_dtm_3man_white_piece(piece_name, wk, wp, bk, turn_idx):
    tb = _WV_3MAN_TABLES.get(piece_name)
    if tb is None:
        return None
    q_idx = (wk[0]*8+wk[1], wp[0]*8+wp[1], bk[0]*8+bk[1], turn_idx)
    return _white_win_dtm_from_raw_tb_value(int(tb[q_idx]), turn_idx)

def _white_win_dtm_3man_black_piece(piece_name, wk, bk, bp, turn_idx):
    tb = _WV_3MAN_TABLES.get(piece_name)
    if tb is None:
        return None
    t_turn_idx = 1 - turn_idx
    atk_k = _flip(bk)
    atk_p = _flip(bp)
    def_k = _flip(wk)
    q_idx = (atk_k[0]*8+atk_k[1], atk_p[0]*8+atk_p[1], def_k[0]*8+def_k[1], t_turn_idx)
    val = int(tb[q_idx])
    if val == 0:
        return None
    attacker_wins = (t_turn_idx == 0 and val > 0) or (t_turn_idx == 1 and val < 0)
    if attacker_wins:
        return None
    return abs(val)

def _white_win_dtm_promo_vs(w_name, b_name, wk, wp, bk, bp, turn_idx):
    key = tuple(sorted((w_name, b_name)))
    tb = _WV_PROMO_TABLES.get(key)
    if tb is None:
        return None

    if w_name > b_name:
        t_turn_idx = 1 - turn_idx
        bk_f, bp_f = _flip(bk), _flip(bp)
        wk_f, wp_f = _flip(wk), _flip(wp)
        q_idx = (bk_f[0]*8+bk_f[1], bp_f[0]*8+bp_f[1],
                 wk_f[0]*8+wk_f[1], wp_f[0]*8+wp_f[1], t_turn_idx)
        val = int(tb.flat[_flat_idx_raw_4(*q_idx)])
        if val == 0:
            return None
        attacker_wins = (t_turn_idx == 0 and val > 0) or (t_turn_idx == 1 and val < 0)
        return None if attacker_wins else abs(val)
    else:
        q_idx = (wk[0]*8+wk[1], wp[0]*8+wp[1], bk[0]*8+bk[1], bp[0]*8+bp[1], turn_idx)
        val = int(tb.flat[_flat_idx_raw_4(*q_idx)])
        return _white_win_dtm_from_raw_tb_value(val, turn_idx)

def _white_win_dtm_kvk(child, turn_idx):
    turn_color = 'white' if turn_idx == 0 else 'black'
    if not has_legal_moves(child, turn_color):
        return 1 if turn_color == 'black' else None
    return None

def _external_white_win_dtm_4vs(child, turn_idx):
    w_nk = [p for p in child.white_pieces if not isinstance(p, King)]
    b_nk = [p for p in child.black_pieces if not isinstance(p, King)]

    if len(w_nk) == 1 and len(b_nk) == 1:
        wn, bn = type(w_nk[0]).__name__, type(b_nk[0]).__name__
        if wn == _WV_W_NAME and bn == _WV_B_NAME:
            return _IN_TABLE_SENTINEL
        return _white_win_dtm_promo_vs(
            wn, bn, child.white_king_pos, w_nk[0].pos,
            child.black_king_pos, b_nk[0].pos, turn_idx)
    if len(w_nk) == 1 and len(b_nk) == 0:
        return _white_win_dtm_3man_white_piece(
            type(w_nk[0]).__name__, child.white_king_pos, w_nk[0].pos, child.black_king_pos, turn_idx)
    if len(w_nk) == 0 and len(b_nk) == 1:
        return _white_win_dtm_3man_black_piece(
            type(b_nk[0]).__name__, child.white_king_pos, child.black_king_pos, b_nk[0].pos, turn_idx)
    if len(w_nk) == 0 and len(b_nk) == 0:
        return _white_win_dtm_kvk(child, turn_idx)
    return None

def _build_transition_worker_4vs(flat):
    i4, rest = flat % 2, flat // 2
    i3, rest = rest % 64, rest // 64
    i2, rest = rest % 64, rest // 64
    i1, i0   = rest % 64, rest // 64
    wk, wp, bk, bp = (i0//8, i0%8), (i1//8, i1%8), (i2//8, i2%8), (i3//8, i3%8)
    turn_is_white, opp_turn_idx = (i4 == 0), 1 - i4

    board = _W4V_BOARD
    g = board.grid
    for p in (_W4V_WK_OBJ, _W4V_WP_OBJ, _W4V_BK_OBJ, _W4V_BP_OBJ):
        if p.pos:
            g[p.pos[0]][p.pos[1]] = None
    _W4V_WK_OBJ.pos, _W4V_WP_OBJ.pos = wk, wp
    _W4V_BK_OBJ.pos, _W4V_BP_OBJ.pos = bk, bp
    _W4V_WK_OBJ._list_pos, _W4V_WP_OBJ._list_pos = 0, 1
    _W4V_BK_OBJ._list_pos, _W4V_BP_OBJ._list_pos = 0, 1
    g[wk[0]][wk[1]], g[wp[0]][wp[1]] = _W4V_WK_OBJ, _W4V_WP_OBJ
    g[bk[0]][bk[1]], g[bp[0]][bp[1]] = _W4V_BK_OBJ, _W4V_BP_OBJ
    board.white_king_pos, board.black_king_pos = wk, bk
    board.white_pieces[:] = [_W4V_WK_OBJ, _W4V_WP_OBJ]
    board.black_pieces[:] = [_W4V_BK_OBJ, _W4V_BP_OBJ]

    passive_color = 'black' if turn_is_white else 'white'
    if is_in_check(board, passive_color):
        return None

    moves = get_all_pseudo_legal_moves(board, 'white' if turn_is_white else 'black')

    if turn_is_white:
        immediate_win = False
        known_win_vals = []
        child_flats = []

        for start, end in moves:
            record = board.make_move_track(start, end)
            
            # --- FIX: Strict Legality Guard ---
            if not board.white_king_pos or not board.black_king_pos or is_in_check(board, 'white'):
                board.unmake_move(record)
                continue

            # True Checkmate (No legal moves for Black)
            if not has_legal_moves(board, 'black'):
                immediate_win = True
                board.unmake_move(record)
                break

            ext_dtm = _external_white_win_dtm_4vs(board, opp_turn_idx)
            if ext_dtm == _IN_TABLE_SENTINEL:
                w_piece = next((p for p in board.white_pieces if not isinstance(p, King)), None)
                b_piece = next((p for p in board.black_pieces if not isinstance(p, King)), None)
                if w_piece is not None and b_piece is not None:
                    c0 = board.white_king_pos[0]*8 + board.white_king_pos[1]
                    c1 = w_piece.pos[0]*8 + w_piece.pos[1]
                    c2 = board.black_king_pos[0]*8 + board.black_king_pos[1]
                    c3 = b_piece.pos[0]*8 + b_piece.pos[1]
                    child_flats.append(_flat_idx_raw_4(c0, c1, c2, c3, opp_turn_idx))
            elif ext_dtm is not None and ext_dtm > 0:
                known_win_vals.append(ext_dtm + 1)
            board.unmake_move(record)

        return (flat, ('w', immediate_win, array('H', known_win_vals), array('I', child_flats)))

    # --- Black to move ---
    legal_moves_count = 0
    has_non_losing_escape = False
    known_win_vals = []
    child_flats = []

    for start, end in moves:
        record = board.make_move_track(start, end)
        
        # --- FIX: Strict Legality Guard ---
        if not board.black_king_pos or not board.white_king_pos or is_in_check(board, 'black'):
            board.unmake_move(record)
            continue

        legal_moves_count += 1

        if not has_legal_moves(board, 'white'):
            has_non_losing_escape = True
            board.unmake_move(record)
            continue

        ext_dtm = _external_white_win_dtm_4vs(board, opp_turn_idx)
        if ext_dtm == _IN_TABLE_SENTINEL:
            w_piece = next((p for p in board.white_pieces if not isinstance(p, King)), None)
            b_piece = next((p for p in board.black_pieces if not isinstance(p, King)), None)
            if w_piece is None or b_piece is None:
                has_non_losing_escape = True
            else:
                c0 = board.white_king_pos[0]*8 + board.white_king_pos[1]
                c1 = w_piece.pos[0]*8 + w_piece.pos[1]
                c2 = board.black_king_pos[0]*8 + board.black_king_pos[1]
                c3 = b_piece.pos[0]*8 + b_piece.pos[1]
                child_flats.append(_flat_idx_raw_4(c0, c1, c2, c3, opp_turn_idx))
        elif ext_dtm is not None and ext_dtm > 0:
            known_win_vals.append(ext_dtm)
        else:
            has_non_losing_escape = True
        board.unmake_move(record)

    return (flat, ('b', legal_moves_count, has_non_losing_escape,
                   array('H', known_win_vals), array('I', child_flats)))


def _gen_valid_4man_vs_indices_numpy(total_positions, w_name, b_name, chunk_size=8_000_000):
    for start in range(0, total_positions, chunk_size):
        end_c = min(start + chunk_size, total_positions)
        flat = np.arange(start, end_c, dtype=np.int64)
        rest = flat // 2
        bp_arr = rest % 64; rest //= 64
        bk_arr = rest % 64; rest //= 64
        wp_arr = rest % 64
        wk_arr = rest // 64

        mask = ((wk_arr != wp_arr) & (wk_arr != bk_arr) & (wk_arr != bp_arr) &
                (wp_arr != bk_arr) & (wp_arr != bp_arr) & (bk_arr != bp_arr))
        if w_name == "Pawn": mask &= (wp_arr >= 8) & (wp_arr < 56)
        if b_name == "Pawn": mask &= (bp_arr >= 8) & (bp_arr < 56)

        if not mask.any():
            continue
        yield flat[mask].tolist()


class Generator4Vs:
    def __init__(self, w_piece_class, b_piece_class):
        names = sorted([w_piece_class.__name__, b_piece_class.__name__])
        self.w_name, self.b_name = names[0], names[1]
        self.w_piece_class = PIECE_CLASS_BY_NAME[self.w_name]
        self.b_piece_class = PIECE_CLASS_BY_NAME[self.b_name]
        self.filename = _tb_file_4man_vs(self.w_name, self.b_name)
        self.total_positions = 64 * 64 * 64 * 64 * 2
        self.table = np.zeros((64, 64, 64, 64, 2), dtype=np.int16)
        self.transition_workers = max(1, (os.cpu_count() or 2) - TB_THREADS_SUBTRACT)

    def generate(self):
        if os.path.exists(self.filename):
            print(f"[SKIP] {self.filename} already exists.")
            return

        start_time = time.time()
        print(f"\n{'='*60}\n GENERATING: King + {self.w_name} vs King + {self.b_name}\n{'='*60}")

        print("[Stage 1] Enumerating structural candidates...")
        stage1_start = time.time()
        all_flats = array('I')
        for chunk in _gen_valid_4man_vs_indices_numpy(self.total_positions, self.w_name, self.b_name):
            all_flats.extend(chunk)
        print(f"[Stage 1] Found {len(all_flats):,} candidates in {time.time()-stage1_start:.1f}s.")

        print(f"[Stage 2 & 3] Merged legality filter + transition cache ({len(all_flats):,} states)...")
        stage23_start = time.time()
        unsolved_flats = array('I')
        transitions = []
        with ProcessPoolExecutor(
            max_workers=self.transition_workers,
            initializer=_init_transition_worker_4vs,
            initargs=(self.w_name, self.b_name)
        ) as ex:
            for done, result in enumerate(
                ex.map(_build_transition_worker_4vs, all_flats, chunksize=4096), 1
            ):
                if result is not None:
                    unsolved_flats.append(result[0])
                    transitions.append(result[1])
                if done % 100_000 == 0:
                    elapsed = max(0.001, time.time() - stage23_start)
                    speed = done / elapsed
                    eta = (len(all_flats) - done) / speed
                    print(f"  > {done / len(all_flats) * 100:.1f}% | "
                          f"{speed:,.0f} st/s | ETA: {eta/60:.1f}m", end='\r', flush=True)
        candidate_states = len(unsolved_flats)
        print(f"\n[Stage 2 & 3] Valid states: {candidate_states:,} | "
              f"Time: {(time.time()-stage23_start)/60:.1f}m")

        print("[Stage 4] Pre-solving & building retrograde parent maps...")
        table_flat = self.table.reshape(-1)
        btm_to_wtm = defaultdict(list)
        wtm_to_btm = defaultdict(list)
        pre_solved_set = set()
        work_set = set()
        pre_solved = 0

        for flat, trans in zip(unsolved_flats, transitions):
            t_idx = flat & 1
            if t_idx == 1 and trans[1] == 0:
                table_flat[flat] = -1
                pre_solved += 1
                pre_solved_set.add(flat)
            else:
                if t_idx == 0:
                    for cflat in trans[3]:
                        btm_to_wtm[cflat].append(flat)
                    if trans[1] or len(trans[2]) > 0:
                        work_set.add(flat)
                else:
                    if not trans[2]:
                        for cflat in trans[4]:
                            wtm_to_btm[cflat].append(flat)

        for flat in pre_solved_set:
            for parent in btm_to_wtm.get(flat, []):
                if table_flat[parent] == 0:
                    work_set.add(parent)

        trans_lookup = dict(zip(unsolved_flats, transitions))
        print(f"[Stage 4] Pre-solved: {pre_solved:,} | Initial work_set: {len(work_set):,}")

        print("[Stage 5] O(N) Dirty-Set Retrograde Analysis...")
        iteration = 1
        max_dtm_solved = 0
        solved_count = pre_solved

        while work_set:
            iter_start = time.time()
            changed = 0
            next_work_set = set()
            snapshot = table_flat.copy()

            for flat in work_set:
                if table_flat[flat] != 0:
                    continue
                trans = trans_lookup[flat]

                if (flat & 1) == 0:  # White to move
                    _, imm_win, known_wins, child_vals = trans
                    best_win = 1 if imm_win else 0
                    for val in known_wins:
                        if best_win == 0 or val < best_win:
                            best_win = val
                    for cflat in child_vals:
                        res = int(snapshot[cflat])
                        if res < 0:
                            val = abs(res) + 1
                            if best_win == 0 or val < best_win:
                                best_win = val
                    if best_win > 0:
                        table_flat[flat] = best_win
                        changed += 1
                        for parent in wtm_to_btm.get(flat, []):
                            if table_flat[parent] == 0:
                                next_work_set.add(parent)

                else:  # Black to move
                    _, moves_count, escape, known_win_vals, child_vals = trans
                    if moves_count == 0:
                        table_flat[flat] = -1
                        changed += 1
                        for parent in btm_to_wtm.get(flat, []):
                            if table_flat[parent] == 0:
                                next_work_set.add(parent)
                        continue
                    if escape:
                        continue
                    max_win = 0
                    for val in known_win_vals:
                        if val > max_win:
                            max_win = val
                    all_res = True
                    for cflat in child_vals:
                        res = int(snapshot[cflat])
                        if res <= 0:
                            all_res = False
                            break
                        if res > max_win:
                            max_win = res
                    if all_res:
                        table_flat[flat] = -(max_win + 1)
                        changed += 1
                        for parent in btm_to_wtm.get(flat, []):
                            if table_flat[parent] == 0:
                                next_work_set.add(parent)

            work_set = next_work_set
            solved_count += changed
            if changed > 0:
                max_dtm_solved = iteration
            print(f"[Iteration {iteration}] {changed:,} new | Total: {solved_count:,} | "
                  f"Next: {len(work_set):,} | Time: {time.time()-iter_start:.1f}s")
            iteration += 1

        with open(self.filename, 'wb') as f:
            self.table.tofile(f)
        elapsed = time.time() - start_time
        remaining = candidate_states - solved_count
        _safe_record_longest_mate(
            f"K_{self.w_name}_vs_{self.b_name}_K", max_dtm_solved, solved_count, remaining, elapsed)
        print(f"[LongestMate] K_{self.w_name}_vs_{self.b_name}_K: "
              f"max_dtm={max_dtm_solved}, decisive={solved_count:,}, remaining={remaining:,}")
        print(f"\nSUCCESS: Generated in {elapsed / 60:.1f} minutes.")


# ==============================================================================
# ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    _install_main_interrupt_ignores()
    print("=== Tablebase Generator (v9.0 - Absolute Perfection) ===")

    Q, R, B, N, P = Queen, Rook, Bishop, Knight, Pawn

    print("\n--- TIER 1: 3-MAN NON-PAWN TABLES ---")
    for piece_class in [Q, R, B, N]:
        Generator(piece_class).generate()

    print("\n--- TIER 2: 3-MAN PAWN TABLE (needs K_Queen_K) ---")
    Generator(P).generate()

    print("\n--- TIER 3: 4-MAN NON-PAWN TABLES ---")
    non_pawn = [Q, R, B, N]
    for p1, p2 in combinations_with_replacement(non_pawn, 2):
        Generator4(p1, p2).generate()
        
    seen_vs = set()
    for w in non_pawn:
        for b in non_pawn:
            key = tuple(sorted([w.__name__, b.__name__]))
            if key not in seen_vs:
                seen_vs.add(key)
                Generator4Vs(w, b).generate()

    print("\n--- TIER 4: 4-MAN PAWN TABLES ---")
    for piece_class in [Q, R, B, N, P]:
        Generator4(P, piece_class).generate()
        
    seen_vs4 = set()
    for piece_class in [Q, R, B, N, P]:
        key = tuple(sorted(["Pawn", piece_class.__name__]))
        if key not in seen_vs4:
            seen_vs4.add(key)
            Generator4Vs(P, piece_class).generate()

    print("\n\n=== ALL TABLEBASE GENERATION COMPLETE ===")