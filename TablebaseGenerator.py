# TablebaseGenerator.py (v5.10 - always-loss cleanup + simpler transition tuples)

import os
import time
import __main__
import signal
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from GameLogic import *
import struct
from array import array
import cProfile
import pstats
import io

# --- CONFIGURATION ---
TB_DIR = "tablebases"
os.makedirs(TB_DIR, exist_ok=True)

ENABLE_RETRO_FILTER_4MAN = True # True adds a new stop that makes for smaller file size and more correctness. It costs a litle time, but actually saves so much computation its allways worth it.
LONGEST_MATES_NOTE_FILE = os.path.join(TB_DIR, "new_longest_mates.tsv")
LONGEST_MATE_KEY_PREFIX = "regen_"

 # How many CPUs to subtract from the total when choosing worker count.
 # The generator will use max(1, os.cpu_count() - TB_THREADS_SUBTRACT).
 # This lets you specify "total - n" threads by changing this value.
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
    print("\n[Interrupt ignored] Tablebase generation is protected from Ctrl+C. Close the terminal or kill the process explicitly if you really want to stop it.", flush=True)

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

def _jung_king_threatens(board, r1, c1, r2, c2, vacated=None):
    """
    Returns True if a Jungle King at (r1,c1) threatens square (r2,c2).
    Adjacent kings always threaten each other. Two-square king threats only
    exist when the midpoint is empty, except during move prescreens where the
    moving king's start square counts as vacated.
    """
    dr_raw = r2 - r1
    dc_raw = c2 - c1
    dr = abs(dr_raw)
    dc = abs(dc_raw)
    mx = max(dr, dc)
    mn = min(dr, dc)
    if mx == 1:
        return True
    if mx != 2 or mn == 1:
        return False

    mr = r1 + ((dr_raw > 0) - (dr_raw < 0))
    mc = c1 + ((dc_raw > 0) - (dc_raw < 0))
    if vacated is not None and (mr, mc) == vacated:
        return True
    return board.grid[mr][mc] is None

def _load_table_file_any_dtype(filename):
    data16 = np.fromfile(filename, dtype=np.int16)
    if data16.size == EXPECTED_TABLE_ENTRIES:
        return data16.reshape((64, 64, 64, 2))
    data8 = np.fromfile(filename, dtype=np.int8)
    if data8.size == EXPECTED_TABLE_ENTRIES:
        return data8.astype(np.int16).reshape((64, 64, 64, 2))
    raise ValueError(f"Invalid tablebase file size: {filename}")

def _flat_idx_raw(i0, i1, i2, i3):
    return (((i0 * 64 + i1) * 64 + i2) * 2 + i3)

def _pack_uint_list(lst):
    n = len(lst)
    if n == 0: return struct.pack('<I', 0)
    return struct.pack('<I', n) + struct.pack(f'<{n}I', *lst)

def _unpack_uint_list(b):
    if not b: return ()
    n = struct.unpack_from('<I', b, 0)[0]
    if n == 0: return ()
    return struct.unpack_from(f'<{n}I', b, 4)

def _read_longest_mate_records():
    records = {}
    if not os.path.exists(LONGEST_MATES_NOTE_FILE):
        return records
    try:
        with open(LONGEST_MATES_NOTE_FILE, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) < 6:
                    continue
                key, max_dtm, decisive, remaining, elapsed_min, updated_utc = parts[:6]
                records[key] = {
                    "max_dtm": str(max_dtm),
                    "decisive": str(decisive),
                    "remaining": str(remaining),
                    "elapsed_min": str(elapsed_min),
                    "updated_utc": str(updated_utc),
                }
    except Exception:
        pass
    return records

def _write_longest_mate_records(records):
    with open(LONGEST_MATES_NOTE_FILE, "w", encoding="utf-8") as f:
        f.write("# table_key\tmax_dtm\tdecisive\tremaining\telapsed_min\tupdated_utc\n")
        for key in sorted(records.keys()):
            rec = records[key]
            f.write(
                f"{key}\t{rec['max_dtm']}\t{rec['decisive']}\t{rec['remaining']}\t"
                f"{rec['elapsed_min']}\t{rec['updated_utc']}\n"
            )

def _upsert_longest_mate_record(table_key, max_dtm, decisive, remaining, elapsed_seconds):
    records = _read_longest_mate_records()
    records[f"{LONGEST_MATE_KEY_PREFIX}{table_key}"] = {
        "max_dtm": str(int(max_dtm)),
        "decisive": str(int(decisive)),
        "remaining": str(int(remaining)),
        "elapsed_min": f"{elapsed_seconds/60.0:.1f}",
        "updated_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
    }
    _write_longest_mate_records(records)

def _safe_record_longest_mate(table_key, max_dtm, decisive, remaining, elapsed_seconds):
    try:
        _upsert_longest_mate_record(table_key, max_dtm, decisive, remaining, elapsed_seconds)
    except Exception as e:
        print(f"[LongestMate] Warning: failed to update note file ({e})")

# ==============================================================================
# 3-MAN GENERATOR
# ==============================================================================

_W_PIECE_NAME = None
_W_PIECE_CLASS = None
_W_QUEEN_TABLE = None

def _init_transition_worker(piece_name, queen_tb_file):
    _install_worker_interrupt_ignores()
    global _W_PIECE_NAME, _W_PIECE_CLASS, _W_QUEEN_TABLE
    _W_PIECE_NAME = piece_name
    _W_PIECE_CLASS = PIECE_CLASS_BY_NAME[piece_name]
    _W_QUEEN_TABLE = None
    if piece_name == "Pawn" and queen_tb_file:
        _W_QUEEN_TABLE = _load_table_file_any_dtype(queen_tb_file)

def _build_transition_worker(idx):
    i0, i1, i2, i3 = idx
    wk = (i0 // 8, i0 % 8)
    wp = (i1 // 8, i1 % 8)
    bk = (i2 // 8, i2 % 8)
    turn_is_white = (i3 == 0)
    opp_turn_idx = 1 - i3

    board = Board(setup=False)
    board.add_piece(King('white'), wk[0], wk[1])
    board.add_piece(_W_PIECE_CLASS('white'), wp[0], wp[1])
    board.add_piece(King('black'), bk[0], bk[1])
    moves = get_all_pseudo_legal_moves(board, 'white' if turn_is_white else 'black')

    if turn_is_white:
        immediate_win = False
        promo_win_vals =[]
        child_flats =[]

        for m in moves:
            start, end = m
            
            # [Opt-2] Fast Pre-Screen
            if isinstance(board.grid[start[0]][start[1]], King) and end != bk:
                if _jung_king_threatens(board, end[0], end[1], bk[0], bk[1], vacated=start):
                    continue

            child = board.clone()
            child.make_move(start, end)
            
            # [Opt-1] Inline Check Validation (Black only has a King)
            wkp = child.white_king_pos
            bkp = child.black_king_pos
            if wkp and bkp and _jung_king_threatens(child, wkp[0], wkp[1], bkp[0], bkp[1]):
                continue

            if not bkp:
                immediate_win = True
                continue

            if _W_PIECE_NAME == "Pawn" and isinstance(child.grid[m[1][0]][m[1][1]], Queen):
                q_idx = (child.white_king_pos[0] * 8 + child.white_king_pos[1],
                         m[1][0] * 8 + m[1][1],
                         child.black_king_pos[0] * 8 + child.black_king_pos[1], 1)
                q_val = int(_W_QUEEN_TABLE[q_idx])
                if q_val < 0: promo_win_vals.append(abs(q_val) + 1)
                continue

            if len(child.white_pieces) >= 2:
                p = next((x for x in child.white_pieces if not isinstance(x, King)), None)
                if p is None: continue
                c0 = child.white_king_pos[0] * 8 + child.white_king_pos[1]
                c1 = p.pos[0] * 8 + p.pos[1]
                c2 = child.black_king_pos[0] * 8 + child.black_king_pos[1]
                child_flats.append(_flat_idx_raw(c0, c1, c2, opp_turn_idx))

        return _flat_idx_raw(i0, i1, i2, i3), ('w', immediate_win, _pack_uint_list(promo_win_vals), _pack_uint_list(child_flats))

    legal_moves_count = 0
    has_non_losing_escape = False
    child_flats =[]

    for m in moves:
        if m[1] != wk and _jung_king_threatens(board, m[1][0], m[1][1], wk[0], wk[1], vacated=m[0]):
            continue
        child = board.clone()
        child.make_move(m[0], m[1])
        if is_in_check(child, 'black'):
            continue

        legal_moves_count += 1
        if not child.white_king_pos or len(child.white_pieces) < 2:
            has_non_losing_escape = True; continue
            
        if _W_PIECE_NAME == "Pawn" and isinstance(child.grid[child.white_pieces[1].pos[0]][child.white_pieces[1].pos[1]], Queen):
            has_non_losing_escape = True; continue

        p = next((x for x in child.white_pieces if not isinstance(x, King)), None)
        if p is None:
            has_non_losing_escape = True; continue
            
        c0 = child.white_king_pos[0] * 8 + child.white_king_pos[1]
        c1 = p.pos[0] * 8 + p.pos[1]
        c2 = child.black_king_pos[0] * 8 + child.black_king_pos[1]
        child_flats.append(_flat_idx_raw(c0, c1, c2, opp_turn_idx))

    return _flat_idx_raw(i0, i1, i2, i3), ('b', legal_moves_count, has_non_losing_escape, _pack_uint_list(child_flats))

class Generator:
    def __init__(self, piece_class):
        self.piece_class = piece_class
        self.piece_name = piece_class.__name__
        self.filename = os.path.join(TB_DIR, f"K_{self.piece_name}_K.bin")
        self.queen_tb_file = os.path.join(TB_DIR, "K_Queen_K.bin")

        self.total_positions = 64 * 64 * 64 * 2
        self.table = np.zeros((64, 64, 64, 2), dtype=np.int16)

        self.sim_board = Board(setup=False)
        self.wk_obj = King('white')
        self.wp_obj = self.piece_class('white')
        self.bk_obj = King('black')

        self.queen_table = None
        if self.piece_name == "Pawn":
            if os.path.exists(self.queen_tb_file):
                print(f"[Pawn Support] Loading Queen Tablebase for promotion lookups...")
                self.queen_table = _load_table_file_any_dtype(self.queen_tb_file)
            else:
                print(f"CRITICAL ERROR: Queen Tablebase not found! Generate Queen first.")
                exit()

        cpu_default = max(1, (os.cpu_count() or 2) - TB_THREADS_SUBTRACT)
        self.transition_workers = min(8, cpu_default)

    def decode(self, idx):
        wk = (idx[0] // 8, idx[0] % 8)
        wp = (idx[1] // 8, idx[1] % 8)
        bk = (idx[2] // 8, idx[2] % 8)
        return wk, wp, bk, idx[3]

    def _flat_idx(self, idx):
        return (((idx[0] * 64 + idx[1]) * 64 + idx[2]) * 2 + idx[3])

    def setup_sim(self, wk, wp, bk):
        self.sim_board.grid = [[None for _ in range(8)] for _ in range(8)]
        self.wk_obj.pos = wk
        self.sim_board.grid[wk[0]][wk[1]] = self.wk_obj
        self.sim_board.white_king_pos = wk
        self.wp_obj.pos = wp
        self.sim_board.grid[wp[0]][wp[1]] = self.wp_obj
        self.bk_obj.pos = bk
        self.sim_board.grid[bk[0]][bk[1]] = self.bk_obj
        self.sim_board.black_king_pos = bk
        self.sim_board.white_pieces = [self.wk_obj, self.wp_obj]
        self.sim_board.black_pieces = [self.bk_obj]
        self.sim_board.piece_counts = {
            'white': {Pawn:0, Knight:0, Bishop:0, Rook:0, Queen:0, King:1},
            'black': {Pawn:0, Knight:0, Bishop:0, Rook:0, Queen:0, King:1}
        }
        self.sim_board.piece_counts['white'][type(self.wp_obj)] += 1

    def generate(self):
        if os.path.exists(self.filename):
            print(f"[SKIP] {self.filename} already exists.")
            return

        start_time = time.time()
        print(f"\n{'='*60}\n GENERATING: King + {self.piece_name} vs King\n{'='*60}")
        unsolved_indices = []
        candidate_flats = array('I')
        raw_candidate_states = 0
        candidate_states = 0
        retro_illegal_discarded = 0
        phase1_presolved = 0

        print(f"[Phase 1] Initialization...")
        for count, idx in enumerate(np.ndindex(self.table.shape)):
            if count % 200000 == 0 and count > 0: 
                print(f"  > Scanned {count}/{self.total_positions} positions...", end='\r')

            wk, wp, bk, t_idx = self.decode(idx)
            if wk == wp or wk == bk or wp == bk: continue
            if self.piece_name == "Pawn" and (wp[0] == 0 or wp[0] == 7): continue

            raw_candidate_states += 1
            
            # [Optimization restored] Handle Phase 1 checkmates locally to save worker time
            self.setup_sim(wk, wp, bk)
            # Retro-legality filter:
            # The side that just moved (the opponent of side-to-move) cannot be in check.
            if t_idx == 0:
                # White to move -> Black just moved, so Black must not be in check.
                if is_in_check(self.sim_board, 'black'):
                    retro_illegal_discarded += 1
                    continue
            else:
                # Black to move -> White just moved, so White must not be in check.
                if is_in_check(self.sim_board, 'white'):
                    retro_illegal_discarded += 1
                    continue

            candidate_states += 1
            candidate_flats.append(self._flat_idx(idx))

            if t_idx == 1: # Black
                if not self.sim_board.black_king_pos:
                    self.table[idx] = -1
                    phase1_presolved += 1
                    continue
                if not has_legal_moves(self.sim_board, 'black'):
                    self.table[idx] = -1
                    phase1_presolved += 1
                    continue
            else: # White
                if not self.sim_board.white_king_pos:
                    self.table[idx] = -1
                    phase1_presolved += 1
                    continue

            unsolved_indices.append(idx)
        print()
        print(
            f"[Phase 1] Raw candidates: {raw_candidate_states:,} | "
            f"Retro-illegal discarded: {retro_illegal_discarded:,} | "
            f"Legal candidates: {candidate_states:,} | "
            f"Pre-solved: {phase1_presolved:,} | Remaining: {len(unsolved_indices):,}"
        )

        print(f"[Phase 1.5] Building transition cache ({len(unsolved_indices)} states)...")
        cache = {}
        
        can_parallel = True
        main_file = getattr(__main__, "__file__", "")
        if not main_file or main_file == "<stdin>": can_parallel = False

        if can_parallel:
            try:
                with ProcessPoolExecutor(max_workers=self.transition_workers,
                                         initializer=_init_transition_worker,
                                         initargs=(self.piece_name, self.queen_tb_file)) as ex:
                    for done, (flat, transition) in enumerate(
                            ex.map(_build_transition_worker, unsolved_indices, chunksize=1024), start=1):
                        cache[flat] = transition
                        if done % 50000 == 0:
                            speed = done / max(0.001, time.time() - start_time)
                            print(f"  > Cache: {done/len(unsolved_indices)*100:.1f}% | {speed:.0f} st/s | ETA: {(len(unsolved_indices)-done)/speed/60:.1f}m", end='\r')
                print()
            except Exception as e:
                print(f"[{self.piece_name}] Parallel build failed ({e}). Falling back to single-process.")
                cache.clear()
                can_parallel = False

        if not can_parallel:
            _init_transition_worker(self.piece_name, self.queen_tb_file)
            for done, idx in enumerate(unsolved_indices, start=1):
                cache[self._flat_idx(idx)] = _build_transition_worker(idx)
                if done % 50000 == 0:
                    speed = done / max(0.001, time.time() - start_time)
                    print(f"  > Cache: {done/len(unsolved_indices)*100:.1f}% | {speed:.0f} st/s", end='\r')
            print()
        
        iteration = 1
        max_dtm_solved = 0
        while True:
            iter_start = time.time()
            changed = 0
            new_unsolved =[]
            prev_table = self.table.copy()
            prev_flat = prev_table.reshape(-1)

            print(f"\n[{self.piece_name}] Iteration {iteration} (DTM {iteration})")

            for idx in unsolved_indices:
                t_idx = idx[3]
                flat = self._flat_idx(idx)
                transition = cache[flat]

                if t_idx == 0:
                    best_win = 0
                    _, immediate_win, promo_packed, child_packed = transition
                    if immediate_win: best_win = 1
                    for val in _unpack_uint_list(promo_packed):
                        if best_win == 0 or val < best_win: best_win = val
                    for cflat in _unpack_uint_list(child_packed):
                        res_val = int(prev_flat[cflat])
                        if res_val < 0:
                            val = abs(res_val) + 1
                            if best_win == 0 or val < best_win: best_win = val

                    if best_win > 0:
                        self.table[idx] = best_win
                        changed += 1
                    else:
                        new_unsolved.append(idx)
                else:
                    _, legal_moves_count, has_non_losing_escape, child_packed = transition
                    
                    if legal_moves_count == 0:
                        self.table[idx] = -1
                        changed += 1
                        continue

                    all_moves_lead_to_win = not has_non_losing_escape
                    max_win_val = 0

                    if all_moves_lead_to_win:
                        for cflat in _unpack_uint_list(child_packed):
                            res_val = int(prev_flat[cflat])
                            if res_val <= 0:
                                all_moves_lead_to_win = False; break
                            if res_val > max_win_val: max_win_val = res_val

                    if all_moves_lead_to_win:
                        self.table[idx] = -(max_win_val + 1)
                        changed += 1
                    else:
                        new_unsolved.append(idx)

            unsolved_indices = new_unsolved
            solved_candidates = candidate_states - len(unsolved_indices)
            percent_candidates = (solved_candidates / candidate_states * 100.0) if candidate_states else 0.0
            print(f"[{self.piece_name}] Round {iteration} Summary: {changed:,} new solved | Total: {solved_candidates:,}/{candidate_states:,} ({percent_candidates:.1f}%) | Time: {time.time()-iter_start:.1f}s")
            if changed > 0:
                max_dtm_solved = iteration
            if changed == 0: break
            iteration += 1

        with open(self.filename, 'wb') as f:
            self.table.tofile(f)
        table_flat = self.table.reshape(-1)
        wins = 0
        losses = 0
        unresolved = 0
        for cflat in candidate_flats:
            val = int(table_flat[cflat])
            if val > 0:
                wins += 1
            elif val < 0:
                losses += 1
            else:
                unresolved += 1
        solved = wins + losses
        solve_rate = (solved / candidate_states * 100.0) if candidate_states else 0.0
        draws = unresolved
        print(
            f"[{self.piece_name}] Final Stats: "
            f"Candidates={candidate_states:,} | "
            f"Decisive={solved:,} ({solve_rate:.1f}%) | "
            f"Wins={wins:,} | Losses={losses:,} | Draws={draws:,}"
        )
        elapsed_seconds = time.time() - start_time
        _safe_record_longest_mate(
            f"K_{self.piece_name}_K",
            max_dtm_solved,
            solved,
            draws,
            elapsed_seconds
        )
        print(f"[{self.piece_name}] Longest decisive DTM: {max_dtm_solved}")
        print(f"\nSUCCESS: Generated in {elapsed_seconds/60:.1f} minutes.")


# ==============================================================================
# OPTIMIZED 4-MAN GENERATOR
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

def _flat_idx_raw_4(i0, i1, i2, i3, i4):
    return (((((i0 * 64 + i1) * 64 + i2) * 64 + i3) * 2) + i4)

def _init_transition_worker_4(p1_name, p2_name, promo_tb_file):
    _install_worker_interrupt_ignores()
    global _W4_P1_CLASS, _W4_P2_CLASS, _W4_P1_NAME, _W4_P2_NAME
    global _W4_3MAN_TABLES, _W4_PROMO_TABLE, _W4_SAME_PIECE, _W4_PROMO_SAME_PIECE
    _W4_P1_NAME = p1_name
    _W4_P2_NAME = p2_name
    _W4_P1_CLASS = PIECE_CLASS_BY_NAME[p1_name]
    _W4_P2_CLASS = PIECE_CLASS_BY_NAME[p2_name]
    _W4_SAME_PIECE = (p1_name == p2_name)               
    _W4_3MAN_TABLES = {}
    _W4_PROMO_TABLE = None
    _W4_PROMO_SAME_PIECE = False

    for name in PIECE_CLASS_BY_NAME:
        path = os.path.join(TB_DIR, f"K_{name}_K.bin")
        if os.path.exists(path):
            _W4_3MAN_TABLES[name] = np.memmap(path, dtype=np.int16, mode='r', shape=(64, 64, 64, 2))

    if promo_tb_file and os.path.exists(promo_tb_file):
        _W4_PROMO_TABLE = np.memmap(promo_tb_file, dtype=np.int16, mode='r', shape=(64, 64, 64, 64, 2))
        base = os.path.basename(promo_tb_file)          
        parts = base[2:-6].split('_')                   
        _W4_PROMO_SAME_PIECE = (len(parts) == 2 and parts[0] == parts[1])

def _build_transition_worker_4(flat):
    i4 = flat % 2
    rest = flat // 2
    i3 = rest % 64
    rest //= 64
    i2 = rest % 64
    rest //= 64
    i1 = rest % 64
    i0 = rest // 64

    wk = (i0 // 8, i0 % 8)
    p1 = (i1 // 8, i1 % 8)
    p2 = (i2 // 8, i2 % 8)
    bk = (i3 // 8, i3 % 8)
    turn_is_white = (i4 == 0)
    opp_turn_idx = 1 - i4

    board = Board(setup=False)
    board.add_piece(King('white'), wk[0], wk[1])
    board.add_piece(_W4_P1_CLASS('white'), p1[0], p1[1])
    board.add_piece(_W4_P2_CLASS('white'), p2[0], p2[1])
    board.add_piece(King('black'), bk[0], bk[1])

    moves = get_all_pseudo_legal_moves(board, 'white' if turn_is_white else 'black')

    if turn_is_white:
        immediate_win = False
        known_win_vals = []
        child_flats =[]

        for m in moves:
            start, end = m

            # [Opt-2] Fast Pre-Screen (King only)
            if isinstance(board.grid[start[0]][start[1]], King) and end != bk:
                if _jung_king_threatens(board, end[0], end[1], bk[0], bk[1], vacated=start):
                    continue

            child = board.clone()
            child.make_move(start, end)

            # [Opt-1] Inline Legality Check (Black only has King)
            wkp = child.white_king_pos
            bkp = child.black_king_pos
            if wkp and bkp and _jung_king_threatens(child, wkp[0], wkp[1], bkp[0], bkp[1]):
                continue

            if not bkp:
                immediate_win = True
                continue

            w_pieces = [p for p in child.white_pieces if not isinstance(p, King)]

            # Double evaporation crash fix included!
            if len(w_pieces) < 2:
                if len(w_pieces) == 1:
                    rem_name = type(w_pieces[0]).__name__
                    if rem_name in _W4_3MAN_TABLES:
                        q_idx = (child.white_king_pos[0] * 8 + child.white_king_pos[1],
                                 w_pieces[0].pos[0] * 8 + w_pieces[0].pos[1],
                                 child.black_king_pos[0] * 8 + child.black_king_pos[1], 1)
                        val = int(_W4_3MAN_TABLES[rem_name][q_idx])
                        if val < 0:
                            known_win_vals.append(abs(val) + 1)
                continue

            p0_ord = _PIECE_CANONICAL_ORDER.get(type(w_pieces[0]), 99)
            p1_ord = _PIECE_CANONICAL_ORDER.get(type(w_pieces[1]), 99)
            if p0_ord > p1_ord:
                w_pieces[0], w_pieces[1] = w_pieces[1], w_pieces[0]

            curr_n1 = type(w_pieces[0]).__name__
            curr_n2 = type(w_pieces[1]).__name__

            if curr_n1 != _W4_P1_NAME or curr_n2 != _W4_P2_NAME:
                if _W4_PROMO_TABLE is not None:
                    idx1 = w_pieces[0].pos[0] * 8 + w_pieces[0].pos[1]
                    idx2 = w_pieces[1].pos[0] * 8 + w_pieces[1].pos[1]
                    if _W4_PROMO_SAME_PIECE and idx1 > idx2:
                        idx1, idx2 = idx2, idx1
                    q_idx = (child.white_king_pos[0] * 8 + child.white_king_pos[1],
                             idx1, idx2,
                             child.black_king_pos[0] * 8 + child.black_king_pos[1], 1)
                    val = int(_W4_PROMO_TABLE.flat[_flat_idx_raw_4(*q_idx)])
                    if val < 0:
                        known_win_vals.append(abs(val) + 1)
                continue

            c0 = child.white_king_pos[0] * 8 + child.white_king_pos[1]
            c1 = w_pieces[0].pos[0] * 8 + w_pieces[0].pos[1]
            c2 = w_pieces[1].pos[0] * 8 + w_pieces[1].pos[1]
            c3 = child.black_king_pos[0] * 8 + child.black_king_pos[1]
            if _W4_SAME_PIECE and c1 > c2:
                c1, c2 = c2, c1
            child_flats.append(_flat_idx_raw_4(c0, c1, c2, c3, opp_turn_idx))

        return ('w', immediate_win, _pack_uint_list(known_win_vals), _pack_uint_list(child_flats))

    legal_moves_count = 0
    has_non_losing_escape = False
    child_flats =[]

    for m in moves:
        if m[1] != wk and _jung_king_threatens(board, m[1][0], m[1][1], wk[0], wk[1], vacated=m[0]):
            continue
        child = board.clone()
        child.make_move(m[0], m[1])
        if is_in_check(child, 'black'):
            continue

        legal_moves_count += 1
        if not child.white_king_pos:
            has_non_losing_escape = True
            continue

        w_pieces = [p for p in child.white_pieces if not isinstance(p, King)]

        if len(w_pieces) < 2:
            if len(w_pieces) == 1:
                rem_name = type(w_pieces[0]).__name__
                if rem_name in _W4_3MAN_TABLES:
                    q_idx = (child.white_king_pos[0] * 8 + child.white_king_pos[1],
                             w_pieces[0].pos[0] * 8 + w_pieces[0].pos[1],
                             child.black_king_pos[0] * 8 + child.black_king_pos[1], 0)
                    val = int(_W4_3MAN_TABLES[rem_name][q_idx])
                    if val <= 0:
                        has_non_losing_escape = True
                        continue
                    else:
                        continue
            has_non_losing_escape = True
            continue

        p0_ord = _PIECE_CANONICAL_ORDER.get(type(w_pieces[0]), 99)
        p1_ord = _PIECE_CANONICAL_ORDER.get(type(w_pieces[1]), 99)
        if p0_ord > p1_ord:
            w_pieces[0], w_pieces[1] = w_pieces[1], w_pieces[0]

        c0 = child.white_king_pos[0] * 8 + child.white_king_pos[1]
        c1 = w_pieces[0].pos[0] * 8 + w_pieces[0].pos[1]
        c2 = w_pieces[1].pos[0] * 8 + w_pieces[1].pos[1]
        c3 = child.black_king_pos[0] * 8 + child.black_king_pos[1]

        if _W4_SAME_PIECE and c1 > c2:
            c1, c2 = c2, c1
        child_flats.append(_flat_idx_raw_4(c0, c1, c2, c3, opp_turn_idx))

    return ('b', legal_moves_count, has_non_losing_escape, _pack_uint_list(child_flats))


def _gen_valid_4man_indices_numpy(total_positions, p1_name, p2_name, same_piece, chunk_size=8_000_000):
    for start in range(0, total_positions, chunk_size):
        end_c = min(start + chunk_size, total_positions)
        flat = np.arange(start, end_c, dtype=np.int64)

        t_arr  = flat % 2;       rest = flat // 2
        bk_arr = rest % 64;      rest //= 64
        p2_arr = rest % 64;      rest //= 64
        p1_arr = rest % 64
        wk_arr = rest // 64

        mask = ((wk_arr != p1_arr) & (wk_arr != p2_arr) & (wk_arr != bk_arr) &
                (p1_arr != p2_arr) & (p1_arr != bk_arr) & (p2_arr != bk_arr))

        if p1_name == "Pawn": mask &= (p1_arr >= 8) & (p1_arr < 56)
        if p2_name == "Pawn": mask &= (p2_arr >= 8) & (p2_arr < 56)
        if same_piece: mask &= (p1_arr <= p2_arr)

        if not mask.any(): continue

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

        cpu_default = max(1, (os.cpu_count() or 2) - TB_THREADS_SUBTRACT)
        self.transition_workers = cpu_default
        self.sim_board = Board(setup=False)
        self.wk_obj = King('white')
        self.p1_obj = self.p1_class('white')
        self.p2_obj = self.p2_class('white')
        self.bk_obj = King('black')

    def _is_retro_legal_flat(self, flat):
        i4 = flat % 2
        rest = flat // 2
        i3 = rest % 64
        rest //= 64
        i2 = rest % 64
        rest //= 64
        i1 = rest % 64
        i0 = rest // 64

        wk = (i0 // 8, i0 % 8)
        p1 = (i1 // 8, i1 % 8)
        p2 = (i2 // 8, i2 % 8)
        bk = (i3 // 8, i3 % 8)

        board = self.sim_board
        board.grid = [[None for _ in range(8)] for _ in range(8)]
        self.wk_obj.pos = wk
        self.p1_obj.pos = p1
        self.p2_obj.pos = p2
        self.bk_obj.pos = bk
        board.grid[wk[0]][wk[1]] = self.wk_obj
        board.grid[p1[0]][p1[1]] = self.p1_obj
        board.grid[p2[0]][p2[1]] = self.p2_obj
        board.grid[bk[0]][bk[1]] = self.bk_obj
        board.white_king_pos = wk
        board.black_king_pos = bk
        board.white_pieces = [self.wk_obj, self.p1_obj, self.p2_obj]
        board.black_pieces = [self.bk_obj]
        board.piece_counts = {
            'white': {Pawn:0, Knight:0, Bishop:0, Rook:0, Queen:0, King:1},
            'black': {Pawn:0, Knight:0, Bishop:0, Rook:0, Queen:0, King:1}
        }
        board.piece_counts['white'][type(self.p1_obj)] += 1
        board.piece_counts['white'][type(self.p2_obj)] += 1

        # Retro-legality: the side that just moved (opponent of side-to-move) cannot be in check.
        passive_color = 'black' if i4 == 0 else 'white'
        return not is_in_check(board, passive_color)

    def generate(self):
        if os.path.exists(self.filename):
            print(f"[SKIP] {self.filename} already exists.")
            return

        start_time = time.time()
        print(f"\n{'='*60}\n GENERATING: King + {self.p1_name} + {self.p2_name} vs King")
        if self.same_piece:
            print(f" [Symmetry] Same-piece table -- using canonical half (p1<=p2).")
        print(f"{'='*60}")

        print(f"[Phase 1] Scanning positions (numpy-chunked)...")
        phase1_start = time.time()
        unsolved_flats = array('I')
        for chunk in _gen_valid_4man_indices_numpy(
                self.total_positions, self.p1_name, self.p2_name, self.same_piece
        ):
            unsolved_flats.extend(chunk)
        phase1_elapsed = time.time() - phase1_start
        print(f"[Phase 1] Found {len(unsolved_flats):,} candidate positions in "
              f"{phase1_elapsed:.1f}s.")

        if ENABLE_RETRO_FILTER_4MAN:
            print(f"[Phase 1.1] Applying retro-legality filter (4-man)...")
            filter_start = time.time()
            filtered = array('I')
            retro_illegal = 0
            total = len(unsolved_flats)
            for done, flat in enumerate(unsolved_flats, start=1):
                if self._is_retro_legal_flat(flat):
                    filtered.append(flat)
                else:
                    retro_illegal += 1
                if done % 500000 == 0:
                    pct = done / total * 100.0
                    print(f"  > Retro filter: {pct:.1f}% ({done:,}/{total:,})", end='\r')
            if total >= 500000:
                print()
            unsolved_flats = filtered
            print(f"[Phase 1.1] Retro-illegal discarded: {retro_illegal:,} | "
                  f"Remaining: {len(unsolved_flats):,} | "
                  f"Time: {time.time()-filter_start:.1f}s.\n")
        else:
            print()

        candidate_states = len(unsolved_flats)

        print(f"[Phase 1.5] Building transition cache ({len(unsolved_flats):,} states)...")
        cache_start = time.time()
        transitions = []
        with ProcessPoolExecutor(
            max_workers=self.transition_workers,
            initializer=_init_transition_worker_4,
            initargs=(self.p1_name, self.p2_name, self.promo_tb_file)
        ) as ex:
            for done, trans in enumerate(
                ex.map(_build_transition_worker_4, unsolved_flats, chunksize=4096), 1
            ):
                transitions.append(trans)
                if done % 100_000 == 0:
                    elapsed = max(0.001, time.time() - cache_start)
                    speed = done / elapsed
                    eta = (len(unsolved_flats) - done) / speed
                    print(f"  > Cache: {done/len(unsolved_flats)*100:.1f}% | "
                          f"{speed:,.0f} st/s | ETA: {eta/60:.1f}m", end='\r', flush=True)
        print(f"\n[Phase 1.5] Cache built in {(time.time()-cache_start)/60:.1f}m.")

        # ----------------------------------------------------------------
        # Phase 2: Pre-solve initial no-legal-move losses
        # ----------------------------------------------------------------
        print(f"\n[Phase 2] Resolving initial no-legal-move losses...")
        pre_solved = 0
        surviving_flats = array('I')
        surviving_transitions = []
        table_flat = self.table.reshape(-1)
        for flat, trans in zip(unsolved_flats, transitions):
            if (flat & 1) == 1:  # black's turn
                legal_moves_count = trans[1]
                if legal_moves_count == 0:
                    table_flat[flat] = -1
                    pre_solved += 1
                    continue
                         
            surviving_flats.append(flat)
            surviving_transitions.append(trans)

        unsolved_flats = surviving_flats
        transitions = surviving_transitions
        print(f"[Phase 2] Pre-solved {pre_solved:,} positions.")

        iteration = 1
        max_dtm_solved = 0
        while True:
            iter_start = time.time()
            changed = 0
            new_unsolved = array('I')
            new_transitions =[]
            prev_flat = self.table.reshape(-1).copy()

            print(f"\n[Iteration {iteration}] (DTM {iteration})")

            for flat, transition in zip(unsolved_flats, transitions):
                t_idx = flat & 1

                if t_idx == 0:
                    best_win = 0
                    _, immediate_win, known_win_packed, child_packed = transition

                    if immediate_win:
                        best_win = 1
                    for val in _unpack_uint_list(known_win_packed):
                        if best_win == 0 or val < best_win:
                            best_win = val
                    for cflat in _unpack_uint_list(child_packed):
                        res_val = int(prev_flat[cflat])
                        if res_val < 0:
                            val = abs(res_val) + 1
                            if best_win == 0 or val < best_win:
                                best_win = val

                    if best_win > 0:
                        table_flat[flat] = best_win
                        changed += 1
                    else:
                        new_unsolved.append(flat)
                        new_transitions.append(transition)

                else:
                    _, legal_moves_count, has_non_losing_escape, child_packed = transition
                    
                    if legal_moves_count == 0:
                        table_flat[flat] = -1
                        changed += 1
                        continue

                    all_moves_lead_to_win = not has_non_losing_escape
                    max_win_val = 0

                    if all_moves_lead_to_win:
                        for cflat in _unpack_uint_list(child_packed):
                            res_val = int(prev_flat[cflat])
                            if res_val <= 0:
                                all_moves_lead_to_win = False
                                break
                            if res_val > max_win_val:
                                max_win_val = res_val

                    if all_moves_lead_to_win:
                        table_flat[flat] = -(max_win_val + 1)
                        changed += 1
                    else:
                        new_unsolved.append(flat)
                        new_transitions.append(transition)

            unsolved_flats = new_unsolved
            transitions = new_transitions
            total_solved = int((self.table != 0).sum())
            print(f"  > Summary: {changed:,} new solved | "
                  f"Total: {total_solved:,} | "
                  f"Remaining: {len(unsolved_flats):,} | "
                  f"Time: {time.time()-iter_start:.1f}s")
            if changed > 0:
                max_dtm_solved = iteration
            if changed == 0:
                break
            iteration += 1

        with open(self.filename, 'wb') as f:
            self.table.tofile(f)
        elapsed = time.time() - start_time
        remaining = len(unsolved_flats)
        decisive = candidate_states - remaining
        _safe_record_longest_mate(
            f"K_{self.p1_name}_{self.p2_name}_K",
            max_dtm_solved,
            decisive,
            remaining,
            elapsed
        )
        print(f"[LongestMate] K_{self.p1_name}_{self.p2_name}_K: max_dtm={max_dtm_solved}, decisive={decisive:,}, remaining={remaining:,}")
        print(f"\nSUCCESS: Generated in {elapsed/60:.1f} minutes.")

# ==============================================================================
# 4-MAN CROSS GENERATOR (K + WPiece vs K + BPiece)
# ==============================================================================

_WV_W_NAME = None
_WV_B_NAME = None
_WV_W_CLASS = None
_WV_B_CLASS = None
_WV_3MAN_TABLES = {}
_WV_PROMO_TABLES = {}
_IN_TABLE_SENTINEL = "IN_TABLE"

def _tb_file_4man_vs(w_name, b_name):
    return os.path.join(TB_DIR, f"K_{w_name}_vs_{b_name}_K.bin")

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
    _WV_W_NAME = w_name
    _WV_B_NAME = b_name
    _WV_W_CLASS = PIECE_CLASS_BY_NAME[w_name]
    _WV_B_CLASS = PIECE_CLASS_BY_NAME[b_name]

    _WV_3MAN_TABLES = {}
    names_needed = {w_name, b_name}
    if w_name == "Pawn" or b_name == "Pawn":
        names_needed.add("Queen")
    for name in names_needed:
        path = os.path.join(TB_DIR, f"K_{name}_K.bin")
        if os.path.exists(path):
            _WV_3MAN_TABLES[name] = np.memmap(path, dtype=np.int16, mode='r', shape=(64, 64, 64, 2))

    _WV_PROMO_TABLES = {}
    promo_targets = set()
    if w_name == "Pawn":
        promo_targets.add(("Queen", b_name))
    if b_name == "Pawn":
        promo_targets.add((w_name, "Queen"))
    for key in promo_targets:
        path = _tb_file_4man_vs(key[0], key[1])
        if os.path.exists(path):
            _WV_PROMO_TABLES[key] = np.memmap(path, dtype=np.int16, mode='r', shape=(64, 64, 64, 64, 2))

def _white_win_dtm_3man_white_piece(piece_name, wk, wp, bk, turn_idx):
    tb = _WV_3MAN_TABLES.get(piece_name)
    if tb is None:
        return None
    q_idx = (wk[0] * 8 + wk[1], wp[0] * 8 + wp[1], bk[0] * 8 + bk[1], turn_idx)
    return _white_win_dtm_from_raw_tb_value(int(tb[q_idx]), turn_idx)

def _white_win_dtm_3man_black_piece(piece_name, wk, bk, bp, turn_idx):
    tb = _WV_3MAN_TABLES.get(piece_name)
    if tb is None:
        return None
    def flip(pos):
        return (7 - pos[0], pos[1])
    # In transformed attacker table: attacker is side with the black piece.
    t_turn_idx = 0 if turn_idx == 1 else 1
    atk_k = flip(bk)
    atk_p = flip(bp)
    def_k = flip(wk)
    q_idx = (atk_k[0] * 8 + atk_k[1], atk_p[0] * 8 + atk_p[1], def_k[0] * 8 + def_k[1], t_turn_idx)
    val = int(tb[q_idx])
    if val == 0:
        return None
    attacker_wins = (t_turn_idx == 0 and val > 0) or (t_turn_idx == 1 and val < 0)
    if attacker_wins:
        return None
    return abs(val)

def _white_win_dtm_promo_vs(w_name, b_name, wk, wp, bk, bp, turn_idx):
    tb = _WV_PROMO_TABLES.get((w_name, b_name))
    if tb is None:
        return None
    q_idx = (wk[0] * 8 + wk[1], wp[0] * 8 + wp[1], bk[0] * 8 + bk[1], bp[0] * 8 + bp[1], turn_idx)
    val = int(tb.flat[_flat_idx_raw_4(*q_idx)])
    return _white_win_dtm_from_raw_tb_value(val, turn_idx)

def _white_win_dtm_kvk(child, turn_idx):
    if not child.white_king_pos:
        return None
    if not child.black_king_pos:
        return 1
    turn_color = 'white' if turn_idx == 0 else 'black'
    if has_legal_moves(child, turn_color):
        return None
    return 1 if turn_color == 'black' else None

def _external_white_win_dtm_4vs(child, turn_idx):
    white_non_king = [p for p in child.white_pieces if not isinstance(p, King)]
    black_non_king = [p for p in child.black_pieces if not isinstance(p, King)]

    if len(white_non_king) == 1 and len(black_non_king) == 1:
        wn = type(white_non_king[0]).__name__
        bn = type(black_non_king[0]).__name__
        if wn == _WV_W_NAME and bn == _WV_B_NAME:
            return _IN_TABLE_SENTINEL
        return _white_win_dtm_promo_vs(
            wn, bn, child.white_king_pos, white_non_king[0].pos, child.black_king_pos, black_non_king[0].pos, turn_idx
        )

    if len(white_non_king) == 1 and len(black_non_king) == 0:
        return _white_win_dtm_3man_white_piece(
            type(white_non_king[0]).__name__, child.white_king_pos, white_non_king[0].pos, child.black_king_pos, turn_idx
        )

    if len(white_non_king) == 0 and len(black_non_king) == 1:
        return _white_win_dtm_3man_black_piece(
            type(black_non_king[0]).__name__, child.white_king_pos, child.black_king_pos, black_non_king[0].pos, turn_idx
        )

    if len(white_non_king) == 0 and len(black_non_king) == 0:
        return _white_win_dtm_kvk(child, turn_idx)

    return None

def _build_transition_worker_4vs(flat):
    i4 = flat % 2
    rest = flat // 2
    i3 = rest % 64
    rest //= 64
    i2 = rest % 64
    rest //= 64
    i1 = rest % 64
    i0 = rest // 64

    wk = (i0 // 8, i0 % 8)
    wp = (i1 // 8, i1 % 8)
    bk = (i2 // 8, i2 % 8)
    bp = (i3 // 8, i3 % 8)
    turn_is_white = (i4 == 0)
    opp_turn_idx = 1 - i4

    board = Board(setup=False)
    board.add_piece(King('white'), wk[0], wk[1])
    board.add_piece(_WV_W_CLASS('white'), wp[0], wp[1])
    board.add_piece(King('black'), bk[0], bk[1])
    board.add_piece(_WV_B_CLASS('black'), bp[0], bp[1])
    moves = get_all_pseudo_legal_moves(board, 'white' if turn_is_white else 'black')

    if turn_is_white:
        immediate_win = False
        known_win_vals = []
        child_flats = []

        for start, end in moves:
            moving_piece = board.grid[start[0]][start[1]]
            if isinstance(moving_piece, King) and end != bk:
                if _jung_king_threatens(board, end[0], end[1], bk[0], bk[1], vacated=start):
                    continue
            child = board.clone()
            child.make_move(start, end)

            if not child.white_king_pos:
                continue
            if is_in_check(child, 'white'):
                continue
            if child.white_king_pos and child.black_king_pos:
                if _jung_king_threatens(child, child.white_king_pos[0], child.white_king_pos[1],
                                         child.black_king_pos[0], child.black_king_pos[1]):
                    continue

            if not child.black_king_pos:
                immediate_win = True
                continue

            ext_dtm = _external_white_win_dtm_4vs(child, opp_turn_idx)
            if ext_dtm == _IN_TABLE_SENTINEL:
                w_piece = next((p for p in child.white_pieces if not isinstance(p, King)), None)
                b_piece = next((p for p in child.black_pieces if not isinstance(p, King)), None)
                if w_piece is None or b_piece is None:
                    continue
                c0 = child.white_king_pos[0] * 8 + child.white_king_pos[1]
                c1 = w_piece.pos[0] * 8 + w_piece.pos[1]
                c2 = child.black_king_pos[0] * 8 + child.black_king_pos[1]
                c3 = b_piece.pos[0] * 8 + b_piece.pos[1]
                child_flats.append(_flat_idx_raw_4(c0, c1, c2, c3, opp_turn_idx))
            elif ext_dtm is not None and ext_dtm > 0:
                known_win_vals.append(ext_dtm + 1)

        return ('w', immediate_win, _pack_uint_list(known_win_vals), _pack_uint_list(child_flats))

    legal_moves_count = 0
    has_non_losing_escape = False
    known_win_vals = []
    child_flats = []

    for start, end in moves:
        moving_piece = board.grid[start[0]][start[1]]
        if isinstance(moving_piece, King) and end != wk:
            if _jung_king_threatens(board, end[0], end[1], wk[0], wk[1], vacated=start):
                continue
        child = board.clone()
        child.make_move(start, end)

        if not child.black_king_pos:
            continue
        if is_in_check(child, 'black'):
            continue

        legal_moves_count += 1
        if not child.white_king_pos:
            has_non_losing_escape = True
            continue

        ext_dtm = _external_white_win_dtm_4vs(child, opp_turn_idx)
        if ext_dtm == _IN_TABLE_SENTINEL:
            w_piece = next((p for p in child.white_pieces if not isinstance(p, King)), None)
            b_piece = next((p for p in child.black_pieces if not isinstance(p, King)), None)
            if w_piece is None or b_piece is None:
                has_non_losing_escape = True
                continue
            c0 = child.white_king_pos[0] * 8 + child.white_king_pos[1]
            c1 = w_piece.pos[0] * 8 + w_piece.pos[1]
            c2 = child.black_king_pos[0] * 8 + child.black_king_pos[1]
            c3 = b_piece.pos[0] * 8 + b_piece.pos[1]
            child_flats.append(_flat_idx_raw_4(c0, c1, c2, c3, opp_turn_idx))
        elif ext_dtm is not None and ext_dtm > 0:
            known_win_vals.append(ext_dtm)
        else:
            has_non_losing_escape = True

    return ('b', legal_moves_count, has_non_losing_escape, _pack_uint_list(known_win_vals), _pack_uint_list(child_flats))

def _gen_valid_4man_vs_indices_numpy(total_positions, w_name, b_name, chunk_size=8_000_000):
    for start in range(0, total_positions, chunk_size):
        end_c = min(start + chunk_size, total_positions)
        flat = np.arange(start, end_c, dtype=np.int64)

        t_arr = flat % 2
        rest = flat // 2
        bp_arr = rest % 64
        rest //= 64
        bk_arr = rest % 64
        rest //= 64
        wp_arr = rest % 64
        wk_arr = rest // 64

        mask = ((wk_arr != wp_arr) & (wk_arr != bk_arr) & (wk_arr != bp_arr) &
                (wp_arr != bk_arr) & (wp_arr != bp_arr) & (bk_arr != bp_arr))
        if w_name == "Pawn":
            mask &= (wp_arr >= 8) & (wp_arr < 56)
        if b_name == "Pawn":
            mask &= (bp_arr >= 8) & (bp_arr < 56)

        if not mask.any():
            continue
        yield flat[mask].tolist()

class Generator4Vs:
    def __init__(self, w_piece_class, b_piece_class):
        self.w_name = w_piece_class.__name__
        self.b_name = b_piece_class.__name__
        self.filename = _tb_file_4man_vs(self.w_name, self.b_name)
        self.total_positions = 64 * 64 * 64 * 64 * 2
        self.table = np.zeros((64, 64, 64, 64, 2), dtype=np.int16)
        cpu_default = max(1, (os.cpu_count() or 2) - TB_THREADS_SUBTRACT)
        self.transition_workers = cpu_default

    def generate(self):
        if os.path.exists(self.filename):
            print(f"[SKIP] {self.filename} already exists.")
            return

        start_time = time.time()
        print(f"\n{'='*60}\n GENERATING: King + {self.w_name} vs King + {self.b_name}\n{'='*60}")

        print(f"[Phase 1] Scanning positions (numpy-chunked)...")
        phase1_start = time.time()
        unsolved_flats = array('I')
        for chunk in _gen_valid_4man_vs_indices_numpy(self.total_positions, self.w_name, self.b_name):
            unsolved_flats.extend(chunk)
        print(f"[Phase 1] Found {len(unsolved_flats):,} candidate positions in {time.time()-phase1_start:.1f}s.\n")
        candidate_states = len(unsolved_flats)

        print(f"[Phase 1.5] Building transition cache ({len(unsolved_flats):,} states)...")
        cache_start = time.time()
        transitions = []
        with ProcessPoolExecutor(
            max_workers=self.transition_workers,
            initializer=_init_transition_worker_4vs,
            initargs=(self.w_name, self.b_name)
        ) as ex:
            for done, trans in enumerate(ex.map(_build_transition_worker_4vs, unsolved_flats, chunksize=4096), 1):
                transitions.append(trans)
                if done % 100_000 == 0:
                    elapsed = max(0.001, time.time() - cache_start)
                    speed = done / elapsed
                    eta = (len(unsolved_flats) - done) / speed
                    print(f"  > Cache: {done/len(unsolved_flats)*100:.1f}% | {speed:,.0f} st/s | ETA: {eta/60:.1f}m", end='\r', flush=True)
        print(f"\n[Phase 1.5] Cache built in {(time.time()-cache_start)/60:.1f}m.")

        print(f"\n[Phase 2] Resolving initial no-legal-move losses...")
        pre_solved = 0
        surviving_flats = array('I')
        surviving_transitions = []
        table_flat = self.table.reshape(-1)
        for flat, trans in zip(unsolved_flats, transitions):
            if (flat & 1) == 1:
                legal_moves_count = trans[1]
                if legal_moves_count == 0:
                    table_flat[flat] = -1
                    pre_solved += 1
                    continue
            surviving_flats.append(flat)
            surviving_transitions.append(trans)
        unsolved_flats = surviving_flats
        transitions = surviving_transitions
        print(f"[Phase 2] Pre-solved {pre_solved:,} positions.")

        iteration = 1
        max_dtm_solved = 0
        while True:
            iter_start = time.time()
            changed = 0
            new_unsolved = array('I')
            new_transitions = []
            prev_flat = self.table.reshape(-1).copy()

            print(f"\n[Iteration {iteration}] (DTM {iteration})")
            for flat, transition in zip(unsolved_flats, transitions):
                t_idx = flat & 1
                if t_idx == 0:
                    best_win = 0
                    _, immediate_win, known_win_packed, child_packed = transition
                    if immediate_win:
                        best_win = 1
                    for val in _unpack_uint_list(known_win_packed):
                        if best_win == 0 or val < best_win:
                            best_win = val
                    for cflat in _unpack_uint_list(child_packed):
                        res_val = int(prev_flat[cflat])
                        if res_val < 0:
                            val = abs(res_val) + 1
                            if best_win == 0 or val < best_win:
                                best_win = val
                    if best_win > 0:
                        table_flat[flat] = best_win
                        changed += 1
                    else:
                        new_unsolved.append(flat)
                        new_transitions.append(transition)
                else:
                    _, legal_moves_count, has_non_losing_escape, known_win_packed, child_packed = transition
                    if legal_moves_count == 0:
                        table_flat[flat] = -1
                        changed += 1
                        continue
                    all_moves_lead_to_win = not has_non_losing_escape
                    max_win_val = 0
                    if all_moves_lead_to_win:
                        for val in _unpack_uint_list(known_win_packed):
                            if val > max_win_val:
                                max_win_val = val
                        for cflat in _unpack_uint_list(child_packed):
                            res_val = int(prev_flat[cflat])
                            if res_val <= 0:
                                all_moves_lead_to_win = False
                                break
                            if res_val > max_win_val:
                                max_win_val = res_val
                    if all_moves_lead_to_win:
                        table_flat[flat] = -(max_win_val + 1)
                        changed += 1
                    else:
                        new_unsolved.append(flat)
                        new_transitions.append(transition)

            unsolved_flats = new_unsolved
            transitions = new_transitions
            total_solved = int((self.table != 0).sum())
            print(f"  > Summary: {changed:,} new solved | Total: {total_solved:,} | Remaining: {len(unsolved_flats):,} | Time: {time.time()-iter_start:.1f}s")
            if changed > 0:
                max_dtm_solved = iteration
            if changed == 0:
                break
            iteration += 1

        with open(self.filename, 'wb') as f:
            self.table.tofile(f)
        elapsed = time.time() - start_time
        remaining = len(unsolved_flats)
        decisive = candidate_states - remaining
        _safe_record_longest_mate(
            f"K_{self.w_name}_vs_{self.b_name}_K",
            max_dtm_solved,
            decisive,
            remaining,
            elapsed
        )
        print(f"[LongestMate] K_{self.w_name}_vs_{self.b_name}_K: max_dtm={max_dtm_solved}, decisive={decisive:,}, remaining={remaining:,}")
        print(f"\nSUCCESS: Generated in {elapsed/60:.1f} minutes.")

def _promo_tb_file_for_4man(p1_name, p2_name):
    if p1_name == "Pawn" or p2_name == "Pawn":
        other = p2_name if p1_name == "Pawn" else p1_name
        if p1_name == "Pawn" and p2_name == "Pawn":
            p_names = sorted(["Pawn", "Queen"])
        else:
            p_names = sorted(["Queen", other])
        return os.path.join(TB_DIR, f"K_{p_names[0]}_{p_names[1]}_K.bin")
    return None

def _sample_valid_flats_4man(p1_name, p2_name, same_piece, sample):
    flats = []
    for chunk in _gen_valid_4man_indices_numpy(64 * 64 * 64 * 64 * 2, p1_name, p2_name, same_piece):
        flats.extend(chunk)
        if len(flats) >= sample:
            return flats[:sample]
    return flats

def profile_4man(p1_name, p2_name, sample=2000):
    # Profile a small sample of transition builds to locate hotspots.
    names = sorted([p1_name, p2_name])
    p1_name, p2_name = names[0], names[1]
    same_piece = (p1_name == p2_name)
    promo_tb = _promo_tb_file_for_4man(p1_name, p2_name)
    _init_transition_worker_4(p1_name, p2_name, promo_tb)

    flats = _sample_valid_flats_4man(p1_name, p2_name, same_piece, sample)
    if not flats:
        print("No sample flats found.")
        return

    pr = cProfile.Profile()
    pr.enable()
    for flat in flats:
        _build_transition_worker_4(flat)
    pr.disable()

    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats('cumtime').print_stats(25)
    print(s.getvalue())

if __name__ == "__main__":
    _install_main_interrupt_ignores()
    print("=== Tablebase Generator (v5.10: always-loss cleanup + simpler transition tuples) ===")
    mode = input(
        "1. Generate 3-Man Tables\n"
        "2. Generate 4-Man Tables (K + 2 pieces vs K)\n"
        "3. Generate 4-Man Cross Tables (K + X vs K + Y)\n"
        "4. Profile 4-Man (sample)\nSelect: "
    )

    if mode == '1':
        for c in [Queen, Rook, Knight, Bishop, Pawn]:
            if not os.path.exists(os.path.join(TB_DIR, f"K_{c.__name__}_K.bin")):
                Generator(c).generate()
            else:
                print(f"Skipping K_{c.__name__}_K.bin (exists)")

    elif mode == '2':
        pieces = [Queen, Rook, Knight, Bishop, Pawn]
        from itertools import combinations_with_replacement
        for p1, p2 in combinations_with_replacement(pieces, 2):
            Generator4(p1, p2).generate()
    elif mode == '3':
        pieces = [Queen, Rook, Knight, Bishop, Pawn]
        for w in pieces:
            for b in pieces:
                Generator4Vs(w, b).generate()
    elif mode == '4':
        p1 = input("Piece 1 (Queen/Rook/Knight/Bishop/Pawn): ").strip().title()
        p2 = input("Piece 2 (Queen/Rook/Knight/Bishop/Pawn): ").strip().title()
        try:
            sample = int(input("Sample size (e.g. 2000): ").strip())
        except Exception:
            sample = 2000
        if p1 not in PIECE_CLASS_BY_NAME or p2 not in PIECE_CLASS_BY_NAME:
            print("Invalid piece name(s).")
        else:
            profile_4man(p1, p2, sample=sample)
