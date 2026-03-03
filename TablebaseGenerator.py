# TablebaseGenerator.py (v5.22 - Faster 4-man pipeline: flat indices + list cache)
# - Corrects 3-Man Phase 1 efficiency (Claude's note)
# - Maintains "Stalemate = Loss" logic
# - Fully integrated with GameLogic v42+

import os
import time
import __main__
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from GameLogic import *
import struct
from array import array

# --- CONFIGURATION ---
TB_DIR = "tablebases"
os.makedirs(TB_DIR, exist_ok=True)

# Jungle Chess Rule: 0 legal moves is a loss, not a draw.
STALEMATE_IS_LOSS = True 

 # How many CPUs to subtract from the total when choosing worker count.
 # The generator will use max(1, os.cpu_count() - TB_THREADS_SUBTRACT).
 # This lets you specify "total - n" threads by changing this value.
TB_THREADS_SUBTRACT = 2
EXPECTED_TABLE_ENTRIES = 64 * 64 * 64 * 2
PIECE_CLASS_BY_NAME = {
    "Queen": Queen, "Rook": Rook, "Knight": Knight, "Bishop": Bishop, "Pawn": Pawn,
}

# ==============================================================================
# SHARED HELPERS
# ==============================================================================

def _jung_king_threatens(r1, c1, r2, c2):
    """
    Returns True if a Jungle King at (r1,c1) threatens square (r2,c2).
    Correctly excludes Knight-distance offsets (max 2, min 1).
    """
    dr = abs(r2 - r1)
    dc = abs(c2 - c1)
    mx = max(dr, dc)
    mn = min(dr, dc)
    return mx == 1 or (mx == 2 and mn != 1)

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

# ==============================================================================
# 3-MAN GENERATOR
# ==============================================================================

_W_PIECE_NAME = None
_W_PIECE_CLASS = None
_W_QUEEN_TABLE = None

def _init_transition_worker(piece_name, queen_tb_file):
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
                if _jung_king_threatens(end[0], end[1], bk[0], bk[1]):
                    continue

            child = board.clone()
            child.make_move(start, end)
            
            # [Opt-1] Inline Check Validation (Black only has a King)
            wkp = child.white_king_pos
            bkp = child.black_king_pos
            if wkp and bkp and _jung_king_threatens(wkp[0], wkp[1], bkp[0], bkp[1]):
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

    # is_cm calculation for retrograde loop
    is_cm = (legal_moves_count == 0) and is_in_check(board, 'black')
    return _flat_idx_raw(i0, i1, i2, i3), ('b', legal_moves_count, has_non_losing_escape, is_cm, _pack_uint_list(child_flats))

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

    def generate(self):
        if os.path.exists(self.filename):
            print(f"[SKIP] {self.filename} already exists.")
            return

        start_time = time.time()
        print(f"\n{'='*60}\n GENERATING: King + {self.piece_name} vs King\n{'='*60}")
        unsolved_indices = []

        print(f"[Phase 1] Initialization...")
        for count, idx in enumerate(np.ndindex(self.table.shape)):
            if count % 200000 == 0 and count > 0: 
                print(f"  > Scanned {count}/{self.total_positions} positions...", end='\r')

            wk, wp, bk, t_idx = self.decode(idx)
            if wk == wp or wk == bk or wp == bk: continue
            if self.piece_name == "Pawn" and (wp[0] == 0 or wp[0] == 7): continue
            
            # [Optimization restored] Handle Phase 1 checkmates locally to save worker time
            self.setup_sim(wk, wp, bk)
            if t_idx == 1: # Black
                if not self.sim_board.black_king_pos:
                    self.table[idx] = -1; continue
                # With STALEMATE_IS_LOSS, we can mark any 0-move state as loss
                if is_in_check(self.sim_board, 'black') and not has_legal_moves(self.sim_board, 'black'):
                    self.table[idx] = -1; continue
                if STALEMATE_IS_LOSS and not has_legal_moves(self.sim_board, 'black'):
                    self.table[idx] = -1; continue
            else: # White
                if not self.sim_board.white_king_pos:
                    self.table[idx] = -1; continue

            unsolved_indices.append(idx)
        print()

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
                    _, legal_moves_count, has_non_losing_escape, is_cm, child_packed = transition
                    
                    if legal_moves_count == 0:
                        if STALEMATE_IS_LOSS or is_cm:
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
            total_solved = (self.table != 0).sum()
            percent = (total_solved / self.total_positions) * 100
            print(f"[{self.piece_name}] Round {iteration} Summary: {changed:,} new solved | Total: {total_solved:,} ({percent:.1f}%) | Time: {time.time()-iter_start:.1f}s")
            
            if changed == 0: break
            iteration += 1

        with open(self.filename, 'wb') as f:
            self.table.tofile(f)
        print(f"\nSUCCESS: Generated in {(time.time() - start_time)/60:.1f} minutes.")


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
                if _jung_king_threatens(end[0], end[1], bk[0], bk[1]):
                    continue

            child = board.clone()
            child.make_move(start, end)

            # [Opt-1] Inline Legality Check (Black only has King)
            wkp = child.white_king_pos
            bkp = child.black_king_pos
            if wkp and bkp and _jung_king_threatens(wkp[0], wkp[1], bkp[0], bkp[1]):
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
        child = board.clone()
        child.make_move(m[0], m[1])
        
        if is_in_check(child, 'black'):
            continue

        legal_moves_count += 1
        if not child.white_king_pos:
            has_non_losing_escape = True
            continue

        w_pieces =[p for p in child.white_pieces if not isinstance(p, King)]

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

    # Claude's Fix: Explicitly track is_cm for the retrograde loop
    is_cm = (legal_moves_count == 0) and is_in_check(board, 'black')
    return ('b', legal_moves_count, has_non_losing_escape, is_cm, _pack_uint_list(child_flats))


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
        print(f"[Phase 1] Found {len(unsolved_flats):,} candidate positions in "
              f"{time.time()-phase1_start:.1f}s.\n")

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
        # Phase 2: Pre-solve initial checkmates AND stalemates
        # ----------------------------------------------------------------
        print(f"\n[Phase 2] Resolving initial checkmates/stalemates...")
        pre_solved = 0
        surviving_flats = array('I')
        surviving_transitions = []
        table_flat = self.table.reshape(-1)
        for flat, trans in zip(unsolved_flats, transitions):
            if (flat & 1) == 1:  # black's turn
                # Transition: ('b', lmc, hnle, is_checkmate, packed)
                legal_moves_count = trans[1]
                is_cm = trans[3]
                
                # STALEMATE = LOSS logic applied here
                if legal_moves_count == 0:
                    if STALEMATE_IS_LOSS or is_cm:
                        table_flat[flat] = -1
                        pre_solved += 1
                        continue
                         
            surviving_flats.append(flat)
            surviving_transitions.append(trans)

        unsolved_flats = surviving_flats
        transitions = surviving_transitions
        print(f"[Phase 2] Pre-solved {pre_solved:,} positions.")

        iteration = 1
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
                    _, legal_moves_count, has_non_losing_escape, is_cm, child_packed = transition
                    
                    # Already handled in Phase 2, but safe to keep here
                    if legal_moves_count == 0:
                        if STALEMATE_IS_LOSS or is_cm:
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
            
            if changed == 0:
                break
            iteration += 1

        with open(self.filename, 'wb') as f:
            self.table.tofile(f)
        elapsed = time.time() - start_time
        print(f"\nSUCCESS: Generated in {elapsed/60:.1f} minutes.")


if __name__ == "__main__":
    print("=== Tablebase Generator (v5.2) ===")
    mode = input("1. Generate 3-Man Tables\n2. Generate 4-Man Tables\nSelect: ")

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
