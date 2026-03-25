# TablebaseGenerator.py (v11.1 - SML / Full Symmetry Compression + Stability Fixes)

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

LONGEST_MATES_NOTE_FILE = os.path.join(TB_DIR, "new_longest_mates_sml.tsv")
LONGEST_MATE_KEY_PREFIX = "regen_"

TB_THREADS_SUBTRACT = 2
PIECE_CLASS_BY_NAME = {
    "Queen": Queen, "Rook": Rook, "Knight": Knight, "Bishop": Bishop, "Pawn": Pawn,
}
_PIECE_CANONICAL_ORDER = {Bishop: 0, Knight: 1, Pawn: 2, Queen: 3, Rook: 4}

_CONSOLE_INTERRUPT_SIGNALS = [signal.SIGINT]
if hasattr(signal, "SIGBREAK"):
    _CONSOLE_INTERRUPT_SIGNALS.append(signal.SIGBREAK)

# ==============================================================================
# FAST SYMMETRY MAPPINGS (Shared with TablebaseManager)
# ==============================================================================
SYMMETRY_MAP = [[0]*8 for _ in range(64)]
for r in range(8):
    for c in range(8):
        sq = r * 8 + c
        SYMMETRY_MAP[sq][0] = r * 8 + c
        SYMMETRY_MAP[sq][1] = r * 8 + (7 - c)
        SYMMETRY_MAP[sq][2] = (7 - r) * 8 + c
        SYMMETRY_MAP[sq][3] = (7 - r) * 8 + (7 - c)
        SYMMETRY_MAP[sq][4] = c * 8 + r
        SYMMETRY_MAP[sq][5] = c * 8 + (7 - r)
        SYMMETRY_MAP[sq][6] = (7 - c) * 8 + r
        SYMMETRY_MAP[sq][7] = (7 - c) * 8 + (7 - r)

# Pawn constraints (WK must be c <= 3)
PAWN_VALID_T = [0 if (sq % 8) <= 3 else 1 for sq in range(64)]
PAWN_WK_SQUARES = [r*8+c for r in range(8) for c in range(4)]
PAWN_WK_IDX = {sq: i for i, sq in enumerate(PAWN_WK_SQUARES)}

# Non-Pawn constraints (WK must be in a1-d4 triangle: c<=3, r<=c)
NON_PAWN_WK_SQUARES = []
for c in range(4):
    for r in range(c + 1):
        NON_PAWN_WK_SQUARES.append(r * 8 + c)
NON_PAWN_WK_IDX = {sq: i for i, sq in enumerate(NON_PAWN_WK_SQUARES)}

NON_PAWN_VALID_TS = [[] for _ in range(64)]
for sq in range(64):
    for t in range(8):
        if SYMMETRY_MAP[sq][t] in NON_PAWN_WK_IDX:
            NON_PAWN_VALID_TS[sq].append(t)

def _canonical_flat_3(wk, p1, bk, turn, has_pawn):
    if has_pawn:
        t = PAWN_VALID_T[wk]
        m_wk = PAWN_WK_IDX[SYMMETRY_MAP[wk][t]]
        m_p1, m_bk = SYMMETRY_MAP[p1][t], SYMMETRY_MAP[bk][t]
        return ((m_wk * 64 + m_p1) * 64 + m_bk) * 2 + turn
    else:
        best = None
        for t in NON_PAWN_VALID_TS[wk]:
            m = (NON_PAWN_WK_IDX[SYMMETRY_MAP[wk][t]], SYMMETRY_MAP[p1][t], SYMMETRY_MAP[bk][t])
            if best is None or m < best: best = m
        return ((best[0] * 64 + best[1]) * 64 + best[2]) * 2 + turn

def _canonical_flat_4(wk, p1, p2, bk, turn, has_pawn, same_piece):
    if has_pawn:
        t = PAWN_VALID_T[wk]
        m_wk = PAWN_WK_IDX[SYMMETRY_MAP[wk][t]]
        m_p1, m_p2, m_bk = SYMMETRY_MAP[p1][t], SYMMETRY_MAP[p2][t], SYMMETRY_MAP[bk][t]
        if same_piece and m_p1 > m_p2: m_p1, m_p2 = m_p2, m_p1
        return (((m_wk * 64 + m_p1) * 64 + m_p2) * 64 + m_bk) * 2 + turn
    else:
        best = None
        for t in NON_PAWN_VALID_TS[wk]:
            m_p1, m_p2 = SYMMETRY_MAP[p1][t], SYMMETRY_MAP[p2][t]
            if same_piece and m_p1 > m_p2: m_p1, m_p2 = m_p2, m_p1
            m = (NON_PAWN_WK_IDX[SYMMETRY_MAP[wk][t]], m_p1, m_p2, SYMMETRY_MAP[bk][t])
            if best is None or m < best: best = m
        return (((best[0] * 64 + best[1]) * 64 + best[2]) * 64 + best[3]) * 2 + turn

def _canonical_flat_4vs(wk, wp, bk, bp, turn, has_pawn):
    if has_pawn:
        t = PAWN_VALID_T[wk]
        m_wk = PAWN_WK_IDX[SYMMETRY_MAP[wk][t]]
        m_wp, m_bk, m_bp = SYMMETRY_MAP[wp][t], SYMMETRY_MAP[bk][t], SYMMETRY_MAP[bp][t]
        return (((m_wk * 64 + m_wp) * 64 + m_bk) * 64 + m_bp) * 2 + turn
    else:
        best = None
        for t in NON_PAWN_VALID_TS[wk]:
            m = (NON_PAWN_WK_IDX[SYMMETRY_MAP[wk][t]], SYMMETRY_MAP[wp][t], SYMMETRY_MAP[bk][t], SYMMETRY_MAP[bp][t])
            if best is None or m < best: best = m
        return (((best[0] * 64 + best[1]) * 64 + best[2]) * 64 + best[3]) * 2 + turn


# ==============================================================================
# SHARED HELPERS
# ==============================================================================

def _main_interrupt_handler(signum, frame):
    print("\n[Interrupt ignored] Use Ctrl+C again if stuck.", flush=True)

def _install_main_interrupt_ignores():
    for sig in _CONSOLE_INTERRUPT_SIGNALS:
        try: signal.signal(sig, _main_interrupt_handler)
        except Exception: pass

def _install_worker_interrupt_ignores():
    for sig in _CONSOLE_INTERRUPT_SIGNALS:
        try: signal.signal(sig, signal.SIG_IGN)
        except Exception: pass

def _load_3man_table_file(filename):
    data16 = np.fromfile(filename, dtype=np.int16)
    has_pawn = "Pawn" in filename
    sz = 32 if has_pawn else 10
    return data16.reshape((sz, 64, 64, 2))

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

_W_PIECE_NAME = None; _W_HAS_PAWN = False; _W_QUEEN_TABLE = None
_W3_BOARD = None; _W3_WK_OBJ = None; _W3_WP_OBJ = None; _W3_BK_OBJ = None

def _init_transition_worker(piece_name, queen_tb_file):
    _install_worker_interrupt_ignores()
    global _W_PIECE_NAME, _W_HAS_PAWN, _W_QUEEN_TABLE
    global _W3_BOARD, _W3_WK_OBJ, _W3_WP_OBJ, _W3_BK_OBJ
    _W_PIECE_NAME = piece_name
    _W_HAS_PAWN = (piece_name == "Pawn")
    _W_QUEEN_TABLE = _load_3man_table_file(queen_tb_file) if (_W_HAS_PAWN and queen_tb_file) else None
    
    _W3_BOARD = Board(setup=False)
    _W3_WK_OBJ, _W3_WP_OBJ, _W3_BK_OBJ = King('white'), PIECE_CLASS_BY_NAME[piece_name]('white'), King('black')
    pc = _W3_BOARD.piece_counts
    pc['white'][King] = pc['black'][King] = 1
    pc['white'][type(_W3_WP_OBJ)] += 1

def _build_transition_worker(flat):
    rest = flat // 2
    bk_sq = rest % 64; rest //= 64
    p1_sq = rest % 64; wk_idx = rest // 64
    wk_sq = PAWN_WK_SQUARES[wk_idx] if _W_HAS_PAWN else NON_PAWN_WK_SQUARES[wk_idx]
    
    turn_is_white = (flat % 2 == 0)
    opp_turn_idx = 1 - (flat % 2)

    wk, p1, bk = (wk_sq//8, wk_sq%8), (p1_sq//8, p1_sq%8), (bk_sq//8, bk_sq%8)

    board = _W3_BOARD
    g = board.grid
    for p in (_W3_WK_OBJ, _W3_WP_OBJ, _W3_BK_OBJ):
        if p.pos: g[p.pos[0]][p.pos[1]] = None
    _W3_WK_OBJ.pos, _W3_WP_OBJ.pos, _W3_BK_OBJ.pos = wk, p1, bk
    _W3_WK_OBJ._list_pos, _W3_WP_OBJ._list_pos, _W3_BK_OBJ._list_pos = 0, 1, 0
    g[wk[0]][wk[1]], g[p1[0]][p1[1]], g[bk[0]][bk[1]] = _W3_WK_OBJ, _W3_WP_OBJ, _W3_BK_OBJ
    board.white_king_pos, board.black_king_pos = wk, bk
    board.white_pieces[:] = [_W3_WK_OBJ, _W3_WP_OBJ]
    board.black_pieces[:] = [_W3_BK_OBJ]

    if is_in_check(board, 'black' if turn_is_white else 'white'): return None

    moves = get_all_pseudo_legal_moves(board, 'white' if turn_is_white else 'black')

    if turn_is_white:
        immediate_win, promo_win_vals, child_flats = False, [], []
        for m in moves:
            record = board.make_move_track(m[0], m[1])
            if is_in_check(board, 'white'):
                board.unmake_move(record); continue

            wkp, bkp = board.white_king_pos, board.black_king_pos
            if not bkp or not has_legal_moves(board, 'black'):
                immediate_win = True
                board.unmake_move(record); break

            if _W_HAS_PAWN:
                nk = next((x for x in board.white_pieces if not isinstance(x, King)), None)
                if nk is not None and isinstance(nk, Queen):
                    q_idx = _canonical_flat_3(wkp[0]*8+wkp[1], nk.pos[0]*8+nk.pos[1], bkp[0]*8+bkp[1], 1, False)
                    q_val = int(_W_QUEEN_TABLE.flat[q_idx])
                    if q_val < 0: promo_win_vals.append(abs(q_val) + 1)
                    board.unmake_move(record); continue

            p = next((x for x in board.white_pieces if not isinstance(x, King)), None)
            if p is not None:
                cflat = _canonical_flat_3(wkp[0]*8+wkp[1], p.pos[0]*8+p.pos[1], bkp[0]*8+bkp[1], opp_turn_idx, _W_HAS_PAWN)
                child_flats.append(cflat)
            board.unmake_move(record)
        return (flat, ('w', immediate_win, array('H', promo_win_vals), array('I', child_flats)))

    else:
        legal_moves, escape, child_flats = 0, False, []
        for m in moves:
            record = board.make_move_track(m[0], m[1])
            if is_in_check(board, 'black'):
                board.unmake_move(record); continue

            legal_moves += 1
            if not board.white_king_pos or len(board.white_pieces) < 2 or not has_legal_moves(board, 'white'):
                escape = True
                board.unmake_move(record); continue

            if _W_HAS_PAWN:
                nk = next((x for x in board.white_pieces if not isinstance(x, King)), None)
                if nk is None or isinstance(nk, Queen):
                    escape = True
                    board.unmake_move(record); continue

            p = next((x for x in board.white_pieces if not isinstance(x, King)), None)
            if p is None:
                escape = True
                board.unmake_move(record); continue

            wkp, bkp = board.white_king_pos, board.black_king_pos
            cflat = _canonical_flat_3(wkp[0]*8+wkp[1], p.pos[0]*8+p.pos[1], bkp[0]*8+bkp[1], opp_turn_idx, _W_HAS_PAWN)
            child_flats.append(cflat)
            board.unmake_move(record)
        return (flat, ('b', legal_moves, escape, array('I', child_flats)))

class Generator:
    def __init__(self, piece_class):
        self.piece_name = piece_class.__name__
        self.has_pawn = (self.piece_name == "Pawn")
        self.filename = os.path.join(TB_DIR, f"K_{self.piece_name}_K_sml.bin")
        self.queen_tb_file = os.path.join(TB_DIR, "K_Queen_K_sml.bin")
        
        self.wk_size = 32 if self.has_pawn else 10
        self.total_positions = self.wk_size * 64 * 64 * 2
        self.table = np.zeros((self.wk_size, 64, 64, 2), dtype=np.int16)

        self.transition_workers = min(8, max(1, (os.cpu_count() or 2) - TB_THREADS_SUBTRACT))

    def generate(self):
        if os.path.exists(self.filename):
            print(f"[SKIP] {self.filename} already exists.")
            return

        start_time = time.time()
        print(f"\n{'='*60}\n GENERATING: King + {self.piece_name} vs King (SML)\n{'='*60}")

        print("[Stage 1] Enumerating candidate positions...")
        raw_candidates = []
        
        for flat in range(self.total_positions):
            rest = flat // 2
            bk_sq = rest % 64; rest //= 64
            p1_sq = rest % 64; wk_idx = rest // 64
            wk_sq = PAWN_WK_SQUARES[wk_idx] if self.has_pawn else NON_PAWN_WK_SQUARES[wk_idx]
            
            if wk_sq == p1_sq or wk_sq == bk_sq or p1_sq == bk_sq: continue
            if self.has_pawn and (p1_sq // 8 == 0 or p1_sq // 8 == 7): continue
            raw_candidates.append(flat)

        print(f"[Stage 1] Found {len(raw_candidates):,} structural candidates.")

        print(f"[Stage 2 & 3] Pipelining Legality Filter + Maps...")
        stage23_start = time.time()
        unsolved_flats = array('I')
        transitions = []
        trans_lookup = {}
        btm_to_wtm = defaultdict(list)
        wtm_to_btm = defaultdict(list)
        work_set = set()

        with ProcessPoolExecutor(max_workers=self.transition_workers,
                                 initializer=_init_transition_worker,
                                 initargs=(self.piece_name, self.queen_tb_file)) as ex:
            for done, result in enumerate(ex.map(_build_transition_worker, raw_candidates, chunksize=1024), 1):
                if result is not None:
                    flat, trans = result
                    unsolved_flats.append(flat)
                    transitions.append(trans)
                    trans_lookup[flat] = trans
                    
                    t_idx = flat & 1
                    if t_idx == 0:
                        for cflat in trans[3]: btm_to_wtm[cflat].append(flat)
                        if trans[1] or len(trans[2]) > 0: work_set.add(flat)
                    else:
                        if not trans[2]:
                            for cflat in trans[3]: wtm_to_btm[cflat].append(flat)

                if done % 50000 == 0:
                    speed = done / max(0.001, time.time() - stage23_start)
                    print(f"  > {done / len(raw_candidates) * 100:.1f}% | {speed:,.0f} st/s", end='\r', flush=True)

        candidate_states = len(unsolved_flats)
        print(f"\n[Stage 2 & 3] Valid states: {candidate_states:,} | Time: {time.time()-stage23_start:.1f}s")

        print("[Stage 4 & 5] Retrograde Analysis...")
        table_flat = self.table.reshape(-1)
        
        for flat, trans in zip(unsolved_flats, transitions):
            if (flat & 1) == 1 and trans[1] == 0:
                table_flat[flat] = -1
                for parent in btm_to_wtm.get(flat, []):
                    if table_flat[parent] == 0: work_set.add(parent)

        iteration, max_dtm_solved = 1, 0
        while work_set:
            changed = 0
            next_work_set = set()
            snapshot = table_flat.copy()

            for flat in work_set:
                if table_flat[flat] != 0: continue
                trans = trans_lookup[flat]

                if (flat & 1) == 0:
                    _, imm_win, promo_vals, child_vals = trans
                    best_win = 1 if imm_win else 0
                    for val in promo_vals:
                        if best_win == 0 or val < best_win: best_win = val
                    for cflat in child_vals:
                        res = int(snapshot[cflat])
                        if res < 0:
                            val = abs(res) + 1
                            if best_win == 0 or val < best_win: best_win = val
                    if best_win > 0:
                        table_flat[flat] = best_win; changed += 1
                        for parent in wtm_to_btm.get(flat, []):
                            if table_flat[parent] == 0: next_work_set.add(parent)
                else:
                    _, moves_count, escape, child_vals = trans
                    if moves_count == 0 or escape: continue
                    max_win = 0
                    all_res = True
                    for cflat in child_vals:
                        res = int(snapshot[cflat])
                        if res <= 0: all_res = False; break
                        if res > max_win: max_win = res
                    if all_res:
                        table_flat[flat] = -(max_win + 1); changed += 1
                        for parent in btm_to_wtm.get(flat, []):
                            if table_flat[parent] == 0: next_work_set.add(parent)

            work_set = next_work_set
            if changed > 0: max_dtm_solved = iteration
            iteration += 1

        with open(self.filename, 'wb') as f:
            self.table.tofile(f)
            
        elapsed = time.time() - start_time
        decisive = np.count_nonzero(table_flat)
        _safe_record_longest_mate(f"K_{self.piece_name}_K", max_dtm_solved, decisive, candidate_states - decisive, elapsed)
        print(f"\nSUCCESS: Generated in {elapsed / 60:.1f} minutes.")


# ==============================================================================
# 4-MAN GENERATOR
# ==============================================================================

_W4_HAS_PAWN = False; _W4_SAME_PIECE = False
_W4_P1_NAME = None; _W4_P2_NAME = None
_W4_3MAN_TABLES = {}; _W4_PROMO_TABLE = None
_W4_BOARD = None; _W4_WK_OBJ = None; _W4_P1_OBJ = None; _W4_P2_OBJ = None; _W4_BK_OBJ = None

def _init_transition_worker_4(p1_name, p2_name, promo_tb_file):
    _install_worker_interrupt_ignores()
    global _W4_P1_NAME, _W4_P2_NAME, _W4_HAS_PAWN, _W4_SAME_PIECE
    global _W4_3MAN_TABLES, _W4_PROMO_TABLE
    global _W4_BOARD, _W4_WK_OBJ, _W4_P1_OBJ, _W4_P2_OBJ, _W4_BK_OBJ

    _W4_P1_NAME, _W4_P2_NAME = p1_name, p2_name
    _W4_HAS_PAWN = (p1_name == "Pawn" or p2_name == "Pawn")
    _W4_SAME_PIECE = (p1_name == p2_name)
    _W4_3MAN_TABLES, _W4_PROMO_TABLE = {}, None

    for name in PIECE_CLASS_BY_NAME:
        path = os.path.join(TB_DIR, f"K_{name}_K_sml.bin")
        if os.path.exists(path): _W4_3MAN_TABLES[name] = _load_3man_table_file(path)

    if promo_tb_file and os.path.exists(promo_tb_file):
        data16 = np.fromfile(promo_tb_file, dtype=np.int16)
        sz = 32 if ("Pawn" in os.path.basename(promo_tb_file)) else 10
        _W4_PROMO_TABLE = data16.reshape((sz, 64, 64, 64, 2))

    _W4_BOARD = Board(setup=False)
    _W4_WK_OBJ, _W4_P1_OBJ, _W4_P2_OBJ, _W4_BK_OBJ = King('white'), PIECE_CLASS_BY_NAME[p1_name]('white'), PIECE_CLASS_BY_NAME[p2_name]('white'), King('black')
    pc = _W4_BOARD.piece_counts
    pc['white'][King] = pc['black'][King] = 1
    pc['white'][type(_W4_P1_OBJ)] += 1
    pc['white'][type(_W4_P2_OBJ)] += 1

def _build_transition_worker_4(flat):
    rest = flat // 2
    bk_sq = rest % 64; rest //= 64
    p2_sq = rest % 64; rest //= 64
    p1_sq = rest % 64; wk_idx = rest // 64
    wk_sq = PAWN_WK_SQUARES[wk_idx] if _W4_HAS_PAWN else NON_PAWN_WK_SQUARES[wk_idx]
    turn_is_white = (flat % 2 == 0)
    opp_turn_idx = 1 - (flat % 2)

    wk, p1, p2, bk = (wk_sq//8, wk_sq%8), (p1_sq//8, p1_sq%8), (p2_sq//8, p2_sq%8), (bk_sq//8, bk_sq%8)
    
    board = _W4_BOARD
    g = board.grid
    for p in (_W4_WK_OBJ, _W4_P1_OBJ, _W4_P2_OBJ, _W4_BK_OBJ):
        if p.pos: g[p.pos[0]][p.pos[1]] = None
    _W4_WK_OBJ.pos, _W4_P1_OBJ.pos, _W4_P2_OBJ.pos, _W4_BK_OBJ.pos = wk, p1, p2, bk
    _W4_WK_OBJ._list_pos, _W4_P1_OBJ._list_pos, _W4_P2_OBJ._list_pos, _W4_BK_OBJ._list_pos = 0, 1, 2, 0
    g[wk[0]][wk[1]], g[p1[0]][p1[1]], g[p2[0]][p2[1]], g[bk[0]][bk[1]] = _W4_WK_OBJ, _W4_P1_OBJ, _W4_P2_OBJ, _W4_BK_OBJ
    board.white_king_pos, board.black_king_pos = wk, bk
    board.white_pieces[:] = [_W4_WK_OBJ, _W4_P1_OBJ, _W4_P2_OBJ]
    board.black_pieces[:] = [_W4_BK_OBJ]

    if is_in_check(board, 'black' if turn_is_white else 'white'): return None
    moves = get_all_pseudo_legal_moves(board, 'white' if turn_is_white else 'black')

    if turn_is_white:
        immediate_win, known_wins, child_flats = False, [], []
        for start, end in moves:
            record = board.make_move_track(start, end)
            if is_in_check(board, 'white'):
                board.unmake_move(record); continue

            wkp, bkp = board.white_king_pos, board.black_king_pos
            if not bkp or not has_legal_moves(board, 'black'):
                immediate_win = True
                board.unmake_move(record); break

            w_pieces = [p for p in board.white_pieces if not isinstance(p, King)]
            if len(w_pieces) < 2:
                if len(w_pieces) == 1:
                    rem_name = type(w_pieces[0]).__name__
                    if rem_name in _W4_3MAN_TABLES:
                        q_idx = _canonical_flat_3(wkp[0]*8+wkp[1], w_pieces[0].pos[0]*8+w_pieces[0].pos[1], bkp[0]*8+bkp[1], 1, rem_name=="Pawn")
                        val = int(_W4_3MAN_TABLES[rem_name].flat[q_idx])
                        if val < 0: known_wins.append(abs(val) + 1)
                board.unmake_move(record); continue

            if _PIECE_CANONICAL_ORDER.get(type(w_pieces[0]), 99) > _PIECE_CANONICAL_ORDER.get(type(w_pieces[1]), 99):
                w_pieces[0], w_pieces[1] = w_pieces[1], w_pieces[0]
            n1, n2 = type(w_pieces[0]).__name__, type(w_pieces[1]).__name__

            if n1 != _W4_P1_NAME or n2 != _W4_P2_NAME:
                if _W4_PROMO_TABLE is not None:
                    sq1, sq2 = w_pieces[0].pos[0]*8+w_pieces[0].pos[1], w_pieces[1].pos[0]*8+w_pieces[1].pos[1]
                    q_idx = _canonical_flat_4(wkp[0]*8+wkp[1], sq1, sq2, bkp[0]*8+bkp[1], 1, "Pawn" in {n1, n2}, n1==n2)
                    val = int(_W4_PROMO_TABLE.flat[q_idx])
                    if val < 0: known_wins.append(abs(val) + 1)
                board.unmake_move(record); continue

            cflat = _canonical_flat_4(wkp[0]*8+wkp[1], w_pieces[0].pos[0]*8+w_pieces[0].pos[1], w_pieces[1].pos[0]*8+w_pieces[1].pos[1], bkp[0]*8+bkp[1], opp_turn_idx, _W4_HAS_PAWN, _W4_SAME_PIECE)
            child_flats.append(cflat)
            board.unmake_move(record)
        return (flat, ('w', immediate_win, array('H', known_wins), array('I', child_flats)))

    else:
        legal_moves, escape, known_3man, child_flats = 0, False, [], []
        for start, end in moves:
            record = board.make_move_track(start, end)
            if is_in_check(board, 'black'):
                board.unmake_move(record); continue

            legal_moves += 1
            if not board.white_king_pos or len(board.white_pieces) < 2 or not has_legal_moves(board, 'white'):
                escape = True
                board.unmake_move(record); continue

            w_pieces = [p for p in board.white_pieces if not isinstance(p, King)]
            if len(w_pieces) < 2:
                if len(w_pieces) == 1:
                    rem_name = type(w_pieces[0]).__name__
                    if rem_name in _W4_3MAN_TABLES:
                        wkp, bkp = board.white_king_pos, board.black_king_pos
                        q_idx = _canonical_flat_3(wkp[0]*8+wkp[1], w_pieces[0].pos[0]*8+w_pieces[0].pos[1], bkp[0]*8+bkp[1], 0, rem_name=="Pawn")
                        val = int(_W4_3MAN_TABLES[rem_name].flat[q_idx])
                        if val <= 0: escape = True
                        else: known_3man.append(val)
                else: escape = True
                board.unmake_move(record); continue

            if _PIECE_CANONICAL_ORDER.get(type(w_pieces[0]), 99) > _PIECE_CANONICAL_ORDER.get(type(w_pieces[1]), 99):
                w_pieces[0], w_pieces[1] = w_pieces[1], w_pieces[0]

            cflat = _canonical_flat_4(board.white_king_pos[0]*8+board.white_king_pos[1], w_pieces[0].pos[0]*8+w_pieces[0].pos[1], w_pieces[1].pos[0]*8+w_pieces[1].pos[1], board.black_king_pos[0]*8+board.black_king_pos[1], opp_turn_idx, _W4_HAS_PAWN, _W4_SAME_PIECE)
            child_flats.append(cflat)
            board.unmake_move(record)
        return (flat, ('b', legal_moves, escape, array('H', known_3man), array('I', child_flats)))


def _gen_valid_4man_indices_numpy(total_positions, p1_name, p2_name, same_piece, has_pawn, chunk_size=8_000_000):
    for start in range(0, total_positions, chunk_size):
        end_c = min(start + chunk_size, total_positions)
        flat = np.arange(start, end_c, dtype=np.int64)
        rest = flat // 2
        bk_arr = rest % 64; rest //= 64
        p2_arr = rest % 64; rest //= 64
        p1_arr = rest % 64
        wk_idx_arr = rest // 64
        
        wk_arr = np.array(PAWN_WK_SQUARES if has_pawn else NON_PAWN_WK_SQUARES)[wk_idx_arr]

        mask = ((wk_arr != p1_arr) & (wk_arr != p2_arr) & (wk_arr != bk_arr) &
                (p1_arr != p2_arr) & (p1_arr != bk_arr) & (p2_arr != bk_arr))

        if p1_name == "Pawn": mask &= (p1_arr >= 8) & (p1_arr < 56)
        if p2_name == "Pawn": mask &= (p2_arr >= 8) & (p2_arr < 56)
        if same_piece:        mask &= (p1_arr <= p2_arr)

        if p1_name == "Bishop" and p2_name == "Bishop":
            p1_r, p1_c = p1_arr // 8, p1_arr % 8
            p2_r, p2_c = p2_arr // 8, p2_arr % 8
            mask &= ((p1_r + p1_c) % 2) != ((p2_r + p2_c) % 2)

        if not mask.any(): continue
        yield flat[mask].tolist()

class Generator4:
    def __init__(self, p1_class, p2_class):
        names = sorted([p1_class.__name__, p2_class.__name__])
        self.p1_name, self.p2_name = names[0], names[1]
        self.same_piece = (self.p1_name == self.p2_name)
        self.has_pawn = (self.p1_name == "Pawn" or self.p2_name == "Pawn")
        self.filename = os.path.join(TB_DIR, f"K_{self.p1_name}_{self.p2_name}_K_sml.bin")
        
        self.wk_size = 32 if self.has_pawn else 10
        self.total_positions = self.wk_size * 64 * 64 * 64 * 2
        self.table = np.zeros((self.wk_size, 64, 64, 64, 2), dtype=np.int16)

        self.promo_tb_file = None
        if self.has_pawn:
            other = self.p2_name if self.p1_name == "Pawn" else self.p1_name
            p_names = sorted(["Pawn", "Queen"]) if self.same_piece else sorted(["Queen", other])
            self.promo_tb_file = os.path.join(TB_DIR, f"K_{p_names[0]}_{p_names[1]}_K_sml.bin")

        self.transition_workers = max(1, (os.cpu_count() or 2) - TB_THREADS_SUBTRACT)

    def generate(self):
        if os.path.exists(self.filename):
            print(f"[SKIP] {self.filename} already exists.")
            return

        start_time = time.time()
        print(f"\n{'='*60}\n GENERATING: King + {self.p1_name} + {self.p2_name} vs King (SML)")

        print("[Stage 1] Enumerating structural candidates...")
        all_flats = array('I')
        for chunk in _gen_valid_4man_indices_numpy(self.total_positions, self.p1_name, self.p2_name, self.same_piece, self.has_pawn):
            all_flats.extend(chunk)

        print(f"[Stage 2 & 3] Pipelining Legality Filter + Maps ({len(all_flats):,} states)...")
        stage23_start = time.time()
        unsolved_flats = array('I'); transitions = []
        trans_lookup = {}; btm_to_wtm = defaultdict(list); wtm_to_btm = defaultdict(list)
        work_set = set()

        with ProcessPoolExecutor(max_workers=self.transition_workers,
                                 initializer=_init_transition_worker_4,
                                 initargs=(self.p1_name, self.p2_name, self.promo_tb_file)) as ex:
            for done, result in enumerate(ex.map(_build_transition_worker_4, all_flats, chunksize=4096), 1):
                if result is not None:
                    flat, trans = result
                    unsolved_flats.append(flat); transitions.append(trans); trans_lookup[flat] = trans
                    if (flat & 1) == 0:
                        for cflat in trans[3]: btm_to_wtm[cflat].append(flat)
                        if trans[1] or len(trans[2]) > 0: work_set.add(flat)
                    else:
                        if not trans[2]:
                            for cflat in trans[4]: wtm_to_btm[cflat].append(flat)

                if done % 100_000 == 0:
                    speed = done / max(0.001, time.time() - stage23_start)
                    print(f"  > {done / len(all_flats) * 100:.1f}% | {speed:,.0f} st/s", end='\r', flush=True)

        candidate_states = len(unsolved_flats)
        print(f"\n[Stage 2 & 3] Valid states: {candidate_states:,} | Time: {(time.time()-stage23_start)/60:.1f}m")

        print("[Stage 4 & 5] Retrograde Analysis...")
        table_flat = self.table.reshape(-1)
        for flat, trans in zip(unsolved_flats, transitions):
            if (flat & 1) == 1 and trans[1] == 0:
                table_flat[flat] = -1
                for parent in btm_to_wtm.get(flat, []):
                    if table_flat[parent] == 0: work_set.add(parent)

        iteration, max_dtm_solved = 1, 0
        while work_set:
            changed = 0; next_work_set = set(); snapshot = table_flat.copy()
            for flat in work_set:
                if table_flat[flat] != 0: continue
                trans = trans_lookup[flat]

                if (flat & 1) == 0: # WTM
                    _, imm_win, known_wins, child_vals = trans
                    best_win = 1 if imm_win else 0
                    for val in known_wins:
                        if best_win == 0 or val < best_win: best_win = val
                    for cflat in child_vals:
                        res = int(snapshot[cflat])
                        if res < 0:
                            val = abs(res) + 1
                            if best_win == 0 or val < best_win: best_win = val
                    if best_win > 0:
                        table_flat[flat] = best_win; changed += 1
                        for parent in wtm_to_btm.get(flat, []):
                            if table_flat[parent] == 0: next_work_set.add(parent)
                else: # BTM
                    _, moves_count, escape, known_3man, child_vals = trans
                    if escape or moves_count == 0: continue
                    max_win = 0
                    for val in known_3man:
                        if val > max_win: max_win = val
                    all_res = True
                    for cflat in child_vals:
                        res = int(snapshot[cflat])
                        if res <= 0: all_res = False; break
                        if res > max_win: max_win = res
                    if all_res:
                        table_flat[flat] = -(max_win + 1); changed += 1
                        for parent in btm_to_wtm.get(flat, []):
                            if table_flat[parent] == 0: next_work_set.add(parent)

            work_set = next_work_set
            if changed > 0: max_dtm_solved = iteration
            iteration += 1

        with open(self.filename, 'wb') as f:
            self.table.tofile(f)
            
        elapsed = time.time() - start_time
        decisive = np.count_nonzero(table_flat)
        _safe_record_longest_mate(f"K_{self.p1_name}_{self.p2_name}_K", max_dtm_solved, decisive, candidate_states - decisive, elapsed)
        print(f"\nSUCCESS: Generated in {elapsed / 60:.1f} minutes.")

# ==============================================================================
# 4-MAN CROSS GENERATOR
# ==============================================================================

_W4V_HAS_PAWN = False; _W4V_W_NAME = None; _W4V_B_NAME = None
_W4V_3MAN_TABLES = {}; _W4V_PROMO_TABLES = {}
_W4V_BOARD = None; _W4V_WK_OBJ = None; _W4V_WP_OBJ = None; _W4V_BK_OBJ = None; _W4V_BP_OBJ = None
_IN_TABLE_SENTINEL = "IN_TABLE"

def _init_transition_worker_4vs(w_name, b_name):
    _install_worker_interrupt_ignores()
    global _W4V_W_NAME, _W4V_B_NAME, _W4V_HAS_PAWN
    global _W4V_3MAN_TABLES, _W4V_PROMO_TABLES
    global _W4V_BOARD, _W4V_WK_OBJ, _W4V_WP_OBJ, _W4V_BK_OBJ, _W4V_BP_OBJ

    names = sorted([w_name, b_name])
    _W4V_W_NAME, _W4V_B_NAME = names[0], names[1]
    _W4V_HAS_PAWN = (w_name == "Pawn" or b_name == "Pawn")

    _W4V_3MAN_TABLES = {}
    names_needed = {w_name, b_name}
    if _W4V_HAS_PAWN: names_needed.add("Queen")
    for name in names_needed:
        path = os.path.join(TB_DIR, f"K_{name}_K_sml.bin")
        if os.path.exists(path): _W4V_3MAN_TABLES[name] = _load_3man_table_file(path)

    _W4V_PROMO_TABLES = {}
    promo_targets = set()
    if w_name == "Pawn": promo_targets.add(("Queen", b_name))
    if b_name == "Pawn": promo_targets.add((w_name, "Queen"))
    for key in promo_targets:
        sk = tuple(sorted(key))
        path = os.path.join(TB_DIR, f"K_{sk[0]}_vs_{sk[1]}_K_sml.bin")
        if os.path.exists(path):
            data16 = np.fromfile(path, dtype=np.int16)
            sz = 32 if ("Pawn" in path) else 10
            _W4V_PROMO_TABLES[sk] = data16.reshape((sz, 64, 64, 64, 2))

    _W4V_BOARD = Board(setup=False)
    _W4V_WK_OBJ, _W4V_WP_OBJ = King('white'), PIECE_CLASS_BY_NAME[_W4V_W_NAME]('white')
    _W4V_BK_OBJ, _W4V_BP_OBJ = King('black'), PIECE_CLASS_BY_NAME[_W4V_B_NAME]('black')
    pc = _W4V_BOARD.piece_counts
    pc['white'][King] = pc['black'][King] = 1
    pc['white'][type(_W4V_WP_OBJ)] += 1
    pc['black'][type(_W4V_BP_OBJ)] += 1

def _white_win_dtm_3man(piece_name, wk, p, bk, turn_idx, is_white_piece):
    tb = _W4V_3MAN_TABLES.get(piece_name)
    if tb is None: return None
    
    if is_white_piece:
        q_idx = _canonical_flat_3(wk[0]*8+wk[1], p[0]*8+p[1], bk[0]*8+bk[1], turn_idx, piece_name=="Pawn")
        val = int(tb.flat[q_idx])
        if val == 0: return None
        return abs(val) if ((turn_idx == 0 and val > 0) or (turn_idx == 1 and val < 0)) else None
    else:
        t_turn = 1 - turn_idx
        atk_k, atk_p, def_k = _flip(bk), _flip(p), _flip(wk)
        q_idx = _canonical_flat_3(atk_k[0]*8+atk_k[1], atk_p[0]*8+atk_p[1], def_k[0]*8+def_k[1], t_turn, piece_name=="Pawn")
        val = int(tb.flat[q_idx])
        if val == 0: return None
        b_wins = (t_turn == 0 and val > 0) or (t_turn == 1 and val < 0)
        return None if b_wins else abs(val)

def _white_win_dtm_promo_vs(w_name, b_name, wk, wp, bk, bp, turn_idx):
    key = tuple(sorted((w_name, b_name)))
    tb = _W4V_PROMO_TABLES.get(key)
    if tb is None: return None
    has_pawn = (w_name == "Pawn" or b_name == "Pawn")
    
    if w_name > b_name:
        t_turn = 1 - turn_idx
        bk_f, bp_f, wk_f, wp_f = _flip(bk), _flip(bp), _flip(wk), _flip(wp)
        q_idx = _canonical_flat_4vs(bk_f[0]*8+bk_f[1], bp_f[0]*8+bp_f[1], wk_f[0]*8+wk_f[1], wp_f[0]*8+wp_f[1], t_turn, has_pawn)
        val = int(tb.flat[q_idx])
        if val == 0: return None
        b_wins = (t_turn == 0 and val > 0) or (t_turn == 1 and val < 0)
        return None if b_wins else abs(val)
    else:
        q_idx = _canonical_flat_4vs(wk[0]*8+wk[1], wp[0]*8+wp[1], bk[0]*8+bk[1], bp[0]*8+bp[1], turn_idx, has_pawn)
        val = int(tb.flat[q_idx])
        if val == 0: return None
        return abs(val) if ((turn_idx == 0 and val > 0) or (turn_idx == 1 and val < 0)) else None

def _ext_white_win_4vs(child, turn_idx):
    w_nk = [p for p in child.white_pieces if not isinstance(p, King)]
    b_nk = [p for p in child.black_pieces if not isinstance(p, King)]
    if len(w_nk) == 1 and len(b_nk) == 1:
        wn, bn = type(w_nk[0]).__name__, type(b_nk[0]).__name__
        if wn == _W4V_W_NAME and bn == _W4V_B_NAME: return _IN_TABLE_SENTINEL
        return _white_win_dtm_promo_vs(wn, bn, child.white_king_pos, w_nk[0].pos, child.black_king_pos, b_nk[0].pos, turn_idx)
    if len(w_nk) == 1 and len(b_nk) == 0:
        return _white_win_dtm_3man(type(w_nk[0]).__name__, child.white_king_pos, w_nk[0].pos, child.black_king_pos, turn_idx, True)
    if len(w_nk) == 0 and len(b_nk) == 1:
        return _white_win_dtm_3man(type(b_nk[0]).__name__, child.white_king_pos, b_nk[0].pos, child.black_king_pos, turn_idx, False)
    if len(w_nk) == 0 and len(b_nk) == 0:
        if not child.white_king_pos or not child.black_king_pos: return None
        return 1 if (turn_idx == 1 and not has_legal_moves(child, 'black')) else None
    return None

def _build_transition_worker_4vs(flat):
    rest = flat // 2
    bp_sq = rest % 64; rest //= 64
    bk_sq = rest % 64; rest //= 64
    wp_sq = rest % 64; wk_idx = rest // 64
    wk_sq = PAWN_WK_SQUARES[wk_idx] if _W4V_HAS_PAWN else NON_PAWN_WK_SQUARES[wk_idx]
    
    turn_is_white = (flat % 2 == 0)
    opp_turn_idx = 1 - (flat % 2)

    wk, wp, bk, bp = (wk_sq//8, wk_sq%8), (wp_sq//8, wp_sq%8), (bk_sq//8, bk_sq%8), (bp_sq//8, bp_sq%8)

    board = _W4V_BOARD
    g = board.grid
    for p in (_W4V_WK_OBJ, _W4V_WP_OBJ, _W4V_BK_OBJ, _W4V_BP_OBJ):
        if p.pos: g[p.pos[0]][p.pos[1]] = None
    _W4V_WK_OBJ.pos, _W4V_WP_OBJ.pos = wk, wp
    _W4V_BK_OBJ.pos, _W4V_BP_OBJ.pos = bk, bp
    _W4V_WK_OBJ._list_pos, _W4V_WP_OBJ._list_pos = 0, 1
    _W4V_BK_OBJ._list_pos, _W4V_BP_OBJ._list_pos = 0, 1
    g[wk[0]][wk[1]], g[wp[0]][wp[1]] = _W4V_WK_OBJ, _W4V_WP_OBJ
    g[bk[0]][bk[1]], g[bp[0]][bp[1]] = _W4V_BK_OBJ, _W4V_BP_OBJ
    board.white_king_pos, board.black_king_pos = wk, bk
    board.white_pieces[:] = [_W4V_WK_OBJ, _W4V_WP_OBJ]
    board.black_pieces[:] = [_W4V_BK_OBJ, _W4V_BP_OBJ]

    if is_in_check(board, 'black' if turn_is_white else 'white'): return None
    moves = get_all_pseudo_legal_moves(board, 'white' if turn_is_white else 'black')

    if turn_is_white:
        immediate_win, known_wins, child_flats = False, [], []
        for start, end in moves:
            record = board.make_move_track(start, end)
            if not board.white_king_pos or not board.black_king_pos or is_in_check(board, 'white'):
                board.unmake_move(record); continue
                
            if not has_legal_moves(board, 'black'):
                immediate_win = True
                board.unmake_move(record); break

            ext_dtm = _ext_white_win_4vs(board, opp_turn_idx)
            if ext_dtm == _IN_TABLE_SENTINEL:
                w_piece = next((p for p in board.white_pieces if not isinstance(p, King)), None)
                b_piece = next((p for p in board.black_pieces if not isinstance(p, King)), None)
                if w_piece and b_piece:
                    c0 = board.white_king_pos[0]*8+board.white_king_pos[1]
                    c1 = w_piece.pos[0]*8+w_piece.pos[1]
                    c2 = board.black_king_pos[0]*8+board.black_king_pos[1]
                    c3 = b_piece.pos[0]*8+b_piece.pos[1]
                    child_flats.append(_canonical_flat_4vs(c0,c1,c2,c3,opp_turn_idx, _W4V_HAS_PAWN))
            elif ext_dtm is not None and ext_dtm > 0:
                known_wins.append(ext_dtm + 1)
            board.unmake_move(record)
        return (flat, ('w', immediate_win, array('H', known_wins), array('I', child_flats)))

    else:
        legal_moves, escape, known_wins, child_flats = 0, False, [], []
        for start, end in moves:
            record = board.make_move_track(start, end)
            if not board.black_king_pos or not board.white_king_pos or is_in_check(board, 'black'):
                board.unmake_move(record); continue

            legal_moves += 1
            if not has_legal_moves(board, 'white'):
                escape = True
                board.unmake_move(record); continue

            ext_dtm = _ext_white_win_4vs(board, opp_turn_idx)
            if ext_dtm == _IN_TABLE_SENTINEL:
                w_piece = next((p for p in board.white_pieces if not isinstance(p, King)), None)
                b_piece = next((p for p in board.black_pieces if not isinstance(p, King)), None)
                if not w_piece or not b_piece: escape = True
                else:
                    c0 = board.white_king_pos[0]*8+board.white_king_pos[1]
                    c1 = w_piece.pos[0]*8+w_piece.pos[1]
                    c2 = board.black_king_pos[0]*8+board.black_king_pos[1]
                    c3 = b_piece.pos[0]*8+b_piece.pos[1]
                    child_flats.append(_canonical_flat_4vs(c0,c1,c2,c3,opp_turn_idx, _W4V_HAS_PAWN))
            elif ext_dtm is not None and ext_dtm > 0: known_wins.append(ext_dtm)
            else: escape = True
            board.unmake_move(record)
        return (flat, ('b', legal_moves, escape, array('H', known_wins), array('I', child_flats)))

def _gen_valid_4man_vs_indices_numpy(total_positions, w_name, b_name, has_pawn, chunk_size=8_000_000):
    for start in range(0, total_positions, chunk_size):
        end_c = min(start + chunk_size, total_positions)
        flat = np.arange(start, end_c, dtype=np.int64)
        rest = flat // 2
        bp_arr = rest % 64; rest //= 64
        bk_arr = rest % 64; rest //= 64
        wp_arr = rest % 64
        wk_idx_arr = rest // 64

        wk_arr = np.array(PAWN_WK_SQUARES if has_pawn else NON_PAWN_WK_SQUARES)[wk_idx_arr]

        mask = ((wk_arr != wp_arr) & (wk_arr != bk_arr) & (wk_arr != bp_arr) &
                (wp_arr != bk_arr) & (wp_arr != bp_arr) & (bk_arr != bp_arr))
        if w_name == "Pawn": mask &= (wp_arr >= 8) & (wp_arr < 56)
        if b_name == "Pawn": mask &= (bp_arr >= 8) & (bp_arr < 56)
        if not mask.any(): continue
        yield flat[mask].tolist()

class Generator4Vs:
    def __init__(self, w_piece_class, b_piece_class):
        names = sorted([w_piece_class.__name__, b_piece_class.__name__])
        self.w_name, self.b_name = names[0], names[1]
        self.has_pawn = (self.w_name == "Pawn" or self.b_name == "Pawn")
        self.filename = os.path.join(TB_DIR, f"K_{self.w_name}_vs_{self.b_name}_K_sml.bin")
        
        self.wk_size = 32 if self.has_pawn else 10
        self.total_positions = self.wk_size * 64 * 64 * 64 * 2
        self.table = np.zeros((self.wk_size, 64, 64, 64, 2), dtype=np.int16)
        self.transition_workers = max(1, (os.cpu_count() or 2) - TB_THREADS_SUBTRACT)

    def generate(self):
        if os.path.exists(self.filename):
            print(f"[SKIP] {self.filename} already exists.")
            return

        start_time = time.time()
        print(f"\n{'='*60}\n GENERATING: King + {self.w_name} vs King + {self.b_name} (SML)")
        
        print("[Stage 1] Enumerating structural candidates...")
        all_flats = array('I')
        for chunk in _gen_valid_4man_vs_indices_numpy(self.total_positions, self.w_name, self.b_name, self.has_pawn):
            all_flats.extend(chunk)

        print(f"[Stage 2 & 3] Pipelining Legality Filter + Maps ({len(all_flats):,} states)...")
        stage23_start = time.time()
        unsolved_flats = array('I'); transitions = []
        trans_lookup = {}; btm_to_wtm = defaultdict(list); wtm_to_btm = defaultdict(list)
        work_set = set()

        with ProcessPoolExecutor(max_workers=self.transition_workers,
                                 initializer=_init_transition_worker_4vs,
                                 initargs=(self.w_name, self.b_name)) as ex:
            for done, result in enumerate(ex.map(_build_transition_worker_4vs, all_flats, chunksize=4096), 1):
                if result is not None:
                    flat, trans = result
                    unsolved_flats.append(flat); transitions.append(trans); trans_lookup[flat] = trans
                    if (flat & 1) == 0:
                        for cflat in trans[3]: btm_to_wtm[cflat].append(flat)
                        if trans[1] or len(trans[2]) > 0: work_set.add(flat)
                    else:
                        if not trans[2]:
                            for cflat in trans[4]: wtm_to_btm[cflat].append(flat)

                if done % 100_000 == 0:
                    speed = done / max(0.001, time.time() - stage23_start)
                    print(f"  > {done / len(all_flats) * 100:.1f}% | {speed:,.0f} st/s", end='\r', flush=True)

        candidate_states = len(unsolved_flats)
        print(f"\n[Stage 2 & 3] Valid states: {candidate_states:,} | Time: {(time.time()-stage23_start)/60:.1f}m")

        print("[Stage 4 & 5] Retrograde Analysis...")
        table_flat = self.table.reshape(-1)
        for flat, trans in zip(unsolved_flats, transitions):
            if (flat & 1) == 1 and trans[1] == 0:
                table_flat[flat] = -1
                for parent in btm_to_wtm.get(flat, []):
                    if table_flat[parent] == 0: work_set.add(parent)

        iteration, max_dtm_solved = 1, 0
        while work_set:
            changed = 0; next_work_set = set(); snapshot = table_flat.copy()
            for flat in work_set:
                if table_flat[flat] != 0: continue
                trans = trans_lookup[flat]

                if (flat & 1) == 0: # WTM
                    _, imm_win, known_wins, child_vals = trans
                    best_win = 1 if imm_win else 0
                    for val in known_wins:
                        if best_win == 0 or val < best_win: best_win = val
                    for cflat in child_vals:
                        res = int(snapshot[cflat])
                        if res < 0:
                            val = abs(res) + 1
                            if best_win == 0 or val < best_win: best_win = val
                    if best_win > 0:
                        table_flat[flat] = best_win; changed += 1
                        for parent in wtm_to_btm.get(flat, []):
                            if table_flat[parent] == 0: next_work_set.add(parent)
                else: # BTM
                    _, moves_count, escape, known_wins, child_vals = trans
                    if escape or moves_count == 0: continue
                    max_win = 0
                    for val in known_wins:
                        if val > max_win: max_win = val
                    all_res = True
                    for cflat in child_vals:
                        res = int(snapshot[cflat])
                        if res <= 0: all_res = False; break
                        if res > max_win: max_win = res
                    if all_res:
                        table_flat[flat] = -(max_win + 1); changed += 1
                        for parent in btm_to_wtm.get(flat, []):
                            if table_flat[parent] == 0: next_work_set.add(parent)

            work_set = next_work_set
            if changed > 0: max_dtm_solved = iteration
            iteration += 1

        with open(self.filename, 'wb') as f:
            self.table.tofile(f)
            
        elapsed = time.time() - start_time
        decisive = np.count_nonzero(table_flat)
        _safe_record_longest_mate(f"K_{self.w_name}_vs_{self.b_name}_K", max_dtm_solved, decisive, candidate_states - decisive, elapsed)
        print(f"\nSUCCESS: Generated in {elapsed / 60:.1f} minutes.")

# ==============================================================================
# ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    _install_main_interrupt_ignores()
    print("=== Tablebase Generator (v11.1 - SML / Symmetry Compression) ===")

    Q, R, B, N, P = Queen, Rook, Bishop, Knight, Pawn

    print("\n--- TIER 1: 3-MAN NON-PAWN TABLES ---")
    for pc in [Q, R, B, N]: Generator(pc).generate()

    print("\n--- TIER 2: 3-MAN PAWN TABLE ---")
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
    for pc in [Q, R, B, N, P]:
        Generator4(P, pc).generate()
        
    seen_vs4 = set()
    for pc in [Q, R, B, N, P]:
        key = tuple(sorted(["Pawn", pc.__name__]))
        if key not in seen_vs4:
            seen_vs4.add(key)
            Generator4Vs(P, pc).generate()

    print("\n\n=== ALL TABLEBASE GENERATION COMPLETE ===")