# TablebaseGenerator.py (v13.31 - tuple IPC, low-RAM 5-man chunksize fix, removed dead code)


import os
import time
import signal
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations_with_replacement
import numpy as np
from GameLogic import *
from array import array
from collections import defaultdict

TB_DIR = "tablebases"
os.makedirs(TB_DIR, exist_ok=True)

LONGEST_MATES_NOTE_FILE = os.path.join(TB_DIR, "longest_mates_xsml.tsv")
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
# FAST SYMMETRY MAPPINGS
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

PAWN_VALID_T = [0 if (sq % 8) <= 3 else 1 for sq in range(64)]
PAWN_WK_SQUARES = [r*8+c for r in range(8) for c in range(4)]
PAWN_WK_IDX = {sq: i for i, sq in enumerate(PAWN_WK_SQUARES)}

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

# ==============================================================================
# CANONICAL FLATS
# ==============================================================================

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

def _canonical_flat_5(wk, p1, p2, p3, bk, turn, has_pawn, p1n, p2n, p3n):
    if has_pawn:
        t = PAWN_VALID_T[wk]
        m_wk = PAWN_WK_IDX[SYMMETRY_MAP[wk][t]]
        m_bk = SYMMETRY_MAP[bk][t]
        m1, m2, m3 = SYMMETRY_MAP[p1][t], SYMMETRY_MAP[p2][t], SYMMETRY_MAP[p3][t]
        if p1n == p2n and m1 > m2: m1, m2 = m2, m1
        if p2n == p3n and m2 > m3: m2, m3 = m3, m2
        if p1n == p2n and m1 > m2: m1, m2 = m2, m1
        return ((((m_wk * 64 + m1) * 64 + m2) * 64 + m3) * 64 + m_bk) * 2 + turn
    else:
        best = None
        for t in NON_PAWN_VALID_TS[wk]:
            m_bk = SYMMETRY_MAP[bk][t]
            m1, m2, m3 = SYMMETRY_MAP[p1][t], SYMMETRY_MAP[p2][t], SYMMETRY_MAP[p3][t]
            if p1n == p2n and m1 > m2: m1, m2 = m2, m1
            if p2n == p3n and m2 > m3: m2, m3 = m3, m2
            if p1n == p2n and m1 > m2: m1, m2 = m2, m1
            m = (NON_PAWN_WK_IDX[SYMMETRY_MAP[wk][t]], m1, m2, m3, m_bk)
            if best is None or m < best: best = m
        return ((((best[0] * 64 + best[1]) * 64 + best[2]) * 64 + best[3]) * 64 + best[4]) * 2 + turn

def _canonical_flat_5vs(wk, wp1, wp2, bk, bp, turn, has_pawn, wp1n, wp2n):
    if has_pawn:
        t = PAWN_VALID_T[wk]
        m_wk = PAWN_WK_IDX[SYMMETRY_MAP[wk][t]]
        m1, m2 = SYMMETRY_MAP[wp1][t], SYMMETRY_MAP[wp2][t]
        if wp1n == wp2n and m1 > m2: m1, m2 = m2, m1
        m_bk, m_bp = SYMMETRY_MAP[bk][t], SYMMETRY_MAP[bp][t]
        return ((((m_wk * 64 + m1) * 64 + m2) * 64 + m_bk) * 64 + m_bp) * 2 + turn
    else:
        best = None
        for t in NON_PAWN_VALID_TS[wk]:
            m1, m2 = SYMMETRY_MAP[wp1][t], SYMMETRY_MAP[wp2][t]
            if wp1n == wp2n and m1 > m2: m1, m2 = m2, m1
            m_bk, m_bp = SYMMETRY_MAP[bk][t], SYMMETRY_MAP[bp][t]
            m = (NON_PAWN_WK_IDX[SYMMETRY_MAP[wk][t]], m1, m2, m_bk, m_bp)
            if best is None or m < best: best = m
        return ((((best[0] * 64 + best[1]) * 64 + best[2]) * 64 + best[3]) * 64 + best[4]) * 2 + turn

# ==============================================================================
# SHARED HELPERS & LOADERS
# ==============================================================================

def _fmt_elapsed(seconds):
    if seconds >= 60:
        m = int(seconds // 60); s = seconds - m * 60
        return f"{m}m {s:.1f}s"
    return f"{seconds:.1f}s"

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
    data8 = np.fromfile(filename, dtype=np.int8)
    has_pawn = "Pawn" in os.path.basename(filename)
    return data8.reshape((32 if has_pawn else 10, 64, 64, 2))

def _load_4man_table_file(filename):
    data8 = np.fromfile(filename, dtype=np.int8)
    has_pawn = "Pawn" in os.path.basename(filename)
    return data8.reshape((32 if has_pawn else 10, 64, 64, 64, 2))

def _load_5man_table_file(filename):
    data8 = np.fromfile(filename, dtype=np.int8)
    has_pawn = "Pawn" in os.path.basename(filename)
    return data8.reshape((32 if has_pawn else 10, 64, 64, 64, 64, 2))

def _flip(pos):
    return (7 - pos[0], pos[1])

def _safe_record_longest_mate(table_key, max_dtm, decisive, remaining, elapsed_seconds):
    try:
        records = {}
        if os.path.exists(LONGEST_MATES_NOTE_FILE):
            import re
            with open(LONGEST_MATES_NOTE_FILE, "r", encoding="utf-8") as f:
                for raw in f:
                    line = raw.strip()
                    if not line or any(line.startswith(x) for x in ("#", "Table", "---")): continue
                    parts = re.split(r'\s{2,}|\t', line)
                    if len(parts) >= 6:
                        parts[2] = parts[2].replace(',', ''); parts[3] = parts[3].replace(',', '')
                        records[parts[0]] = parts[1:6]
        records[f"{LONGEST_MATE_KEY_PREFIX}{table_key}"] = [
            str(int(max_dtm)), str(int(decisive)), str(int(remaining)),
            f"{elapsed_seconds / 60.0:.1f}", time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        ]
        w_table = 35; w_dtm = 6; w_wins = 14; w_draws = 14; w_time = 10; w_utc = 20
        with open(LONGEST_MATES_NOTE_FILE, "w", encoding="utf-8") as f:
            header = (f"{'Table':<{w_table}}  {'DTM':>{w_dtm}}  {'Decisive':>{w_wins}}  "
                      f"{'Drawn':>{w_draws}}  {'Min':>{w_time}}  {'Updated (UTC)':<{w_utc}}")
            f.write(header + "\n"); f.write("-" * len(header) + "\n")
            for key in sorted(records.keys()):
                r = records[key]
                f.write(f"{key:<{w_table}}  {r[0]:>{w_dtm}}  {int(r[1]):>{w_wins},}  "
                        f"{int(r[2]):>{w_draws},}  {r[3]:>{w_time}}  {r[4]:<{w_utc}}\n")
    except Exception as e:
        print(f"[LongestMate] Warning: failed to update note file ({e})")

def _run_bfs_retrograde(table, unsolved_flats, transitions, trans_lookup,
                        btm_to_wtm, wtm_to_btm, candidate_states, start_time):
    table_flat = table.reshape(-1)
    pre_solved_set = set()
    pending_work = defaultdict(list)

    for flat, trans in zip(unsolved_flats, transitions):
        t_idx = flat & 1
        if t_idx == 0:  # WTM
            if trans[1]:
                table_flat[flat] = 1; pre_solved_set.add(flat)
            elif len(trans[2]) > 0:
                pending_work[min(trans[2])].append(flat)
        else:  # BTM
            if trans[1] == 0:
                table_flat[flat] = -1; pre_solved_set.add(flat)
            elif not trans[2]:
                # FIX-3: was array('H'), now () since workers return tuples
                known_wins = trans[3] if len(trans) == 5 else ()
                child_vals  = trans[4] if len(trans) == 5 else trans[3]
                if len(child_vals) == 0 and len(known_wins) > 0:
                    pending_work[max(known_wins) + 1].append(flat)

    work_set = set()
    for flat in pre_solved_set:
        targets = btm_to_wtm if (flat & 1 == 1) else wtm_to_btm
        for parent in targets.get(flat, []):
            if table_flat[parent] == 0: work_set.add(parent)

    iteration = 2
    max_dtm_solved = 1 if pre_solved_set else 0
    solved_count = len(pre_solved_set)
    bfs_start = time.time()

    print(f"[Stage 3] BFS start | Terminal (DTM 1): {solved_count:,} | "
          f"Elapsed so far: {_fmt_elapsed(bfs_start - start_time)}", flush=True)

    warning_triggered = False
    while work_set or pending_work:
        iter_start = time.time()

        if not warning_triggered and iteration >= 120:
            print("\n" + "="*70)
            print(f"[!!!] CRITICAL DTM WARNING: Ply count has reached {iteration}.")
            print("      Data type 'int8' is at risk of overflowing (max is 127).")
            print("      Recommended: stop (Ctrl+C) and switch to np.int16.")
            print("="*70 + "\n", flush=True)
            warning_triggered = True

        if iteration in pending_work:
            work_set.update(pending_work.pop(iteration))

        if not work_set:
            iteration += 1; continue

        changed = 0; next_work_set = set()
        snapshot = table_flat.copy()

        for flat in work_set:
            if table_flat[flat] != 0: continue
            trans = trans_lookup[flat]

            if (flat & 1) == 0:  # WTM
                _, imm_win, known_wins, child_vals = trans[0], trans[1], trans[2], trans[3]
                best_win = None
                for val in known_wins:
                    if best_win is None or val < best_win: best_win = val
                for cflat in child_vals:
                    res = int(snapshot[cflat])
                    if res < 0:
                        val = abs(res) + 1
                        if best_win is None or val < best_win: best_win = val
                if best_win == iteration:
                    table_flat[flat] = iteration; changed += 1
                    for parent in wtm_to_btm.get(flat, []):
                        if table_flat[parent] == 0: next_work_set.add(parent)
                elif best_win is not None and best_win > iteration:
                    pending_work[best_win].append(flat)

            else:  # BTM
                escape = trans[2]
                if escape: continue
                known_wins = trans[3] if len(trans) == 5 else ()
                child_vals  = trans[4] if len(trans) == 5 else trans[3]
                max_win = 0
                for val in known_wins:
                    if val > max_win: max_win = val
                all_res = True
                for cflat in child_vals:
                    res = int(snapshot[cflat])
                    if res <= 0: all_res = False; break
                    if res > max_win: max_win = res
                if all_res:
                    if max_win + 1 == iteration:
                        table_flat[flat] = -iteration; changed += 1
                        for parent in btm_to_wtm.get(flat, []):
                            if table_flat[parent] == 0: next_work_set.add(parent)
                    elif max_win + 1 > iteration:
                        pending_work[max_win + 1].append(flat)

        work_set = next_work_set
        solved_count += changed
        if changed > 0: max_dtm_solved = iteration
        pending_count = sum(len(v) for v in pending_work.values())
        elapsed_overall = time.time() - start_time
        print(f"  [DTM {iteration:^3}] New: {changed:<8,} | Total: {solved_count:,}/{candidate_states:,} "
              f"| Next Q: {len(work_set):<7,} | Held: {pending_count:<6,} "
              f"| Iter: {time.time()-iter_start:.1f}s | Elapsed: {_fmt_elapsed(elapsed_overall)}", flush=True)
        iteration += 1

    bfs_elapsed = time.time() - bfs_start
    print(f"[Stage 3] BFS complete | {_fmt_elapsed(bfs_elapsed)} | "
          f"max_dtm={max_dtm_solved} | decisive={solved_count:,}/{candidate_states:,}", flush=True)
    return max_dtm_solved, solved_count

# ==============================================================================
# 3-MAN GENERATOR
# ==============================================================================
_W_PIECE_NAME = None; _W_HAS_PAWN = False; _W_QUEEN_TABLE = None
_W3_BOARD = None; _W3_WK_OBJ = None; _W3_WP_OBJ = None; _W3_BK_OBJ = None

def _init_transition_worker(piece_name, queen_tb_file):
    _install_worker_interrupt_ignores()
    global _W_PIECE_NAME, _W_HAS_PAWN, _W_QUEEN_TABLE, _W3_BOARD, _W3_WK_OBJ, _W3_WP_OBJ, _W3_BK_OBJ
    _W_PIECE_NAME = piece_name; _W_HAS_PAWN = (piece_name == "Pawn")
    _W_QUEEN_TABLE = _load_3man_table_file(queen_tb_file) if (_W_HAS_PAWN and queen_tb_file) else None
    _W3_BOARD = Board(setup=False)
    _W3_WK_OBJ, _W3_WP_OBJ, _W3_BK_OBJ = King('white'), PIECE_CLASS_BY_NAME[piece_name]('white'), King('black')
    pc = _W3_BOARD.piece_counts; pc['white'][King] = pc['black'][King] = 1; pc['white'][type(_W3_WP_OBJ)] += 1

def _build_transition_worker(flat):
    rest = flat // 2; bk_sq = rest % 64; rest //= 64; p1_sq = rest % 64; wk_idx = rest // 64
    wk_sq = PAWN_WK_SQUARES[wk_idx] if _W_HAS_PAWN else NON_PAWN_WK_SQUARES[wk_idx]
    turn_is_white = (flat % 2 == 0); opp_turn_idx = 1 - (flat % 2)
    wk, p1, bk = (wk_sq//8, wk_sq%8), (p1_sq//8, p1_sq%8), (bk_sq//8, bk_sq%8)
    board = _W3_BOARD; g = board.grid
    for p in (_W3_WK_OBJ, _W3_WP_OBJ, _W3_BK_OBJ):
        if p.pos: g[p.pos[0]][p.pos[1]] = None
    _W3_WK_OBJ.pos, _W3_WP_OBJ.pos, _W3_BK_OBJ.pos = wk, p1, bk
    _W3_WK_OBJ._list_pos, _W3_WP_OBJ._list_pos, _W3_BK_OBJ._list_pos = 0, 1, 0
    g[wk[0]][wk[1]], g[p1[0]][p1[1]], g[bk[0]][bk[1]] = _W3_WK_OBJ, _W3_WP_OBJ, _W3_BK_OBJ
    board.white_king_pos, board.black_king_pos = wk, bk
    board.white_pieces[:] = [_W3_WK_OBJ, _W3_WP_OBJ]; board.black_pieces[:] = [_W3_BK_OBJ]
    if is_in_check(board, 'black' if turn_is_white else 'white'): return None
    moves = get_all_pseudo_legal_moves(board, 'white' if turn_is_white else 'black')
    if turn_is_white:
        immediate_win, promo_win_vals, child_flats = False, [], []
        for m in moves:
            record = board.make_move_track(m[0], m[1])
            if is_in_check(board, 'white'): board.unmake_move(record); continue
            wkp, bkp = board.white_king_pos, board.black_king_pos
            if not bkp or not has_legal_moves(board, 'black'):
                immediate_win = True; board.unmake_move(record); break
            if _W_HAS_PAWN:
                nk = next((x for x in board.white_pieces if not isinstance(x, King)), None)
                if nk is not None and isinstance(nk, Queen):
                    q_idx = _canonical_flat_3(wkp[0]*8+wkp[1], nk.pos[0]*8+nk.pos[1], bkp[0]*8+bkp[1], 1, False)
                    q_val = int(_W_QUEEN_TABLE.flat[q_idx])
                    if q_val < 0: promo_win_vals.append(abs(q_val) + 1)
                    board.unmake_move(record); continue
            p = next((x for x in board.white_pieces if not isinstance(x, King)), None)
            if p is not None:
                child_flats.append(_canonical_flat_3(wkp[0]*8+wkp[1], p.pos[0]*8+p.pos[1], bkp[0]*8+bkp[1], opp_turn_idx, _W_HAS_PAWN))
            board.unmake_move(record)
        # FIX-3: tuple instead of array for IPC
        return (flat, ('w', immediate_win, tuple(promo_win_vals), tuple(child_flats)))
    else:
        legal_moves, escape, child_flats = 0, False, []
        for m in moves:
            record = board.make_move_track(m[0], m[1])
            if is_in_check(board, 'black'): board.unmake_move(record); continue
            legal_moves += 1
            if not board.white_king_pos or len(board.white_pieces) < 2 or not has_legal_moves(board, 'white'):
                escape = True; board.unmake_move(record); continue
            if _W_HAS_PAWN:
                nk = next((x for x in board.white_pieces if not isinstance(x, King)), None)
                if nk is None: escape = True; board.unmake_move(record); continue
            p = next((x for x in board.white_pieces if not isinstance(x, King)), None)
            if p is None: escape = True; board.unmake_move(record); continue
            wkp, bkp = board.white_king_pos, board.black_king_pos
            child_flats.append(_canonical_flat_3(wkp[0]*8+wkp[1], p.pos[0]*8+p.pos[1], bkp[0]*8+bkp[1], opp_turn_idx, _W_HAS_PAWN))
            board.unmake_move(record)
        # FIX-3: tuple instead of array for IPC
        return (flat, ('b', legal_moves, escape, tuple(child_flats)))

class Generator:
    def __init__(self, piece_class):
        self.piece_name = piece_class.__name__; self.has_pawn = (self.piece_name == "Pawn")
        self.filename = os.path.join(TB_DIR, f"K_{self.piece_name}_K_xsml.bin")
        self.queen_tb_file = os.path.join(TB_DIR, "K_Queen_K_xsml.bin")
        self.wk_size = 32 if self.has_pawn else 10
        self.total_positions = self.wk_size * 64 * 64 * 2
        self.table = np.zeros((self.wk_size, 64, 64, 2), dtype=np.int8)
        self.transition_workers = min(8, max(1, (os.cpu_count() or 2) - TB_THREADS_SUBTRACT))

    def generate(self):
        if os.path.exists(self.filename): return
        start_time = time.time()
        print(f"\n{'='*60}\n GENERATING: K+{self.piece_name} vs K (XSML)\n{'='*60}")
        s1 = time.time(); print(f"[Stage 1] Enumerating candidates...", flush=True)
        raw_candidates = []
        for flat in range(self.total_positions):
            rest = flat // 2; bk_sq = rest % 64; rest //= 64; p1_sq = rest % 64; wk_idx = rest // 64
            wk_sq = PAWN_WK_SQUARES[wk_idx] if self.has_pawn else NON_PAWN_WK_SQUARES[wk_idx]
            if wk_sq == p1_sq or wk_sq == bk_sq or p1_sq == bk_sq: continue
            if self.has_pawn and (p1_sq // 8 == 0 or p1_sq // 8 == 7): continue
            raw_candidates.append(flat)
        total = len(raw_candidates)
        print(f"[Stage 1] Done: {total:,} candidates | {_fmt_elapsed(time.time()-s1)}", flush=True)
        s2 = time.time(); print(f"[Stage 2] Building transition cache ({total:,} states)...", flush=True)
        unsolved_flats = array('I'); transitions = []
        trans_lookup = {}; btm_to_wtm = defaultdict(list); wtm_to_btm = defaultdict(list)
        with ProcessPoolExecutor(max_workers=self.transition_workers, initializer=_init_transition_worker,
                                 initargs=(self.piece_name, self.queen_tb_file)) as ex:
            for done, result in enumerate(ex.map(_build_transition_worker, raw_candidates, chunksize=1024), 1):
                if result is not None:
                    flat, trans = result
                    unsolved_flats.append(flat); transitions.append(trans); trans_lookup[flat] = trans
                    if (flat & 1) == 0:
                        for cflat in trans[3]: btm_to_wtm[cflat].append(flat)
                    else:
                        if not trans[2]:
                            for cflat in trans[3]: wtm_to_btm[cflat].append(flat)
                if done % 10_000 == 0:
                    elapsed = time.time() - start_time; speed = done / max(0.001, time.time() - s2)
                    print(f"  > {done/total*100:.1f}% ({done:,}/{total:,}) | {speed:,.0f} st/s | Elapsed: {_fmt_elapsed(elapsed)}", end='\r', flush=True)
        print(); s2e = time.time() - s2
        print(f"[Stage 2] Cache built: {len(unsolved_flats):,} valid states | {_fmt_elapsed(s2e)} ({total/max(0.001,s2e):,.0f} st/s avg)", flush=True)
        candidate_states = len(unsolved_flats)
        max_dtm, decisive = _run_bfs_retrograde(self.table, unsolved_flats, transitions, trans_lookup, btm_to_wtm, wtm_to_btm, candidate_states, start_time)
        with open(self.filename, 'wb') as f: self.table.tofile(f)
        elapsed = time.time() - start_time
        _safe_record_longest_mate(f"K_{self.piece_name}_K", max_dtm, decisive, candidate_states - decisive, elapsed)
        print(f"SUCCESS: {self.filename} | Total time: {_fmt_elapsed(elapsed)}", flush=True)

# ==============================================================================
# 4-MAN GENERATOR
# ==============================================================================
_W4_HAS_PAWN = False; _W4_SAME_PIECE = False; _W4_P1_NAME = None; _W4_P2_NAME = None
_W4_3MAN_TABLES = {}; _W4_PROMO_TABLE = None
_W4_BOARD = None; _W4_WK_OBJ = None; _W4_P1_OBJ = None; _W4_P2_OBJ = None; _W4_BK_OBJ = None

def _init_transition_worker_4(p1_name, p2_name, promo_tb_file):
    _install_worker_interrupt_ignores()
    global _W4_P1_NAME, _W4_P2_NAME, _W4_HAS_PAWN, _W4_SAME_PIECE, _W4_3MAN_TABLES, _W4_PROMO_TABLE
    global _W4_BOARD, _W4_WK_OBJ, _W4_P1_OBJ, _W4_P2_OBJ, _W4_BK_OBJ
    _W4_P1_NAME, _W4_P2_NAME = p1_name, p2_name
    _W4_HAS_PAWN = (p1_name == "Pawn" or p2_name == "Pawn"); _W4_SAME_PIECE = (p1_name == p2_name)
    _W4_3MAN_TABLES, _W4_PROMO_TABLE = {}, None
    for name in PIECE_CLASS_BY_NAME:
        path = os.path.join(TB_DIR, f"K_{name}_K_xsml.bin")
        if os.path.exists(path): _W4_3MAN_TABLES[name] = _load_3man_table_file(path)
    if promo_tb_file and os.path.exists(promo_tb_file): _W4_PROMO_TABLE = _load_4man_table_file(promo_tb_file)
    _W4_BOARD = Board(setup=False)
    _W4_WK_OBJ, _W4_P1_OBJ, _W4_P2_OBJ, _W4_BK_OBJ = King('white'), PIECE_CLASS_BY_NAME[p1_name]('white'), PIECE_CLASS_BY_NAME[p2_name]('white'), King('black')
    pc = _W4_BOARD.piece_counts; pc['white'][King] = pc['black'][King] = 1
    pc['white'][type(_W4_P1_OBJ)] += 1; pc['white'][type(_W4_P2_OBJ)] += 1

def _build_transition_worker_4(flat):
    rest = flat // 2; bk_sq = rest % 64; rest //= 64; p2_sq = rest % 64; rest //= 64; p1_sq = rest % 64; wk_idx = rest // 64
    wk_sq = PAWN_WK_SQUARES[wk_idx] if _W4_HAS_PAWN else NON_PAWN_WK_SQUARES[wk_idx]
    turn_is_white = (flat % 2 == 0); opp_turn_idx = 1 - (flat % 2)
    wk, p1, p2, bk = (wk_sq//8,wk_sq%8),(p1_sq//8,p1_sq%8),(p2_sq//8,p2_sq%8),(bk_sq//8,bk_sq%8)
    board = _W4_BOARD; g = board.grid
    for p in (_W4_WK_OBJ, _W4_P1_OBJ, _W4_P2_OBJ, _W4_BK_OBJ):
        if p.pos: g[p.pos[0]][p.pos[1]] = None
    _W4_WK_OBJ.pos, _W4_P1_OBJ.pos, _W4_P2_OBJ.pos, _W4_BK_OBJ.pos = wk, p1, p2, bk
    _W4_WK_OBJ._list_pos, _W4_P1_OBJ._list_pos, _W4_P2_OBJ._list_pos, _W4_BK_OBJ._list_pos = 0,1,2,0
    g[wk[0]][wk[1]],g[p1[0]][p1[1]],g[p2[0]][p2[1]],g[bk[0]][bk[1]] = _W4_WK_OBJ,_W4_P1_OBJ,_W4_P2_OBJ,_W4_BK_OBJ
    board.white_king_pos, board.black_king_pos = wk, bk
    board.white_pieces[:] = [_W4_WK_OBJ, _W4_P1_OBJ, _W4_P2_OBJ]; board.black_pieces[:] = [_W4_BK_OBJ]
    if is_in_check(board, 'black' if turn_is_white else 'white'): return None
    moves = get_all_pseudo_legal_moves(board, 'white' if turn_is_white else 'black')
    if turn_is_white:
        immediate_win, known_wins, child_flats = False, [], []
        for start, end in moves:
            record = board.make_move_track(start, end)
            if is_in_check(board, 'white'): board.unmake_move(record); continue
            wkp, bkp = board.white_king_pos, board.black_king_pos
            if not bkp or not has_legal_moves(board, 'black'):
                immediate_win = True; board.unmake_move(record); break
            w_pieces = [p for p in board.white_pieces if not isinstance(p, King)]
            if len(w_pieces) < 2:
                if len(w_pieces) == 1:
                    rem_name = type(w_pieces[0]).__name__
                    if rem_name in _W4_3MAN_TABLES:
                        q_idx = _canonical_flat_3(wkp[0]*8+wkp[1], w_pieces[0].pos[0]*8+w_pieces[0].pos[1], bkp[0]*8+bkp[1], 1, rem_name=="Pawn")
                        val = int(_W4_3MAN_TABLES[rem_name].flat[q_idx])
                        if val < 0: known_wins.append(abs(val) + 1)
                board.unmake_move(record); continue
            if _PIECE_CANONICAL_ORDER.get(type(w_pieces[0]),99) > _PIECE_CANONICAL_ORDER.get(type(w_pieces[1]),99):
                w_pieces[0], w_pieces[1] = w_pieces[1], w_pieces[0]
            n1, n2 = type(w_pieces[0]).__name__, type(w_pieces[1]).__name__
            if n1 != _W4_P1_NAME or n2 != _W4_P2_NAME:
                if _W4_PROMO_TABLE is not None:
                    sq1, sq2 = w_pieces[0].pos[0]*8+w_pieces[0].pos[1], w_pieces[1].pos[0]*8+w_pieces[1].pos[1]
                    q_idx = _canonical_flat_4(wkp[0]*8+wkp[1], sq1, sq2, bkp[0]*8+bkp[1], 1, "Pawn" in {n1,n2}, n1==n2)
                    val = int(_W4_PROMO_TABLE.flat[q_idx])
                    if val < 0: known_wins.append(abs(val) + 1)
                board.unmake_move(record); continue
            cflat = _canonical_flat_4(wkp[0]*8+wkp[1], w_pieces[0].pos[0]*8+w_pieces[0].pos[1], w_pieces[1].pos[0]*8+w_pieces[1].pos[1], bkp[0]*8+bkp[1], opp_turn_idx, _W4_HAS_PAWN, _W4_SAME_PIECE)
            child_flats.append(cflat); board.unmake_move(record)
        # FIX-3: tuple instead of array for IPC
        return (flat, ('w', immediate_win, tuple(known_wins), tuple(child_flats)))
    else:
        legal_moves, escape, known_3man, child_flats = 0, False, [], []
        for start, end in moves:
            record = board.make_move_track(start, end)
            if is_in_check(board, 'black'): board.unmake_move(record); continue
            legal_moves += 1
            if not board.white_king_pos or len(board.white_pieces) < 2 or not has_legal_moves(board, 'white'):
                escape = True; board.unmake_move(record); continue
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
            if _PIECE_CANONICAL_ORDER.get(type(w_pieces[0]),99) > _PIECE_CANONICAL_ORDER.get(type(w_pieces[1]),99):
                w_pieces[0], w_pieces[1] = w_pieces[1], w_pieces[0]
            wkp, bkp = board.white_king_pos, board.black_king_pos
            cflat = _canonical_flat_4(wkp[0]*8+wkp[1], w_pieces[0].pos[0]*8+w_pieces[0].pos[1], w_pieces[1].pos[0]*8+w_pieces[1].pos[1], bkp[0]*8+bkp[1], opp_turn_idx, _W4_HAS_PAWN, _W4_SAME_PIECE)
            child_flats.append(cflat); board.unmake_move(record)
        # FIX-3: tuple instead of array for IPC
        return (flat, ('b', legal_moves, escape, tuple(known_3man), tuple(child_flats)))

def _gen_valid_4man_indices_numpy(total_positions, p1_name, p2_name, same_piece, has_pawn, chunk_size=8_000_000):
    for start in range(0, total_positions, chunk_size):
        end_c = min(start + chunk_size, total_positions)
        flat = np.arange(start, end_c, dtype=np.int64); rest = flat // 2
        bk_arr = rest % 64; rest //= 64; p2_arr = rest % 64; rest //= 64
        p1_arr = rest % 64; wk_idx_arr = rest // 64
        wk_arr = np.array(PAWN_WK_SQUARES if has_pawn else NON_PAWN_WK_SQUARES)[wk_idx_arr]
        mask = ((wk_arr != p1_arr) & (wk_arr != p2_arr) & (wk_arr != bk_arr) &
                (p1_arr != p2_arr) & (p1_arr != bk_arr) & (p2_arr != bk_arr))
        if p1_name == "Pawn": mask &= (p1_arr >= 8) & (p1_arr < 56)
        if p2_name == "Pawn": mask &= (p2_arr >= 8) & (p2_arr < 56)
        if same_piece: mask &= (p1_arr <= p2_arr)
        if p1_name == "Bishop" and p2_name == "Bishop":
            mask &= ((p1_arr // 8 + p1_arr % 8) % 2) != ((p2_arr // 8 + p2_arr % 8) % 2)
        if not mask.any(): continue
        yield flat[mask].tolist()

class Generator4:
    def __init__(self, p1_class, p2_class):
        names = sorted([p1_class.__name__, p2_class.__name__])
        self.p1_name, self.p2_name = names[0], names[1]
        self.same_piece = (self.p1_name == self.p2_name)
        self.has_pawn   = (self.p1_name == "Pawn" or self.p2_name == "Pawn")
        self.filename   = os.path.join(TB_DIR, f"K_{self.p1_name}_{self.p2_name}_K_xsml.bin")
        self.wk_size = 32 if self.has_pawn else 10
        self.total_positions = self.wk_size * 64 * 64 * 64 * 2
        self.table = np.zeros((self.wk_size, 64, 64, 64, 2), dtype=np.int8)
        self.promo_tb_file = None
        if self.has_pawn:
            other = self.p2_name if self.p1_name == "Pawn" else self.p1_name
            p_names = sorted(["Pawn", "Queen"]) if self.same_piece else sorted(["Queen", other])
            self.promo_tb_file = os.path.join(TB_DIR, f"K_{p_names[0]}_{p_names[1]}_K_xsml.bin")
        self.transition_workers = max(1, (os.cpu_count() or 2) - TB_THREADS_SUBTRACT)

    def generate(self):
        if os.path.exists(self.filename): return
        start_time = time.time()
        print(f"\n{'='*60}\n GENERATING: K+{self.p1_name}+{self.p2_name} vs K (XSML)\n{'='*60}", flush=True)
        s1 = time.time(); print(f"[Stage 1] Enumerating candidates...", flush=True)
        all_flats = array('I')
        for chunk in _gen_valid_4man_indices_numpy(self.total_positions, self.p1_name, self.p2_name, self.same_piece, self.has_pawn):
            all_flats.extend(chunk)
        total = len(all_flats)
        print(f"[Stage 1] Done: {total:,} candidates | {_fmt_elapsed(time.time()-s1)}", flush=True)
        s2 = time.time(); print(f"[Stage 2] Building transition cache ({total:,} states)...", flush=True)
        unsolved_flats = array('I'); transitions = []
        trans_lookup = {}; btm_to_wtm = defaultdict(list); wtm_to_btm = defaultdict(list)
        with ProcessPoolExecutor(max_workers=self.transition_workers, initializer=_init_transition_worker_4,
                                 initargs=(self.p1_name, self.p2_name, self.promo_tb_file)) as ex:
            for done, result in enumerate(ex.map(_build_transition_worker_4, all_flats, chunksize=4096), 1):
                if result is not None:
                    flat, trans = result
                    unsolved_flats.append(flat); transitions.append(trans); trans_lookup[flat] = trans
                    if (flat & 1) == 0:
                        for cflat in trans[3]: btm_to_wtm[cflat].append(flat)
                    else:
                        if not trans[2]:
                            for cflat in trans[4]: wtm_to_btm[cflat].append(flat)
                if done % 50_000 == 0:
                    elapsed = time.time() - start_time; speed = done / max(0.001, time.time() - s2)
                    print(f"  > {done/total*100:.1f}% ({done:,}/{total:,}) | {speed:,.0f} st/s | Elapsed: {_fmt_elapsed(elapsed)}", end='\r', flush=True)
        print(); s2e = time.time() - s2
        print(f"[Stage 2] Cache built: {len(unsolved_flats):,} valid states | {_fmt_elapsed(s2e)} ({total/max(0.001,s2e):,.0f} st/s avg)", flush=True)
        candidate_states = len(unsolved_flats)
        max_dtm, decisive = _run_bfs_retrograde(self.table, unsolved_flats, transitions, trans_lookup, btm_to_wtm, wtm_to_btm, candidate_states, start_time)
        with open(self.filename, 'wb') as f: self.table.tofile(f)
        elapsed = time.time() - start_time
        _safe_record_longest_mate(f"K_{self.p1_name}_{self.p2_name}_K", max_dtm, decisive, candidate_states - decisive, elapsed)
        print(f"SUCCESS: {self.filename} | Total time: {_fmt_elapsed(elapsed)}", flush=True)

# ==============================================================================
# 4-MAN CROSS GENERATOR
# ==============================================================================
_W4V_HAS_PAWN = False; _W4V_W_NAME = None; _W4V_B_NAME = None
_W4V_3MAN_TABLES = {}; _W4V_PROMO_TABLES = {}
_W4V_BOARD = None; _W4V_WK_OBJ = None; _W4V_WP_OBJ = None; _W4V_BK_OBJ = None; _W4V_BP_OBJ = None
_IN_TABLE_SENTINEL = "IN_TABLE"

def _init_transition_worker_4vs(w_name, b_name):
    _install_worker_interrupt_ignores()
    global _W4V_W_NAME, _W4V_B_NAME, _W4V_HAS_PAWN, _W4V_3MAN_TABLES, _W4V_PROMO_TABLES
    global _W4V_BOARD, _W4V_WK_OBJ, _W4V_WP_OBJ, _W4V_BK_OBJ, _W4V_BP_OBJ
    names = sorted([w_name, b_name]); _W4V_W_NAME, _W4V_B_NAME = names[0], names[1]
    _W4V_HAS_PAWN = ("Pawn" in {w_name, b_name}); _W4V_3MAN_TABLES.clear(); _W4V_PROMO_TABLES.clear()
    names_needed = {w_name, b_name}
    if _W4V_HAS_PAWN: names_needed.add("Queen")
    for name in names_needed:
        path = os.path.join(TB_DIR, f"K_{name}_K_xsml.bin")
        if os.path.exists(path): _W4V_3MAN_TABLES[name] = _load_3man_table_file(path)
    promo_targets = set()
    if w_name == "Pawn": promo_targets.add(("Queen", b_name))
    if b_name == "Pawn": promo_targets.add((w_name, "Queen"))
    for key in promo_targets:
        sk = tuple(sorted(key)); path = os.path.join(TB_DIR, f"K_{sk[0]}_vs_{sk[1]}_K_xsml.bin")
        if os.path.exists(path): _W4V_PROMO_TABLES[sk] = _load_4man_table_file(path)
    _W4V_BOARD = Board(setup=False)
    _W4V_WK_OBJ, _W4V_WP_OBJ = King('white'), PIECE_CLASS_BY_NAME[_W4V_W_NAME]('white')
    _W4V_BK_OBJ, _W4V_BP_OBJ = King('black'), PIECE_CLASS_BY_NAME[_W4V_B_NAME]('black')
    pc = _W4V_BOARD.piece_counts; pc['white'][King] = pc['black'][King] = 1
    pc['white'][type(_W4V_WP_OBJ)] += 1; pc['black'][type(_W4V_BP_OBJ)] += 1

def _white_win_dtm_3man(piece_name, wk, p, bk, turn_idx, is_white_piece):
    tb = _W4V_3MAN_TABLES.get(piece_name)
    if tb is None: return None
    if is_white_piece:
        q_idx = _canonical_flat_3(wk[0]*8+wk[1], p[0]*8+p[1], bk[0]*8+bk[1], turn_idx, piece_name=="Pawn")
        val = int(tb.flat[q_idx]); return None if val == 0 else (abs(val) if ((turn_idx==0 and val>0) or (turn_idx==1 and val<0)) else None)
    else:
        t2 = 1 - turn_idx; atk_k, atk_p, def_k = _flip(bk), _flip(p), _flip(wk)
        q_idx = _canonical_flat_3(atk_k[0]*8+atk_k[1], atk_p[0]*8+atk_p[1], def_k[0]*8+def_k[1], t2, piece_name=="Pawn")
        val = int(tb.flat[q_idx])
        if val == 0: return None
        return None if ((t2==0 and val>0) or (t2==1 and val<0)) else abs(val)

def _white_win_dtm_promo_vs(w_name, b_name, wk, wp, bk, bp, turn_idx):
    key = tuple(sorted((w_name, b_name))); tb = _W4V_PROMO_TABLES.get(key)
    if tb is None: return None
    hp = ("Pawn" in {w_name, b_name})
    if w_name > b_name:
        t2 = 1 - turn_idx; bk_f,bp_f,wk_f,wp_f = _flip(bk),_flip(bp),_flip(wk),_flip(wp)
        q_idx = _canonical_flat_4vs(bk_f[0]*8+bk_f[1], bp_f[0]*8+bp_f[1], wk_f[0]*8+wk_f[1], wp_f[0]*8+wp_f[1], t2, hp)
        val = int(tb.flat[q_idx])
        if val == 0: return None
        return None if ((t2==0 and val>0) or (t2==1 and val<0)) else abs(val)
    else:
        q_idx = _canonical_flat_4vs(wk[0]*8+wk[1], wp[0]*8+wp[1], bk[0]*8+bk[1], bp[0]*8+bp[1], turn_idx, hp)
        val = int(tb.flat[q_idx]); return None if val == 0 else (abs(val) if ((turn_idx==0 and val>0) or (turn_idx==1 and val<0)) else None)

def _ext_white_win_4vs(child, turn_idx):
    w_nk = [p for p in child.white_pieces if not isinstance(p, King)]
    b_nk = [p for p in child.black_pieces if not isinstance(p, King)]
    wkp, bkp = child.white_king_pos, child.black_king_pos
    if len(w_nk)==1 and len(b_nk)==1:
        wn, bn = type(w_nk[0]).__name__, type(b_nk[0]).__name__
        if wn==_W4V_W_NAME and bn==_W4V_B_NAME: return _IN_TABLE_SENTINEL
        return _white_win_dtm_promo_vs(wn, bn, wkp, w_nk[0].pos, bkp, b_nk[0].pos, turn_idx)
    if len(w_nk)==1 and len(b_nk)==0: return _white_win_dtm_3man(type(w_nk[0]).__name__, wkp, w_nk[0].pos, bkp, turn_idx, True)
    if len(w_nk)==0 and len(b_nk)==1: return _white_win_dtm_3man(type(b_nk[0]).__name__, wkp, b_nk[0].pos, bkp, turn_idx, False)
    if len(w_nk)==0 and len(b_nk)==0:
        if not bkp: return 1
        if turn_idx==1 and not has_legal_moves(child, 'black'): return 1
        return None
    return None

def _build_transition_worker_4vs(flat):
    rest = flat // 2; bp_sq = rest % 64; rest //= 64; bk_sq = rest % 64; rest //= 64
    wp_sq = rest % 64; wk_idx = rest // 64
    wk_sq = PAWN_WK_SQUARES[wk_idx] if _W4V_HAS_PAWN else NON_PAWN_WK_SQUARES[wk_idx]
    turn_is_white = (flat % 2 == 0); opp_turn_idx = 1 - (flat % 2)
    wk = (wk_sq//8,wk_sq%8); wp = (wp_sq//8,wp_sq%8); bk = (bk_sq//8,bk_sq%8); bp = (bp_sq//8,bp_sq%8)
    board = _W4V_BOARD; g = board.grid
    for p in (_W4V_WK_OBJ,_W4V_WP_OBJ,_W4V_BK_OBJ,_W4V_BP_OBJ):
        if p.pos: g[p.pos[0]][p.pos[1]] = None
    _W4V_WK_OBJ.pos,_W4V_WP_OBJ.pos = wk,wp; _W4V_BK_OBJ.pos,_W4V_BP_OBJ.pos = bk,bp
    _W4V_WK_OBJ._list_pos,_W4V_WP_OBJ._list_pos = 0,1; _W4V_BK_OBJ._list_pos,_W4V_BP_OBJ._list_pos = 0,1
    g[wk[0]][wk[1]],g[wp[0]][wp[1]] = _W4V_WK_OBJ,_W4V_WP_OBJ; g[bk[0]][bk[1]],g[bp[0]][bp[1]] = _W4V_BK_OBJ,_W4V_BP_OBJ
    board.white_king_pos,board.black_king_pos = wk,bk
    board.white_pieces[:] = [_W4V_WK_OBJ,_W4V_WP_OBJ]; board.black_pieces[:] = [_W4V_BK_OBJ,_W4V_BP_OBJ]
    if is_in_check(board, 'black' if turn_is_white else 'white'): return None
    moves = get_all_pseudo_legal_moves(board, 'white' if turn_is_white else 'black')
    if turn_is_white:
        immediate_win, known_wins, child_flats = False, [], []
        for start, end in moves:
            record = board.make_move_track(start, end)
            if not board.white_king_pos or not board.black_king_pos or is_in_check(board, 'white'):
                board.unmake_move(record); continue
            if not has_legal_moves(board, 'black'):
                immediate_win = True; board.unmake_move(record); break
            ext = _ext_white_win_4vs(board, opp_turn_idx)
            if ext == _IN_TABLE_SENTINEL:
                w_p = next((p for p in board.white_pieces if not isinstance(p, King)), None)
                b_p = next((p for p in board.black_pieces if not isinstance(p, King)), None)
                if w_p and b_p:
                    child_flats.append(_canonical_flat_4vs(board.white_king_pos[0]*8+board.white_king_pos[1], w_p.pos[0]*8+w_p.pos[1], board.black_king_pos[0]*8+board.black_king_pos[1], b_p.pos[0]*8+b_p.pos[1], opp_turn_idx, _W4V_HAS_PAWN))
            elif ext is not None and ext > 0: known_wins.append(ext + 1)
            board.unmake_move(record)
        # FIX-3: tuple instead of array for IPC
        return (flat, ('w', immediate_win, tuple(known_wins), tuple(child_flats)))
    else:
        legal_moves, escape, known_wins, child_flats = 0, False, [], []
        for start, end in moves:
            record = board.make_move_track(start, end)
            if not board.black_king_pos or not board.white_king_pos or is_in_check(board, 'black'):
                board.unmake_move(record); continue
            legal_moves += 1
            if not has_legal_moves(board, 'white'): escape = True; board.unmake_move(record); continue
            ext = _ext_white_win_4vs(board, opp_turn_idx)
            if ext == _IN_TABLE_SENTINEL:
                w_p = next((p for p in board.white_pieces if not isinstance(p, King)), None)
                b_p = next((p for p in board.black_pieces if not isinstance(p, King)), None)
                if not w_p or not b_p: escape = True
                else:
                    child_flats.append(_canonical_flat_4vs(board.white_king_pos[0]*8+board.white_king_pos[1], w_p.pos[0]*8+w_p.pos[1], board.black_king_pos[0]*8+board.black_king_pos[1], b_p.pos[0]*8+b_p.pos[1], opp_turn_idx, _W4V_HAS_PAWN))
            elif ext is not None and ext > 0: known_wins.append(ext)
            else: escape = True
            board.unmake_move(record)
        # FIX-3: tuple instead of array for IPC
        return (flat, ('b', legal_moves, escape, tuple(known_wins), tuple(child_flats)))

def _gen_valid_4man_vs_indices_numpy(total_positions, w_name, b_name, has_pawn, chunk_size=8_000_000):
    for start in range(0, total_positions, chunk_size):
        end_c = min(start + chunk_size, total_positions)
        flat = np.arange(start, end_c, dtype=np.int64); rest = flat // 2
        bp_arr = rest % 64; rest //= 64; bk_arr = rest % 64; rest //= 64
        wp_arr = rest % 64; wk_idx_arr = rest // 64
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
        self.has_pawn = ("Pawn" in {self.w_name, self.b_name})
        self.filename  = os.path.join(TB_DIR, f"K_{self.w_name}_vs_{self.b_name}_K_xsml.bin")
        self.wk_size   = 32 if self.has_pawn else 10
        self.total_positions = self.wk_size * 64 * 64 * 64 * 2
        self.table     = np.zeros((self.wk_size, 64, 64, 64, 2), dtype=np.int8)
        self.transition_workers = max(1, (os.cpu_count() or 2) - TB_THREADS_SUBTRACT)

    def generate(self):
        if os.path.exists(self.filename): return
        start_time = time.time()
        print(f"\n{'='*60}\n GENERATING: K+{self.w_name} vs K+{self.b_name} (XSML)\n{'='*60}", flush=True)
        s1 = time.time(); print(f"[Stage 1] Enumerating candidates...", flush=True)
        all_flats = array('I')
        for chunk in _gen_valid_4man_vs_indices_numpy(self.total_positions, self.w_name, self.b_name, self.has_pawn):
            all_flats.extend(chunk)
        total = len(all_flats)
        print(f"[Stage 1] Done: {total:,} candidates | {_fmt_elapsed(time.time()-s1)}", flush=True)
        s2 = time.time(); print(f"[Stage 2] Building transition cache ({total:,} states)...", flush=True)
        unsolved_flats = array('I'); transitions = []
        trans_lookup = {}; btm_to_wtm = defaultdict(list); wtm_to_btm = defaultdict(list)
        with ProcessPoolExecutor(max_workers=self.transition_workers, initializer=_init_transition_worker_4vs,
                                 initargs=(self.w_name, self.b_name)) as ex:
            for done, result in enumerate(ex.map(_build_transition_worker_4vs, all_flats, chunksize=4096), 1):
                if result is not None:
                    flat, trans = result
                    unsolved_flats.append(flat); transitions.append(trans); trans_lookup[flat] = trans
                    if (flat & 1) == 0:
                        for cflat in trans[3]: btm_to_wtm[cflat].append(flat)
                    else:
                        if not trans[2]:
                            for cflat in trans[4]: wtm_to_btm[cflat].append(flat)
                if done % 50_000 == 0:
                    elapsed = time.time() - start_time; speed = done / max(0.001, time.time() - s2)
                    print(f"  > {done/total*100:.1f}% ({done:,}/{total:,}) | {speed:,.0f} st/s | Elapsed: {_fmt_elapsed(elapsed)}", end='\r', flush=True)
        print(); s2e = time.time() - s2
        print(f"[Stage 2] Cache built: {len(unsolved_flats):,} valid states | {_fmt_elapsed(s2e)} ({total/max(0.001,s2e):,.0f} st/s avg)", flush=True)
        candidate_states = len(unsolved_flats)
        max_dtm, decisive = _run_bfs_retrograde(self.table, unsolved_flats, transitions, trans_lookup, btm_to_wtm, wtm_to_btm, candidate_states, start_time)
        with open(self.filename, 'wb') as f: self.table.tofile(f)
        elapsed = time.time() - start_time
        _safe_record_longest_mate(f"K_{self.w_name}_vs_{self.b_name}_K", max_dtm, decisive, candidate_states - decisive, elapsed)
        print(f"SUCCESS: {self.filename} | Total time: {_fmt_elapsed(elapsed)}", flush=True)

# ==============================================================================
# 5-MAN SAME-SIDE GENERATOR
# ==============================================================================
_W5_HAS_PAWN = False; _W5_P1_NAME = None; _W5_P2_NAME = None; _W5_P3_NAME = None
_W5_3MAN_TABLES = {}; _W5_4MAN_TABLES = {}; _W5_PROMO_TABLES = {}
_W5_BOARD = None; _W5_WK_OBJ = None; _W5_P1_OBJ = None; _W5_P2_OBJ = None; _W5_P3_OBJ = None; _W5_BK_OBJ = None

def _init_transition_worker_5(p1_name, p2_name, p3_name):
    _install_worker_interrupt_ignores()
    global _W5_HAS_PAWN, _W5_P1_NAME, _W5_P2_NAME, _W5_P3_NAME
    global _W5_3MAN_TABLES, _W5_4MAN_TABLES, _W5_PROMO_TABLES
    global _W5_BOARD, _W5_WK_OBJ, _W5_P1_OBJ, _W5_P2_OBJ, _W5_P3_OBJ, _W5_BK_OBJ
    _W5_P1_NAME, _W5_P2_NAME, _W5_P3_NAME = p1_name, p2_name, p3_name
    _W5_HAS_PAWN = ("Pawn" in {p1_name, p2_name, p3_name})
    _W5_3MAN_TABLES.clear(); _W5_4MAN_TABLES.clear(); _W5_PROMO_TABLES.clear()
    for name in PIECE_CLASS_BY_NAME:
        path = os.path.join(TB_DIR, f"K_{name}_K_xsml.bin")
        if os.path.exists(path): _W5_3MAN_TABLES[name] = _load_3man_table_file(path)
    pieces = sorted(PIECE_CLASS_BY_NAME.keys()); seen = set()
    for n1 in pieces:
        for n2 in pieces:
            key = tuple(sorted([n1, n2]))
            if key in seen: continue
            seen.add(key)
            path = os.path.join(TB_DIR, f"K_{key[0]}_{key[1]}_K_xsml.bin")
            if os.path.exists(path): _W5_4MAN_TABLES[f"{key[0]}_{key[1]}"] = _load_4man_table_file(path)
    if _W5_HAS_PAWN:
        names = [p1_name, p2_name, p3_name]
        if "Pawn" in names:
            idx = names.index("Pawn"); promo_names = names.copy(); promo_names[idx] = "Queen"
            promo_names.sort(key=lambda n: (_PIECE_CANONICAL_ORDER.get(PIECE_CLASS_BY_NAME[n], 99), n))
            tb_name = f"K_{promo_names[0]}_{promo_names[1]}_{promo_names[2]}_K_xsml.bin"
            path = os.path.join(TB_DIR, tb_name)
            if os.path.exists(path): _W5_PROMO_TABLES[tuple(promo_names)] = _load_5man_table_file(path)
    _W5_BOARD = Board(setup=False)
    _W5_WK_OBJ = King('white'); _W5_P1_OBJ = PIECE_CLASS_BY_NAME[p1_name]('white')
    _W5_P2_OBJ = PIECE_CLASS_BY_NAME[p2_name]('white'); _W5_P3_OBJ = PIECE_CLASS_BY_NAME[p3_name]('white')
    _W5_BK_OBJ = King('black')
    pc = _W5_BOARD.piece_counts; pc['white'][King] = pc['black'][King] = 1
    pc['white'][type(_W5_P1_OBJ)] += 1; pc['white'][type(_W5_P2_OBJ)] += 1; pc['white'][type(_W5_P3_OBJ)] += 1

def _w5_ext_dtm(board, turn_idx):
    w_nk = [p for p in board.white_pieces if not isinstance(p, King)]
    wkp, bkp = board.white_king_pos, board.black_king_pos; nw = len(w_nk)
    if nw == 2:
        names = sorted([type(p).__name__ for p in w_nk]); key = f"{names[0]}_{names[1]}"
        tb = _W5_4MAN_TABLES.get(key)
        if tb is None: return None
        ws = sorted(w_nk, key=lambda p: (_PIECE_CANONICAL_ORDER.get(type(p),99), p.pos[0]*8+p.pos[1]))
        n1,n2 = type(ws[0]).__name__, type(ws[1]).__name__; hp = ("Pawn" in {n1, n2})
        q_idx = _canonical_flat_4(wkp[0]*8+wkp[1], ws[0].pos[0]*8+ws[0].pos[1], ws[1].pos[0]*8+ws[1].pos[1], bkp[0]*8+bkp[1], turn_idx, hp, n1==n2)
        val = int(tb.flat[q_idx]); return None if val == 0 else (abs(val) if ((turn_idx==0 and val>0) or (turn_idx==1 and val<0)) else None)
    if nw == 1:
        wn = type(w_nk[0]).__name__; tb = _W5_3MAN_TABLES.get(wn)
        if tb is None: return None
        q_idx = _canonical_flat_3(wkp[0]*8+wkp[1], w_nk[0].pos[0]*8+w_nk[0].pos[1], bkp[0]*8+bkp[1], turn_idx, wn=="Pawn")
        val = int(tb.flat[q_idx]); return None if val == 0 else (abs(val) if ((turn_idx==0 and val>0) or (turn_idx==1 and val<0)) else None)
    if nw == 0:
        if not bkp: return 1
        if turn_idx==1 and not has_legal_moves(board, 'black'): return 1
        return None
    return None

def _build_transition_worker_5(flat):
    rest = flat // 2; bk_sq = rest % 64; rest //= 64; p3_sq = rest % 64; rest //= 64
    p2_sq = rest % 64; rest //= 64; p1_sq = rest % 64; wk_idx = rest // 64
    wk_sq = PAWN_WK_SQUARES[wk_idx] if _W5_HAS_PAWN else NON_PAWN_WK_SQUARES[wk_idx]
    turn_is_white = (flat % 2 == 0); opp_turn_idx = 1 - (flat % 2)
    wk=(wk_sq//8,wk_sq%8); p1=(p1_sq//8,p1_sq%8); p2=(p2_sq//8,p2_sq%8); p3=(p3_sq//8,p3_sq%8); bk=(bk_sq//8,bk_sq%8)
    board = _W5_BOARD; g = board.grid
    for p in (_W5_WK_OBJ,_W5_P1_OBJ,_W5_P2_OBJ,_W5_P3_OBJ,_W5_BK_OBJ):
        if p.pos: g[p.pos[0]][p.pos[1]] = None
    _W5_WK_OBJ.pos,_W5_P1_OBJ.pos,_W5_P2_OBJ.pos = wk,p1,p2; _W5_P3_OBJ.pos,_W5_BK_OBJ.pos = p3,bk
    _W5_WK_OBJ._list_pos,_W5_P1_OBJ._list_pos,_W5_P2_OBJ._list_pos = 0,1,2; _W5_P3_OBJ._list_pos,_W5_BK_OBJ._list_pos = 3,0
    g[wk[0]][wk[1]],g[p1[0]][p1[1]],g[p2[0]][p2[1]] = _W5_WK_OBJ,_W5_P1_OBJ,_W5_P2_OBJ
    g[p3[0]][p3[1]],g[bk[0]][bk[1]] = _W5_P3_OBJ,_W5_BK_OBJ
    board.white_king_pos,board.black_king_pos = wk,bk
    board.white_pieces[:] = [_W5_WK_OBJ,_W5_P1_OBJ,_W5_P2_OBJ,_W5_P3_OBJ]; board.black_pieces[:] = [_W5_BK_OBJ]
    if is_in_check(board, 'black' if turn_is_white else 'white'): return None
    moves = get_all_pseudo_legal_moves(board, 'white' if turn_is_white else 'black')
    if turn_is_white:
        immediate_win, known_wins, child_flats = False, [], []
        for start, end in moves:
            record = board.make_move_track(start, end)
            if not board.white_king_pos or is_in_check(board, 'white'): board.unmake_move(record); continue
            bkp = board.black_king_pos
            if not bkp or not has_legal_moves(board, 'black'):
                immediate_win = True; board.unmake_move(record); break
            wkp = board.white_king_pos
            w_nk = [p for p in board.white_pieces if not isinstance(p, King)]
            if len(w_nk) == 3:
                ws = sorted(w_nk, key=lambda p: (_PIECE_CANONICAL_ORDER.get(type(p),99), p.pos[0]*8+p.pos[1]))
                cn1,cn2,cn3 = type(ws[0]).__name__,type(ws[1]).__name__,type(ws[2]).__name__
                if cn1==_W5_P1_NAME and cn2==_W5_P2_NAME and cn3==_W5_P3_NAME:
                    child_flats.append(_canonical_flat_5(wkp[0]*8+wkp[1], ws[0].pos[0]*8+ws[0].pos[1], ws[1].pos[0]*8+ws[1].pos[1], ws[2].pos[0]*8+ws[2].pos[1], bkp[0]*8+bkp[1], opp_turn_idx, _W5_HAS_PAWN, cn1, cn2, cn3))
                else:
                    key = (cn1, cn2, cn3)
                    if key in _W5_PROMO_TABLES:
                        tb = _W5_PROMO_TABLES[key]; hp = ("Pawn" in key)
                        q_idx = _canonical_flat_5(wkp[0]*8+wkp[1], ws[0].pos[0]*8+ws[0].pos[1], ws[1].pos[0]*8+ws[1].pos[1], ws[2].pos[0]*8+ws[2].pos[1], bkp[0]*8+bkp[1], opp_turn_idx, hp, cn1, cn2, cn3)
                        val = int(tb.flat[q_idx])
                        if val < 0: known_wins.append(abs(val) + 1)
            else:
                ext = _w5_ext_dtm(board, opp_turn_idx)
                if ext is not None and ext > 0: known_wins.append(ext + 1)
            board.unmake_move(record)
        # FIX-3: tuple instead of array for IPC
        return (flat, ('w', immediate_win, tuple(known_wins), tuple(child_flats)))
    else:
        legal_moves, escape, known_wins, child_flats = 0, False, [], []
        for start, end in moves:
            record = board.make_move_track(start, end)
            if not board.black_king_pos or is_in_check(board, 'black'): board.unmake_move(record); continue
            legal_moves += 1
            wkp = board.white_king_pos
            if not wkp: escape = True; board.unmake_move(record); continue
            bkp = board.black_king_pos
            w_nk = [p for p in board.white_pieces if not isinstance(p, King)]
            if len(w_nk) == 3:
                ws = sorted(w_nk, key=lambda p: (_PIECE_CANONICAL_ORDER.get(type(p),99), p.pos[0]*8+p.pos[1]))
                cn1,cn2,cn3 = type(ws[0]).__name__,type(ws[1]).__name__,type(ws[2]).__name__
                if cn1==_W5_P1_NAME and cn2==_W5_P2_NAME and cn3==_W5_P3_NAME:
                    child_flats.append(_canonical_flat_5(wkp[0]*8+wkp[1], ws[0].pos[0]*8+ws[0].pos[1], ws[1].pos[0]*8+ws[1].pos[1], ws[2].pos[0]*8+ws[2].pos[1], bkp[0]*8+bkp[1], opp_turn_idx, _W5_HAS_PAWN, cn1, cn2, cn3))
                else: escape = True
            else:
                ext = _w5_ext_dtm(board, opp_turn_idx)
                if ext is not None and ext > 0: known_wins.append(ext)
                else: escape = True
            board.unmake_move(record)
        # FIX-3: tuple instead of array for IPC
        return (flat, ('b', legal_moves, escape, tuple(known_wins), tuple(child_flats)))

def _gen_valid_5same_indices_numpy(total, p1n, p2n, p3n, has_pawn, chunk_size=2_000_000):
    wk_list = np.array(PAWN_WK_SQUARES if has_pawn else NON_PAWN_WK_SQUARES)
    same_12 = (p1n == p2n); same_23 = (p2n == p3n)
    for start in range(0, total, chunk_size):
        end_c = min(start + chunk_size, total); flat = np.arange(start, end_c, dtype=np.int64); rest = flat // 2
        bk_arr = rest % 64; rest //= 64; p3_arr = rest % 64; rest //= 64
        p2_arr = rest % 64; rest //= 64; p1_arr = rest % 64; wk_idx = rest // 64; wk_arr = wk_list[wk_idx]
        mask = ((wk_arr != p1_arr) & (wk_arr != p2_arr) & (wk_arr != p3_arr) & (wk_arr != bk_arr) &
                (p1_arr != p2_arr) & (p1_arr != p3_arr) & (p1_arr != bk_arr) &
                (p2_arr != p3_arr) & (p2_arr != bk_arr) & (p3_arr != bk_arr))
        if p1n == "Pawn": mask &= (p1_arr >= 8) & (p1_arr < 56)
        if p2n == "Pawn": mask &= (p2_arr >= 8) & (p2_arr < 56)
        if p3n == "Pawn": mask &= (p3_arr >= 8) & (p3_arr < 56)
        if same_12: mask &= (p1_arr <= p2_arr)
        if same_23: mask &= (p2_arr <= p3_arr)
        if p1n == "Bishop" and p2n == "Bishop":
            mask &= ((p1_arr // 8 + p1_arr % 8) % 2) != ((p2_arr // 8 + p2_arr % 8) % 2)
        if p2n == "Bishop" and p3n == "Bishop":
            mask &= ((p2_arr // 8 + p2_arr % 8) % 2) != ((p3_arr // 8 + p3_arr % 8) % 2)
        if not mask.any(): continue
        yield flat[mask].tolist()

class Generator5:
    def __init__(self, p1_class, p2_class, p3_class):
        ps = sorted([p1_class, p2_class, p3_class], key=lambda c: (_PIECE_CANONICAL_ORDER.get(c, 99), c.__name__))
        self.p1_name = ps[0].__name__; self.p2_name = ps[1].__name__; self.p3_name = ps[2].__name__
        self.has_pawn = ("Pawn" in {self.p1_name, self.p2_name, self.p3_name})
        self.filename  = os.path.join(TB_DIR, f"K_{self.p1_name}_{self.p2_name}_{self.p3_name}_K_xsml.bin")
        self.wk_size = 32 if self.has_pawn else 10
        self.total_positions = self.wk_size * 64 * 64 * 64 * 64 * 2
        self.table = np.zeros((self.wk_size, 64, 64, 64, 64, 2), dtype=np.int8)
        self.transition_workers = max(1, (os.cpu_count() or 2) - TB_THREADS_SUBTRACT)

    def generate(self):
        if os.path.exists(self.filename): return
        start_time = time.time()
        sz_mb = self.wk_size * 64**4 * 2 * 1 / 1024**2  # int8 = 1 byte
        print(f"\n{'='*60}\n GENERATING: K+{self.p1_name}+{self.p2_name}+{self.p3_name} vs K (XSML)", flush=True)
        print(f" Estimated file size: {sz_mb:.0f} MB\n{'='*60}", flush=True)
        s1 = time.time(); print(f"[Stage 1] Enumerating candidates...", flush=True)
        all_flats = array('I')
        for chunk in _gen_valid_5same_indices_numpy(self.total_positions, self.p1_name, self.p2_name, self.p3_name, self.has_pawn):
            all_flats.extend(chunk)
        total = len(all_flats)
        print(f"[Stage 1] Done: {total:,} candidates | {_fmt_elapsed(time.time()-s1)}", flush=True)
        s2 = time.time(); print(f"[Stage 2] Building transition cache ({total:,} states)...", flush=True)
        unsolved_flats = array('I'); transitions = []
        trans_lookup = {}; btm_to_wtm = defaultdict(list); wtm_to_btm = defaultdict(list)
        with ProcessPoolExecutor(max_workers=self.transition_workers, initializer=_init_transition_worker_5,
                                 initargs=(self.p1_name, self.p2_name, self.p3_name)) as ex:
            # FIX-4: chunksize reduced from 2048 to 256 for low-RAM machines
            for done, result in enumerate(ex.map(_build_transition_worker_5, all_flats, chunksize=256), 1):
                if result is not None:
                    flat, trans = result
                    unsolved_flats.append(flat); transitions.append(trans); trans_lookup[flat] = trans
                    if (flat & 1) == 0:
                        for cflat in trans[3]: btm_to_wtm[cflat].append(flat)
                    else:
                        if not trans[2]:
                            # FIX-1: trans[3]=known_wins, trans[4]=child_flats
                            for cflat in trans[4]: wtm_to_btm[cflat].append(flat)
                if done % 100_000 == 0:
                    elapsed = time.time() - start_time; speed = done / max(0.001, time.time() - s2)
                    print(f"  > {done/total*100:.1f}% ({done:,}/{total:,}) | {speed:,.0f} st/s | Elapsed: {_fmt_elapsed(elapsed)}", end='\r', flush=True)
        print(); s2e = time.time() - s2
        print(f"[Stage 2] Cache built: {len(unsolved_flats):,} valid states | {_fmt_elapsed(s2e)} ({total/max(0.001,s2e):,.0f} st/s avg)", flush=True)
        candidate_states = len(unsolved_flats)
        max_dtm, decisive = _run_bfs_retrograde(self.table, unsolved_flats, transitions, trans_lookup, btm_to_wtm, wtm_to_btm, candidate_states, start_time)
        with open(self.filename, 'wb') as f: self.table.tofile(f)
        elapsed = time.time() - start_time
        _safe_record_longest_mate(f"K_{self.p1_name}_{self.p2_name}_{self.p3_name}_K", max_dtm, decisive, candidate_states - decisive, elapsed)
        print(f"SUCCESS: {self.filename} | Total time: {_fmt_elapsed(elapsed)}", flush=True)

# ==============================================================================
# 5-MAN CROSS GENERATOR
# ==============================================================================
_W5V_HAS_PAWN = False; _W5V_W1_NAME = None; _W5V_W2_NAME = None; _W5V_B_NAME = None
_W5V_3MAN_TABLES = {}; _W5V_4MAN_TABLES = {}; _W5V_4VS_TABLES = {}; _W5V_PROMO_TABLES = {}
_W5V_BOARD = None; _W5V_WK_OBJ = None; _W5V_WP1_OBJ = None; _W5V_WP2_OBJ = None; _W5V_BK_OBJ = None; _W5V_BP_OBJ = None

def _init_transition_worker_5vs(w1_name, w2_name, b_name):
    _install_worker_interrupt_ignores()
    global _W5V_HAS_PAWN, _W5V_W1_NAME, _W5V_W2_NAME, _W5V_B_NAME
    global _W5V_3MAN_TABLES, _W5V_4MAN_TABLES, _W5V_4VS_TABLES, _W5V_PROMO_TABLES
    global _W5V_BOARD, _W5V_WK_OBJ, _W5V_WP1_OBJ, _W5V_WP2_OBJ, _W5V_BK_OBJ, _W5V_BP_OBJ
    _W5V_W1_NAME, _W5V_W2_NAME, _W5V_B_NAME = w1_name, w2_name, b_name
    _W5V_HAS_PAWN = ("Pawn" in {w1_name, w2_name, b_name})
    _W5V_3MAN_TABLES.clear(); _W5V_4MAN_TABLES.clear(); _W5V_4VS_TABLES.clear(); _W5V_PROMO_TABLES.clear()
    for name in PIECE_CLASS_BY_NAME:
        path = os.path.join(TB_DIR, f"K_{name}_K_xsml.bin")
        if os.path.exists(path): _W5V_3MAN_TABLES[name] = _load_3man_table_file(path)
    pieces = sorted(PIECE_CLASS_BY_NAME.keys()); seen4 = set()
    for n1 in pieces:
        for n2 in pieces:
            key = tuple(sorted([n1, n2]))
            if key in seen4: continue; seen4.add(key)
            path = os.path.join(TB_DIR, f"K_{key[0]}_{key[1]}_K_xsml.bin")
            if os.path.exists(path): _W5V_4MAN_TABLES[f"{key[0]}_{key[1]}"] = _load_4man_table_file(path)
    seen4v = set()
    for wn in pieces:
        for bn in pieces:
            key = tuple(sorted([wn, bn]))
            if key in seen4v: continue; seen4v.add(key)
            path = os.path.join(TB_DIR, f"K_{key[0]}_vs_{key[1]}_K_xsml.bin")
            if os.path.exists(path): _W5V_4VS_TABLES[f"{key[0]}_vs_{key[1]}"] = _load_4man_table_file(path)
    if _W5V_HAS_PAWN:
        if "Pawn" in [w1_name, w2_name]:
            w_names = [w1_name, w2_name]; w_names[w_names.index("Pawn")] = "Queen"
            w_names.sort(key=lambda n: (_PIECE_CANONICAL_ORDER.get(PIECE_CLASS_BY_NAME[n],99), n))
            tb_name = f"K_{w_names[0]}_{w_names[1]}_vs_{b_name}_K_xsml.bin"
            path = os.path.join(TB_DIR, tb_name)
            if os.path.exists(path): _W5V_PROMO_TABLES[("w", tuple(w_names), b_name)] = _load_5man_table_file(path)
        if b_name == "Pawn":
            tb_name = f"K_{w1_name}_{w2_name}_vs_Queen_K_xsml.bin"
            path = os.path.join(TB_DIR, tb_name)
            if os.path.exists(path): _W5V_PROMO_TABLES[("b", (w1_name, w2_name), "Queen")] = _load_5man_table_file(path)
    _W5V_BOARD = Board(setup=False)
    _W5V_WK_OBJ = King('white'); _W5V_WP1_OBJ = PIECE_CLASS_BY_NAME[w1_name]('white'); _W5V_WP2_OBJ = PIECE_CLASS_BY_NAME[w2_name]('white')
    _W5V_BK_OBJ = King('black'); _W5V_BP_OBJ = PIECE_CLASS_BY_NAME[b_name]('black')
    pc = _W5V_BOARD.piece_counts; pc['white'][King] = pc['black'][King] = 1
    pc['white'][type(_W5V_WP1_OBJ)] += 1; pc['white'][type(_W5V_WP2_OBJ)] += 1; pc['black'][type(_W5V_BP_OBJ)] += 1

def _w5v_ext_dtm(board, turn_idx):
    w_nk = [p for p in board.white_pieces if not isinstance(p, King)]
    b_nk = [p for p in board.black_pieces if not isinstance(p, King)]
    wkp, bkp = board.white_king_pos, board.black_king_pos; nw, nb = len(w_nk), len(b_nk)
    if nw == 2 and nb == 1:
        ws = sorted(w_nk, key=lambda p: (_PIECE_CANONICAL_ORDER.get(type(p),99), p.pos[0]*8+p.pos[1]))
        cw1,cw2 = type(ws[0]).__name__, type(ws[1]).__name__; cb = type(b_nk[0]).__name__
        if cw1==_W5V_W1_NAME and cw2==_W5V_W2_NAME and cb==_W5V_B_NAME: return _IN_TABLE_SENTINEL
        key = ("w" if cb == _W5V_B_NAME else "b", (cw1, cw2), cb)
        tb = _W5V_PROMO_TABLES.get(key)
        if tb is None: return None
        hp = ("Pawn" in {cw1, cw2, cb})
        q_idx = _canonical_flat_5vs(wkp[0]*8+wkp[1], ws[0].pos[0]*8+ws[0].pos[1], ws[1].pos[0]*8+ws[1].pos[1], bkp[0]*8+bkp[1], b_nk[0].pos[0]*8+b_nk[0].pos[1], turn_idx, hp, cw1, cw2)
        val = int(tb.flat[q_idx]); return None if val == 0 else (abs(val) if ((turn_idx==0 and val>0) or (turn_idx==1 and val<0)) else None)
    if nw == 2 and nb == 0:
        names = sorted([type(p).__name__ for p in w_nk]); tb = _W5V_4MAN_TABLES.get(f"{names[0]}_{names[1]}")
        if tb is None: return None
        ws = sorted(w_nk, key=lambda p: (_PIECE_CANONICAL_ORDER.get(type(p),99), p.pos[0]*8+p.pos[1]))
        n1,n2 = type(ws[0]).__name__, type(ws[1]).__name__; hp = ("Pawn" in {n1, n2})
        q_idx = _canonical_flat_4(wkp[0]*8+wkp[1], ws[0].pos[0]*8+ws[0].pos[1], ws[1].pos[0]*8+ws[1].pos[1], bkp[0]*8+bkp[1], turn_idx, hp, n1==n2)
        val = int(tb.flat[q_idx]); return None if val == 0 else (abs(val) if ((turn_idx==0 and val>0) or (turn_idx==1 and val<0)) else None)
    if nw == 1 and nb == 1:
        wn, bn = type(w_nk[0]).__name__, type(b_nk[0]).__name__
        names = sorted([wn, bn]); sn1,sn2 = names[0], names[1]; tb = _W5V_4VS_TABLES.get(f"{sn1}_vs_{sn2}")
        if tb is None: return None
        hp = ("Pawn" in {wn, bn})
        if wn <= bn:
            q_idx = _canonical_flat_4vs(wkp[0]*8+wkp[1], w_nk[0].pos[0]*8+w_nk[0].pos[1], bkp[0]*8+bkp[1], b_nk[0].pos[0]*8+b_nk[0].pos[1], turn_idx, hp)
            val = int(tb.flat[q_idx]); return None if val == 0 else (abs(val) if ((turn_idx==0 and val>0) or (turn_idx==1 and val<0)) else None)
        else:
            bk_f,bp_f = _flip(bkp),_flip(b_nk[0].pos); wk_f,wp_f = _flip(wkp),_flip(w_nk[0].pos)
            t2 = 1 - turn_idx
            q_idx = _canonical_flat_4vs(bk_f[0]*8+bk_f[1], bp_f[0]*8+bp_f[1], wk_f[0]*8+wk_f[1], wp_f[0]*8+wp_f[1], t2, hp)
            val = int(tb.flat[q_idx])
            if val == 0: return None
            return None if ((t2==0 and val>0) or (t2==1 and val<0)) else abs(val)
    if nw == 1 and nb == 0:
        wn = type(w_nk[0]).__name__; tb = _W5V_3MAN_TABLES.get(wn)
        if tb is None: return None
        q_idx = _canonical_flat_3(wkp[0]*8+wkp[1], w_nk[0].pos[0]*8+w_nk[0].pos[1], bkp[0]*8+bkp[1], turn_idx, wn=="Pawn")
        val = int(tb.flat[q_idx]); return None if val == 0 else (abs(val) if ((turn_idx==0 and val>0) or (turn_idx==1 and val<0)) else None)
    if nw == 0 and nb == 0:
        if not bkp: return 1
        if turn_idx==1 and not has_legal_moves(board, 'black'): return 1
        return None
    return None

def _build_transition_worker_5vs(flat):
    rest = flat // 2; bp_sq = rest % 64; rest //= 64; bk_sq = rest % 64; rest //= 64
    wp2_sq = rest % 64; rest //= 64; wp1_sq = rest % 64; wk_idx = rest // 64
    wk_sq = PAWN_WK_SQUARES[wk_idx] if _W5V_HAS_PAWN else NON_PAWN_WK_SQUARES[wk_idx]
    turn_is_white = (flat % 2 == 0); opp_turn_idx = 1 - (flat % 2)
    wk=(wk_sq//8,wk_sq%8); wp1=(wp1_sq//8,wp1_sq%8); wp2=(wp2_sq//8,wp2_sq%8); bk=(bk_sq//8,bk_sq%8); bp=(bp_sq//8,bp_sq%8)
    board = _W5V_BOARD; g = board.grid
    for p in (_W5V_WK_OBJ,_W5V_WP1_OBJ,_W5V_WP2_OBJ,_W5V_BK_OBJ,_W5V_BP_OBJ):
        if p.pos: g[p.pos[0]][p.pos[1]] = None
    _W5V_WK_OBJ.pos,_W5V_WP1_OBJ.pos,_W5V_WP2_OBJ.pos = wk,wp1,wp2; _W5V_BK_OBJ.pos,_W5V_BP_OBJ.pos = bk,bp
    _W5V_WK_OBJ._list_pos,_W5V_WP1_OBJ._list_pos,_W5V_WP2_OBJ._list_pos = 0,1,2; _W5V_BK_OBJ._list_pos,_W5V_BP_OBJ._list_pos = 0,1
    g[wk[0]][wk[1]],g[wp1[0]][wp1[1]],g[wp2[0]][wp2[1]] = _W5V_WK_OBJ,_W5V_WP1_OBJ,_W5V_WP2_OBJ
    g[bk[0]][bk[1]],g[bp[0]][bp[1]] = _W5V_BK_OBJ,_W5V_BP_OBJ
    board.white_king_pos,board.black_king_pos = wk,bk
    board.white_pieces[:] = [_W5V_WK_OBJ,_W5V_WP1_OBJ,_W5V_WP2_OBJ]; board.black_pieces[:] = [_W5V_BK_OBJ,_W5V_BP_OBJ]
    if is_in_check(board, 'black' if turn_is_white else 'white'): return None
    moves = get_all_pseudo_legal_moves(board, 'white' if turn_is_white else 'black')
    if turn_is_white:
        immediate_win, known_wins, child_flats = False, [], []
        for start, end in moves:
            record = board.make_move_track(start, end)
            if not board.white_king_pos or is_in_check(board, 'white'): board.unmake_move(record); continue
            if not has_legal_moves(board, 'black'):
                immediate_win = True; board.unmake_move(record); break
            ext = _w5v_ext_dtm(board, opp_turn_idx)
            if ext == _IN_TABLE_SENTINEL:
                w_nk = [p for p in board.white_pieces if not isinstance(p, King)]
                b_nk = [p for p in board.black_pieces if not isinstance(p, King)]
                ws = sorted(w_nk, key=lambda p: (_PIECE_CANONICAL_ORDER.get(type(p),99), p.pos[0]*8+p.pos[1]))
                child_flats.append(_canonical_flat_5vs(board.white_king_pos[0]*8+board.white_king_pos[1], ws[0].pos[0]*8+ws[0].pos[1], ws[1].pos[0]*8+ws[1].pos[1], board.black_king_pos[0]*8+board.black_king_pos[1], b_nk[0].pos[0]*8+b_nk[0].pos[1], opp_turn_idx, _W5V_HAS_PAWN, type(ws[0]).__name__, type(ws[1]).__name__))
            elif ext is not None and ext > 0: known_wins.append(ext + 1)
            board.unmake_move(record)
        # FIX-3: tuple instead of array for IPC
        return (flat, ('w', immediate_win, tuple(known_wins), tuple(child_flats)))
    else:
        legal_moves, escape, known_wins, child_flats = 0, False, [], []
        for start, end in moves:
            record = board.make_move_track(start, end)
            if not board.black_king_pos or not board.white_king_pos or is_in_check(board, 'black'):
                board.unmake_move(record); continue
            legal_moves += 1
            if not has_legal_moves(board, 'white'): escape = True; board.unmake_move(record); continue
            ext = _w5v_ext_dtm(board, opp_turn_idx)
            if ext == _IN_TABLE_SENTINEL:
                w_nk = [p for p in board.white_pieces if not isinstance(p, King)]
                b_nk = [p for p in board.black_pieces if not isinstance(p, King)]
                ws = sorted(w_nk, key=lambda p: (_PIECE_CANONICAL_ORDER.get(type(p),99), p.pos[0]*8+p.pos[1]))
                child_flats.append(_canonical_flat_5vs(board.white_king_pos[0]*8+board.white_king_pos[1], ws[0].pos[0]*8+ws[0].pos[1], ws[1].pos[0]*8+ws[1].pos[1], board.black_king_pos[0]*8+board.black_king_pos[1], b_nk[0].pos[0]*8+b_nk[0].pos[1], opp_turn_idx, _W5V_HAS_PAWN, type(ws[0]).__name__, type(ws[1]).__name__))
            elif ext is not None and ext > 0: known_wins.append(ext)
            else: escape = True
            board.unmake_move(record)
        # FIX-3: tuple instead of array for IPC
        return (flat, ('b', legal_moves, escape, tuple(known_wins), tuple(child_flats)))

def _gen_valid_5vs_indices_numpy(total, w1n, w2n, bn, has_pawn, same_wp, chunk_size=2_000_000):
    wk_list = np.array(PAWN_WK_SQUARES if has_pawn else NON_PAWN_WK_SQUARES)
    for start in range(0, total, chunk_size):
        end_c = min(start + chunk_size, total); flat = np.arange(start, end_c, dtype=np.int64); rest = flat // 2
        bp_arr = rest % 64; rest //= 64; bk_arr = rest % 64; rest //= 64
        wp2_arr = rest % 64; rest //= 64; wp1_arr = rest % 64; wk_idx = rest // 64; wk_arr = wk_list[wk_idx]
        mask = ((wk_arr != wp1_arr) & (wk_arr != wp2_arr) & (wk_arr != bk_arr) & (wk_arr != bp_arr) &
                (wp1_arr != wp2_arr) & (wp1_arr != bk_arr) & (wp1_arr != bp_arr) &
                (wp2_arr != bk_arr) & (wp2_arr != bp_arr) & (bk_arr != bp_arr))
        if w1n == "Pawn": mask &= (wp1_arr >= 8) & (wp1_arr < 56)
        if w2n == "Pawn": mask &= (wp2_arr >= 8) & (wp2_arr < 56)
        if bn  == "Pawn": mask &= (bp_arr  >= 8) & (bp_arr  < 56)
        if same_wp: mask &= (wp1_arr <= wp2_arr)
        if w1n == "Bishop" and w2n == "Bishop":
            mask &= ((wp1_arr // 8 + wp1_arr % 8) % 2) != ((wp2_arr // 8 + wp2_arr % 8) % 2)
        if not mask.any(): continue
        yield flat[mask].tolist()

class Generator5Vs:
    def __init__(self, w1_class, w2_class, b_class):
        ws = sorted([w1_class, w2_class], key=lambda c: (_PIECE_CANONICAL_ORDER.get(c,99), c.__name__))
        self.w1_name, self.w2_name = ws[0].__name__, ws[1].__name__; self.b_name = b_class.__name__
        self.same_wp = (self.w1_name == self.w2_name)
        self.has_pawn = ("Pawn" in {self.w1_name, self.w2_name, self.b_name})
        self.filename = os.path.join(TB_DIR, f"K_{self.w1_name}_{self.w2_name}_vs_{self.b_name}_K_xsml.bin")
        self.wk_size = 32 if self.has_pawn else 10
        self.total_positions = self.wk_size * 64 * 64 * 64 * 64 * 2
        self.table = np.zeros((self.wk_size, 64, 64, 64, 64, 2), dtype=np.int8)
        self.transition_workers = max(1, (os.cpu_count() or 2) - TB_THREADS_SUBTRACT)

    def generate(self):
        if os.path.exists(self.filename): return
        start_time = time.time()
        sz_mb = self.wk_size * 64**4 * 2 * 1 / 1024**2  # FIX-2: int8 = 1 byte
        print(f"\n{'='*60}\n GENERATING: K+{self.w1_name}+{self.w2_name} vs K+{self.b_name} (XSML)", flush=True)
        print(f" Estimated file size: {sz_mb:.0f} MB\n{'='*60}", flush=True)
        s1 = time.time(); print(f"[Stage 1] Enumerating candidates...", flush=True)
        all_flats = array('I')
        for chunk in _gen_valid_5vs_indices_numpy(self.total_positions, self.w1_name, self.w2_name, self.b_name, self.has_pawn, self.same_wp):
            all_flats.extend(chunk)
        total = len(all_flats)
        print(f"[Stage 1] Done: {total:,} candidates | {_fmt_elapsed(time.time()-s1)}", flush=True)
        s2 = time.time(); print(f"[Stage 2] Building transition cache ({total:,} states)...", flush=True)
        unsolved_flats = array('I'); transitions = []
        trans_lookup = {}; btm_to_wtm = defaultdict(list); wtm_to_btm = defaultdict(list)
        with ProcessPoolExecutor(max_workers=self.transition_workers, initializer=_init_transition_worker_5vs,
                                 initargs=(self.w1_name, self.w2_name, self.b_name)) as ex:
            # FIX-4: chunksize reduced from 2048 to 256 for low-RAM machines
            for done, result in enumerate(ex.map(_build_transition_worker_5vs, all_flats, chunksize=256), 1):
                if result is not None:
                    flat, trans = result
                    unsolved_flats.append(flat); transitions.append(trans); trans_lookup[flat] = trans
                    if (flat & 1) == 0:
                        for cflat in trans[3]: btm_to_wtm[cflat].append(flat)
                    else:
                        if not trans[2]:
                            for cflat in trans[4]: wtm_to_btm[cflat].append(flat)
                if done % 100_000 == 0:
                    elapsed = time.time() - start_time; speed = done / max(0.001, time.time() - s2)
                    print(f"  > {done/total*100:.1f}% ({done:,}/{total:,}) | {speed:,.0f} st/s | Elapsed: {_fmt_elapsed(elapsed)}", end='\r', flush=True)
        print(); s2e = time.time() - s2
        print(f"[Stage 2] Cache built: {len(unsolved_flats):,} valid states | {_fmt_elapsed(s2e)} ({total/max(0.001,s2e):,.0f} st/s avg)", flush=True)
        candidate_states = len(unsolved_flats)
        max_dtm, decisive = _run_bfs_retrograde(self.table, unsolved_flats, transitions, trans_lookup, btm_to_wtm, wtm_to_btm, candidate_states, start_time)
        with open(self.filename, 'wb') as f: self.table.tofile(f)
        elapsed = time.time() - start_time
        _safe_record_longest_mate(f"K_{self.w1_name}_{self.w2_name}_vs_{self.b_name}_K", max_dtm, decisive, candidate_states - decisive, elapsed)
        print(f"SUCCESS: {self.filename} | Total time: {_fmt_elapsed(elapsed)}", flush=True)

# ==============================================================================
# ENTRY POINT
# ==============================================================================

def is_possible(pieces):
    """Filters out impossible piece combinations (e.g. 3 Knights)"""
    counts = {Queen: 0, Rook: 0, Bishop: 0, Knight: 0, Pawn: 0}
    for p in pieces: counts[p] += 1
    if counts[Rook] > 2 or counts[Bishop] > 2 or counts[Knight] > 2: return False
    return True

if __name__ == "__main__":
    _install_main_interrupt_ignores()
    overall_start = time.time()
    print("=== Tablebase Generator (v13.3 - Tuple IPC / Low-RAM fix) ===")

    Q, R, B, N, P = Queen, Rook, Bishop, Knight, Pawn
    non_pawn = [Q, R, B, N]

    print("\n--- TIER 1: 3-MAN NON-PAWN ---")
    for pc in non_pawn: Generator(pc).generate()

    print("\n--- TIER 2: 3-MAN PAWN ---")
    Generator(P).generate()

    print("\n--- TIER 3: 4-MAN NON-PAWN SAME-SIDE ---")
    for p1, p2 in combinations_with_replacement(non_pawn, 2):
        if is_possible([p1, p2]): Generator4(p1, p2).generate()

    print("\n--- TIER 3b: 4-MAN NON-PAWN CROSS ---")
    seen_vs = set()
    for w in non_pawn:
        for b in non_pawn:
            key = tuple(sorted([w.__name__, b.__name__]))
            if key not in seen_vs: seen_vs.add(key); Generator4Vs(w, b).generate()

    print("\n--- TIER 4: 4-MAN PAWN SAME-SIDE ---")
    for pc in [Q, R, B, N, P]:
        if is_possible([P, pc]): Generator4(P, pc).generate()

    print("\n--- TIER 4b: 4-MAN PAWN CROSS ---")
    seen_vs4 = set()
    for pc in [Q, R, B, N, P]:
        key = tuple(sorted(["Pawn", pc.__name__]))
        if key not in seen_vs4: seen_vs4.add(key); Generator4Vs(P, pc).generate()

    print("\n--- TIER 5: 5-MAN NON-PAWN SAME-SIDE (K+3 vs K) ---")
    for p1, p2, p3 in combinations_with_replacement(non_pawn, 3):
        if is_possible([p1, p2, p3]): Generator5(p1, p2, p3).generate()

    print("\n--- TIER 5b: 5-MAN NON-PAWN CROSS (K+2 vs K+1) ---")
    seen_5vs = set()
    for w1 in non_pawn:
        for w2 in non_pawn:
            if not is_possible([w1, w2]): continue
            for b in non_pawn:
                ws = sorted([w1.__name__, w2.__name__], key=lambda n: (_PIECE_CANONICAL_ORDER.get(PIECE_CLASS_BY_NAME[n],99), n))
                key = (ws[0], ws[1], b.__name__)
                if key not in seen_5vs: seen_5vs.add(key); Generator5Vs(PIECE_CLASS_BY_NAME[ws[0]], PIECE_CLASS_BY_NAME[ws[1]], b).generate()

    print("\n--- TIER 6: 5-MAN PAWN SAME-SIDE ---")
    for p1, p2 in combinations_with_replacement(non_pawn, 2):
        if is_possible([p1, p2, P]): Generator5(p1, p2, P).generate()
    for pc in non_pawn:
        if is_possible([pc, P, P]): Generator5(pc, P, P).generate()
    Generator5(P, P, P).generate()

    print("\n--- TIER 6b: 5-MAN PAWN CROSS ---")
    seen_5vp = set()
    all_pieces = [Q, R, B, N, P]
    for w1 in all_pieces:
        for w2 in all_pieces:
            if not is_possible([w1, w2]): continue
            for b in all_pieces:
                if "Pawn" not in {w1.__name__, w2.__name__, b.__name__}: continue
                ws = sorted([w1.__name__, w2.__name__], key=lambda n: (_PIECE_CANONICAL_ORDER.get(PIECE_CLASS_BY_NAME[n],99), n))
                key = (ws[0], ws[1], b.__name__)
                if key not in seen_5vp: seen_5vp.add(key); Generator5Vs(PIECE_CLASS_BY_NAME[ws[0]], PIECE_CLASS_BY_NAME[ws[1]], b).generate()

    overall_elapsed = time.time() - overall_start
    print(f"\n\n=== ALL TABLEBASE GENERATION COMPLETE ===")
    print(f"=== Total overall runtime: {_fmt_elapsed(overall_elapsed)} ===")