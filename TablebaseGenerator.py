# TablebaseGenerator.py (v17.0 - Unified 16-bit Tablebase Generator)
#
# Features:
# - External Disk Binning: Zero RAM, zero disk-thrashing O(N log N) graph sort.
# - Out-of-Core Memmap Streaming: Solves 16GB graphs using OS Page Caching.
# - Bidirectional Topological Out-Degree BFS Retrograde Analysis.
# - Unified 16-bit table storage and loading across the entire pipeline.
# - Standardized 5-element Tuple IPC for all processes.
# - Mathematical Topological Sorting for flawless Pawn Promotion routing.

import os
import time
import signal
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations_with_replacement
import numpy as np
from GameLogic import *
from array import array

TB_DIR = "tablebases"
os.makedirs(TB_DIR, exist_ok=True)

TB_SUFFIX = "_tb16.bin"
TB_DTYPE = np.int16
MAX_DTM = 32766
LONGEST_MATES_NOTE_FILE = os.path.join(TB_DIR, "longest_mates_tb16.tsv")
LONGEST_MATE_KEY_PREFIX = "regen_"
TB_THREADS_SUBTRACT = 2
_IN_TABLE_SENTINEL = "IN_TABLE"
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
    data = np.fromfile(filename, dtype=TB_DTYPE)
    has_pawn = "Pawn" in os.path.basename(filename)
    return data.reshape((32 if has_pawn else 10, 64, 64, 2))

def _load_4man_table_file(filename):
    data = np.fromfile(filename, dtype=TB_DTYPE)
    has_pawn = "Pawn" in os.path.basename(filename)
    return data.reshape((32 if has_pawn else 10, 64, 64, 64, 2))

def _load_5man_table_file(filename):
    data = np.fromfile(filename, dtype=TB_DTYPE)
    has_pawn = "Pawn" in os.path.basename(filename)
    return data.reshape((32 if has_pawn else 10, 64, 64, 64, 64, 2))

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

def _write_dense_table(filename, total_positions, solved_flats_np, solved_values):
    dense_table = np.zeros(total_positions, dtype=TB_DTYPE)
    dense_table[solved_flats_np] = solved_values
    with open(filename, 'wb') as f:
        dense_table.tofile(f)

def _make_temp_graph_paths(output_filename):
    base_name = os.path.basename(output_filename).replace(TB_SUFFIX, "")
    run_prefix = f"{base_name}_{os.getpid()}_{time.time_ns()}"
    tmp_b2w = os.path.join(TB_DIR, f"{run_prefix}_tmp_b2w.bin")
    tmp_w2b = os.path.join(TB_DIR, f"{run_prefix}_tmp_w2b.bin")
    return run_prefix, tmp_b2w, tmp_w2b

# ==============================================================================
# DISK-BACKED ZERO-RAM CSR BUILDER & FLAT BFS
# ==============================================================================

def _build_csr_from_disk(unsolved_flats_np, edges_file, prefix):
    """
    Reads a temporary binary file containing (cflat, parent_idx) uint32 pairs.
    Splits into bins, sorts in-memory sequentially, and writes to memmap.
    Keeps RAM footprint < 2GB even for 16GB edge files.
    """
    num_states = len(unsolved_flats_np)
    num_bins = 64
    bin_files = [open(os.path.join(TB_DIR, f"{prefix}_bin_{b}.bin"), "wb") for b in range(num_bins)]
    
    chunk_size = 10_000_000 
    with open(edges_file, "rb") as f:
        while True:
            buf = f.read(chunk_size * 8)
            if not buf: break
            data = np.frombuffer(buf, dtype=np.uint32).reshape(-1, 2)
            cflats = data[:, 0]
            parents = data[:, 1]
            
            idxs = np.searchsorted(unsolved_flats_np, cflats)
            valid = (idxs < num_states) & (unsolved_flats_np[np.minimum(idxs, num_states-1)] == cflats)
            
            v_idx = idxs[valid]
            v_par = parents[valid]
            
            if len(v_idx) == 0: continue
            
            bin_assignments = (v_idx.astype(np.uint64) * num_bins) // num_states
            
            for b in range(num_bins):
                mask = (bin_assignments == b)
                if mask.any():
                    arr = np.empty((np.count_nonzero(mask), 2), dtype=np.uint32)
                    arr[:, 0] = v_idx[mask]
                    arr[:, 1] = v_par[mask]
                    bin_files[b].write(arr.tobytes())
                    
    for f in bin_files: f.close()
    
    head = np.zeros(num_states + 1, dtype=np.uint32)
    edges_out_file = os.path.join(TB_DIR, f"{prefix}_edges.bin")
    
    with open(edges_out_file, "wb") as f_out:
        for b in range(num_bins):
            bin_path = os.path.join(TB_DIR, f"{prefix}_bin_{b}.bin")
            data = np.fromfile(bin_path, dtype=np.uint32).reshape(-1, 2)
            if len(data) > 0:
                v_idx = data[:, 0]
                v_par = data[:, 1]
                
                sort_i = np.argsort(v_idx)
                v_idx = v_idx[sort_i]
                v_par = v_par[sort_i]
                
                unique_idx, counts = np.unique(v_idx, return_counts=True)
                head[unique_idx + 1] = counts
                
                f_out.write(v_par.tobytes())
            os.remove(bin_path)
            
    np.cumsum(head, out=head)
    edges_out = np.memmap(edges_out_file, dtype=np.uint32, mode='r')
    
    return head, edges_out, edges_out_file

def _run_flat_bfs(unsolved_flats_np, init_table_np, init_out_degree_np,
                  wtm_promo_wins_idx, wtm_promo_wins_val,
                  btm_promo_losses_idx, btm_promo_losses_val,
                  b2w_head, b2w_edges, w2b_head, w2b_edges,
                  start_time):
                  
    num_states = len(unsolved_flats_np)
    local_table = init_table_np.astype(TB_DTYPE)
    out_degree = init_out_degree_np.copy()
    max_child_dtm = np.zeros(num_states, dtype=np.uint16)
    
    queues = [array('I') for _ in range(MAX_DTM + 2)]

    def _queue_forced_win(state_idx, dtm_value):
        dtm_int = int(dtm_value)
        if dtm_int > MAX_DTM:
            raise OverflowError(f"16-bit DTM overflow: {dtm_int}")
        local_table[state_idx] = dtm_int
        queues[dtm_int].append(int(state_idx))

    def _queue_forced_loss(state_idx, dtm_value):
        dtm_int = int(dtm_value)
        if dtm_int > MAX_DTM:
            raise OverflowError(f"16-bit DTM overflow: {dtm_int}")
        local_table[state_idx] = -dtm_int
        queues[dtm_int].append(int(state_idx))
    
    solved_idxs = np.where(local_table != 0)[0]
    solved_count = len(solved_idxs)
    for idx in solved_idxs:
        queues[1].append(idx)
        
    for i in range(len(wtm_promo_wins_idx)):
        idx = wtm_promo_wins_idx[i]
        val = wtm_promo_wins_val[i]
        if local_table[idx] == 0:
            _queue_forced_win(idx, val)
            
    for i in range(len(btm_promo_losses_idx)):
        idx = btm_promo_losses_idx[i]
        val = btm_promo_losses_val[i]
        if local_table[idx] == 0 and out_degree[idx] != 65535:
            if val > max_child_dtm[idx]:
                max_child_dtm[idx] = val
            out_degree[idx] -= 1
            if out_degree[idx] == 0:
                dtm = int(max_child_dtm[idx]) + 1
                _queue_forced_loss(idx, dtm)
                solved_count += 1
            
    max_dtm_solved = 1 if solved_count > 0 else 0
    
    print(f"[Stage 3] BFS start (16-bit) | Terminal (DTM 1): {solved_count:,} | "
          f"Elapsed so far: {_fmt_elapsed(time.time() - start_time)}", flush=True)

    for dtm in range(1, MAX_DTM + 1):
        q = queues[dtm]
        if not q: continue
        max_dtm_solved = dtm
        
        q_arr = np.frombuffer(q, dtype=np.uint32)
        changed = 0
        
        for idx in q_arr:
            t_idx = unsolved_flats_np[idx] & 1
            
            if local_table[idx] == 0:
                local_table[idx] = dtm if t_idx == 0 else -dtm
                changed += 1
            elif abs(local_table[idx]) != dtm:
                continue 
                
            if t_idx == 1: 
                start_e, end_e = b2w_head[idx], b2w_head[idx+1]
                for p_idx in b2w_edges[start_e:end_e]:
                    if local_table[p_idx] == 0:
                        _queue_forced_win(p_idx, dtm + 1)
                        changed += 1
                        
            else: 
                start_e, end_e = w2b_head[idx], w2b_head[idx+1]
                for p_idx in w2b_edges[start_e:end_e]:
                    if out_degree[p_idx] == 65535: continue
                    if max_child_dtm[p_idx] < dtm:
                        max_child_dtm[p_idx] = dtm
                    out_degree[p_idx] -= 1
                    if out_degree[p_idx] == 0:
                        p_dtm = int(max_child_dtm[p_idx]) + 1
                        _queue_forced_loss(p_idx, p_dtm)
                        changed += 1
                        
        if changed > 0:
            solved_count = np.count_nonzero(local_table)
            
        pending_count = sum(len(queues[i]) for i in range(dtm + 1, MAX_DTM + 1))
        
        if dtm == MAX_DTM and pending_count > 0:
            raise OverflowError("16-bit DTM overflowed the supported tablebase range.")

        print(f"  [DTM {dtm:^3}] Solved: {solved_count:,}/{num_states:,} "
              f"| Next Q: {len(queues[dtm+1]):<6,} | Held: {pending_count:<6,} ", flush=True)

    return local_table, max_dtm_solved, solved_count

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
                if nk is None or isinstance(nk, Queen): escape = True; board.unmake_move(record); continue
            p = next((x for x in board.white_pieces if not isinstance(x, King)), None)
            if p is None: escape = True; board.unmake_move(record); continue
            wkp, bkp = board.white_king_pos, board.black_king_pos
            child_flats.append(_canonical_flat_3(wkp[0]*8+wkp[1], p.pos[0]*8+p.pos[1], bkp[0]*8+bkp[1], opp_turn_idx, _W_HAS_PAWN))
            board.unmake_move(record)
        return (flat, ('b', legal_moves, escape, (), tuple(child_flats)))

class Generator:
    def __init__(self, piece_class):
        self.piece_name = piece_class.__name__; self.has_pawn = (self.piece_name == "Pawn")
        base_name = f"K_{self.piece_name}_K"
        self.filename = os.path.join(TB_DIR, f"{base_name}{TB_SUFFIX}")
        self.queen_tb_file = os.path.join(TB_DIR, f"K_Queen_K{TB_SUFFIX}")
        self.wk_size = 32 if self.has_pawn else 10
        self.total_positions = self.wk_size * 64 * 64 * 2
        self.table = None
        self.transition_workers = min(8, max(1, (os.cpu_count() or 2) - TB_THREADS_SUBTRACT))

    def generate(self):
        if os.path.exists(self.filename): return
        self.table = np.zeros((self.wk_size, 64, 64, 2), dtype=TB_DTYPE)
        start_time = time.time()
        print(f"\n{'='*60}\n GENERATING: K+{self.piece_name} vs K (TB16)\n{'='*60}")
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
        
        s2 = time.time(); print(f"[Stage 2] Building flat disk cache ({total:,} states)...", flush=True)
        unsolved_flats = array('I')
        init_table = array('h'); init_out_degree = array('H')
        wtm_promo_idx = array('I'); wtm_promo_val = array('H')
        btm_promo_idx = array('I'); btm_promo_val = array('H')
        
        run_prefix, tmp_b2w, tmp_w2b = _make_temp_graph_paths(self.filename)
        f_b2w = open(tmp_b2w, "wb")
        f_w2b = open(tmp_w2b, "wb")
        
        with ProcessPoolExecutor(max_workers=self.transition_workers, initializer=_init_transition_worker,
                                 initargs=(self.piece_name, self.queen_tb_file)) as ex:
            for done, result in enumerate(ex.map(_build_transition_worker, raw_candidates, chunksize=1024), 1):
                if result is not None:
                    flat, trans = result
                    i = len(unsolved_flats)
                    unsolved_flats.append(flat)
                    if (flat & 1) == 0:
                        if trans[1]: init_table.append(1); init_out_degree.append(0)
                        else:
                            init_table.append(0); init_out_degree.append(len(trans[3]))
                            if trans[2]: wtm_promo_idx.append(i); wtm_promo_val.append(min(trans[2]))
                        if trans[3]:
                            arr = np.empty((len(trans[3]), 2), dtype=np.uint32)
                            arr[:, 0] = trans[3]; arr[:, 1] = i
                            f_b2w.write(arr.tobytes())
                    else:
                        if trans[1] == 0: init_table.append(-1); init_out_degree.append(0)
                        elif trans[2]: init_table.append(0); init_out_degree.append(65535)
                        else:
                            promo_losses, children = trans[3], trans[4]
                            init_table.append(0); init_out_degree.append(len(children) + len(promo_losses))
                            for p_val in promo_losses: btm_promo_idx.append(i); btm_promo_val.append(p_val)
                            if children:
                                arr = np.empty((len(children), 2), dtype=np.uint32)
                                arr[:, 0] = children; arr[:, 1] = i
                                f_w2b.write(arr.tobytes())
                if done % 10_000 == 0:
                    elapsed = time.time() - start_time; speed = done / max(0.001, time.time() - s2)
                    print(f"  > {done/total*100:.1f}% ({done:,}/{total:,}) | {speed:,.0f} st/s | Elapsed: {_fmt_elapsed(elapsed)}", end='\r', flush=True)
        
        f_b2w.close(); f_w2b.close()
        print(); s2e = time.time() - s2
        print(f"[Stage 2] Cache built: {len(unsolved_flats):,} valid states | {_fmt_elapsed(s2e)} ({total/max(0.001,s2e):,.0f} st/s avg)", flush=True)
        
        s3 = time.time(); print(f"[Stage 2b] Flattening reverse graph via Disk CSR...", flush=True)
        unsolved_flats_np = np.array(unsolved_flats, dtype=np.uint32)
        b2w_head, b2w_edges, b2w_file = _build_csr_from_disk(unsolved_flats_np, tmp_b2w, f"{run_prefix}_b2w")
        w2b_head, w2b_edges, w2b_file = _build_csr_from_disk(unsolved_flats_np, tmp_w2b, f"{run_prefix}_w2b")
        print(f"[Stage 2b] Done | {_fmt_elapsed(time.time()-s3)}", flush=True)
        
        try:
            table_flat, max_dtm, decisive = _run_flat_bfs(
                unsolved_flats_np, np.array(init_table, dtype=TB_DTYPE), np.array(init_out_degree, dtype=np.uint16),
                np.array(wtm_promo_idx, dtype=np.uint32), np.array(wtm_promo_val, dtype=np.uint16),
                np.array(btm_promo_idx, dtype=np.uint32), np.array(btm_promo_val, dtype=np.uint16),
                b2w_head, b2w_edges, w2b_head, w2b_edges, start_time)

            _write_dense_table(self.filename, self.total_positions, unsolved_flats_np, table_flat)
                
            elapsed = time.time() - start_time
            tb_key = os.path.basename(self.filename).replace(TB_SUFFIX, '')
            _safe_record_longest_mate(tb_key, max_dtm, decisive, len(unsolved_flats_np) - decisive, elapsed)
            print(f"SUCCESS: {self.filename} | Total time: {_fmt_elapsed(elapsed)}", flush=True)

        finally:
            if hasattr(b2w_edges, '_mmap'): b2w_edges._mmap.close()
            if hasattr(w2b_edges, '_mmap'): w2b_edges._mmap.close()
            del b2w_edges; del w2b_edges
            try:
                os.remove(tmp_b2w); os.remove(tmp_w2b)
                os.remove(b2w_file); os.remove(w2b_file)
            except Exception as e:
                print(f"  [Warning] Temporary file cleanup failed: {e}")

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
        path = os.path.join(TB_DIR, f"K_{name}_K{TB_SUFFIX}")
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
        base_name = f"K_{self.p1_name}_{self.p2_name}_K"
        self.filename = os.path.join(TB_DIR, f"{base_name}{TB_SUFFIX}")
        self.wk_size = 32 if self.has_pawn else 10
        self.total_positions = self.wk_size * 64 * 64 * 64 * 2
        self.table = None
        self.promo_tb_file = None
        if self.has_pawn:
            other = self.p2_name if self.p1_name == "Pawn" else self.p1_name
            p_names = sorted(["Pawn", "Queen"]) if self.same_piece else sorted(["Queen", other])
            self.promo_tb_file = os.path.join(TB_DIR, f"K_{p_names[0]}_{p_names[1]}_K{TB_SUFFIX}")
        self.transition_workers = max(1, (os.cpu_count() or 2) - TB_THREADS_SUBTRACT)

    def generate(self):
        if os.path.exists(self.filename): return
        self.table = np.zeros((self.wk_size, 64, 64, 64, 2), dtype=TB_DTYPE)
        start_time = time.time()
        print(f"\n{'='*60}\n GENERATING: K+{self.p1_name}+{self.p2_name} vs K (TB16)\n{'='*60}", flush=True)
        s1 = time.time(); print(f"[Stage 1] Enumerating candidates...", flush=True)
        all_flats = array('I')
        for chunk in _gen_valid_4man_indices_numpy(self.total_positions, self.p1_name, self.p2_name, self.same_piece, self.has_pawn):
            all_flats.extend(chunk)
        total = len(all_flats)
        print(f"[Stage 1] Done: {total:,} candidates | {_fmt_elapsed(time.time()-s1)}", flush=True)
        
        s2 = time.time(); print(f"[Stage 2] Building flat disk cache ({total:,} states)...", flush=True)
        unsolved_flats = array('I')
        init_table = array('h'); init_out_degree = array('H')
        wtm_promo_idx = array('I'); wtm_promo_val = array('H')
        btm_promo_idx = array('I'); btm_promo_val = array('H')
        
        run_prefix, tmp_b2w, tmp_w2b = _make_temp_graph_paths(self.filename)
        f_b2w = open(tmp_b2w, "wb")
        f_w2b = open(tmp_w2b, "wb")
        
        with ProcessPoolExecutor(max_workers=self.transition_workers, initializer=_init_transition_worker_4,
                                 initargs=(self.p1_name, self.p2_name, self.promo_tb_file)) as ex:
            for done, result in enumerate(ex.map(_build_transition_worker_4, all_flats, chunksize=4096), 1):
                if result is not None:
                    flat, trans = result
                    i = len(unsolved_flats)
                    unsolved_flats.append(flat)
                    if (flat & 1) == 0:
                        if trans[1]: init_table.append(1); init_out_degree.append(0)
                        else:
                            init_table.append(0); init_out_degree.append(len(trans[3]))
                            if trans[2]: wtm_promo_idx.append(i); wtm_promo_val.append(min(trans[2]))
                        if trans[3]:
                            arr = np.empty((len(trans[3]), 2), dtype=np.uint32)
                            arr[:, 0] = trans[3]; arr[:, 1] = i
                            f_b2w.write(arr.tobytes())
                    else:
                        if trans[1] == 0: init_table.append(-1); init_out_degree.append(0)
                        elif trans[2]: init_table.append(0); init_out_degree.append(65535)
                        else:
                            promo_losses, children = trans[3], trans[4]
                            init_table.append(0); init_out_degree.append(len(children) + len(promo_losses))
                            for p_val in promo_losses: btm_promo_idx.append(i); btm_promo_val.append(p_val)
                            if children:
                                arr = np.empty((len(children), 2), dtype=np.uint32)
                                arr[:, 0] = children; arr[:, 1] = i
                                f_w2b.write(arr.tobytes())
                if done % 50_000 == 0:
                    elapsed = time.time() - start_time; speed = done / max(0.001, time.time() - s2)
                    print(f"  > {done/total*100:.1f}% ({done:,}/{total:,}) | {speed:,.0f} st/s | Elapsed: {_fmt_elapsed(elapsed)}", end='\r', flush=True)
        
        f_b2w.close(); f_w2b.close()
        print(); s2e = time.time() - s2
        print(f"[Stage 2] Cache built: {len(unsolved_flats):,} valid states | {_fmt_elapsed(s2e)} ({total/max(0.001,s2e):,.0f} st/s avg)", flush=True)
        
        s3 = time.time(); print(f"[Stage 2b] Flattening reverse graph via Disk CSR...", flush=True)
        unsolved_flats_np = np.array(unsolved_flats, dtype=np.uint32)
        b2w_head, b2w_edges, b2w_file = _build_csr_from_disk(unsolved_flats_np, tmp_b2w, f"{run_prefix}_b2w")
        w2b_head, w2b_edges, w2b_file = _build_csr_from_disk(unsolved_flats_np, tmp_w2b, f"{run_prefix}_w2b")
        print(f"[Stage 2b] Done | {_fmt_elapsed(time.time()-s3)}", flush=True)
        
        try:
            table_flat, max_dtm, decisive = _run_flat_bfs(
                unsolved_flats_np, np.array(init_table, dtype=TB_DTYPE), np.array(init_out_degree, dtype=np.uint16),
                np.array(wtm_promo_idx, dtype=np.uint32), np.array(wtm_promo_val, dtype=np.uint16),
                np.array(btm_promo_idx, dtype=np.uint32), np.array(btm_promo_val, dtype=np.uint16),
                b2w_head, b2w_edges, w2b_head, w2b_edges, start_time)

            _write_dense_table(self.filename, self.total_positions, unsolved_flats_np, table_flat)
                
            elapsed = time.time() - start_time
            tb_key = os.path.basename(self.filename).replace(TB_SUFFIX, '')
            _safe_record_longest_mate(tb_key, max_dtm, decisive, len(unsolved_flats_np) - decisive, elapsed)
            print(f"SUCCESS: {self.filename} | Total time: {_fmt_elapsed(elapsed)}", flush=True)

        finally:
            if hasattr(b2w_edges, '_mmap'): b2w_edges._mmap.close()
            if hasattr(w2b_edges, '_mmap'): w2b_edges._mmap.close()
            del b2w_edges; del w2b_edges
            try:
                os.remove(tmp_b2w); os.remove(tmp_w2b)
                os.remove(b2w_file); os.remove(w2b_file)
            except Exception as e:
                print(f"  [Warning] Temporary file cleanup failed: {e}")

# ==============================================================================
# 4-MAN CROSS GENERATOR
# ==============================================================================
_W4V_HAS_PAWN = False; _W4V_W_NAME = None; _W4V_B_NAME = None
_W4V_3MAN_TABLES = {}; _W4V_PROMO_TABLES = {}
_W4V_BOARD = None; _W4V_WK_OBJ = None; _W4V_WP_OBJ = None; _W4V_BK_OBJ = None; _W4V_BP_OBJ = None

def _init_transition_worker_4vs(w_name, b_name):
    _install_worker_interrupt_ignores()
    global _W4V_W_NAME, _W4V_B_NAME, _W4V_HAS_PAWN, _W4V_3MAN_TABLES, _W4V_PROMO_TABLES
    global _W4V_BOARD, _W4V_WK_OBJ, _W4V_WP_OBJ, _W4V_BK_OBJ, _W4V_BP_OBJ
    names = sorted([w_name, b_name]); _W4V_W_NAME, _W4V_B_NAME = names[0], names[1]
    _W4V_HAS_PAWN = ("Pawn" in {w_name, b_name}); _W4V_3MAN_TABLES.clear(); _W4V_PROMO_TABLES.clear()
    names_needed = {w_name, b_name}
    if _W4V_HAS_PAWN: names_needed.add("Queen")
    for name in names_needed:
        path = os.path.join(TB_DIR, f"K_{name}_K{TB_SUFFIX}")
        if os.path.exists(path): _W4V_3MAN_TABLES[name] = _load_3man_table_file(path)
    promo_targets = set()
    if w_name == "Pawn": promo_targets.add(("Queen", b_name))
    if b_name == "Pawn": promo_targets.add((w_name, "Queen"))
    for key in promo_targets:
        sk = tuple(sorted(key)); path = os.path.join(TB_DIR, f"K_{sk[0]}_vs_{sk[1]}_K{TB_SUFFIX}")
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
        base_name = f"K_{self.w_name}_vs_{self.b_name}_K"
        self.filename = os.path.join(TB_DIR, f"{base_name}{TB_SUFFIX}")
        self.wk_size   = 32 if self.has_pawn else 10
        self.total_positions = self.wk_size * 64 * 64 * 64 * 2
        self.table     = None
        self.transition_workers = max(1, (os.cpu_count() or 2) - TB_THREADS_SUBTRACT)

    def generate(self):
        if os.path.exists(self.filename): return
        self.table = np.zeros((self.wk_size, 64, 64, 64, 2), dtype=TB_DTYPE)
        start_time = time.time()
        print(f"\n{'='*60}\n GENERATING: K+{self.w_name} vs K+{self.b_name} (TB16)\n{'='*60}", flush=True)
        s1 = time.time(); print(f"[Stage 1] Enumerating candidates...", flush=True)
        all_flats = array('I')
        for chunk in _gen_valid_4man_vs_indices_numpy(self.total_positions, self.w_name, self.b_name, self.has_pawn):
            all_flats.extend(chunk)
        total = len(all_flats)
        print(f"[Stage 1] Done: {total:,} candidates | {_fmt_elapsed(time.time()-s1)}", flush=True)
        
        s2 = time.time(); print(f"[Stage 2] Building flat disk cache ({total:,} states)...", flush=True)
        unsolved_flats = array('I')
        init_table = array('h'); init_out_degree = array('H')
        wtm_promo_idx = array('I'); wtm_promo_val = array('H')
        btm_promo_idx = array('I'); btm_promo_val = array('H')
        
        run_prefix, tmp_b2w, tmp_w2b = _make_temp_graph_paths(self.filename)
        f_b2w = open(tmp_b2w, "wb")
        f_w2b = open(tmp_w2b, "wb")
        
        with ProcessPoolExecutor(max_workers=self.transition_workers, initializer=_init_transition_worker_4vs,
                                 initargs=(self.w_name, self.b_name)) as ex:
            for done, result in enumerate(ex.map(_build_transition_worker_4vs, all_flats, chunksize=4096), 1):
                if result is not None:
                    flat, trans = result
                    i = len(unsolved_flats)
                    unsolved_flats.append(flat)
                    if (flat & 1) == 0:
                        if trans[1]: init_table.append(1); init_out_degree.append(0)
                        else:
                            init_table.append(0); init_out_degree.append(len(trans[3]))
                            if trans[2]: wtm_promo_idx.append(i); wtm_promo_val.append(min(trans[2]))
                        if trans[3]:
                            arr = np.empty((len(trans[3]), 2), dtype=np.uint32)
                            arr[:, 0] = trans[3]; arr[:, 1] = i
                            f_b2w.write(arr.tobytes())
                    else:
                        if trans[1] == 0: init_table.append(-1); init_out_degree.append(0)
                        elif trans[2]: init_table.append(0); init_out_degree.append(65535)
                        else:
                            promo_losses, children = trans[3], trans[4]
                            init_table.append(0); init_out_degree.append(len(children) + len(promo_losses))
                            for p_val in promo_losses: btm_promo_idx.append(i); btm_promo_val.append(p_val)
                            if children:
                                arr = np.empty((len(children), 2), dtype=np.uint32)
                                arr[:, 0] = children; arr[:, 1] = i
                                f_w2b.write(arr.tobytes())
                if done % 50_000 == 0:
                    elapsed = time.time() - start_time; speed = done / max(0.001, time.time() - s2)
                    print(f"  > {done/total*100:.1f}% ({done:,}/{total:,}) | {speed:,.0f} st/s | Elapsed: {_fmt_elapsed(elapsed)}", end='\r', flush=True)
        
        f_b2w.close(); f_w2b.close()
        print(); s2e = time.time() - s2
        print(f"[Stage 2] Cache built: {len(unsolved_flats):,} valid states | {_fmt_elapsed(s2e)} ({total/max(0.001,s2e):,.0f} st/s avg)", flush=True)
        
        s3 = time.time(); print(f"[Stage 2b] Flattening reverse graph via Disk CSR...", flush=True)
        unsolved_flats_np = np.array(unsolved_flats, dtype=np.uint32)
        b2w_head, b2w_edges, b2w_file = _build_csr_from_disk(unsolved_flats_np, tmp_b2w, f"{run_prefix}_b2w")
        w2b_head, w2b_edges, w2b_file = _build_csr_from_disk(unsolved_flats_np, tmp_w2b, f"{run_prefix}_w2b")
        print(f"[Stage 2b] Done | {_fmt_elapsed(time.time()-s3)}", flush=True)
        
        try:
            table_flat, max_dtm, decisive = _run_flat_bfs(
                unsolved_flats_np, np.array(init_table, dtype=TB_DTYPE), np.array(init_out_degree, dtype=np.uint16),
                np.array(wtm_promo_idx, dtype=np.uint32), np.array(wtm_promo_val, dtype=np.uint16),
                np.array(btm_promo_idx, dtype=np.uint32), np.array(btm_promo_val, dtype=np.uint16),
                b2w_head, b2w_edges, w2b_head, w2b_edges, start_time)

            _write_dense_table(self.filename, self.total_positions, unsolved_flats_np, table_flat)
                
            elapsed = time.time() - start_time
            tb_key = os.path.basename(self.filename).replace(TB_SUFFIX, '')
            _safe_record_longest_mate(tb_key, max_dtm, decisive, len(unsolved_flats_np) - decisive, elapsed)
            print(f"SUCCESS: {self.filename} | Total time: {_fmt_elapsed(elapsed)}", flush=True)

        finally:
            if hasattr(b2w_edges, '_mmap'): b2w_edges._mmap.close()
            if hasattr(w2b_edges, '_mmap'): w2b_edges._mmap.close()
            del b2w_edges; del w2b_edges
            try:
                os.remove(tmp_b2w); os.remove(tmp_w2b)
                os.remove(b2w_file); os.remove(w2b_file)
            except Exception as e:
                print(f"  [Warning] Temporary file cleanup failed: {e}")

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
        path = os.path.join(TB_DIR, f"K_{name}_K{TB_SUFFIX}")
        if os.path.exists(path): _W5_3MAN_TABLES[name] = _load_3man_table_file(path)
    pieces = sorted(PIECE_CLASS_BY_NAME.keys()); seen = set()
    for n1 in pieces:
        for n2 in pieces:
            key = tuple(sorted([n1, n2]))
            if key in seen: continue
            seen.add(key)
            path = os.path.join(TB_DIR, f"K_{key[0]}_{key[1]}_K{TB_SUFFIX}")
            if os.path.exists(path): _W5_4MAN_TABLES[f"{key[0]}_{key[1]}"] = _load_4man_table_file(path)
    if _W5_HAS_PAWN:
        names = [p1_name, p2_name, p3_name]
        if "Pawn" in names:
            idx = names.index("Pawn"); promo_names = names.copy(); promo_names[idx] = "Queen"
            promo_names.sort(key=lambda n: (_PIECE_CANONICAL_ORDER.get(PIECE_CLASS_BY_NAME[n], 99), n))
            tb_name = f"K_{promo_names[0]}_{promo_names[1]}_{promo_names[2]}_K{TB_SUFFIX}"
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
        base_name = f"K_{self.p1_name}_{self.p2_name}_{self.p3_name}_K"
        self.filename = os.path.join(TB_DIR, f"{base_name}{TB_SUFFIX}")
        self.wk_size = 32 if self.has_pawn else 10
        self.total_positions = self.wk_size * 64 * 64 * 64 * 64 * 2
        self.table = None
        self.transition_workers = max(1, (os.cpu_count() or 2) - TB_THREADS_SUBTRACT)

    def generate(self):
        if os.path.exists(self.filename): return
        self.table = np.zeros((self.wk_size, 64, 64, 64, 64, 2), dtype=TB_DTYPE)
        start_time = time.time()
        sz_mb = self.wk_size * 64**4 * 2 * 1 / 1024**2
        print(f"\n{'='*60}\n GENERATING: K+{self.p1_name}+{self.p2_name}+{self.p3_name} vs K (TB16)", flush=True)
        print(f" Estimated file size: {sz_mb:.0f} MB\n{'='*60}", flush=True)
        s1 = time.time(); print(f"[Stage 1] Enumerating candidates...", flush=True)
        all_flats = array('I')
        for chunk in _gen_valid_5same_indices_numpy(self.total_positions, self.p1_name, self.p2_name, self.p3_name, self.has_pawn):
            all_flats.extend(chunk)
        total = len(all_flats)
        print(f"[Stage 1] Done: {total:,} candidates | {_fmt_elapsed(time.time()-s1)}", flush=True)
        
        s2 = time.time(); print(f"[Stage 2] Building flat disk cache ({total:,} states)...", flush=True)
        unsolved_flats = array('I')
        init_table = array('h'); init_out_degree = array('H')
        wtm_promo_idx = array('I'); wtm_promo_val = array('H')
        btm_promo_idx = array('I'); btm_promo_val = array('H')
        
        run_prefix, tmp_b2w, tmp_w2b = _make_temp_graph_paths(self.filename)
        f_b2w = open(tmp_b2w, "wb")
        f_w2b = open(tmp_w2b, "wb")
        
        with ProcessPoolExecutor(max_workers=self.transition_workers, initializer=_init_transition_worker_5,
                                 initargs=(self.p1_name, self.p2_name, self.p3_name)) as ex:
            for done, result in enumerate(ex.map(_build_transition_worker_5, all_flats, chunksize=256), 1):
                if result is not None:
                    flat, trans = result
                    i = len(unsolved_flats)
                    unsolved_flats.append(flat)
                    if (flat & 1) == 0:
                        if trans[1]: init_table.append(1); init_out_degree.append(0)
                        else:
                            init_table.append(0); init_out_degree.append(len(trans[3]))
                            if trans[2]: wtm_promo_idx.append(i); wtm_promo_val.append(min(trans[2]))
                        if trans[3]:
                            arr = np.empty((len(trans[3]), 2), dtype=np.uint32)
                            arr[:, 0] = trans[3]; arr[:, 1] = i
                            f_b2w.write(arr.tobytes())
                    else:
                        if trans[1] == 0: init_table.append(-1); init_out_degree.append(0)
                        elif trans[2]: init_table.append(0); init_out_degree.append(65535)
                        else:
                            promo_losses, children = trans[3], trans[4]
                            init_table.append(0); init_out_degree.append(len(children) + len(promo_losses))
                            for p_val in promo_losses: btm_promo_idx.append(i); btm_promo_val.append(p_val)
                            if children:
                                arr = np.empty((len(children), 2), dtype=np.uint32)
                                arr[:, 0] = children; arr[:, 1] = i
                                f_w2b.write(arr.tobytes())
                if done % 100_000 == 0:
                    elapsed = time.time() - start_time; speed = done / max(0.001, time.time() - s2)
                    print(f"  > {done/total*100:.1f}% ({done:,}/{total:,}) | {speed:,.0f} st/s | Elapsed: {_fmt_elapsed(elapsed)}", end='\r', flush=True)
        
        f_b2w.close(); f_w2b.close()
        print(); s2e = time.time() - s2
        print(f"[Stage 2] Cache built: {len(unsolved_flats):,} valid states | {_fmt_elapsed(s2e)} ({total/max(0.001,s2e):,.0f} st/s avg)", flush=True)
        
        s3 = time.time(); print(f"[Stage 2b] Flattening reverse graph via Disk CSR...", flush=True)
        unsolved_flats_np = np.array(unsolved_flats, dtype=np.uint32)
        b2w_head, b2w_edges, b2w_file = _build_csr_from_disk(unsolved_flats_np, tmp_b2w, f"{run_prefix}_b2w")
        w2b_head, w2b_edges, w2b_file = _build_csr_from_disk(unsolved_flats_np, tmp_w2b, f"{run_prefix}_w2b")
        print(f"[Stage 2b] Done | {_fmt_elapsed(time.time()-s3)}", flush=True)
        
        try:
            table_flat, max_dtm, decisive = _run_flat_bfs(
                unsolved_flats_np, np.array(init_table, dtype=TB_DTYPE), np.array(init_out_degree, dtype=np.uint16),
                np.array(wtm_promo_idx, dtype=np.uint32), np.array(wtm_promo_val, dtype=np.uint16),
                np.array(btm_promo_idx, dtype=np.uint32), np.array(btm_promo_val, dtype=np.uint16),
                b2w_head, b2w_edges, w2b_head, w2b_edges, start_time)

            _write_dense_table(self.filename, self.total_positions, unsolved_flats_np, table_flat)
                
            elapsed = time.time() - start_time
            tb_key = os.path.basename(self.filename).replace(TB_SUFFIX, '')
            _safe_record_longest_mate(tb_key, max_dtm, decisive, len(unsolved_flats_np) - decisive, elapsed)
            print(f"SUCCESS: {self.filename} | Total time: {_fmt_elapsed(elapsed)}", flush=True)

        finally:
            if hasattr(b2w_edges, '_mmap'): b2w_edges._mmap.close()
            if hasattr(w2b_edges, '_mmap'): w2b_edges._mmap.close()
            del b2w_edges; del w2b_edges
            try:
                os.remove(tmp_b2w); os.remove(tmp_w2b)
                os.remove(b2w_file); os.remove(w2b_file)
            except Exception as e:
                print(f"  [Warning] Temporary file cleanup failed: {e}")

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
        path = os.path.join(TB_DIR, f"K_{name}_K{TB_SUFFIX}")
        if os.path.exists(path): _W5V_3MAN_TABLES[name] = _load_3man_table_file(path)
    pieces = sorted(PIECE_CLASS_BY_NAME.keys()); seen4 = set()
    for n1 in pieces:
        for n2 in pieces:
            key = tuple(sorted([n1, n2]))
            if key in seen4: continue; seen4.add(key)
            path = os.path.join(TB_DIR, f"K_{key[0]}_{key[1]}_K{TB_SUFFIX}")
            if os.path.exists(path): _W5V_4MAN_TABLES[f"{key[0]}_{key[1]}"] = _load_4man_table_file(path)
    seen4v = set()
    for wn in pieces:
        for bn in pieces:
            key = tuple(sorted([wn, bn]))
            if key in seen4v: continue; seen4v.add(key)
            path = os.path.join(TB_DIR, f"K_{key[0]}_vs_{key[1]}_K{TB_SUFFIX}")
            if os.path.exists(path): _W5V_4VS_TABLES[f"{key[0]}_vs_{key[1]}"] = _load_4man_table_file(path)
    if _W5V_HAS_PAWN:
        if "Pawn" in [w1_name, w2_name]:
            w_names = [w1_name, w2_name]; w_names[w_names.index("Pawn")] = "Queen"
            w_names.sort(key=lambda n: (_PIECE_CANONICAL_ORDER.get(PIECE_CLASS_BY_NAME[n],99), n))
            tb_name = f"K_{w_names[0]}_{w_names[1]}_vs_{b_name}_K{TB_SUFFIX}"
            path = os.path.join(TB_DIR, tb_name)
            if os.path.exists(path): _W5V_PROMO_TABLES[("w", tuple(w_names), b_name)] = _load_5man_table_file(path)
        if b_name == "Pawn":
            tb_name = f"K_{w1_name}_{w2_name}_vs_Queen_K{TB_SUFFIX}"
            path = os.path.join(TB_DIR, tb_name)
            if os.path.exists(path): _W5V_PROMO_TABLES[("b", (w1_name, w2_name), "Queen")] = _load_5man_table_file(path)
    _W5V_BOARD = Board(setup=False)
    _W5V_WK_OBJ = King('white'); _W5V_WP1_OBJ = PIECE_CLASS_BY_NAME[w1_name]('white'); _W5V_WP2_OBJ = PIECE_CLASS_BY_NAME[w2_name]('white')
    _W5V_BK_OBJ = King('black'); _W5V_BP_OBJ = PIECE_CLASS_BY_NAME[b_name]('black')
    pc = _W5V_BOARD.piece_counts; pc['white'][King] = pc['black'][King] = 1
    pc['white'][type(_W5V_WP1_OBJ)] += 1; pc['white'][type(_W5V_WP2_OBJ)] += 1; pc['black'][type(_W5V_BP_OBJ)] += 1

def _white_win_dtm_3man(piece_name, wk, p, bk, turn_idx, is_white_piece):
    tb = _W5V_3MAN_TABLES.get(piece_name)
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
    key = tuple(sorted((w_name, b_name))); tb = _W5V_PROMO_TABLES.get(key)
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
        if wn==_W5V_W1_NAME and bn==_W5V_B_NAME: return _IN_TABLE_SENTINEL
        return _white_win_dtm_promo_vs(wn, bn, wkp, w_nk[0].pos, bkp, b_nk[0].pos, turn_idx)
    if len(w_nk)==1 and len(b_nk)==0: return _white_win_dtm_3man(type(w_nk[0]).__name__, wkp, w_nk[0].pos, bkp, turn_idx, True)
    if len(w_nk)==0 and len(b_nk)==1: return _white_win_dtm_3man(type(b_nk[0]).__name__, wkp, b_nk[0].pos, bkp, turn_idx, False)
    if len(w_nk)==0 and len(b_nk)==0:
        if not bkp: return 1
        if turn_idx==1 and not has_legal_moves(child, 'black'): return 1
        return None
    return None

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
        base_name = f"K_{self.w1_name}_{self.w2_name}_vs_{self.b_name}_K"
        self.filename = os.path.join(TB_DIR, f"{base_name}{TB_SUFFIX}")
        self.wk_size = 32 if self.has_pawn else 10
        self.total_positions = self.wk_size * 64 * 64 * 64 * 64 * 2
        self.table = None
        self.transition_workers = max(1, (os.cpu_count() or 2) - TB_THREADS_SUBTRACT)

    def generate(self):
        if os.path.exists(self.filename): return
        self.table = np.zeros((self.wk_size, 64, 64, 64, 64, 2), dtype=TB_DTYPE)
        start_time = time.time()
        sz_mb = self.wk_size * 64**4 * 2 * 1 / 1024**2
        print(f"\n{'='*60}\n GENERATING: K+{self.w1_name}+{self.w2_name} vs K+{self.b_name} (TB16)", flush=True)
        print(f" Estimated file size: {sz_mb:.0f} MB\n{'='*60}", flush=True)
        s1 = time.time(); print(f"[Stage 1] Enumerating candidates...", flush=True)
        all_flats = array('I')
        for chunk in _gen_valid_5vs_indices_numpy(self.total_positions, self.w1_name, self.w2_name, self.b_name, self.has_pawn, self.same_wp):
            all_flats.extend(chunk)
        total = len(all_flats)
        print(f"[Stage 1] Done: {total:,} candidates | {_fmt_elapsed(time.time()-s1)}", flush=True)
        
        s2 = time.time(); print(f"[Stage 2] Building flat disk cache ({total:,} states)...", flush=True)
        unsolved_flats = array('I')
        init_table = array('h'); init_out_degree = array('H')
        wtm_promo_idx = array('I'); wtm_promo_val = array('H')
        btm_promo_idx = array('I'); btm_promo_val = array('H')
        
        run_prefix, tmp_b2w, tmp_w2b = _make_temp_graph_paths(self.filename)
        f_b2w = open(tmp_b2w, "wb")
        f_w2b = open(tmp_w2b, "wb")
        
        with ProcessPoolExecutor(max_workers=self.transition_workers, initializer=_init_transition_worker_5vs,
                                 initargs=(self.w1_name, self.w2_name, self.b_name)) as ex:
            for done, result in enumerate(ex.map(_build_transition_worker_5vs, all_flats, chunksize=256), 1):
                if result is not None:
                    flat, trans = result
                    i = len(unsolved_flats)
                    unsolved_flats.append(flat)
                    if (flat & 1) == 0:
                        if trans[1]: init_table.append(1); init_out_degree.append(0)
                        else:
                            init_table.append(0); init_out_degree.append(len(trans[3]))
                            if trans[2]: wtm_promo_idx.append(i); wtm_promo_val.append(min(trans[2]))
                        if trans[3]:
                            arr = np.empty((len(trans[3]), 2), dtype=np.uint32)
                            arr[:, 0] = trans[3]; arr[:, 1] = i
                            f_b2w.write(arr.tobytes())
                    else:
                        if trans[1] == 0: init_table.append(-1); init_out_degree.append(0)
                        elif trans[2]: init_table.append(0); init_out_degree.append(65535)
                        else:
                            promo_losses, children = trans[3], trans[4]
                            init_table.append(0); init_out_degree.append(len(children) + len(promo_losses))
                            for p_val in promo_losses: btm_promo_idx.append(i); btm_promo_val.append(p_val)
                            if children:
                                arr = np.empty((len(children), 2), dtype=np.uint32)
                                arr[:, 0] = children; arr[:, 1] = i
                                f_w2b.write(arr.tobytes())
                if done % 100_000 == 0:
                    elapsed = time.time() - start_time; speed = done / max(0.001, time.time() - s2)
                    print(f"  > {done/total*100:.1f}% ({done:,}/{total:,}) | {speed:,.0f} st/s | Elapsed: {_fmt_elapsed(elapsed)}", end='\r', flush=True)
        
        f_b2w.close(); f_w2b.close()
        print(); s2e = time.time() - s2
        print(f"[Stage 2] Cache built: {len(unsolved_flats):,} valid states | {_fmt_elapsed(s2e)} ({total/max(0.001,s2e):,.0f} st/s avg)", flush=True)
        
        s3 = time.time(); print(f"[Stage 2b] Flattening reverse graph via Disk CSR...", flush=True)
        unsolved_flats_np = np.array(unsolved_flats, dtype=np.uint32)
        b2w_head, b2w_edges, b2w_file = _build_csr_from_disk(unsolved_flats_np, tmp_b2w, f"{run_prefix}_b2w")
        w2b_head, w2b_edges, w2b_file = _build_csr_from_disk(unsolved_flats_np, tmp_w2b, f"{run_prefix}_w2b")
        print(f"[Stage 2b] Done | {_fmt_elapsed(time.time()-s3)}", flush=True)
        
        try:
            table_flat, max_dtm, decisive = _run_flat_bfs(
                unsolved_flats_np, np.array(init_table, dtype=TB_DTYPE), np.array(init_out_degree, dtype=np.uint16),
                np.array(wtm_promo_idx, dtype=np.uint32), np.array(wtm_promo_val, dtype=np.uint16),
                np.array(btm_promo_idx, dtype=np.uint32), np.array(btm_promo_val, dtype=np.uint16),
                b2w_head, b2w_edges, w2b_head, w2b_edges, start_time)

            _write_dense_table(self.filename, self.total_positions, unsolved_flats_np, table_flat)
                
            elapsed = time.time() - start_time
            tb_key = os.path.basename(self.filename).replace(TB_SUFFIX, '')
            _safe_record_longest_mate(tb_key, max_dtm, decisive, len(unsolved_flats_np) - decisive, elapsed)
            print(f"SUCCESS: {self.filename} | Total time: {_fmt_elapsed(elapsed)}", flush=True)

        finally:
            if hasattr(b2w_edges, '_mmap'): b2w_edges._mmap.close()
            if hasattr(w2b_edges, '_mmap'): w2b_edges._mmap.close()
            del b2w_edges; del w2b_edges
            try:
                os.remove(tmp_b2w); os.remove(tmp_w2b)
                os.remove(b2w_file); os.remove(w2b_file)
            except Exception as e:
                print(f"  [Warning] Temporary file cleanup failed: {e}")

# ==============================================================================
# ENTRY POINT
# ==============================================================================

def is_possible(pieces):
    counts = {Queen: 0, Rook: 0, Bishop: 0, Knight: 0, Pawn: 0}
    for p in pieces: counts[p] += 1
    if counts[Rook] > 2 or counts[Bishop] > 2 or counts[Knight] > 2: return False
    return True

if __name__ == "__main__":
    _install_main_interrupt_ignores()
    overall_start = time.time()
    print("=== Tablebase Generator (v17.0 - Unified 16-bit Retrograde BFS) ===")

    all_pieces = [Queen, Rook, Bishop, Knight, Pawn]
    all_gens = []

    # --- TIER 1 & 2: 3-Man ---
    for pc in all_pieces:
        all_gens.append(Generator(pc))

    # --- TIER 3 & 4: 4-Man Same-Side ---
    for p1, p2 in combinations_with_replacement(all_pieces, 2):
        if is_possible([p1, p2]):
            all_gens.append(Generator4(p1, p2))

    # --- TIER 3b & 4b: 4-Man Cross (K+w vs K+b) ---
    for w, b in combinations_with_replacement(all_pieces, 2):
        all_gens.append(Generator4Vs(w, b))

    # --- TIER 5 & 6: 5-Man Same-Side ---
    for p1, p2, p3 in combinations_with_replacement(all_pieces, 3):
        if is_possible([p1, p2, p3]):
            all_gens.append(Generator5(p1, p2, p3))

    # --- TIER 5b & 6b: 5-Man Cross (K+w1+w2 vs K+b) ---
    seen_5vs = set()
    for w1, w2 in combinations_with_replacement(all_pieces, 2):
        if not is_possible([w1, w2]): continue
        for b in all_pieces:
            ws = tuple(sorted([w1.__name__, w2.__name__]))
            key = (ws[0], ws[1], b.__name__)
            if key not in seen_5vs:
                seen_5vs.add(key)
                all_gens.append(Generator5Vs(w1, w2, b))

    # --- TOPOLOGICAL SORTING ---
    def get_gen_sort_key(gen):
        if isinstance(gen, Generator):
            return (3, 1 if gen.piece_name == "Pawn" else 0)
        elif isinstance(gen, Generator4):
            return (4, sum(1 for n in [gen.p1_name, gen.p2_name] if n == "Pawn"))
        elif isinstance(gen, Generator4Vs):
            return (4, sum(1 for n in [gen.w_name, gen.b_name] if n == "Pawn"))
        elif isinstance(gen, Generator5):
            return (5, sum(1 for n in [gen.p1_name, gen.p2_name, gen.p3_name] if n == "Pawn"))
        elif isinstance(gen, Generator5Vs):
            return (5, sum(1 for n in [gen.w1_name, gen.w2_name, gen.b_name] if n == "Pawn"))
        return (99, 99)

    all_gens.sort(key=get_gen_sort_key)

    # --- EXECUTE ---
    print(f"\nFound {len(all_gens)} valid Tablebases to generate.")
    print("Generation order strictly optimized for Pawn promotion dependencies.")
    
    for i, gen in enumerate(all_gens, 1):
        print(f"\n[{i}/{len(all_gens)}] Queued: {os.path.basename(gen.filename)}")
        gen.generate()

    overall_elapsed = time.time() - overall_start
    print(f"\n\n=== ALL TABLEBASE GENERATION COMPLETE ===")
    print(f"=== Total overall runtime: {_fmt_elapsed(overall_elapsed)} ===")
