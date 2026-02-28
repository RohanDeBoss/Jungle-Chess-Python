# TablebaseGenerator.py (v1.5 - strict DTM, transition cache, optional parallel cache build)
import os
import time
import __main__
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from GameLogic import *

# Configuration
TB_DIR = "tablebases"
os.makedirs(TB_DIR, exist_ok=True)

EXPECTED_TABLE_ENTRIES = 64 * 64 * 64 * 2
PIECE_CLASS_BY_NAME = {
    "Queen": Queen,
    "Rook": Rook,
    "Knight": Knight,
    "Bishop": Bishop,
    "Pawn": Pawn,
}

# Worker globals for ProcessPool transition construction
_W_PIECE_NAME = None
_W_PIECE_CLASS = None
_W_QUEEN_TABLE = None

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

def _init_transition_worker(piece_name, queen_tb_file):
    global _W_PIECE_NAME, _W_PIECE_CLASS, _W_QUEEN_TABLE
    _W_PIECE_NAME = piece_name
    _W_PIECE_CLASS = PIECE_CLASS_BY_NAME[piece_name]
    _W_QUEEN_TABLE = None
    if piece_name == "Pawn":
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
        promo_win_vals = []
        child_flats = []

        for m in moves:
            child = board.clone()
            child.make_move(m[0], m[1])
            if is_in_check(child, 'white'):
                continue
            if not child.black_king_pos:
                immediate_win = True
                continue

            if _W_PIECE_NAME == "Pawn" and isinstance(child.grid[m[1][0]][m[1][1]], Queen):
                q_idx = (
                    child.white_king_pos[0] * 8 + child.white_king_pos[1],
                    m[1][0] * 8 + m[1][1],
                    child.black_king_pos[0] * 8 + child.black_king_pos[1],
                    1
                )
                q_val = int(_W_QUEEN_TABLE[q_idx])
                if q_val < 0:
                    promo_win_vals.append(abs(q_val) + 1)
                continue

            if len(child.white_pieces) >= 2:
                p = next((x for x in child.white_pieces if not isinstance(x, King)), None)
                if p is None:
                    continue
                c0 = child.white_king_pos[0] * 8 + child.white_king_pos[1]
                c1 = p.pos[0] * 8 + p.pos[1]
                c2 = child.black_king_pos[0] * 8 + child.black_king_pos[1]
                child_flats.append(_flat_idx_raw(c0, c1, c2, opp_turn_idx))

        return _flat_idx_raw(i0, i1, i2, i3), ('w', immediate_win, tuple(promo_win_vals), tuple(child_flats))

    legal_moves_count = 0
    has_non_losing_escape = False
    child_flats = []

    for m in moves:
        child = board.clone()
        child.make_move(m[0], m[1])
        if is_in_check(child, 'black'):
            continue

        legal_moves_count += 1
        if not child.white_king_pos or len(child.white_pieces) < 2:
            has_non_losing_escape = True
            continue
        if _W_PIECE_NAME == "Pawn" and isinstance(child.grid[child.white_pieces[1].pos[0]][child.white_pieces[1].pos[1]], Queen):
            has_non_losing_escape = True
            continue

        p = next((x for x in child.white_pieces if not isinstance(x, King)), None)
        if p is None:
            has_non_losing_escape = True
            continue
        c0 = child.white_king_pos[0] * 8 + child.white_king_pos[1]
        c1 = p.pos[0] * 8 + p.pos[1]
        c2 = child.black_king_pos[0] * 8 + child.black_king_pos[1]
        child_flats.append(_flat_idx_raw(c0, c1, c2, opp_turn_idx))

    return _flat_idx_raw(i0, i1, i2, i3), ('b', legal_moves_count, has_non_losing_escape, tuple(child_flats))

class Generator:
    def __init__(self, piece_class):
        self.piece_class = piece_class
        self.piece_name = piece_class.__name__
        self.filename = os.path.join(TB_DIR, f"K_{self.piece_name}_K.bin")
        self.queen_tb_file = os.path.join(TB_DIR, "K_Queen_K.bin")
        
        # 64 WK * 64 WP * 64 BK * 2 Turns = 524,288 entries
        self.total_positions = 64 * 64 * 64 * 2
        # int16 avoids overflow/truncation for long DTM distances.
        self.table = np.zeros((64, 64, 64, 2), dtype=np.int16)
        
        # We reuse one board and pieces to save millions of memory allocations
        self.sim_board = Board(setup=False)
        self.wk_obj = King('white')
        self.wp_obj = self.piece_class('white')
        self.bk_obj = King('black')

        # Pawn specific: Load Queen TB for promotion lookups
        self.queen_table = None
        if self.piece_name == "Pawn":
            if os.path.exists(self.queen_tb_file):
                print(f"[Pawn Support] Loading Queen Tablebase for promotion lookups...")
                self.queen_table = self._load_table_file(self.queen_tb_file)
            else:
                print(f"CRITICAL ERROR: Queen Tablebase not found! Generate Queen first.")
                exit()
        
        cpu_default = max(1, (os.cpu_count() or 2) - 1)
        requested = os.getenv("TB_WORKERS")
        self.transition_workers = max(1, int(requested)) if requested else min(8, cpu_default)
        self.transition_parallel_threshold = 5000

    def _load_table_file(self, filename):
        """
        Load a tablebase file supporting both legacy int8 and strict int16 formats.
        Returns int16 array shaped (64, 64, 64, 2).
        """
        return _load_table_file_any_dtype(filename)

    def encode(self, wk, wp, bk, t_idx):
        return (wk[0]*8+wk[1], wp[0]*8+wp[1], bk[0]*8+bk[1], t_idx)

    def decode(self, idx):
        wk = (idx[0] // 8, idx[0] % 8)
        wp = (idx[1] // 8, idx[1] % 8)
        bk = (idx[2] // 8, idx[2] % 8)
        return wk, wp, bk, idx[3]

    def _flat_idx(self, idx):
        # Fast flatten for shape (64, 64, 64, 2)
        return (((idx[0] * 64 + idx[1]) * 64 + idx[2]) * 2 + idx[3])

    def _build_transition(self, idx):
        """
        Build and return a cached transition descriptor for one state.
        This is the expensive part (move generation + legality checks), so
        we do it once per state and reuse across retrograde iterations.
        """
        wk, wp, bk, t_idx = self.decode(idx)
        self.setup_sim(wk, wp, bk)
        turn = 'white' if t_idx == 0 else 'black'
        opp_turn_idx = 1 - t_idx
        moves = get_all_pseudo_legal_moves(self.sim_board, turn)

        if t_idx == 0:
            # White to move in K+Piece vs K table.
            immediate_win = False
            promo_win_vals = []
            child_flats = []

            for m in moves:
                child = self.sim_board.clone()
                child.make_move(m[0], m[1])
                if is_in_check(child, 'white'):
                    continue

                if not child.black_king_pos:
                    immediate_win = True
                    continue

                if self.piece_name == "Pawn" and isinstance(child.grid[m[1][0]][m[1][1]], Queen):
                    q_idx = (
                        child.white_king_pos[0] * 8 + child.white_king_pos[1],
                        m[1][0] * 8 + m[1][1],
                        child.black_king_pos[0] * 8 + child.black_king_pos[1],
                        1
                    )
                    q_val = int(self.queen_table[q_idx])
                    if q_val < 0:
                        promo_win_vals.append(abs(q_val) + 1)
                    continue

                if len(child.white_pieces) >= 2:
                    c_idx = self.encode(
                        child.white_king_pos,
                        child.white_pieces[1].pos,
                        child.black_king_pos,
                        opp_turn_idx
                    )
                    child_flats.append(self._flat_idx(c_idx))

            return ('w', immediate_win, tuple(promo_win_vals), tuple(child_flats))

        # Black to move in K+Piece vs K table.
        legal_moves_count = 0
        has_non_losing_escape = False
        child_flats = []

        for m in moves:
            child = self.sim_board.clone()
            child.make_move(m[0], m[1])
            if is_in_check(child, 'black'):
                continue

            legal_moves_count += 1

            if not child.white_king_pos or len(child.white_pieces) < 2:
                has_non_losing_escape = True
                continue

            if self.piece_name == "Pawn" and isinstance(child.grid[child.white_pieces[1].pos[0]][child.white_pieces[1].pos[1]], Queen):
                has_non_losing_escape = True
                continue

            c_idx = self.encode(
                child.white_king_pos,
                child.white_pieces[1].pos,
                child.black_king_pos,
                opp_turn_idx
            )
            child_flats.append(self._flat_idx(c_idx))

        return ('b', legal_moves_count, has_non_losing_escape, tuple(child_flats))

    def _build_transition_cache(self, unsolved_indices):
        cache = {}
        total = len(unsolved_indices)
        if total == 0:
            return cache

        print(f"[{self.piece_name}] Phase 1.5: Building transition cache ({total} states)...")
        start = time.time()

        can_parallel = (
            self.transition_workers > 1 and
            total >= self.transition_parallel_threshold and
            not (self.piece_name == "Pawn" and self.queen_table is None)
        )
        # On Windows spawn, parallel workers require a real importable main file.
        main_file = getattr(__main__, "__file__", "")
        if not main_file or main_file == "<stdin>":
            can_parallel = False

        if can_parallel:
            print(f"[{self.piece_name}] Using {self.transition_workers} workers for transition build.")
            try:
                with ProcessPoolExecutor(
                    max_workers=self.transition_workers,
                    initializer=_init_transition_worker,
                    initargs=(self.piece_name, self.queen_tb_file)
                ) as ex:
                    for done, (flat, transition) in enumerate(
                        ex.map(_build_transition_worker, unsolved_indices, chunksize=512), start=1
                    ):
                        cache[flat] = transition
                        if done % 50000 == 0:
                            elapsed = max(0.001, time.time() - start)
                            speed = done / elapsed
                            eta = (total - done) / speed
                            print(f"  > Cache: {done}/{total} ({(done/total)*100:.1f}%) | {speed:.0f} st/s | ETA: {eta:.0f}s")
            except Exception as e:
                print(f"[{self.piece_name}] Parallel cache build failed ({e}). Falling back to single-process.")
                cache.clear()
                can_parallel = False

        if not can_parallel:
            for done, idx in enumerate(unsolved_indices, start=1):
                cache[self._flat_idx(idx)] = self._build_transition(idx)
                if done % 50000 == 0:
                    elapsed = max(0.001, time.time() - start)
                    speed = done / elapsed
                    eta = (total - done) / speed
                    print(f"  > Cache: {done}/{total} ({(done/total)*100:.1f}%) | {speed:.0f} st/s | ETA: {eta:.0f}s")

        elapsed = time.time() - start
        print(f"[{self.piece_name}] Transition cache ready in {elapsed:.1f}s.")
        return cache

    def setup_sim(self, wk, wp, bk):
        """Fast-update the sim_board without creating new objects."""
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
        start_time = time.time()
        print(f"\n{'='*60}")
        print(f" GENERATING: King + {self.piece_name} vs King")
        print(f"{'='*60}")
        
        unsolved_indices = []

        # --- STEP 1: INITIALIZATION ---
        print(f"[{self.piece_name}] Phase 1: Initialization...")
        count = 0
        for idx in np.ndindex(self.table.shape):
            count += 1
            if count % 200000 == 0:
                print(f"  > Scanned {count}/{self.total_positions} positions...")

            wk, wp, bk, t_idx = self.decode(idx)
            if wk == wp or wk == bk or wp == bk: continue

            # Illegal Pawn positions
            if self.piece_name == "Pawn" and (wp[0] == 0 or wp[0] == 7): continue

            self.setup_sim(wk, wp, bk)
            turn = 'white' if t_idx == 0 else 'black'
            
            if turn == 'black':
                # If Black king is vaporized or in checkmate immediately
                if not self.sim_board.black_king_pos or (is_in_check(self.sim_board, 'black') and not has_legal_moves(self.sim_board, 'black')):
                    self.table[idx] = -1 
                    continue
            else:
                if not self.sim_board.white_king_pos:
                    self.table[idx] = -1
                    continue
            
            unsolved_indices.append(idx)

        print(f"[{self.piece_name}] Initialization complete. Solving {len(unsolved_indices)} non-terminal positions.")

        # --- STEP 2: RETROGRADE ANALYSIS ---
        transition_cache = self._build_transition_cache(unsolved_indices)
        iteration = 1
        while True:
            iter_start = time.time()
            changed = 0
            processed = 0
            total_current = len(unsolved_indices)
            new_unsolved = []
            # Strict DTM layering:
            # Read from the previous iteration's solved frontier only.
            # This prevents same-iteration propagation that can corrupt exact DTM.
            prev_table = self.table.copy()
            prev_flat = prev_table.reshape(-1)
            
            print(f"\n[{self.piece_name}] Iteration {iteration} (DTM {iteration})")
            
            for idx in unsolved_indices:
                processed += 1
                if processed % 20000 == 0:
                    elapsed = time.time() - iter_start
                    speed = processed / elapsed
                    eta = (total_current - processed) / speed
                    print(f"  > Progress: {processed}/{total_current} ({(processed/total_current)*100:.1f}%) | {speed:.0f} pos/s | ETA: {eta:.0f}s")

                t_idx = idx[3]
                flat = self._flat_idx(idx)
                transition = transition_cache.get(flat)
                if transition is None:
                    transition = self._build_transition(idx)
                    transition_cache[flat] = transition

                if t_idx == 0: # White's Turn: Looking for a move to a Black LOSS
                    best_win = 0
                    _, immediate_win, promo_vals, child_flats = transition

                    if immediate_win:
                        best_win = 1

                    for val in promo_vals:
                        if best_win == 0 or val < best_win:
                            best_win = val

                    for cflat in child_flats:
                        res_val = int(prev_flat[cflat])
                        if res_val < 0:
                            val = abs(res_val) + 1
                            if best_win == 0 or val < best_win:
                                best_win = val
                    
                    if best_win > 0:
                        self.table[idx] = best_win
                        changed += 1
                    else:
                        new_unsolved.append(idx)

                else: # Black's Turn: Check if ALL legal moves lead to a White WIN
                    _, legal_moves_count, has_non_losing_escape, child_flats = transition
                    all_moves_lead_to_win = legal_moves_count > 0 and not has_non_losing_escape
                    max_win_val = 0

                    if all_moves_lead_to_win:
                        for cflat in child_flats:
                            res_val = int(prev_flat[cflat])
                            if res_val <= 0:
                                all_moves_lead_to_win = False
                                break
                            if res_val > max_win_val:
                                max_win_val = res_val

                    if all_moves_lead_to_win:
                        self.table[idx] = -(max_win_val + 1)
                        changed += 1
                    else:
                        new_unsolved.append(idx)

            unsolved_indices = new_unsolved
            total_solved = (self.table != 0).sum()
            percent = (total_solved / self.total_positions) * 100
            print(f"[{self.piece_name}] Round {iteration} Summary: {changed} new solved | Total: {total_solved} ({percent:.1f}%)")
            
            if changed == 0:
                print(f"[{self.piece_name}] Convergence reached.")
                break
            iteration += 1

        self.save(time.time() - start_time)

    def save(self, elapsed):
        with open(self.filename, 'wb') as f:
            self.table.tofile(f)
        print(f"\nSUCCESS: {self.piece_name} Tablebase generated in {elapsed/60:.1f} minutes.")
        print(f"File size: {self.table.nbytes / (1024*1024):.2f} MB")
        print(f"File saved to: {self.filename}\n")

if __name__ == "__main__":
    pieces_to_generate = [Queen, Rook, Knight, Bishop, Pawn]
    
    for p_class in pieces_to_generate:
        p_name = p_class.__name__
        file_path = os.path.join(TB_DIR, f"K_{p_name}_K.bin")
        
        if os.path.exists(file_path):
            choice = input(f"\n[?] {p_name} tablebase already exists. Overwrite? (y/n): ").lower()
            if choice != 'y':
                print(f"Skipping {p_name}...")
                continue
        
        Generator(p_class).generate()
