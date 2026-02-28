# TablebaseGenerator.py (v1.3 - strict DTM layering + int16 storage)
import os
import time
import numpy as np
from GameLogic import *

# Configuration
TB_DIR = "tablebases"
os.makedirs(TB_DIR, exist_ok=True)

class Generator:
    def __init__(self, piece_class):
        self.piece_class = piece_class
        self.piece_name = piece_class.__name__
        self.filename = os.path.join(TB_DIR, f"K_{self.piece_name}_K.bin")
        
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
            q_file = os.path.join(TB_DIR, "K_Queen_K.bin")
            if os.path.exists(q_file):
                print(f"[Pawn Support] Loading Queen Tablebase for promotion lookups...")
                self.queen_table = self._load_table_file(q_file)
            else:
                print(f"CRITICAL ERROR: Queen Tablebase not found! Generate Queen first.")
                exit()

    def _load_table_file(self, filename):
        """
        Load a tablebase file supporting both legacy int8 and strict int16 formats.
        Returns int16 array shaped (64, 64, 64, 2).
        """
        expected = 64 * 64 * 64 * 2

        data16 = np.fromfile(filename, dtype=np.int16)
        if data16.size == expected:
            return data16.reshape((64, 64, 64, 2))

        data8 = np.fromfile(filename, dtype=np.int8)
        if data8.size == expected:
            return data8.astype(np.int16).reshape((64, 64, 64, 2))

        raise ValueError(f"Invalid tablebase file size: {filename}")

    def encode(self, wk, wp, bk, t_idx):
        return (wk[0]*8+wk[1], wp[0]*8+wp[1], bk[0]*8+bk[1], t_idx)

    def decode(self, idx):
        wk = (idx[0] // 8, idx[0] % 8)
        wp = (idx[1] // 8, idx[1] % 8)
        bk = (idx[2] // 8, idx[2] % 8)
        return wk, wp, bk, idx[3]

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
            if count % 100000 == 0:
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
            
            print(f"\n[{self.piece_name}] Iteration {iteration} (DTM {iteration})")
            
            for idx in unsolved_indices:
                processed += 1
                if processed % 10000 == 0:
                    elapsed = time.time() - iter_start
                    speed = processed / elapsed
                    eta = (total_current - processed) / speed
                    print(f"  > Progress: {processed}/{total_current} ({(processed/total_current)*100:.1f}%) | {speed:.0f} pos/s | ETA: {eta:.0f}s")

                wk, wp, bk, t_idx = self.decode(idx)
                self.setup_sim(wk, wp, bk)
                turn = 'white' if t_idx == 0 else 'black'
                opp_turn_idx = 1 - t_idx
                
                moves = get_all_pseudo_legal_moves(self.sim_board, turn)
                
                if t_idx == 0: # White's Turn: Looking for a move to a Black LOSS
                    best_win = 0
                    for m in moves:
                        child = self.sim_board.clone()
                        child.make_move(m[0], m[1])
                        if is_in_check(child, 'white'): continue
                        
                        if not child.black_king_pos:
                            best_win = 1; break
                        
                        # Handle Pawn Promotion to Queen lookup
                        if self.piece_name == "Pawn" and isinstance(child.grid[m[1][0]][m[1][1]], Queen):
                            q_idx = (child.white_king_pos[0]*8+child.white_king_pos[1], 
                                     m[1][0]*8+m[1][1], 
                                     child.black_king_pos[0]*8+child.black_king_pos[1], 1)
                            res_val = self.queen_table[q_idx]
                            if res_val < 0: # Black loses in Queen TB
                                val = abs(res_val) + 1
                                if best_win == 0 or val < best_win: best_win = val
                            continue
                        
                        if len(child.white_pieces) >= 2:
                            res_idx = self.encode(child.white_king_pos, child.white_pieces[1].pos, child.black_king_pos, opp_turn_idx)
                            res_val = prev_table[res_idx]
                            if res_val < 0:
                                val = abs(res_val) + 1
                                if best_win == 0 or val < best_win: best_win = val
                    
                    if best_win > 0:
                        self.table[idx] = best_win
                        changed += 1
                    else:
                        new_unsolved.append(idx)

                else: # Black's Turn: Check if ALL legal moves lead to a White WIN
                    legal_moves_count = 0
                    all_moves_lead_to_win = True
                    max_win_val = 0
                    
                    for m in moves:
                        child = self.sim_board.clone()
                        child.make_move(m[0], m[1])
                        if is_in_check(child, 'black'): continue
                        legal_moves_count += 1
                        
                        if not child.white_king_pos or len(child.white_pieces) < 2:
                            all_moves_lead_to_win = False; break
                        
                        # Black cannot promote, but handle case where White piece becomes Queen 
                        # (Not usually possible on black's turn in 3-piece, but logic remains for safety)
                        if self.piece_name == "Pawn" and isinstance(child.grid[child.white_pieces[1].pos[0]][child.white_pieces[1].pos[1]], Queen):
                            all_moves_lead_to_win = False; break

                        res_idx = self.encode(child.white_king_pos, child.white_pieces[1].pos, child.black_king_pos, opp_turn_idx)
                        res_val = prev_table[res_idx]
                        if res_val <= 0:
                            all_moves_lead_to_win = False; break
                        max_win_val = max(max_win_val, res_val)
                    
                    if legal_moves_count > 0 and all_moves_lead_to_win:
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
