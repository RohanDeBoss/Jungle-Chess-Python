# TablebaseGenerator.py (v1.0 - TAG: INITIAL_RELEASE)
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
        self.table = np.zeros((64, 64, 64, 2), dtype=np.int8)
        
        # We reuse one board and set of pieces to save millions of allocations
        self.sim_board = Board(setup=False)
        self.wk_obj = King('white')
        self.wp_obj = self.piece_class('white')
        self.bk_obj = King('black')

    def encode(self, wk, wp, bk, t_idx):
        """Converts coordinates to table indices."""
        return (wk[0]*8+wk[1], wp[0]*8+wp[1], bk[0]*8+bk[1], t_idx)

    def decode(self, idx):
        """Converts table indices back to coordinates."""
        wk = (idx[0] // 8, idx[0] % 8)
        wp = (idx[1] // 8, idx[1] % 8)
        bk = (idx[2] // 8, idx[2] % 8)
        return wk, wp, bk, idx[3]

    def setup_sim(self, wk, wp, bk):
        """Fast-update the sim_board without creating new objects."""
        self.sim_board.grid = [[None for _ in range(8)] for _ in range(8)]
        
        # Update White King
        self.wk_obj.pos = wk
        self.sim_board.grid[wk[0]][wk[1]] = self.wk_obj
        self.sim_board.white_king_pos = wk
        
        # Update White Piece
        self.wp_obj.pos = wp
        self.sim_board.grid[wp[0]][wp[1]] = self.wp_obj
        
        # Update Black King
        self.bk_obj.pos = bk
        self.sim_board.grid[bk[0]][bk[1]] = self.bk_obj
        self.sim_board.black_king_pos = bk
        
        self.sim_board.white_pieces = [self.wk_obj, self.wp_obj]
        self.sim_board.black_pieces = [self.bk_obj]

    def generate(self):
        start_time = time.time()
        print(f"\n{'='*50}")
        print(f" GENERATING: King + {self.piece_name} vs King")
        print(f"{'='*50}")
        
        unsolved_indices = []

        # --- STEP 1: INITIALIZATION ---
        print(f"[{self.piece_name}] Phase 1: Initialization...")
        count = 0
        for idx in np.ndindex(self.table.shape):
            count += 1
            if count % 50000 == 0:
                print(f"  > Scanned {count}/{self.total_positions} positions...")

            wk, wp, bk, t_idx = self.decode(idx)
            
            # Skip physically impossible positions
            if wk == wp or wk == bk or wp == bk:
                continue

            self.setup_sim(wk, wp, bk)
            turn = 'white' if t_idx == 0 else 'black'
            
            # Detect immediate terminal states (Mates/Ghost Kings)
            if turn == 'black':
                # If Black king is already vaporized or in mate with no moves
                if not self.sim_board.black_king_pos or (is_in_check(self.sim_board, 'black') and not has_legal_moves(self.sim_board, 'black')):
                    self.table[idx] = -1 # Loss for side to move
                    continue
            else:
                # If White king is already vaporized (rare but possible in initialization scan)
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
                
                # Get pseudo-moves for speed; we filter legality inside the loop
                moves = get_all_pseudo_legal_moves(self.sim_board, turn)
                
                if t_idx == 0: # White's Turn: Search for a move to a Black LOSS
                    best_win = 0
                    for m in moves:
                        child = self.sim_board.clone()
                        child.make_move(m[0], m[1])
                        
                        if is_in_check(child, 'white'): continue # Illegal move
                        
                        # Win by direct king capture/vaporization
                        if not child.black_king_pos:
                            best_win = 1; break
                        
                        # Win by moving to a position where Black is in a Loss state
                        if len(child.white_pieces) >= 2:
                            res_idx = self.encode(child.white_king_pos, child.white_pieces[1].pos, child.black_king_pos, opp_turn_idx)
                            res_val = self.table[res_idx]
                            if res_val < 0: # Opponent is in a LOSS state
                                val = abs(res_val) + 1
                                if best_win == 0 or val < best_win:
                                    best_win = val
                    
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
                        
                        if is_in_check(child, 'black'): continue # Illegal
                        
                        legal_moves_count += 1
                        
                        # If Black can draw (capture piece) or win (capture White King), not a loss
                        if not child.white_king_pos or len(child.white_pieces) < 2:
                            all_moves_lead_to_win = False; break
                            
                        res_idx = self.encode(child.white_king_pos, child.white_pieces[1].pos, child.black_king_pos, opp_turn_idx)
                        res_val = self.table[res_idx]
                        
                        if res_val <= 0: # Leading to draw or unknown is not a forced loss
                            all_moves_lead_to_win = False; break
                            
                        max_win_val = max(max_win_val, res_val)
                    
                    if legal_moves_count > 0 and all_moves_lead_to_win:
                        self.table[idx] = -(max_win_val + 1)
                        changed += 1
                    else:
                        new_unsolved.append(idx)

            unsolved_indices = new_unsolved
            print(f"[{self.piece_name}] Round {iteration} Summary: {changed} new positions solved.")
            
            if changed == 0:
                print(f"[{self.piece_name}] Convergence reached.")
                break
            iteration += 1

        total_time = time.time() - start_time
        self.save(total_time)

    def save(self, elapsed):
        with open(self.filename, 'wb') as f:
            self.table.tofile(f)
        print(f"\nSUCCESS: {self.piece_name} Tablebase generated in {elapsed/60:.1f} minutes.")
        print(f"File saved to: {self.filename}\n")

if __name__ == "__main__":
    # You can comment/uncomment these to generate specific tables
    Generator(Queen).generate()
    Generator(Rook).generate()
    Generator(Knight).generate()
    Generator(Bishop).generate()