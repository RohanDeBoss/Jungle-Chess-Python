# TablebaseManager.py (v2.0 base + 4-Man Support)

import os
import numpy as np
from GameLogic import King, Board


def _flat_idx_raw_4(i0, i1, i2, i3, i4):
    return (((((i0 * 64 + i1) * 64 + i2) * 64 + i3) * 2) + i4)

class TablebaseManager:
    def __init__(self):
        self.tables = {}
        self.tb_dir = "tablebases"
        self.pre_load_all()

    def pre_load_all(self):
        pieces = ['Queen', 'Rook', 'Knight', 'Bishop', 'Pawn']
        # Load 3-man
        for p in pieces:
            self.load_table(f"K_{p}_K")
            
        # Optional: Preload 4-man if they exist (can be heavy on RAM if too many)
        from itertools import combinations_with_replacement
        for p1, p2 in combinations_with_replacement(pieces, 2):
            names = sorted([p1, p2])
            self.load_table(f"K_{names[0]}_{names[1]}_K")

    def load_table(self, name):
        filename = os.path.join(self.tb_dir, f"{name}.bin")
        if os.path.exists(filename):
            try:
                data16 = np.fromfile(filename, dtype=np.int16)
                if data16.size == 64 * 64 * 64 * 2: # 3-man
                    self.tables[name] = data16.reshape((64, 64, 64, 2))
                    return True
                elif data16.size == 64 * 64 * 64 * 64 * 2: # 4-man
                    self.tables[name] = data16.reshape((64, 64, 64, 64, 2))
                    return True
                
                # Support legacy int8 for 3-man
                data8 = np.fromfile(filename, dtype=np.int8)
                if data8.size == 64 * 64 * 64 * 2:
                    self.tables[name] = data8.astype(np.int16).reshape((64, 64, 64, 2))
                    return True
            except Exception:
                pass
        return False

    def _to_white_perspective(self, signed_tb_score, turn):
        """
        Convert a table score into white-centric absolute evaluation.
        """
        if signed_tb_score == 0:
            return 0

        abs_val = 1000000 - abs(int(signed_tb_score))
        side_to_move_wins = signed_tb_score > 0

        if turn == 'white':
            return abs_val if side_to_move_wins else -abs_val
        return -abs_val if side_to_move_wins else abs_val

    def probe(self, board, turn):
        white_count = len(board.white_pieces)
        black_count = len(board.black_pieces)
        
        # --- 3-Man Probe ---
        if white_count + black_count == 3:
            if white_count == 2 and black_count == 1:
                try:
                    wp = next(p for p in board.white_pieces if not isinstance(p, King))
                    wk, bk = board.white_king_pos, board.black_king_pos
                    p_name = f"K_{wp.__class__.__name__}_K"
                    if p_name not in self.tables: return None
                    
                    t_idx = 0 if turn == 'white' else 1
                    idx = (wk[0]*8+wk[1], wp.pos[0]*8+wp.pos[1], bk[0]*8+bk[1], t_idx)
                    score = int(self.tables[p_name][idx])
                    return self._to_white_perspective(score, turn)
                except Exception:
                    return None

            elif black_count == 2 and white_count == 1:
                try:
                    bp = next(p for p in board.black_pieces if not isinstance(p, King))
                    bk, wk = board.black_king_pos, board.white_king_pos
                    p_name = f"K_{bp.__class__.__name__}_K"
                    if p_name not in self.tables: return None
                    
                    def flip(p): return (7-p[0], p[1])
                    t_idx = 0 if turn == 'black' else 1
                    idx = (flip(bk)[0]*8+flip(bk)[1], flip(bp.pos)[0]*8+flip(bp.pos)[1], flip(wk)[0]*8+flip(wk)[1], t_idx)
                    score = int(self.tables[p_name][idx])
                    return self._to_white_perspective(score, turn)
                except Exception:
                    return None
                    
        # --- 4-Man Probe (Logic Corrected) ---
        elif white_count + black_count == 4:
            # Tablebases only exist for 3 attackers vs 1 king
            if white_count == 3 and black_count == 1:
                atk_pieces = sorted([p for p in board.white_pieces if not isinstance(p, King)], key=lambda x: type(x).__name__)
                wk, bk = board.white_king_pos, board.black_king_pos
                table_name = f"K_{type(atk_pieces[0]).__name__}_{type(atk_pieces[1]).__name__}_K"
                
                if table_name not in self.tables: return None
                
                idx = _flat_idx_raw_4(wk[0]*8+wk[1], atk_pieces[0].pos[0]*8+atk_pieces[0].pos[1], 
                                     atk_pieces[1].pos[0]*8+atk_pieces[1].pos[1], bk[0]*8+bk[1], 
                                     0 if turn == 'white' else 1)
                return self._to_white_perspective(self.tables[table_name].flat[idx], turn)

            elif black_count == 3 and white_count == 1:
                atk_pieces = sorted([p for p in board.black_pieces if not isinstance(p, King)], key=lambda x: type(x).__name__)
                bk, wk = board.black_king_pos, board.white_king_pos
                table_name = f"K_{type(atk_pieces[0]).__name__}_{type(atk_pieces[1]).__name__}_K"
                
                if table_name not in self.tables: return None
                
                # Mirroring: y = 7-y. This maps Black Attacker to the White Attacker table.
                def m(p): return (7-p[0])*8 + p[1]
                idx = _flat_idx_raw_4(m(bk), m(atk_pieces[0].pos), m(atk_pieces[1].pos), m(wk), 
                                     0 if turn == 'black' else 1)
                
                score = int(self.tables[table_name].flat[idx])
                return self._to_white_perspective(score, turn)

        return None