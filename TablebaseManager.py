# TablebaseManager.py (v5.0 - Unified Canonical Probing)

import os
import numpy as np
from GameLogic import King, Board

def _flip(pos):
    """Helper to flip board coordinates for Black's perspective."""
    return (7 - pos[0], pos[1])

class TablebaseManager:
    def __init__(self):
        self.tables = {}
        self.tb_dir = "tablebases"
        self.pre_load_all()

    def pre_load_all(self):
        if not os.path.exists(self.tb_dir):
            return
            
        pieces = ['Queen', 'Rook', 'Knight', 'Bishop', 'Pawn']
        
        # 1. Load 3-man tables
        for p in pieces:
            self.load_table(f"K_{p}_K")
            
        # 2. Load 4-man Same-Side tables
        from itertools import combinations_with_replacement
        for p1, p2 in combinations_with_replacement(pieces, 2):
            names = sorted([p1, p2])
            self.load_table(f"K_{names[0]}_{names[1]}_K")
            
        # 3. Load 4-man Cross tables (Canonical names only)
        for i in range(len(pieces)):
            for j in range(i, len(pieces)):
                w, b = pieces[i], pieces[j]
                self.load_table(f"K_{w}_vs_{b}_K")

    def load_table(self, name):
        if name in self.tables: return True
        filename = os.path.join(self.tb_dir, f"{name}.bin")
        if os.path.exists(filename):
            try:
                file_size = os.path.getsize(filename)
                # 3-man ~1MB, 4-man ~67MB
                expected_3man = 64 * 64 * 64 * 2 * 2
                expected_4man = 64 * 64 * 64 * 64 * 2 * 2

                if file_size == expected_3man:
                    self.tables[name] = np.memmap(filename, dtype=np.int16, mode='r', shape=(64, 64, 64, 2))
                    return True
                elif file_size == expected_4man:
                    self.tables[name] = np.memmap(filename, dtype=np.int16, mode='r', shape=(64, 64, 64, 64, 2))
                    return True
            except Exception as e:
                print(f"[TablebaseManager] Failed to memmap {name}: {e}")
        return False

    def _tb_score_to_ai_score(self, tb_val, is_win_for_white):
        """Converts DTM to internal engine score from White's perspective."""
        if tb_val == 0: return 0
        dtm = abs(int(tb_val))
        score = 1000000 - dtm
        return score if is_win_for_white else -score

    @staticmethod
    def _flat_idx_3(i0, i1, i2, i3):
        return (((i0 * 64 + i1) * 64 + i2) * 2 + i3)

    @staticmethod
    def _flat_idx_4(i0, i1, i2, i3, i4):
        return (((((i0 * 64 + i1) * 64 + i2) * 64 + i3) * 2) + i4)

    def probe(self, board, turn_to_move):
        w_objs = [p for p in board.white_pieces if not isinstance(p, King)]
        b_objs = [p for p in board.black_pieces if not isinstance(p, King)]
        wk, bk = board.white_king_pos, board.black_king_pos
        if not wk or not bk: return None

        t_idx = 0 if turn_to_move == 'white' else 1
        w_cnt, b_cnt = len(w_objs), len(b_objs)

        # --- 3-MAN PROBE ---
        if w_cnt == 1 and b_cnt == 0:
            p = w_objs[0]
            tb = f"K_{type(p).__name__}_K"
            if tb in self.tables:
                idx = self._flat_idx_3(wk[0]*8+wk[1], p.pos[0]*8+p.pos[1], bk[0]*8+bk[1], t_idx)
                val = int(self.tables[tb].flat[idx])
                is_win = (t_idx == 0 and val > 0) or (t_idx == 1 and val < 0)
                return self._tb_score_to_ai_score(val, is_win)

        elif b_cnt == 1 and w_cnt == 0:
            p = b_objs[0]
            tb = f"K_{type(p).__name__}_K"
            if tb in self.tables:
                # Flip everything to check from Black's win perspective
                idx = self._flat_idx_3(_flip(bk)[0]*8+_flip(bk)[1], _flip(p.pos)[0]*8+_flip(p.pos)[1], _flip(wk)[0]*8+_flip(wk)[1], 1 - t_idx)
                val = int(self.tables[tb].flat[idx])
                # If 'Black-as-attacker' wins, it's a loss for White
                b_wins = ((1 - t_idx) == 0 and val > 0) or ((1 - t_idx) == 1 and val < 0)
                return self._tb_score_to_ai_score(val, not b_wins)

        # --- 4-MAN SAME-SIDE PROBE ---
        elif w_cnt == 2 and b_cnt == 0:
            w_objs.sort(key=lambda x: (type(x).__name__, x.pos[0]*8+x.pos[1]))
            p1, p2 = w_objs
            tb = f"K_{type(p1).__name__}_{type(p2).__name__}_K"
            if tb in self.tables:
                idx = self._flat_idx_4(wk[0]*8+wk[1], p1.pos[0]*8+p1.pos[1], p2.pos[0]*8+p2.pos[1], bk[0]*8+bk[1], t_idx)
                val = int(self.tables[tb].flat[idx])
                is_win = (t_idx == 0 and val > 0) or (t_idx == 1 and val < 0)
                return self._tb_score_to_ai_score(val, is_win)

        elif b_cnt == 2 and w_cnt == 0:
            # Flip to Black's perspective, then sort canonically
            b_flipped = [(_flip(p.pos), type(p).__name__) for p in b_objs]
            b_flipped.sort(key=lambda x: (x[1], x[0][0]*8+x[0][1]))
            p1_pos, p1_name = b_flipped[0]
            p2_pos, p2_name = b_flipped[1]
            tb = f"K_{p1_name}_{p2_name}_K"
            if tb in self.tables:
                idx = self._flat_idx_4(_flip(bk)[0]*8+_flip(bk)[1], p1_pos[0]*8+p1_pos[1], p2_pos[0]*8+p2_pos[1], _flip(wk)[0]*8+_flip(wk)[1], 1-t_idx)
                val = int(self.tables[tb].flat[idx])
                b_wins = ((1-t_idx) == 0 and val > 0) or ((1-t_idx) == 1 and val < 0)
                return self._tb_score_to_ai_score(val, not b_wins)

        # --- 4-MAN CROSS PROBE ---
        elif w_cnt == 1 and b_cnt == 1:
            wp, bp = w_objs[0], b_objs[0]
            wn, bn = type(wp).__name__, type(bp).__name__

            if wn <= bn:
                tb = f"K_{wn}_vs_{bn}_K"
                if tb in self.tables:
                    idx = self._flat_idx_4(wk[0]*8+wk[1], wp.pos[0]*8+wp.pos[1], bk[0]*8+bk[1], bp.pos[0]*8+bp.pos[1], t_idx)
                    val = int(self.tables[tb].flat[idx])
                    is_win = (t_idx == 0 and val > 0) or (t_idx == 1 and val < 0)
                    return self._tb_score_to_ai_score(val, is_win)
            else:
                tb = f"K_{bn}_vs_{wn}_K"
                if tb in self.tables:
                    # Current board is White Queen vs Black Bishop, but file is K_Bishop_vs_Queen_K. 
                    # We must flip everything so Black Bishop becomes the 'White' attacker in the file.
                    idx = self._flat_idx_4(_flip(bk)[0]*8+_flip(bk)[1], _flip(bp.pos)[0]*8+_flip(bp.pos)[1], _flip(wk)[0]*8+_flip(wk)[1], _flip(wp.pos)[0]*8+_flip(wp.pos)[1], 1-t_idx)
                    val = int(self.tables[tb].flat[idx])
                    # If the Bishop side (currently Black) wins in the file, it's a loss for White
                    b_wins = ((1-t_idx) == 0 and val > 0) or ((1-t_idx) == 1 and val < 0)
                    return self._tb_score_to_ai_score(val, not b_wins)

        return None