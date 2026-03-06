# TablebaseManager.py (v4.0 - Full 3-Man, 4-Man Same, & 4-Man Cross Support)

import os
import numpy as np
from GameLogic import King, Board

class TablebaseManager:
    def __init__(self):
        self.tables = {}
        self.tb_dir = "tablebases"
        self.pre_load_all()

    def pre_load_all(self):
        pieces = ['Queen', 'Rook', 'Knight', 'Bishop', 'Pawn']
        
        # Load 3-man (5 tables)
        for p in pieces:
            self.load_table(f"K_{p}_K")
            
        # Load 4-man Same-Side (15 tables)
        from itertools import combinations_with_replacement
        for p1, p2 in combinations_with_replacement(pieces, 2):
            names = sorted([p1, p2])
            self.load_table(f"K_{names[0]}_{names[1]}_K")
            
        # Load 4-man Cross (25 tables)
        for w in pieces:
            for b in pieces:
                self.load_table(f"K_{w}_vs_{b}_K")

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

    def _tb_score_to_ai_score(self, tb_val, attacker_color):
        """
        Converts a Tablebase DTM score into an absolute AI score from White's perspective.
        In the Tablebase, a non-zero value means the attacker wins. 
        """
        if tb_val == 0:
            return 0
        
        dtm = abs(int(tb_val))
        ai_score = 1000000 - dtm
        
        return ai_score if attacker_color == 'white' else -ai_score

    def _flat_idx_raw_4(self, i0, i1, i2, i3, i4):
        return (((((i0 * 64 + i1) * 64 + i2) * 64 + i3) * 2) + i4)

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
                    return self._tb_score_to_ai_score(score, 'white')
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
                    return self._tb_score_to_ai_score(score, 'black')
                except Exception:
                    return None
                    
        # --- 4-Man Probe ---
        elif white_count + black_count == 4:
            
            # 1. 4-Man Same Side (White)
            if white_count == 3 and black_count == 1:
                try:
                    w_p = [p for p in board.white_pieces if not isinstance(p, King)]
                    w_p.sort(key=lambda x: type(x).__name__) 
                    if type(w_p[0]) == type(w_p[1]):
                        w_p.sort(key=lambda p: p.pos[0]*8 + p.pos[1])
                    
                    p_name = f"K_{type(w_p[0]).__name__}_{type(w_p[1]).__name__}_K"
                    if p_name not in self.tables: return None
                    
                    t_idx = 0 if turn == 'white' else 1
                    wk, bk = board.white_king_pos, board.black_king_pos
                    
                    idx = self._flat_idx_raw_4(wk[0]*8+wk[1], w_p[0].pos[0]*8+w_p[0].pos[1], w_p[1].pos[0]*8+w_p[1].pos[1], bk[0]*8+bk[1], t_idx)
                    score = int(self.tables[p_name].flat[idx])
                    return self._tb_score_to_ai_score(score, 'white')
                except Exception:
                    return None
                    
            # 2. 4-Man Same Side (Black)
            elif black_count == 3 and white_count == 1:
                try:
                    b_p = [p for p in board.black_pieces if not isinstance(p, King)]
                    b_p.sort(key=lambda x: type(x).__name__)
                    def flip(p): return (7-p[0], p[1])
                    if type(b_p[0]) == type(b_p[1]):
                        b_p.sort(key=lambda p: flip(p.pos)[0]*8 + flip(p.pos)[1])
                    
                    p_name = f"K_{type(b_p[0]).__name__}_{type(b_p[1]).__name__}_K"
                    if p_name not in self.tables: return None
                    
                    t_idx = 0 if turn == 'black' else 1
                    bk, wk = board.black_king_pos, board.white_king_pos
                    
                    idx = self._flat_idx_raw_4(flip(bk)[0]*8+flip(bk)[1], flip(b_p[0].pos)[0]*8+flip(b_p[0].pos)[1], flip(b_p[1].pos)[0]*8+flip(b_p[1].pos)[1], flip(wk)[0]*8+flip(wk)[1], t_idx)
                    score = int(self.tables[p_name].flat[idx])
                    return self._tb_score_to_ai_score(score, 'black')
                except Exception:
                    return None
                    
            # 3. 4-Man Cross Tables (White Piece vs Black Piece)
            elif white_count == 2 and black_count == 2:
                try:
                    w_p = next(p for p in board.white_pieces if not isinstance(p, King))
                    b_p = next(p for p in board.black_pieces if not isinstance(p, King))
                    
                    w_name = w_p.__class__.__name__
                    b_name = b_p.__class__.__name__
                    t_name_w = f"K_{w_name}_vs_{b_name}_K"
                    t_name_b = f"K_{b_name}_vs_{w_name}_K"
                    
                    # Ensure both reciprocal tables exist so we don't hallucinate a draw
                    if t_name_w not in self.tables or t_name_b not in self.tables:
                        return None
                        
                    wk = board.white_king_pos
                    wp = w_p.pos
                    bk = board.black_king_pos
                    bp = b_p.pos
                    
                    # Check if White is winning
                    t_idx_w = 0 if turn == 'white' else 1
                    idx_w = self._flat_idx_raw_4(wk[0]*8+wk[1], wp[0]*8+wp[1], bk[0]*8+bk[1], bp[0]*8+bp[1], t_idx_w)
                    score_w = int(self.tables[t_name_w].flat[idx_w])
                    
                    if score_w != 0:
                        return self._tb_score_to_ai_score(score_w, 'white')
                        
                    # Check if Black is winning (flip spatial coordinates and invert attacker logic)
                    def flip(p): return (7-p[0], p[1])
                    t_idx_b = 0 if turn == 'black' else 1
                    idx_b = self._flat_idx_raw_4(flip(bk)[0]*8+flip(bk)[1], flip(bp)[0]*8+flip(bp)[1], flip(wk)[0]*8+flip(wk)[1], flip(wp)[0]*8+flip(wp)[1], t_idx_b)
                    score_b = int(self.tables[t_name_b].flat[idx_b])
                    
                    if score_b != 0:
                        return self._tb_score_to_ai_score(score_b, 'black')
                        
                    # If neither side wins, it is mathematically proven to be a dead draw
                    return 0
                    
                except Exception:
                    return None

        return None