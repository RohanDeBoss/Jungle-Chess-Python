# TablebaseManager.py (Fixed 01)
import os
import numpy as np
from GameLogic import King, Board

class TablebaseManager:
    def __init__(self):
        self.tables = {}
        self.tb_dir = "tablebases"
        self.pre_load_all()

    def pre_load_all(self):
        pieces =['Queen', 'Rook', 'Knight', 'Bishop', 'Pawn']
        for p in pieces:
            self.load_table(p)

    def load_table(self, piece_name):
        filename = os.path.join(self.tb_dir, f"K_{piece_name}_K.bin")
        if os.path.exists(filename):
            try:
                # Support both legacy int8 and strict-DTM int16 files.
                expected = 64 * 64 * 64 * 2

                data16 = np.fromfile(filename, dtype=np.int16)
                if data16.size == expected:
                    self.tables[piece_name] = data16.reshape((64, 64, 64, 2))
                    return True

                data8 = np.fromfile(filename, dtype=np.int8)
                if data8.size == expected:
                    self.tables[piece_name] = data8.astype(np.int16).reshape((64, 64, 64, 2))
                    return True

                return False
            except Exception:
                pass
        return False

    def _to_white_perspective(self, signed_tb_score, turn):
        """
        Convert a table score into white-centric absolute evaluation.

        Tablebase sign semantics:
        - Positive: side to move can force a win
        - Negative: side to move is losing
        - Zero: draw/unknown
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
        
        if white_count + black_count != 3:
            return None

        # Case 1: White Attacking
        if white_count == 2 and black_count == 1:
            try:
                wp = next(p for p in board.white_pieces if not isinstance(p, King))
                wk, bk = board.white_king_pos, board.black_king_pos
                p_name = wp.__class__.__name__
                if p_name not in self.tables: return None
                
                t_idx = 0 if turn == 'white' else 1
                idx = (wk[0]*8+wk[1], wp.pos[0]*8+wp.pos[1], bk[0]*8+bk[1], t_idx)
                score = int(self.tables[p_name][idx])
                return self._to_white_perspective(score, turn)
            except Exception:
                return None

        # Case 2: Black Attacking
        elif black_count == 2 and white_count == 1:
            try:
                bp = next(p for p in board.black_pieces if not isinstance(p, King))
                bk, wk = board.black_king_pos, board.white_king_pos
                p_name = bp.__class__.__name__
                if p_name not in self.tables: return None
                
                def flip(p): return (7-p[0], p[1])
                # We mirror Black-attacker positions into the White-attacker table frame.
                t_idx = 0 if turn == 'black' else 1
                idx = (flip(bk)[0]*8+flip(bk)[1], flip(bp.pos)[0]*8+flip(bp.pos)[1], flip(wk)[0]*8+flip(wk)[1], t_idx)
                score = int(self.tables[p_name][idx])
                return self._to_white_perspective(score, turn)
            except Exception:
                return None

        return None
