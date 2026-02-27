import os
import numpy as np
# Import the piece classes so 'King' is recognized
from GameLogic import King, Board

class TablebaseManager:
    def __init__(self):
        self.tables = {}
        # The directory where the generator saves files
        self.tb_dir = "tablebases"

    def load_table(self, piece_name):
        # Match the generator's naming: K_PieceName_K.bin
        filename = os.path.join(self.tb_dir, f"K_{piece_name}_K.bin")
        
        if os.path.exists(filename):
            try:
                # Use memmap for efficiency so we don't hog RAM until we actually probe
                data = np.fromfile(filename, dtype=np.int8)
                # Reshape to (WK, WP, BK, Turn)
                self.tables[piece_name] = data.reshape((64, 64, 64, 2))
                print(f"Loaded Tablebase: {piece_name}")
                return True
            except Exception as e:
                print(f"Error loading tablebase {filename}: {e}")
                return False
        return False

    def probe(self, board, turn):
        white_count = len(board.white_pieces)
        black_count = len(board.black_pieces)
        
        if white_count + black_count != 3:
            return None

        # Case 1: White has King + Piece vs Black King
        if white_count == 2 and black_count == 1:
            try:
                wp = next(p for p in board.white_pieces if not isinstance(p, King))
                wk = board.white_king_pos
                bk = board.black_king_pos
                
                p_name = wp.__class__.__name__
                if p_name not in self.tables:
                    if not self.load_table(p_name): return None
                
                t_idx = 0 if turn == 'white' else 1
                idx = (wk[0]*8+wk[1], wp.pos[0]*8+wp.pos[1], bk[0]*8+bk[1], t_idx)
                
                score = self.tables[p_name][idx]
                return self._process_score(score)
            except (StopIteration, AttributeError, IndexError):
                return None

        # Case 2: Black has King + Piece vs White King
        elif black_count == 2 and white_count == 1:
            try:
                bp = next(p for p in board.black_pieces if not isinstance(p, King))
                bk = board.black_king_pos
                wk = board.white_king_pos
                
                p_name = bp.__class__.__name__
                if p_name not in self.tables:
                    if not self.load_table(p_name): return None
                
                # Flip coordinates to treat Black as White (Attacker)
                # This variant uses row 0-7, so 7-r flips the board perspective
                def flip_sq(pos): return (7-pos[0], pos[1])
                
                f_bk = flip_sq(bk)
                f_bp = flip_sq(bp.pos)
                f_wk = flip_sq(wk)
                
                t_idx = 0 if turn == 'black' else 1
                idx = (f_bk[0]*8+f_bk[1], f_bp[0]*8+f_bp[1], f_wk[0]*8+f_wk[1], t_idx)
                
                score = self.tables[p_name][idx]
                
                processed = self._process_score(score)
                # If the table says "White wins", and Black is the one using the table, it means Black wins.
                # So we return the value inverted for the engine's white-centric evaluation.
                return -processed if processed is not None else None
            except (StopIteration, AttributeError, IndexError):
                return None

        return None

    def _process_score(self, score):
        if score == 0: return 0 
        
        # We use 1,000,000 as the mate score to match ChessBot.MATE_SCORE
        # score is DTM (Distance to Mate in plies)
        if score > 0:
            return 1000000 - int(score)
        else:
            # score is negative, so -1000000 - (-5) = -999,995
            return -1000000 - int(score)