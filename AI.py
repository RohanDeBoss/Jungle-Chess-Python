# AI.py (v5.7 - Transposition Table Fix)

import time
from GameLogic import * # Uses the new, robust gamelogic
import random
import threading
from collections import namedtuple

# --- Versioning ---
# v5.7 (Transposition Table Fix)
# - Removed `self.tt.clear()` from the `make_move` method.
# - The TT now correctly persists throughout an entire game, significantly improving
#   search efficiency on subsequent moves, especially at higher depths.
# - The TT is still correctly reset for each new game because a new `ChessBot`
#   instance is created in the UI's `reset_game` function.

# --- Global Zobrist Hashing & TT Setup (No changes) ---
def initialize_zobrist_table():
    random.seed(42)
    table = {(r, c, p_t, c_t): random.getrandbits(64) for r in range(ROWS) for c in range(COLS) for p_t in [Pawn, Knight, Bishop, Rook, Queen, King, None] for c_t in ['white', 'black', None]}
    table['turn'] = random.getrandbits(64)
    return table
ZOBRIST_TABLE = initialize_zobrist_table()

def board_hash(board, turn):
    h = 0
    for r in range(ROWS):
        for c in range(COLS):
            p = board.grid[r][c]
            k = (r, c, type(p) if p else None, p.color if p else None)
            h ^= ZOBRIST_TABLE.get(k, 0)
    if turn == 'black': h ^= ZOBRIST_TABLE['turn']
    return h

TTEntry = namedtuple('TTEntry', ['score', 'depth', 'flag', 'best_move'])
TT_FLAG_EXACT = 0; TT_FLAG_LOWERBOUND = 1; TT_FLAG_UPPERBOUND = 2
class SearchCancelledException(Exception): pass

class ChessBot:
    search_depth = 3
    MATE_SCORE = 1000000
    DRAW_SCORE = 0
    DRAW_PENALTY = -10
    MAX_Q_SEARCH_DEPTH = 8
    BONUS_PV_MOVE = 2_000_000; BONUS_GOOD_CAPTURE = 1_500_000; BONUS_PROMOTION = 1_200_000
    BONUS_CHECKING_MOVE = 1_000_000; BONUS_KILLER_1 = 900_000; BONUS_KILLER_2 = 850_000
    BONUS_LOSING_CAPTURE = 300_000
    LMR_DEPTH_THRESHOLD = 3; LMR_MOVE_COUNT_THRESHOLD = 4; LMR_REDUCTION = 1

    def __init__(self, board, color, app, cancellation_event, bot_name="AI Bot"):
        self.board = board
        self.color = color
        self.opponent_color = 'black' if color == 'white' else 'white'
        self.app = app
        self.cancellation_event = cancellation_event
        self.bot_name = bot_name
        self.tt = {}
        self.nodes_searched = 0
        self.MAX_PLY_KILLERS = 30
        self.killer_moves = [[None, None] for _ in range(self.MAX_PLY_KILLERS)]
        self.history_heuristic_table = [[[0 for _ in range(ROWS*COLS)] for _ in range(ROWS*COLS)] for _ in range(2)]

    def _is_checking_move(self, board, move, turn):
        sim_board = board.clone()
        sim_board.make_move(move[0], move[1])
        return is_in_check(sim_board, 'black' if turn == 'white' else 'white')

    def is_endgame(self, board):
        material = sum(PIECE_VALUES.get(type(p), 0) for r in board.grid for p in r if p and not isinstance(p, King))
        return material < ENDGAME_MATERIAL_THRESHOLD
    
    def evaluate_board(self, board, turn):
        score = 0
        in_endgame = self.is_endgame(board)
        for r in range(ROWS):
            for c in range(COLS):
                piece = board.grid[r][c]
                if not piece: continue
                value = PIECE_VALUES.get(type(piece), 0)
                pst = PIECE_SQUARE_TABLES['king_endgame'] if isinstance(piece, King) and in_endgame else PIECE_SQUARE_TABLES.get(type(piece))
                if pst:
                    pst_score = pst[r][c] if piece.color == 'white' else pst[ROWS - 1 - r][c]
                    value += pst_score
                if piece.color == turn:
                    score += value
                else:
                    score -= value
        return score

    def order_moves(self, board, moves, ply, hash_move=None, in_qsearch=False):
        if not moves: return []
        move_scores = {}
        moving_color = board.grid[moves[0][0][0]][moves[0][0][1]].color
        color_index = 0 if moving_color == 'white' else 1
        killers = self.killer_moves[ply] if ply < self.MAX_PLY_KILLERS else [None, None]
        for move in moves:
            score = 0
            start_pos, end_pos = move
            moving_piece = board.grid[start_pos[0]][start_pos[1]]
            target_piece = board.grid[end_pos[0]][end_pos[1]]
            is_capture = target_piece is not None
            is_promotion = isinstance(moving_piece, Pawn) and (end_pos[0] == 0 or end_pos[0] == ROWS - 1)

            if move == hash_move: score = self.BONUS_PV_MOVE
            elif is_capture:
                mvv_lva_score = PIECE_VALUES.get(type(target_piece), 0) - PIECE_VALUES.get(type(moving_piece), 0)
                score = (self.BONUS_GOOD_CAPTURE if mvv_lva_score >= 0 else self.BONUS_LOSING_CAPTURE) + mvv_lva_score
            elif is_promotion: score = self.BONUS_PROMOTION
            elif in_qsearch and self._is_checking_move(board, move, moving_color):
                score = self.BONUS_CHECKING_MOVE
            elif move in killers: score = self.BONUS_KILLER_1 if move == killers[0] else self.BONUS_KILLER_2
            else:
                from_sq, to_sq = start_pos[0] * COLS + start_pos[1], end_pos[0] * COLS + end_pos[1]
                score = self.history_heuristic_table[color_index][from_sq][to_sq]
            move_scores[move] = score
        return sorted(moves, key=lambda m: move_scores.get(m, 0), reverse=True)

    def qsearch(self, board, alpha, beta, turn, ply):
        self.nodes_searched += 1
        if self.cancellation_event.is_set(): raise SearchCancelledException()
        if ply >= self.MAX_Q_SEARCH_DEPTH: return self.evaluate_board(board, turn)

        stand_pat = self.evaluate_board(board, turn)
        if stand_pat >= beta: return beta
        alpha = max(alpha, stand_pat)

        legal_moves = get_all_legal_moves(board, turn)
        tactical_moves = [m for m in legal_moves if board.grid[m[1][0]][m[1][1]] or (isinstance(board.grid[m[0][0]][m[0][1]], Pawn) and (m[1][0] in [0, ROWS - 1])) or self._is_checking_move(board, m, turn)]
        
        ordered_tactical_moves = self.order_moves(board, tactical_moves, ply, in_qsearch=True)
        for move in ordered_tactical_moves:
            child_board = board.clone(); child_board.make_move(move[0], move[1])
            score = -self.qsearch(child_board, -beta, -alpha, 'black' if turn == 'white' else 'white', ply + 1)
            if score >= beta: return beta
            alpha = max(alpha, score)
        return alpha

    def negamax(self, board, depth, alpha, beta, turn, ply, search_path):
        self.nodes_searched += 1
        if self.cancellation_event.is_set(): raise SearchCancelledException()
        hash_val = board_hash(board, turn)
        if hash_val in search_path: return self.DRAW_PENALTY

        original_alpha = alpha
        tt_entry = self.tt.get(hash_val)
        if tt_entry and tt_entry.depth >= depth:
            if tt_entry.flag == TT_FLAG_EXACT: return tt_entry.score
            elif tt_entry.flag == TT_FLAG_LOWERBOUND: alpha = max(alpha, tt_entry.score)
            elif tt_entry.flag == TT_FLAG_UPPERBOUND: beta = min(beta, tt_entry.score)
            if alpha >= beta: return tt_entry.score

        if depth <= 0: return self.qsearch(board, alpha, beta, turn, ply)
        if ply > 0 and self.app.position_counts.get(hash_val, 0) >= 2: return self.DRAW_PENALTY

        opponent_turn = 'black' if turn == 'white' else 'white'
        is_check_now = is_in_check(board, turn)
        if is_check_now: depth += 1

        legal_moves = get_all_legal_moves(board, turn)
        if not legal_moves:
            return -self.MATE_SCORE + ply if is_check_now else self.DRAW_SCORE

        hash_move = tt_entry.best_move if tt_entry else None
        ordered_moves = self.order_moves(board, legal_moves, ply, hash_move)
        best_move_for_node = None
        new_search_path = search_path | {hash_val}

        for i, move in enumerate(ordered_moves):
            child_board = board.clone(); child_board.make_move(move[0], move[1])
            child_hash = board_hash(child_board, opponent_turn)
            score = 0
            if child_hash in new_search_path or self.app.position_counts.get(child_hash, 0) >= 2:
                score = self.DRAW_PENALTY
            else:
                self.app.position_counts[child_hash] = self.app.position_counts.get(child_hash, 0) + 1
                reduction = 0
                if depth >= self.LMR_DEPTH_THRESHOLD and i >= self.LMR_MOVE_COUNT_THRESHOLD and not is_check_now and not board.grid[move[1][0]][move[1][1]]:
                    reduction = self.LMR_REDUCTION
                score = -self.negamax(child_board, depth - 1 - reduction, -beta, -alpha, opponent_turn, ply + 1, new_search_path)
                if reduction > 0 and score > alpha:
                    score = -self.negamax(child_board, depth - 1, -beta, -alpha, opponent_turn, ply + 1, new_search_path)
                self.app.position_counts[child_hash] -= 1

            if score > alpha:
                alpha = score; best_move_for_node = move
            if alpha >= beta:
                if not board.grid[move[1][0]][move[1][1]]:
                    if ply < self.MAX_PLY_KILLERS and self.killer_moves[ply][0] != move:
                        self.killer_moves[ply][1] = self.killer_moves[ply][0]; self.killer_moves[ply][0] = move
                    color_index = 0 if turn == 'white' else 1
                    from_sq, to_sq = move[0][0]*COLS+move[0][1], move[1][0]*COLS+move[1][1]
                    self.history_heuristic_table[color_index][from_sq][to_sq] += depth*depth
                self.tt[hash_val] = TTEntry(beta, depth, TT_FLAG_LOWERBOUND, move)
                return beta
        flag = TT_FLAG_EXACT if alpha > original_alpha else TT_FLAG_UPPERBOUND
        self.tt[hash_val] = TTEntry(alpha, depth, flag, best_move_for_node)
        return alpha

    def make_move(self):
        try:
            best_move_overall = None
            # self.tt.clear() # --- THIS LINE IS REMOVED ---
            self.killer_moves = [[None, None] for _ in range(self.MAX_PLY_KILLERS)]
            for i in range(2):
                for j in range(ROWS*COLS):
                    for k in range(ROWS*COLS): self.history_heuristic_table[i][j][k] //= 2
            
            root_moves = get_all_legal_moves(self.board, self.color)
            if not root_moves: return False
            best_move_overall = root_moves[0]
            root_hash = board_hash(self.board, self.color)

            for current_depth in range(1, self.search_depth + 1):
                iter_start_time = time.time(); self.nodes_searched = 0
                best_score_this_iter, best_move_this_iter = -float('inf'), None
                alpha, beta = -float('inf'), float('inf')
                ordered_root_moves = self.order_moves(self.board, root_moves, 0, best_move_overall)

                for move in ordered_root_moves:
                    if self.cancellation_event.is_set(): raise SearchCancelledException()
                    child_board = self.board.clone(); child_board.make_move(move[0], move[1])
                    child_hash = board_hash(child_board, self.opponent_color)
                    self.app.position_counts[child_hash] = self.app.position_counts.get(child_hash, 0) + 1
                    
                    search_path = {root_hash}
                    score = -self.negamax(child_board, current_depth - 1, -beta, -alpha, self.opponent_color, 1, search_path)
                    
                    self.app.position_counts[child_hash] -= 1

                    if score > best_score_this_iter:
                        best_score_this_iter = score; best_move_this_iter = move
                    alpha = max(alpha, best_score_this_iter)
                
                if not self.cancellation_event.is_set():
                    best_move_overall = best_move_this_iter
                    iter_duration = time.time() - iter_start_time
                    knps = (self.nodes_searched / iter_duration / 1000) if iter_duration > 0 else 0
                    eval_for_ui = best_score_this_iter if self.color == 'white' else -best_score_this_iter

                    if self.app:
                        self.app.log_queue.put(f"  > {self.bot_name} (Depth {current_depth}): Eval={eval_for_ui/100:+.2f}, Nodes={self.nodes_searched}, KNPS={knps:.1f}")
                        self.app.master.after(0, self.app.draw_eval_bar, eval_for_ui)
            
            if best_move_overall:
                self.board.make_move(best_move_overall[0], best_move_overall[1])
                return True
            return False
            
        except SearchCancelledException:
            print(f"AI ({self.color}): Search cancelled.")
            return False
        
# -----------------------------------------------------------------------------
# Piece-Square Tables (PSTs) and Material Values for this Variant
# -----------------------------------------------------------------------------

# Values reflect the destructive power of the pieces
PIECE_VALUES = {
    Pawn: 100, Knight: 750, Bishop: 700,
    Rook: 650, Queen: 1000, King: 30000
}
ENDGAME_MATERIAL_THRESHOLD = (PIECE_VALUES[Rook] * 2 + PIECE_VALUES[Bishop] + PIECE_VALUES[Knight])

# FLIPPED: High values are now at the top (low row index), for White's perspective of advancing.
pawn_pst = [
    [0, 0, 0, 0, 0, 0, 0, 0], # Promotion rank handled by move value
    [100, 100, 100, 100, 100, 100, 100, 100],
    [50, 50, 50, 50, 50, 50, 50, 50],
    [30, 30, 40, 50, 50, 40, 30, 30],
    [20, 20, 30, 40, 40, 30, 20, 20],
    [10, 10, 20, 30, 30, 20, 10, 10],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0]
]

# FLIPPED: Penalties are now on the back rank (high row index) for White.
knight_pst = [
    [-50, -40, -30, -30, -30, -30, -40, -50],
    [-40, -20,   5,  10,  10,   5, -20, -40],
    [-30,   5,  20,  25,  25,  20,   5, -30],
    [-30,  10,  25,  35,  35,  25,  10, -30],
    [-30,  10,  25,  35,  35,  25,  10, -30],
    [-30,  10,  20,  25,  25,  20,  10, -30],
    [-40, -20,   5,  10,  10,   5, -20, -40],
    [-50, -40, -30, -30, -30, -30, -40, -50]
]

# FLIPPED:
bishop_pst = [
    [-20, -10, -10, -10, -10, -10, -10, -20],
    [-10,   0,   0,   0,   0,   0,   0, -10],
    [-10,   0,   5,  10,  10,   5,   0, -10],
    [-10,   5,   5,  10,  10,   5,   5, -10],
    [-10,   0,  10,  10,  10,  10,   0, -10],
    [-10,  10,  10,  10,  10,  10,  10, -10],
    [-10,   5,   0,   0,   0,   0,   5, -10],
    [-20, -10, -10, -10, -10, -10, -10, -20]
]

# FLIPPED:
rook_pst = [
    [  0,   0,   0,  10,  10,   0,   0,   0],
    [  5,  10,  10,  20,  20,  10,  10,   5],
    [ -5,   0,   0,   5,   5,   0,   0,  -5],
    [ -5,   0,   0,   5,   5,   0,   0,  -5],
    [ -5,   0,   0,   5,   5,   0,   0,  -5],
    [ -5,   0,   0,   5,   5,   0,   0,  -5],
    [ -5,   0,   0,   5,   5,   0,   0,  -5],
    [  0,   5,   5,  10,  10,   5,   5,   0]
]

# FLIPPED:
queen_pst = [
    [-20, -10, -10,  -5,  -5, -10, -10, -20],
    [-10,   0,   0,   0,   0,   0,   0, -10],
    [-10,   0,   5,   5,   5,   5,   0, -10],
    [ -5,   0,   5,   5,   5,   5,   0,  -5],
    [  0,   0,   5,   5,   5,   5,   0,  -5],
    [-10,   5,   5,   5,   5,   5,   5, -10],
    [-10,   0,   5,   0,   0,   0,   5, -10],
    [-20, -10, -10,  -5,  -5, -10, -10, -20]
]

# FLIPPED: Now correctly rewards king safety on the back rank for White.
king_midgame_pst = [
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-20, -30, -30, -40, -40, -30, -30, -20],
    [-10, -20, -20, -20, -20, -20, -20, -10],
    [ 20,  20,   0,   0,   0,   0,  20,  20],
    [ 20,  30,  10,   0,   0,  10,  30,  20]
]

# FLIPPED: Now correctly encourages the king to centralize from its starting side.
king_endgame_pst = [
    [-50, -40, -30, -20, -20, -30, -40, -50],
    [-30, -20, -10,   0,   0, -10, -20, -30],
    [-30, -10,  20,  30,  30,  20, -10, -30],
    [-30, -10,  30,  40,  40,  30, -10, -30],
    [-30, -10,  30,  40,  40,  30, -10, -30],
    [-30, -10,  20,  30,  30,  20, -10, -30],
    [-30, -30,   0,   0,   0,   0, -30, -30],
    [-50, -30, -30, -30, -30, -30, -30, -50]
]

PIECE_SQUARE_TABLES = {
    Pawn: pawn_pst, Knight: knight_pst, Bishop: bishop_pst, Rook: rook_pst, 
    Queen: queen_pst, 'king_midgame': king_midgame_pst, 'king_endgame': king_endgame_pst
}